import os
import torch
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from features import extract_key_features
from prompt_builder import build_prompt, agent_1, agent_2, generate_description
from utils.utils import encode_features, encode_descriptions
from args import parameter_parser
from contextlib import nullcontext

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Configure environment settings and device"""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = parameter_parser()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return args, device

def load_models(device):
    """Load LLM and embedding models"""
    try:
        # Load tokenizer and LLM model
        logger.info("Loading tokenizer and language model...")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/zephyr-7b-alpha", 
            device_map="auto", 
            low_cpu_mem_usage=True, 
            trust_remote_code=True, 
            offload_buffers=True
        )
        
        # Load sentence embedding model
        logger.info("Loading sentence embedding model...")
        encoder_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        return tokenizer, model, encoder_model
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

def generate_feature_embeddings(df, dataset_name, out_path):
    """Generate feature embeddings for network traffic data"""
    out_feat_emb = f"{out_path}/feature_embeddings.npy"
    
    if os.path.exists(out_feat_emb):
        logger.info(f"Feature embeddings file exists, loading: {out_feat_emb}")
        return np.load(out_feat_emb)
    
    logger.info("Generating feature embeddings...")
    feature_embeddings = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating embeddings"):
        try:
            features = extract_key_features(dataset_name, row)
            feature_embedding = encode_features(features)
            feature_embeddings.append(feature_embedding)
        except Exception as e:
            logger.warning(f"Error processing row: {str(e)}")
            # Add zero vector as placeholder for failed rows
            feature_embeddings.append(np.zeros_like(feature_embeddings[0]) if feature_embeddings else np.zeros(768))
    
    feature_embeddings = np.array(feature_embeddings)
    np.save(out_feat_emb, feature_embeddings)
    logger.info(f"Feature embeddings saved to: {out_feat_emb}")
    return feature_embeddings

def process_batch(batch_rows, dataset_name, model, tokenizer):
    """Process a batch of rows to generate descriptions"""
    batch_descriptions = []
    batch_feature_embeddings = []
    
    for _, row in batch_rows.iterrows():
        try:
            features = extract_key_features(dataset_name, row)
            feature_embedding = encode_features(features)
            batch_feature_embeddings.append(feature_embedding)
            
            # Generate descriptions using both agents
            prompt_1 = agent_1(dataset_name, features)
            prompt_2 = agent_2(dataset_name, features)
            
            description_1 = generate_description(model, tokenizer, prompt_1)
            description_2 = generate_description(model, tokenizer, prompt_2)
            
            # Combine both descriptions
            combined_description = f"Agent 1 Analysis: {description_1}\n\nAgent 2 Analysis: {description_2}"
            batch_descriptions.append(combined_description)
            
            logger.debug(f"Generated description: {combined_description[:100]}...")
        except Exception as e:
            logger.warning(f"Error generating description: {str(e)}")
            batch_descriptions.append("Error generating description")
            batch_feature_embeddings.append(np.zeros(768))
    
    return batch_descriptions, batch_feature_embeddings

def generate_traffic_descriptions(df, dataset_name, model, tokenizer, batch_size=10):
    """Generate descriptions for all traffic data with batch processing"""
    descriptions = []
    feature_embeddings = []
    df["description"] = ""
    
    # Use tqdm for progress tracking
    with tqdm(total=len(df), desc="Generating traffic descriptions") as pbar:
        for i in range(0, len(df), batch_size):
            batch_rows = df.iloc[i:i + batch_size]
            
            # Use torch.cuda.amp.autocast() for mixed precision if on GPU
            ctx = torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext()
            with ctx:
                batch_descriptions, batch_feature_embeddings = process_batch(
                    batch_rows, dataset_name, model, tokenizer
                )
            
            descriptions.extend(batch_descriptions)
            feature_embeddings.extend(batch_feature_embeddings)
            
            # Update DataFrame with new descriptions
            idx_end = min(i + len(batch_descriptions), len(df))
            df.iloc[i:idx_end, df.columns.get_loc("description")] = batch_descriptions
            
            # Clean up CUDA cache periodically
            if torch.cuda.is_available() and i % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
                
            pbar.update(len(batch_rows))
            logger.info(f"Completed batch {i // batch_size + 1}/{(len(df) + batch_size - 1) // batch_size}")
    
    return df, descriptions, np.array(feature_embeddings)

def main():
    # Setup
    args, device = setup_environment()
    dataset_name = args.dataset_name
    
    # Define paths
    out_path = f"datasets/{dataset_name}/outputs"
    os.makedirs(out_path, exist_ok=True)
    data_path = f"{out_path}/text_data.csv"
    out_desc = f"{out_path}/descriptions-concise-pro.csv"
    out_emb = f"{out_path}/embeddings-concise-pro.npy"
    out_feat_emb = f"{out_path}/feature_embeddings.npy"
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows from {data_path}")
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return
    
    # Load models
    tokenizer, model, encoder_model = load_models(device)
    
    # Check if descriptions already exist
    if os.path.exists(out_desc):
        logger.info(f"Descriptions file exists, loading: {out_desc}")
        descriptions = pd.read_csv(out_desc)['description'].tolist()
        
        # Generate embeddings if they don't exist
        if not os.path.exists(out_emb):
            logger.info("Encoding descriptions to embeddings...")
            embeddings = encode_descriptions(descriptions)
            np.save(out_emb, embeddings)
            logger.info(f"Embeddings saved to: {out_emb}")
            
        # Generate feature embeddings if they don't exist
        if not os.path.exists(out_feat_emb):
            generate_feature_embeddings(df, dataset_name, out_path)
    else:
        # Generate descriptions and feature embeddings
        logger.info("Generating traffic descriptions and feature embeddings...")
        df, descriptions, feature_embeddings = generate_traffic_descriptions(
            df, dataset_name, model, tokenizer, batch_size=10
        )
        
        # Save descriptions
        df.to_csv(out_desc, index=False)
        logger.info(f"Descriptions saved to: {out_desc}")
        
        # Save embeddings
        logger.info("Encoding descriptions to embeddings...")
        embeddings = encode_descriptions(descriptions)
        np.save(out_emb, embeddings)
        logger.info(f"Embeddings saved to: {out_emb}")
        
        # Save feature embeddings
        np.save(out_feat_emb, feature_embeddings)
        logger.info(f"Feature embeddings saved to: {out_feat_emb}")
    
    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main()