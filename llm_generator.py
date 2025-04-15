import os
import random
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from features import extract_key_features
from prompt_builder import build_prompt, build_prompt_protocol, generate_description
import pandas as pd
import numpy as np
from args import parameter_parser
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel,AutoTokenizer, AutoModelForCausalLM
from utils.utils import encode_features, encode_descriptions, retrieve_similar_flows

args = parameter_parser()
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
print(device)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# mistralai/Mistral-7B-Instruct-v0.2
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True) # {"": device}
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")  # QwQ-32B  Mistral-7B-Instruct-v0.3 google/gemma-3-27b-it  HuggingFaceH4/zephyr-7b-alpha TurkuNLP/gpt3-finnish-small
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-alpha", device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True, offload_buffers=True)
encoder_model = SentenceTransformer("all-MiniLM-L6-v2")


# texthead = args.texthead # train_sapmles
dataset_name = args.dataset_name  # CICIDS DoHBrw TONIoT
out_path = f"datasets/{dataset_name}/outputs"
DATA_PATH = f"{out_path}/text_data.csv"
OUT_DESC = f"{out_path}/descriptions-concise-pro.csv"
OUT_EMB = f"{out_path}/embeddings-concise-pro.npy"
OUT_FEAT_EMB = f"{out_path}/feature_embeddings.npy"  # 新增：保存特征嵌入的文件

df = pd.read_csv(DATA_PATH)

if os.path.exists(OUT_FEAT_EMB):
    print(f"特征嵌入文件已存在，直接加载: {OUT_FEAT_EMB}")
    feature_embeddings = np.load(OUT_FEAT_EMB)
else:
    print("开始生成特征嵌入...")
    feature_embeddings = []
    for _, row in df.iterrows():
        features = extract_key_features(dataset_name, row)
        feature_embedding = encode_features(features)
        feature_embeddings.append(feature_embedding)
    feature_embeddings = np.array(feature_embeddings)
    np.save(OUT_FEAT_EMB, feature_embeddings)
    print("特征嵌入生成完成，保存至:", OUT_FEAT_EMB)

if os.path.exists(OUT_DESC):
    descriptions = pd.read_csv(OUT_DESC)['description'].tolist()
    print("开始编码描述向量...")
    embeddings = encode_descriptions(descriptions)
    np.save(OUT_EMB, embeddings)
    print("编码完成，保存至:", OUT_EMB)


else:
    descriptions = []
    feature_embeddings = []  # 新增：存储特征嵌入
    print("开始生成流量描述文本和特征嵌入...")
    df["description"] = ""
    batch_size = 10
    for i in range(0, len(df), batch_size):
        batch_rows = df.iloc[i:i + batch_size]
        batch_descriptions = []
        batch_feature_embeddings = []  # 新增：存储批次特征嵌入
        for _, row in batch_rows.iterrows():
            features = extract_key_features(dataset_name, row)
            # 生成特征嵌入
            feature_embedding = encode_features(features)
            batch_feature_embeddings.append(feature_embedding)
            
            prompt_protocol = build_prompt_protocol(dataset_name, features)
            description = generate_description(model, tokenizer, prompt_protocol) 
            batch_descriptions.append(description)
            print(f"~[Describe]: {description}")
        
        descriptions.extend(batch_descriptions)
        feature_embeddings.extend(batch_feature_embeddings)  # 新增：添加批次特征嵌入
        df.iloc[i:i + len(batch_descriptions), df.columns.get_loc("description")] = batch_descriptions
        print(f"~[Batch {i // batch_size + 1}]: Done")
    
    df.to_csv(OUT_DESC, index=False)
    print("开始编码文本...")
    embeddings = encode_descriptions(descriptions)
    np.save(OUT_EMB, embeddings)
    print("编码完成，保存至:", OUT_EMB)
    
    # 保存特征嵌入
    feature_embeddings = np.array(feature_embeddings)
    np.save(OUT_FEAT_EMB, feature_embeddings)
    print("特征嵌入生成完成，保存至:", OUT_FEAT_EMB)




