import os
import random
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from features import extract_key_features
from prompt_builder import build_prompt, generate_description
import pandas as pd
import numpy as np
from args import parameter_parser

from sentence_transformers import SentenceTransformer

from transformers import GPT2LMHeadModel,AutoTokenizer, AutoModelForCausalLM
args = parameter_parser()
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True) # {"": device}
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")  # QwQ-32B
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True, offload_buffers=True)
encoder_model = SentenceTransformer("all-MiniLM-L6-v2")
def encode_descriptions(desc_list):
    embeddings = encoder_model.encode(desc_list, batch_size=16, show_progress_bar=True)
    return np.array(embeddings)


texthead = args.texthead # train_sapmles
dataset_name = args.dataset_name  # CICIDS DoHBrw TONIoT
out_path = f"datasets/{dataset_name}/outputs/"
DATA_PATH = f"{out_path}/text_data.csv"
OUT_DESC = f"{out_path}/descriptions.csv"
OUT_EMB = f"{out_path}/embeddings.npy" # _{texthead}_{index}


df = pd.read_csv(DATA_PATH)


if os.path.exists(OUT_DESC):
    descriptions = pd.read_csv(OUT_DESC)['description']
    print("开始编码描述向量...")
    embeddings = encode_descriptions(descriptions)
    np.save(OUT_EMB, embeddings)
    print("编码完成，保存至:", OUT_EMB)
else:
    descriptions = []
    print("开始生成流量描述文本...")
    df["description"] = ""
    batch_size = 10
    for i in range(0, len(df), batch_size):
        batch_rows = df.iloc[i:i + batch_size]
        batch_descriptions = []
        for _, row in batch_rows.iterrows():
            features = extract_key_features(dataset_name, row)
            # print(f"Extracted features:\n{features}\n")
            prompt = build_prompt(dataset_name, features)
            description = generate_description(model, tokenizer, prompt) 
            batch_descriptions.append(description)
            print(f"~[Describe]: {description}")
        descriptions.extend(batch_descriptions)
        # df.loc[i:i + batch_size - 1, "description"] = batch_descriptions
        df.iloc[i:i + len(batch_descriptions), df.columns.get_loc("description")] = batch_descriptions
    df.to_csv(OUT_DESC, index=False)
    print("开始编码文本...")
    embeddings = encode_descriptions(descriptions)
    np.save(OUT_EMB, embeddings)
    print("编码完成，保存至:", OUT_EMB)




