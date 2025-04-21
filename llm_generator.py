import os
import random
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from features import extract_key_features
# from prompt_builder import build_prompt, build_prompt_protocol, generate_description, agent_1, agent_2
from prompt_builder import generate_description, agent_1, agent_2
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
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)  # QwQ-7B  Mistral-7B-Instruct-v0.3 google/gemma-3-27b-it  HuggingFaceH4/zephyr-7b-alpha TurkuNLP/gpt3-finnish-small
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True, offload_buffers=True)
# encoder_model = SentenceTransformer("all-MiniLM-L6-v2")


# texthead = args.texthead # train_sapmles
dataset_name = args.dataset_name  # CICIDS DoHBrw TONIoT
out_path = f"datasets/{dataset_name}/outputs"
DATA_PATH = f"{out_path}/text_data.csv"

# 只定义agent1和agent2的描述和嵌入文件路径
OUT_DESC_AGENT1 = f"{out_path}/descriptions-agent1.csv"
OUT_DESC_AGENT2 = f"{out_path}/descriptions-agent2.csv"
OUT_EMB_1 = f"{out_path}/embeddings-agent1{args.embedding_type}.npy"
OUT_EMB_2 = f"{out_path}/embeddings-agent2{args.embedding_type}.npy"

df = pd.read_csv(DATA_PATH)

# 检查是否已有描述文件
if os.path.exists(OUT_DESC_AGENT1) and os.path.exists(OUT_DESC_AGENT2):
    # 如果描述文件已存在，直接加载并生成嵌入
    agent1_descriptions = pd.read_csv(OUT_DESC_AGENT1)['description'].tolist()
    agent2_descriptions = pd.read_csv(OUT_DESC_AGENT2)['description'].tolist()
    
    print("开始编码Agent1描述向量...")
    embeddings_1 = encode_descriptions(agent1_descriptions)
    np.save(OUT_EMB_1, embeddings_1)
    print(f"Agent1编码完成，保存至: {OUT_EMB_1}")
    
    print("开始编码Agent2描述向量...")
    embeddings_2 = encode_descriptions(agent2_descriptions)
    np.save(OUT_EMB_2, embeddings_2)
    print(f"Agent2编码完成，保存至: {OUT_EMB_2}")
else:
    # 创建两个新的DataFrame，只保留必要的列
    # 确定要保留的列（通常是用于标识和标签的列）
    keep_columns = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'label', 'description']
    existing_columns = [col for col in keep_columns if col in df.columns]
    
    df_agent1 = df[existing_columns].copy()
    df_agent2 = df[existing_columns].copy()
    
    # 初始化空描述
    df_agent1['description'] = ""
    df_agent2['description'] = ""
    
    agent1_descriptions = []
    agent2_descriptions = []
    
    print("开始生成流量描述文本...")
    batch_size = 10
    for i in range(0, len(df), batch_size):
        batch_rows = df.iloc[i:i + batch_size]
        batch_agent1_descriptions = []
        batch_agent2_descriptions = []
        
        for _, row in batch_rows.iterrows():
            features = extract_key_features(dataset_name, row)
            
            prompt_1 = agent_1(dataset_name, features)
            prompt_2 = agent_2(dataset_name, features)
            
            description_1 = generate_description(model, tokenizer, prompt_1) 
            description_2 = generate_description(model, tokenizer, prompt_2)
            
            # 保存两个agent的描述
            batch_agent1_descriptions.append(description_1)
            batch_agent2_descriptions.append(description_2)
            
            print(f"Agent1: {description_1}")
            print(f"Agent2: {description_2}")
            print("-" * 50)
        
        # 扩展描述列表
        agent1_descriptions.extend(batch_agent1_descriptions)
        agent2_descriptions.extend(batch_agent2_descriptions)
        
        # 更新DataFrame
        df_agent1.iloc[i:i + len(batch_agent1_descriptions), df_agent1.columns.get_loc("description")] = batch_agent1_descriptions
        df_agent2.iloc[i:i + len(batch_agent2_descriptions), df_agent2.columns.get_loc("description")] = batch_agent2_descriptions
        
        print(f"~[Batch {i // batch_size + 1}]: Done")

    # 只保存各agent的描述CSV
    df_agent1.to_csv(OUT_DESC_AGENT1, index=False)
    print(f"Saved Agent 1 descriptions to {OUT_DESC_AGENT1}")
    
    df_agent2.to_csv(OUT_DESC_AGENT2, index=False)
    print(f"Saved Agent 2 descriptions to {OUT_DESC_AGENT2}")

    # 分别为两个agent的描述生成嵌入
    print("开始编码Agent1描述向量...")
    embeddings_1 = encode_descriptions(agent1_descriptions)
    np.save(OUT_EMB_1, embeddings_1)
    print(f"Agent1编码完成，保存至: {OUT_EMB_1}")
    
    print("开始编码Agent2描述向量...")
    embeddings_2 = encode_descriptions(agent2_descriptions)
    np.save(OUT_EMB_2, embeddings_2)
    print(f"Agent2编码完成，保存至: {OUT_EMB_2}")




