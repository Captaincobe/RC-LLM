from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from args import parameter_parser
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 初始化编码器模型
# encoder_model = SentenceTransformer("BAAI/bge-base-en-v1.5") # all-MiniLM-L6-v2 BAAI/bge-base-en-v1.5
# # 替换为其他已知可用的模型
args = parameter_parser()
if args.embedding_type == "par":
    encoder_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    # encoder_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    print("load embedding model")
elif args.embedding_type == "all":
    encoder_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("load embedding model")
elif args.embedding_type == "mpnet":
    encoder_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") # 性能更好，但更大
    print("load embedding model")


def encode_features(features_dict):
    """为流量特征生成嵌入向量"""
    # 将特征字典转换为文本描述
    feature_text = []
    for key, value in features_dict.items():
        if isinstance(value, (int, float, str)):
            feature_text.append(f"{key}: {value}")
    
    # 将特征文本连接成一个字符串
    feature_text = " | ".join(feature_text)
    
    # 生成嵌入向量
    embedding = encoder_model.encode(feature_text, convert_to_tensor=True)
    return embedding.cpu().numpy()

def encode_descriptions(desc_list):
    """为描述文本生成嵌入向量"""
    embeddings = encoder_model.encode(desc_list, batch_size=16, show_progress_bar=True)
    return np.array(embeddings)

def retrieve_similar_flows(current_embedding, feature_embeddings, descriptions, k=9):
    """检索与当前流量最相似的k条流量"""
    # 计算余弦相似度
    similarities = cosine_similarity([current_embedding], feature_embeddings)[0]
    
    # 获取最相似的k个索引
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # 获取对应的描述
    similar_descriptions = [descriptions[i] for i in top_k_indices]
    similar_scores = [similarities[i] for i in top_k_indices]
    
    return similar_descriptions, similar_scores 