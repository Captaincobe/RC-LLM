import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def encode_descriptions(desc_list):
    embeddings = model.encode(desc_list, batch_size=16, show_progress_bar=True)
    return np.array(embeddings)
