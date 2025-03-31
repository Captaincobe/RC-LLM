import random
import numpy as np

import pandas as pd

from utils.Dataloader import generate_multi_view_data
# from .features import load_and_clean_data

def load_and_clean_data(file_path, if_shuffle=False):
    df = pd.read_csv(file_path, header=0)
    df.fillna(0, inplace=True)
    if if_shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

index = 3
texthead = 200 # train_sapmles
dataset_name = "CICIDS"  # CICIDS DoHBrw TONIoT
out_path = f"{dataset_name}/outputs"
DATA_PATH = f"../datasets/{dataset_name}/raw/dataset.csv"

# === 加载完整数据集 ===
df_all = load_and_clean_data(DATA_PATH)
print(f"✅ 总样本数: {len(df_all)}")

# === 生成描述的样本（移除） ===
random.seed(42)
idx_text = sorted(random.sample(range(len(df_all)), texthead))
df_text = df_all.iloc[idx_text].reset_index(drop=True)
df_text.to_csv(f"datasets/{out_path}/text_data.csv", index=False)
np.save(f"datasets/{out_path}/idx_text.npy", np.array(idx_text))
print(f"🧪 已选取 {texthead} 条作为描述生成数据")


# === 从原始数据中剔除已用于生成文本的数据，再得到测试集 ===
df_train = df_all.drop(index=idx_text).reset_index(drop=True)
df_train.to_csv(f"datasets/{out_path}/test_data.csv", index=False)
print(f"📂 剩余训练数据保存至: datasets/{out_path}/test_data.csv")

