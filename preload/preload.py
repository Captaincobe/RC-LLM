import argparse
import os
import random
import numpy as np

import pandas as pd

# from utils.Dataloader import generate_multi_view_data
# from .features import load_and_clean_data
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='CICIDS', choices=["TONIoT","DoHBrw", "CICIDS", "CICMalMen"],
            help='which dataset to use')
parser.add_argument("--cuda", type=str, default='1', help="Device: cuda:num or cpu.")
parser.add_argument('--texthead', dest='texthead', type=int, default=1000)

args = parser.parse_args()

def load_and_clean_data(file_path, if_shuffle=False):
    df = pd.read_csv(file_path, header=0)
    df.fillna(0, inplace=True)
    if if_shuffle:
        random.seed(42)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

texthead = args.texthead # train_sapmles
dataset_name = args.dataset_name  # CICIDS DoHBrw TONIoT
out_path = f"datasets/{dataset_name}/outputs/"
DATA_PATH = f"../datasets/{dataset_name}/raw/{texthead}.csv"

os.makedirs(os.path.dirname(out_path), exist_ok=True)

# === 加载完整数据集 ===
df_all = load_and_clean_data(DATA_PATH, if_shuffle=True)
# print(f"✅ 总样本数: {len(df_all)}")

# === 生成描述的样本（移除） ===
# idx_text = sorted(random.sample(range(len(df_all))))
# df_text = df_all.iloc[idx_text].reset_index(drop=True)
df_all.to_csv(f"{out_path}/text_data.csv", index=False)
# np.save(f"datasets/{out_path}/idx_text.npy", np.array(idx_text))
print(f"已选取 {len(df_all)} 条作为描述生成数据")


# # === 从原始数据中剔除已用于生成文本的数据，再得到测试集 ===
# df_train = df_all.drop(index=idx_text).reset_index(drop=True)
# df_train.to_csv(f"datasets/{out_path}/test_data.csv", index=False)
# print(f"📂 剩余训练数据保存至: datasets/{out_path}/test_data.csv")

# np.save(f"datasets/{out_path}/test_data.npy", df_train.values)
# print(f"📦 剩余训练数据也保存为 NPY 格式: datasets/{out_path}/test_data.npy")
