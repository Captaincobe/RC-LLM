from sklearn.calibration import LabelEncoder
import torch
import pandas as pd
import numpy as np


from torch.utils.data import Dataset



SEPARATOR = '------------------------------------'
# DETECTION_RATE = 'Detection Rate'
# CONFUSION_MATRIX = 'Confusion Matrix'
# BAR_STACKED = 'Bar Stacked'
# LOGGER = 'logger.log'
# TRAINING_LOGGER = 'training.log'
# CWD = os.getcwd()



def generate_multi_view_data(dataset_name):
    desc_embeddings = np.load(f"../datasets/{dataset_name}/outputs/embeddings_200.npy")  # shape: (10, 384)

    df_raw = pd.read_csv("data/text_data.csv")
    X_raw = df_raw.drop(columns=['Timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port'], errors="ignore").values  # shape: (10, D)
    # 可选标签字段，可用于监督任务
    y = df_raw["label"].values  
    multi_view_data = {
        "view_1": desc_embeddings,   # 语义
        "view_2": X_raw,           
        "label": y                 
    } 
    np.savez(f"../datasets/{dataset_name}/outputs/multi_view_200.npz", **multi_view_data)


class ECMLDataset(Dataset):
    def __init__(self, path_npz):
        data = np.load(path_npz, allow_pickle=True)
        self.X = [data["view_1"], data["view_2"]]
        self.Y = data["label"]
        self.num_samples = self.X[0].shape[0]
        self.view1_features = self.X[0].shape[1]
        self.view2_features = self.X[1].shape[1]
        # Add label encoder
        unique_labels = sorted(set(self.Y))
        self.label_encoder = LabelEncoder()
        self.Y = self.label_encoder.fit_transform(self.Y)  # 转成整数数组

        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.inv_label_map = {idx: label for label, idx in self.label_map.items()}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_views = [torch.tensor(x[idx], dtype=torch.float32) for x in self.X]
        y = int(self.Y[idx])
        return x_views, y


def generate_random_partition_indices(num_nodes, train_ratio=0.03, val_ratio=0.47, test_ratio=0.5):
    test_ratio = train_ratio*5
    val_ratio = 1-train_ratio-test_ratio
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."

    all_indices = np.arange(num_nodes)
    np.random.seed(9977) # CIC-IDS 117
    np.random.shuffle(all_indices)

    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices


def create_multi_view_data(args):
    dataset_name = args.dataset_name
    """ Train data"""
    out_path = f"datasets/{dataset_name}/outputs" # datasets/CICIDS/outputs/embeddings_200_3.npy
    texthead = args.texthead
    index = 3
    DATA_PATH = f"../datasets/{dataset_name}/raw/dataset.csv"
    OUT_EMB = f"{out_path}/embeddings_{texthead}_{index}.npy"

    desc_embeddings = np.load(OUT_EMB)
    df_raw = pd.read_csv(DATA_PATH).head(texthead)
    X_raw = df_raw.drop(columns=['Timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'label'], errors="ignore").values  # shape: (10, D)

    # 可选标签字段
    y = df_raw["label"].values  
    multi_view_data = {
        "view_1": desc_embeddings,   # 语义
        "view_2": X_raw,           
        "label": y                 
    }
    np.savez(f"{out_path}/multi_view_{texthead}.npz", **multi_view_data)


def load_data(args, DATA_PATH, logger=None):
    data = ECMLDataset(DATA_PATH)

    # dataset = args.dataset_name
    # device = args.device
    # dataset_path = os.path.join(os.getcwd(), 'datasets', dataset)

    # graph = GraphDataset(
    #     root=dataset_path, dataset_name=dataset, small=args.n_small, base_test=True, num_neighbors=args.n_neigh, binary=args.binary
    # )

    # data = graph[0]

    # num_nodes = data.num_nodes
    num_smaples = data.num_samples
    train_indices, val_indices, test_indices = generate_random_partition_indices(num_smaples, train_ratio=args.train_ratio)
    data.test = test_indices
    data.train = train_indices
    data.val = val_indices
    
    features = data.X
    labels = data.Y
    # num_nodes = data.num_nodes
    n_class = len(set(labels))
    data.X = [torch.tensor(x, dtype=torch.float32) for x in data.X]
    data.Y = torch.tensor(labels, dtype=torch.long)

    if logger is not None:
        logger.info(f'Class: {n_class}')
        logger.info(f'Number of edge: {data.num_edges}')
        logger.info(SEPARATOR)
    return data, n_class, train_indices, val_indices, test_indices, logger
    
