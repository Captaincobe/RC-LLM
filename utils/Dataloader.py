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



class ECMLDataset(Dataset):
    def __init__(self, path_npz):
        data = np.load(path_npz, allow_pickle=True)
        # 动态获取所有视图
        view_keys = sorted([k for k in data.keys() if k.startswith('view_')])
        self.X = [data[key] for key in view_keys]
        self.Y = data["label"]
        self.num_samples = self.X[0].shape[0]
        
        # 记录各个视图的特征维度
        self.view_features = {}
        for i, key in enumerate(view_keys):
            view_num = key.split('_')[1]  # 从"view_1"中提取"1"
            setattr(self, f"view{view_num}_features", self.X[i].shape[1])
            self.view_features[view_num] = self.X[i].shape[1]
        
        # 标签编码
        unique_labels = sorted(set(self.Y))
        self.label_encoder = LabelEncoder()
        self.Y = self.label_encoder.fit_transform(self.Y)
        
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.inv_label_map = {idx: label for label, idx in self.label_map.items()}
        
        # 记录视图数量
        self.num_views = len(self.X)

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

# def generate_random_partition_indices(num_nodes, val_ratio=0.5, test_ratio=0.5):
#     assert val_ratio + test_ratio == 1.0

#     all_indices = np.arange(num_nodes)
#     np.random.seed(9977)  # 为复现实验设置固定种子
#     np.random.shuffle(all_indices)

#     val_size = int(num_nodes * val_ratio)
#     val_indices = all_indices[:val_size]
#     test_indices = all_indices[val_size:]

#     return val_indices, test_indices


def create_multi_view_data(args):
    dataset_name = args.dataset_name
    out_path = f"datasets/{dataset_name}"
    DATA_PATH = f"{out_path}/text_data.csv"
    
    # 解析要使用的视图列表，默认使用1,2
    views_to_use = args.views.split(',') if hasattr(args, 'views') else ['1', '2']
    
    # 准备多视图数据字典
    multi_view_data = {}
    
    # 加载视图嵌入
    for view_id in views_to_use:
        view_id = view_id.strip()
        emb_path = f"{out_path}/{args.pretrain_model}/embeddings-agent{view_id}{args.embedding_type}.npy"
        try:
            embeddings = np.load(emb_path)
            multi_view_data[f"view_{view_id}"] = embeddings
            print(f"Loaded view_{view_id} from {emb_path}")
        except FileNotFoundError:
            print(f"Warning: Could not find embeddings for view_{view_id} at {emb_path}")
    
    # 加载原始数据和标签
    df_raw = pd.read_csv(DATA_PATH)
    
    # 如果需要使用原始特征作为视图
    if '0' in views_to_use:
        X_raw = df_raw.drop(columns=['Timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'label', 'description'], errors="ignore").values
        multi_view_data["view_0"] = X_raw
        print(f"Added original features as view_0")
    
    # 添加标签
    y = df_raw["label"].values
    multi_view_data["label"] = y
    
    # 保存多视图数据
    np.savez(f"{out_path}/{args.pretrain_model}/multi_view-{args.embedding_type}-{args.views}.npz", **multi_view_data)
    
    views_str = ", ".join([f"view_{v}" for v in views_to_use])
    print(f"Created multi-view data file with views: {views_str}")


def load_data(args, DATA_PATH, logger=None):
    data = ECMLDataset(f"{DATA_PATH}/{args.pretrain_model}/multi_view-{args.embedding_type}-{args.views}.npz")
    print(f"Loading data from {DATA_PATH}/{args.pretrain_model}/multi_view-{args.embedding_type}-{args.views}.npz")
    # data = np.load(f"{DATA_PATH}/test_data.npy", allow_pickle=True)

    # dataset = args.dataset_name
    # device = args.device
    # dataset_path = os.path.join(os.getcwd(), 'datasets', dataset)

    # graph = GraphDataset(
    #     root=dataset_path, dataset_name=dataset, small=args.n_small, base_test=True, num_neighbors=args.n_neigh, binary=args.binary
    # )

    # data = graph[0]

    num_smaples = data.num_samples
    train_indices, val_indices, test_indices = generate_random_partition_indices(num_smaples, train_ratio=args.train_ratio)
    data.test = test_indices
    # data.train = train_indices
    data.val = val_indices
    
    features = data.X
    labels = data.Y
    # num_nodes = data.num_nodes
    n_class = len(set(labels))
    data.X = [torch.tensor(x, dtype=torch.float32) for x in features]
    data.Y = torch.tensor(labels, dtype=torch.long)

    if logger is not None:
        logger.info(f'Class: {n_class}')
        logger.info(SEPARATOR)
    return data, n_class, train_indices, val_indices, test_indices, logger
    
