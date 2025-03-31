
import os
import numpy as np
import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
from collections import Counter
from typing import Union, List, Tuple
from torch_geometric.data import Data, Dataset

from torch_geometric.utils.convert import from_networkx


def generate_random_partition_indices_wotest(all_indices, train_ratio=3./50, val_ratio=47./50):
    assert train_ratio + val_ratio == 1.0, "Ratios must sum to 1."

    np.random.seed(117) # CIC-IDS 117
    np.random.shuffle(all_indices)

    train_size = int(len(all_indices) * train_ratio) # 13500
    val_size = int(len(all_indices) * val_ratio) # 4500
    
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    
    return train_indices, val_indices

def generate_random_partition_indices(num_nodes, train_ratio=0.03, val_ratio=0.47, test_ratio=0.5):
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


class GraphDataset(Dataset):
    def __init__(self, root, dataset_name, small=0, base_test=True, num_neighbors=2700, binary: bool = False, augmentation: bool = False,
                 val: bool = False, test: bool = False, transform=None, pre_transform=None):
        self.dataset_name = dataset_name
        self.small = small
        self.file_name = f'{dataset_name}-{small if small>0 else ""}{"-binary" if binary else ""}.csv'
        self.num_neighbors = num_neighbors
        self.binary = binary
        self.augmentation = augmentation
        self.df = None
        self.labels_encoder = []
        self.values_encoded = []
        self.base_test = base_test
        super(GraphDataset, self).__init__(
            root,
            transform,
            pre_transform
        )

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """ If these files are found in processed_dire, processing is skipped. """

        file_path = f'{self.dataset_name}-{self.small if self.small>0 else ""}{"_binary" if self.binary else ""}.npz'

        return [file_path]

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        """ If this file exist in raw_dir, the download is not triggered. """
        return self.file_name

    def process(self):
        #1. Read csv file
        self.df = pd.read_csv(self.raw_paths[0], header=0)
        print(self.df)

        # 2. Define the columns to be extracted from the data frame (attributes for graph nodes)
        extract_col = list(set(self.df.columns) - {'Timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'label'})
        # extract_col = list(set(self.df.columns) - {'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'label'})

        G = nx.Graph()
        iter_df = self.df[extract_col]
        # print("Any NaN in features:", iter_df.isnull())

        y = self.df['label'].values

        print(Counter(y))

        for index, flow_entry in tqdm(iter_df.iterrows(), total=iter_df.shape[0], desc=f'Creating nodes...'):
            # Create attr for each label
            node_attr = {}
            for label, value in flow_entry.items():
                node_attr[label] = value
            node_attr['y'] = y[index]
            G.add_node(index, **node_attr)

        # Create edges
        if self.num_neighbors > 0:
            # Create edges 根据特定特征（'src_ip', 'src_port', 'dst_ip', 'dst_port'）对数据进行分组，并在同一组中的数据项之间创建边
            if self.dataset_name == 'DoHBrw':
                features_to_link = ['src_ip', 'src_port', 'dst_ip', 'dst_port']
            elif self.dataset_name == 'CICIDS':
                features_to_link = ['src_ip', 'src_port', 'dst_ip', 'dst_port','Protocol']
            else:
                features_to_link = ['src_ip', 'src_port', 'dst_ip', 'dst_port','proto']
            # features_to_link = ['src_ip', 'dst_ip']
            groups = self.df.groupby(features_to_link)
            max_edge = 0
            for group in tqdm(groups, total=len(groups), desc=f'Creating edges for features: {features_to_link}'):
                idx_matches = group[1].index
                if (len(idx_matches) > max_edge):
                    max_edge = len(idx_matches)
                if len(idx_matches) < 1:
                    continue
                for idx in range(len(idx_matches)):
                    a = idx_matches[idx]
                    for i in range(self.num_neighbors):
                        if idx + 1 + i < len(idx_matches):
                            b = idx_matches[idx + 1 + i]
                            # If edge (a, b) not exist create
                            if not G.has_edge(a, b):
                                G.add_edge(a, b)
        print("Max edge:", max_edge)

        # Count and store the number of stary nodes
        stary_nodes_count = sum(1 for node in G.nodes if G.degree[node] == 0)
        print(f"Number of stary nodes: {stary_nodes_count}")

        # Create PyTorch Geometric data
        data = from_networkx(G, group_node_attrs=extract_col)
        num_nodes = data.num_nodes 
        test_path = os.path.join(self.processed_dir, f'{self.dataset_name}{"_binary" if self.binary else ""}-test.npz')
        if self.base_test:
            if os.path.exists(test_path):
                test_indices = torch.load(test_path, weights_only=False)
                data.test = test_indices

                all_indices = np.arange(num_nodes)
                all_indices = np.setdiff1d(all_indices, test_indices) 

                train_indices, val_indices = generate_random_partition_indices_wotest(all_indices)

            else:

                train_indices, val_indices, test_indices = generate_random_partition_indices(num_nodes)
                

                data.test = test_indices

                torch.save(data.test, os.path.join(self.processed_dir, f'{self.dataset_name}{"_binary" if self.binary else ""}-test.npz'))
        else:
            train_indices, val_indices, test_indices = generate_random_partition_indices(num_nodes)
            data.test = test_indices
        data.train = train_indices
        data.val = val_indices
        file_path=f'{self.dataset_name}-{self.small if self.small>0 else ""}{"_binary" if self.binary else ""}.npz'
        torch.save(data, os.path.join(self.processed_dir, file_path))

    def len(self) -> int:
        """ Return number of graph """
        return 1

    def get(self, idx: int) -> Data:
        """ Return the idx-th graph. """

        file_path = f'{self.dataset_name}-{self.small if self.small>0 else ""}{"_binary" if self.binary else ""}.npz'
        data = torch.load(os.path.join(self.processed_dir, file_path), weights_only=False)
        return data
