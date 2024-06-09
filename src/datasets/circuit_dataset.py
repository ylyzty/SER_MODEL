from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import networkx as nx

from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from sklearn import preprocessing as pp

from datasets.parse_circuit import parse_circuit_data
from utils.normalize import min_max_scaler


class CircuitDataset(InMemoryDataset):
    """
    自定义内存数据集
    """

    def __init__(self, root, DAG=True, transform=None, pre_transform=None):
        self.root = root
        self.DAG = DAG

        # assert (transform is None) and (pre_transform is None)
        super(CircuitDataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return os.listdir(os.path.join(self.root, 'raw'))

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        """
        生成数据集, 处理 raw_data 后保存至 processed_dir
        :return:
        """
        data_list = []

        circuits = parse_circuit_data(os.path.join(self.root, 'raw'))
        for cir_idx, cir_name in enumerate(circuits):
            print("Circuit Name: ", cir_name)

            x = torch.tensor(circuits[cir_name]['x'], dtype=torch.float)    # [N, 2]
            y = torch.tensor(circuits[cir_name]['y'][:, 1:], dtype=torch.float)    # [N, 1]

            # 归一化
            scaler = pp.MinMaxScaler().fit(y)
            y = torch.tensor(scaler.transform(y), dtype=torch.float)

            edge_index_dag = torch.tensor(circuits[cir_name]['edge_index'], dtype=torch.long).t()   # [2, N]
            edge_feat = torch.tensor(circuits[cir_name]['edge_feat'], dtype=torch.float)    # [N, 1]
            pyg_graph = Data(x=x, edge_index=edge_index_dag, edge_attr=edge_feat, y=y)

            undir_edge_index = to_undirected(edge_index_dag)
            pyg_graph.undir_edge_index = undir_edge_index

            data_new = Data(x=pyg_graph.x, edge_index=pyg_graph.edge_index)
            DG = to_networkx(data_new)

            # 计算 DAG 传递闭包
            TC = nx.transitive_closure_dag(DG)
            data_new = from_networkx(TC)
            edge_index_dag = data_new.edge_index
            if self.DAG:
                pyg_graph.dag_rr_edge_index = to_undirected(edge_index_dag)

            data_list.append(pyg_graph)

        print('Saving...')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


if __name__ == "__main__":
    path = "../../circuit_data"
    dataset = CircuitDataset(root=path, DAG=True)
    dataset.process()
    print(len(dataset))
