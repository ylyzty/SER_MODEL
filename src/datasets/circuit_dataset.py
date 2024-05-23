from typing import List

import torch
import torch.nn as nn
import os

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from build_dataset import parse_circuit_data


class CircuitDataset(InMemoryDataset):
    """
    自定义内存数据集
    """

    def __init__(self, root, transform=None, pre_transform=None, meta_dict=None):
        self.root = root

        assert (transform is None) and (pre_transform is None)
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

            x = torch.tensor(circuits[cir_name]['x'], dtype=torch.float)
            y = torch.tensor(circuits[cir_name]['y'][:, 1:], dtype=torch.float)
            edge_index = torch.tensor(circuits[cir_name]['edge_index'], dtype=torch.float).t().contiguous()
            edge_feat = torch.tensor(circuits[cir_name]['edge_feat'], dtype=torch.float)

            circuit_data = Data(x=x, edge_index=edge_index, edge_attr=edge_feat, y=y)
            data_list.append(circuit_data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    path = "../../circuit_data"
    dataset = CircuitDataset(root=path)
