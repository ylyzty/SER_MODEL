import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.utils import to_undirected

from datasets.circuit_dataset import CircuitDataset


class GraphDataset(object):
    def __init__(self, dataset, cache_path=None, use_mpnn=True, k=1000):
        self.dataset = dataset
        self.n_features = dataset[0].x.shape[-1]
        self.cache_path = cache_path
        self.use_mpnn = use_mpnn
        self.k = k

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        data_new = Data(x=data.x, edge_index=data.edge_index)

        DG = to_networkx(data_new)
        # Statistics
        depth = nx.shortest_path_length(DG, 0)
        data.depth = np.max([depth[i] for i in depth])
        data.maxdegree = np.max([i[1] for i in DG.out_degree])

        TC = nx.transitive_closure_dag(DG)
        TC_copy = TC.copy()
        # Statistics
        # k_list = [TC.number_of_edges() * 2]
        # for i in range(8):
        #     for edge in TC_copy.edges():
        #         if (nx.shortest_path_length(DG, source=edge[0], target=edge[1])) > (8 - i):
        #             TC.remove_edge(edge[0], edge[1])
        #     k_list.append(TC.number_of_edges() * 2)
        #     TC_copy = TC.copy()
        # data.k_list = torch.tensor(k_list)

        # 感受野, 去除感受野以外的边
        if self.k < 1000:
            for edge in TC_copy.edges():
                if nx.shortest_path_length(DG, source=edge[0], target=edge[1]) > self.k:
                    TC.remove_edge(edge[0], edge[1])

        data_new = from_networkx(TC)
        edge_index_dag = data_new.edge_index
        if self.use_mpnn:
            data.dag_rr_edge_index = to_undirected(edge_index_dag)
        if self.n_features == 1:
            data.x = data.x.squeeze(-1)    # 取消某个维度，被取消维度的长度必须为1
        if not isinstance(data.y, list):
            data.y = data.y.view(data.y.shape[0], -1)

        return data


if __name__ == "__main__":
    path = "../../circuit_data"
    dataset = CircuitDataset(root=path)
    graph_dataset = GraphDataset(dataset)
    graph_dataset.__getitem__(0)
