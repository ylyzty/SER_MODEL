import math
import torch
import torch_geometric.nn as gnn

from torch import nn
from dag_transformer.layers import TransformerEncoderLayer


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, SAT, edge_index, mask_dag_, dag_rr_edge_index,
                edge_attr=None, ptr=None, return_attn=False):
        output = x
        for layer in self.layers:
            output = layer(output, SAT, edge_index, mask_dag_, dag_rr_edge_index,
                           edge_attr=edge_attr,
                           ptr=ptr,
                           return_attn=return_attn)

        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    """
    :parameter in_size: 节点特征维度
    :parameter d_model: 节点隐藏层维度
    :parameter num_layers: 卷积层数
    """

    def __init__(self, in_size, num_class, d_model, gps=0, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, abs_pe='dagpe', use_edge_attr=False, num_edge_features=4,
                 in_embed=True, edge_embed=True, use_global_pool=False,
                 global_pool='mean', SAT=True, **kwargs):
        super().__init__()

        self.SAT = SAT
        self.gps = gps
        self.dropout = nn.Dropout(dropout)

        # 节点特征相关嵌入
        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)

        # 边特征相关嵌入
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)  # default 32
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                                                out_features=edge_dim,
                                                bias=False)
        else:
            kwargs['edge_dim'] = None

        self.num_layers = num_layers
        # gps = 0
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)

        self.regression = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, num_class),
            nn.ReLU(True)
        )

    def forward(self, data, return_attn=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        mask_dag_ = data.mask_rc if hasattr(data, 'mask_rc') else None
        dag_rr_edge_index = data.dag_rr_edge_index if hasattr(data, 'dag_rr_edge_index') else None

        output = self.embedding(x)
        output = self.dropout(output)    # abs_pe = 'none'

        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
        else:
            edge_attr = None

        output = self.encoder(
            output,
            self.SAT,
            edge_index,
            mask_dag_,
            dag_rr_edge_index,
            edge_attr=edge_attr,
            ptr=None,
            return_attn=return_attn
        )

        return self.regression(output)
