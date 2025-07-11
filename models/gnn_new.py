import argparse

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool
from utils.graph_handler import *

class EdgeFeatureGCNConv(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, aggr="add"):
        super(EdgeFeatureGCNConv, self).__init__(aggr=aggr)
        self.node_mlp = nn.Linear(node_in_dim, hidden_dim)
        self.edge_mlp = nn.Linear(edge_in_dim, hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.node_mlp(x)
        edge_attr = self.edge_mlp(edge_attr)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return aggr_out


class GCN(nn.Module):
    def __init__(self, args, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(EdgeFeatureGCNConv(args.node_in_dim, args.edge_in_dim, args.out_dim))
        else:
            self.convs.append(EdgeFeatureGCNConv(args.node_in_dim, args.edge_in_dim, args.hidden_dim))  # 第一层
            for _ in range(num_layers - 2):
                self.convs.append(EdgeFeatureGCNConv(args.hidden_dim, args.edge_in_dim, args.hidden_dim))  # 中间层
            self.convs.append(EdgeFeatureGCNConv(args.hidden_dim, args.edge_in_dim, args.out_dim))  # 最后一层

        self.global_pool = global_add_pool
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, node_feature, edge_index, edge_feature):
        x = node_feature

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_feature)
            if i < self.num_layers - 1:
                x = self.relu(x)
                x = self.dropout_layer(x)

        non_constant_mask = (x.max(dim=1).values != x.min(dim=1).values)
        x = x[non_constant_mask]

        graph_feature = self.global_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        return graph_feature.squeeze(0)