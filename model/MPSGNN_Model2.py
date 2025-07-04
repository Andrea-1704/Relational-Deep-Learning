"""
This is a version of MPSGNN that tries to be as coherent 
as possible to the code implementation provided into 
https://arxiv.org/abs/2412.00521.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, MLP
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
from typing import List, Tuple, Dict, Any
from torch_frame.data.stats import StatType

class MetaPathGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add', flow="target_to_source")
        self.w_0 = nn.Linear(in_channels, out_channels)
        self.w_l = nn.Linear(in_channels, out_channels)
        self.w_1 = nn.Linear(in_channels, out_channels)

    def forward(self, x, h, edge_index):
        aggr_out = self.propagate(edge_index=edge_index, x=h)
        return self.w_l(aggr_out) + self.w_0(h) + self.w_1(x)

    def message(self, x_j):
        return x_j



class MetaPathGNN(nn.Module):
    def __init__(self, metapath, hidden_channels, out_channels):
        super().__init__()
        self.metapath = metapath
        self.convs = nn.ModuleList([
            MetaPathGNNLayer(hidden_channels, hidden_channels)
            for _ in metapath
        ])
        self.out_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        h_dict = x_dict.copy()
        for i, (src, rel, dst) in enumerate(reversed(self.metapath)):
            conv_idx = len(self.metapath) - 1 - i
            edge_index = edge_index_dict[(src, rel, dst)]
            h_dst = self.convs[conv_idx](
                x=h_dict[dst],
                h=h_dict[dst],
                edge_index=edge_index
            )
            h_dict[dst] = F.relu(h_dst)
        start_type = self.metapath[0][0]
        return self.out_proj(h_dict[start_type])



class MPSGNN_Original(nn.Module):
    def __init__(self,
                 data: HeteroData,
                 col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 metapaths: List[List[int]],
                 hidden_channels: int = 64,
                 out_channels: int = 64,
                 final_out_channels: int = 1):
        super().__init__()
        self.metapaths = metapaths
        self.target_node_type = "drivers"
        #print(f"batch index dict: {data.edge_index_dict}")
        # Encoder iniziale
        self.encoder = HeteroEncoder(
            channels=hidden_channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict
        )

        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=hidden_channels,
        )

        # MLP iniziale come nel paper
        self.input_mlp = MLP(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels * 2,
            num_layers=3
        )

        # GNN su ciascun metapath
        self.metapath_models = nn.ModuleList([
            MetaPathGNN(
                in_channels=hidden_channels * 2,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                metapath=mp
            ) for mp in metapaths
        ])

        # MLP finale
        self.fc1 = nn.Linear(hidden_channels * len(metapaths), hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, final_out_channels)

    def forward(self, batch: HeteroData, entity_table=None):
        seed_time = batch[entity_table].seed_time

        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x = x_dict[self.target_node_type]  # solo i nodi target
        x = self.input_mlp(x)

        # edge index e type globali
        edge_index = batch.edge_index_dict
        edge_type = batch.edge_types

        embeddings = [
            model(x, edge_index, edge_type) for model in self.metapath_models
        ]
        concat = torch.cat(embeddings, dim=-1)  # [N, hidden * M]

        h = F.relu(self.fc1(concat))
        out = self.fc2(h).squeeze(-1)  # output: [N]

        return out
