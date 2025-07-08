import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, MLP
from torch_geometric.data import HeteroData
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
from torch_frame.data.stats import StatType
from typing import Any, Dict, List, Tuple


class MetaPathGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, relation_index):
        super().__init__(aggr='add', flow='target_to_source')
        self.relation_index = relation_index
        self.w_0 = nn.Linear(in_channels, out_channels)
        self.w_1 = nn.Linear(in_channels, out_channels)
        self.w_l = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_type, h):
        #mask = (edge_type == self.relation_index)
        edge_index_filtered = edge_index[self.relation_index]
        #edge_index_filtered = edge_index[:, mask]

        # Manual message passing
        row, col = edge_index_filtered
        agg = torch.zeros_like(h)
        agg.index_add_(0, row, h[col])

        return self.w_l(agg) + self.w_0(h) + self.w_1(x)


class MetaPathGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, metapath: List[int]):
        super().__init__()
        self.metapath = metapath
        self.input_mlp = MLP(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=hidden_dim * 2, num_layers=3)

        self.layers = nn.ModuleList([
            MetaPathGNNLayer(
                in_channels=hidden_dim * 2 if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                relation_index=rel_idx#edge type
            )
            for i, rel_idx in enumerate(metapath)
        ])
        self.output_proj = nn.Identity()  # lasciamo identità, verrà gestita in MPSGNN

    def forward(self, x, edge_index, edge_type):
        h = self.input_mlp(x)
        for i, layer in enumerate(self.layers):
            h = F.relu(layer(x, edge_index, edge_type, h))
            h = F.dropout(h, p=0.5, training=self.training)
        return self.output_proj(h)


class MPSGNN(nn.Module):
    def __init__(self,
                 data: HeteroData,
                 col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 metapaths: List[List[int]],  # list of metapaths as list of relation indices
                 metapath_counts: Dict[Tuple[int, ...], int],
                 hidden_channels: int = 64,
                 out_channels: int = 64,
                 final_out_channels: int = 1):
        super().__init__()

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

        self.metapath_models = nn.ModuleList([
            MetaPathGNN(
                input_dim=hidden_channels,
                hidden_dim=hidden_channels,
                output_dim=out_channels,
                metapath=mp
            )
            for mp in metapaths
        ])

        self.regressor = nn.Sequential(
            nn.Linear(len(metapaths) * out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, final_out_channels)
        )

        weights = torch.tensor(
            [metapath_counts.get(tuple(mp), 1) for mp in metapaths],
            dtype=torch.float
        )
        weights = weights / weights.sum()
        self.register_buffer("metapath_weights", weights.view(1, -1, 1))  # [1, M, 1]

    def forward(self, batch: HeteroData, entity_table: str):
        seed_time = batch[entity_table].seed_time

        # Encode features and temporal info
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x = x_dict[entity_table]
        edge_index = batch.edge_index_dict
        edge_type = batch.edge_types  #all the relations that are present in the graph

        # Compute embeddings per metapath
        metapath_embeddings = []
        for model in self.metapath_models:
            h = model(x, edge_index, edge_type)
            metapath_embeddings.append(h)  # [N, D]

        concat = torch.cat(metapath_embeddings, dim=1)  # [N, M * D]
        return self.regressor(concat).squeeze(-1)       # [N] → regressione
