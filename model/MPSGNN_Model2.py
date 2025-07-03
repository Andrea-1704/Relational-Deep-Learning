import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, MLP
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
from typing import List, Tuple, Dict, Any
from torch_frame.data.stats import StatType

class MetaPathGNNLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, relation_index: int):
        super().__init__(aggr='add', flow='target_to_source')
        self.relation_index = relation_index
        self.w_l = nn.Linear(in_channels, out_channels)
        self.w_0 = nn.Linear(in_channels, out_channels)
        self.w_1 = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, h: torch.Tensor):
        mask = edge_type == self.relation_index
        print(f"edge_index {edge_index}")
        print(f"edge type {edge_type}")
        filtered_edge_index = edge_index[edge_type]
        print(f"shape di filtered è {filtered_edge_index.shape}")
        agg_messages = self.propagate(filtered_edge_index, x=h)
        return self.w_l(agg_messages) + self.w_0(h) + self.w_1(x)

    def message(self, x_j: torch.Tensor):
        return x_j

class MetaPathGNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 metapath: List[int]):
        super().__init__()
        self.metapath = metapath
        self.layers = nn.ModuleList()

        for i, rel_idx in enumerate(metapath):
            layer = MetaPathGNNLayer(
                in_channels=hidden_channels if i > 0 else hidden_channels * 2,
                out_channels=hidden_channels,
                relation_index=rel_idx
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor):
        h = x
        for i, layer in enumerate(self.layers):
            h = F.relu(layer(x if i == 0 else h, edge_index, edge_type, h))
            h = F.dropout(h, p=0.5, training=self.training)
        return h

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
