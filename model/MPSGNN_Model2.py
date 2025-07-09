import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import MLP
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
from torch_frame.data.stats import StatType
from typing import Any, Dict, List, Tuple

# --- GNN layer che include filtro edge_type ---
class MetaPathGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, relation_index):
        super().__init__(aggr='add', flow="target_to_source")
        self.relation_index = relation_index
        self.w_0 = nn.Linear(in_channels, out_channels)
        self.w_l = nn.Linear(in_channels, out_channels)
        self.w_1 = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, h):
        #mask = (edge_type == self.relation_index)
        #edge index is already the one of the relation
        #edge_index = edge_index[self.relation_index]
        agg = self.propagate(edge_index, x=h)
        return self.w_l(agg) + self.w_0(h) + self.w_1(x)

    def message(self, x_j):
        return x_j

# --- MetaPath GNN per un singolo metapath ---
class MetaPathGNN(nn.Module):
    def __init__(self, metapath: List[int], in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.metapath = metapath
        self.layers = nn.ModuleList()
        for i, rel_idx in enumerate(metapath[::-1]):#inverted order
            if i == 0:
                self.layers.append(MetaPathGNNLayer(in_channels, hidden_channels, rel_idx))
            else:
                self.layers.append(MetaPathGNNLayer(hidden_channels, hidden_channels, rel_idx))
        self.out_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        h_dict = x_dict.copy()
        for i, (src, rel, dst) in enumerate(reversed(self.metapath)):
            conv_idx = len(self.metapath) - 1 - i
            edge_index = edge_index_dict[(src, rel, dst)]
            h_dst = self.layers[conv_idx](
                x=h_dict[dst],
                h=h_dict[dst],
                edge_index=edge_index
            )
            h_dict[dst] = F.relu(h_dst)

        # for layer in self.layers:
        #     h = F.relu(layer(x, edge_index, edge_type, h))
        # return self.out_proj(h)
        start_type = self.metapath[0][0]
        return self.out_proj(h_dict[start_type])

# --- Attention tra metapath ---
class MetaPathSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, embeddings):  # [N, M, D]
        attn = self.encoder(embeddings)  # [N, M, D]
        pooled = attn.mean(dim=1)        # [N, D]
        return self.out_proj(pooled).squeeze(-1)  # [N]

# --- MPS-GNN completo ---
class MPSGNN(nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        metapaths: List[List[int]],  # list of list of relation indices
        metapath_counts: Dict[Tuple, int],
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_heads: int = 8,
        final_out_channels: int = 1,
    ):
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=hidden_channels * 2,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )

        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]],
            channels=hidden_channels * 2,
        )

        self.metapath_models = nn.ModuleList([
            MetaPathGNN(
                mp, in_channels=hidden_channels * 2,
                hidden_channels=hidden_channels,
                out_channels=out_channels
            )
            for mp in metapaths
        ])

        weights = torch.tensor(
            [metapath_counts.get(tuple(mp), 1) for mp in metapaths],
            dtype=torch.float
        )
        weights = weights / weights.sum()
        self.register_buffer("metapath_weights_tensor", weights)

        self.regressor = MetaPathSelfAttention(out_channels, num_heads=num_heads)

    def forward(self, batch: HeteroData, entity_table: str):
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] += rel_time

        embeddings = [
          model(x_dict, batch.edge_index_dict)
          for model in self.metapath_models 
      ] #create a list of the embeddings, one for each metapath

        #we start the metapath from the target node
        target_type = entity_table
        x_target = x_dict[target_type]
        edge_index = batch.edge_index_dict
        edge_type = batch.edge_types

        embeddings = [
            model(x_target, edge_index)
            for model in self.metapath_models
        ]

        # all_embeds = torch.stack(embeddings, dim=1)                     # [N, M, D]
        # weighted_embeds = all_embeds * self.metapath_weights_tensor.view(1, -1, 1)
        # out = self.regressor(weighted_embeds)                           # [N]
        # return out  # logits 
        concat = torch.stack(embeddings, dim=1) #concatenate the embeddings 
        weighted = concat * self.metapath_weights_tensor.view(1, -1, 1)
        
        return self.regressor(weighted) #finally apply regression








