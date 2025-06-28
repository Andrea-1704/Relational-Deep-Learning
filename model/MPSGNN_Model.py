import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import List, Tuple, Dict
from relbench.modeling.nn import HeteroEncoder
from torch_frame.data.stats import StatType
from typing import Any, Dict, List
from torch_geometric.data import HeteroData



class MetaPathGNN(nn.Module):
    def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 metapath: List[Tuple[str, str, str]],
                 hidden_channels: int,
                 out_channels: int):
        super().__init__()
        
        self.metapath = metapath
        self.convs = nn.ModuleList()

        for _ in metapath:
            conv = SAGEConv((-1, -1), hidden_channels)
            self.convs.append(conv)

        self.out_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        h_dict = x_dict.copy()

        for i, (src, rel, dst) in enumerate(self.metapath):
            edge_index = edge_index_dict[(src, rel, dst)]
            h_dst = self.convs[i]((h_dict[src], h_dict[dst]), edge_index)
            h_dict[dst] = F.relu(h_dst)

        start_type = self.metapath[0][0]
        return self.out_proj(h_dict[start_type])


class MPSGNN(nn.Module):
    def __init__(self,
                 data: HeteroData,
                 col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 metapaths: List[List[Tuple[str, str, str]]],
                 hidden_channels: int = 64,
                 out_channels: int = 64,
                 final_out_channels: int = 1):
        super().__init__()
        self.metapath_models = nn.ModuleList([
            MetaPathGNN(metadata, mp, hidden_channels, out_channels)
            for mp in metapaths
        ])
        self.regressor = nn.Linear(out_channels * len(metapaths), final_out_channels)

        self.encoder = HeteroEncoder(
            channels=hidden_channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict  
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict  
        )

    def forward(self, batch: HeteroData):
        x_dict = self.encoder(batch.tf_dict)
        embeddings = [model(x_dict, batch.edge_index_dict) for model in self.metapath_models]
        concat = torch.cat(embeddings, dim=-1)
        return self.regressor(concat).squeeze(-1)
