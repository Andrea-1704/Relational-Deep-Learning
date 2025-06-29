import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import MLP
from relbench.modeling.nn import HeteroEncoder
from torch_geometric.data import HeteroData
from torch_frame.data.stats import StatType
from typing import Any, Dict, List, Tuple
from torch_geometric.nn import SAGEConv



class MetaPathGNNLayer(MessagePassing):
    """
    This model follows the section 4.3 formulation. 
    The W0, W1, Wneigh appearing in equation 7 are threated as Linear 
    layers. 
    This allows to distinguish between the different contribution that 
    each component of eq 7 share.
    """
    def __init__(self, in_channels, out_channels, relation_index):
        super().__init__(aggr='add', flow="target_to_source")
        #we use the add function as aggregation function
        self.relation_index = relation_index
        self.w_0 = nn.Linear(in_channels, out_channels) # W_0 appears here: W_0 · h (in eq 7)
        self.w_l = nn.Linear(in_channels, out_channels) # W_l appears here: W_l * ∑ u∈N h_u 
        self.w_1 = nn.Linear(in_channels, out_channels) # W_1 appears here: W_1 * h(0) 

    def forward(self, x, edge_index, edge_type, h):
        edge_mask = edge_type == self.relation_index
        out = self.propagate(edge_index[:, edge_mask], x=h)
        #the propagate function call the message, aggregate and update function
        return self.w_l(out) + self.w_0(h) + self.w_1(x)

    def message(self, x_j):
        return x_j


class MetaPathGNN(nn.Module):
    """
    This is the network that express the GNN operations over a meta path.
    We create a GNN layer for each relation in the metapath. Then, we 
    propagate over the metapath using convolutions.
    Finally we apply a final prejection to the initial node embeddings.

    So, we generate embeddings considering the metapath "metapath".
    A metapath is passed, and is a list of tuple (src, rel, dst).

    Here, we use SAGEConv as GNN layer, but we can change this choice.
    """
    def __init__(self,
                 metapath: List[Tuple[str, str, str]],
                 hidden_channels: int,  #dimension of the hidden state, 
                 #after each aggregation
                 out_channels: int #final dimension of the 
                 #embeddings produced by the GNN
        ):
        super().__init__()
        self.metapath = metapath
        self.convs = nn.ModuleList()

        for _ in metapath:
            #for each relation in the metapath we consider 
            #a SAGEConv layer
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
                 metapaths: List[List[int]],  # rel_indices
                 hidden_channels: int = 64,
                 out_channels: int = 64,
                 final_out_channels: int = 1):
        super().__init__()
        
        self.metapath_models = nn.ModuleList([
            MetaPathGNN(mp, hidden_channels, out_channels)
            for mp in metapaths
        ])


        self.regressor = nn.Sequential(
            nn.Linear(out_channels * len(metapaths), out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, final_out_channels)
        )

        self.encoder = HeteroEncoder(
            channels=hidden_channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict
        )

    def forward(self, batch: HeteroData, entity_table=None):
      x_dict = self.encoder(batch.tf_dict)
      embeddings = [
          model(x_dict, batch.edge_index_dict)
          for model in self.metapath_models
      ]
      concat = torch.cat(embeddings, dim=-1)
      return self.regressor(concat).squeeze(-1)

