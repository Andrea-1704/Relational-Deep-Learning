import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import MLP
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
from torch_geometric.data import HeteroData
from torch_frame.data.stats import StatType
from typing import Any, Dict, List, Tuple
from torch_geometric.nn import SAGEConv

class MetaPathGNNLayer(MessagePassing):
    """
    MetaPathGNNLayer implements equation 7 from the MPS-GNN paper.

    h'_v = W_l * sum_{u in N(v)} h_u + W_0 * h_v + W_1 * x_v
    where:
      - h_v is the current hidden state of node v,
      - x_v is the original embedding of v,
      - h_u are the neighbors' embeddings.
    """
    def __init__(self, in_channels: int, out_channels: int, relation_index: int):
        super().__init__(aggr='add', flow='target_to_source')
        self.relation_index = relation_index

        # Linear layers for each component of equation 7
        self.w_l = nn.Linear(in_channels, out_channels)  # for neighbor aggregation
        self.w_0 = nn.Linear(in_channels, out_channels)  # for current hidden state
        self.w_1 = nn.Linear(in_channels, out_channels)  # for original input features

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Original input features of the node (x_v), shape [N_dst, in_channels]
            edge_index: Edge index for this relation, shape [2, num_edges]
            h: Current hidden representation of the node (h_v), shape [N_dst, in_channels]
        Returns:
            Updated node representation after applying the layer, shape [N_dst, out_channels]
        """

        agg_messages = self.propagate(edge_index, x=h)

        return self.w_l(agg_messages) + self.w_0(h) + self.w_1(x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j

#version 2 
class MetaPathGNN(nn.Module):
    """
    This is the network that express the GNN operations over a meta path.
    We create a GNN layer for each relation in the metapath. Then, we 
    propagate over the metapath using convolutions.
    Finally we apply a final prejection to the initial node embeddings.

    So, we generate embeddings considering the metapath "metapath".
    A metapath is passed, and is a list of tuple (src, rel, dst).

    Here, we use MetaPathGNNLayer as GNN layer, which follows the paper 
    implementation.
    """
    def __init__(self, metapath, hidden_channels, out_channels):
        super().__init__()
        self.metapath = metapath
        self.convs = nn.ModuleList()
        for i in range(len(metapath)):
            self.convs.append(MetaPathGNNLayer(hidden_channels, hidden_channels, relation_index=i))
        self.out_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_type_dict):
        #edge_type_dict is the list of edge types
        #edge_index_dict contains for each edge_type the edges
        h_dict = x_dict.copy()
        for i, (src, rel, dst) in enumerate(reversed(self.metapath)): #reversed
            conv_idx = len(self.metapath) - 1 - i
            edge_index = edge_index_dict[(src, rel, dst)]
            #only the one of the relation specified
            h_dst = self.convs[conv_idx](
                x=h_dict[dst],
                h=h_dict[dst],
                edge_index=edge_index
            )
            #residual information (MLP post-residual):
            res = h_dst + h_dict[dst]
            h_dict[dst] = self.mlp[conv_idx](F.relu(res))

            #h_dict[dst] = F.relu(h_dst)
        start_type = self.metapath[0][0]
        return self.out_proj(h_dict[start_type])



#Version one, using SAGEConv:
# class MetaPathGNN(nn.Module):
#     """
#     This is the network that express the GNN operations over a meta path.
#     We create a GNN layer for each relation in the metapath. Then, we 
#     propagate over the metapath using convolutions.
#     Finally we apply a final prejection to the initial node embeddings.

#     So, we generate embeddings considering the metapath "metapath".
#     A metapath is passed, and is a list of tuple (src, rel, dst).

#     Here, we use SAGEConv as GNN layer, but we can change this choice.

#     In Section 4.2 of the aforementioned paper, is indicated that they
#     use apply GNN layers by starting from the last layer, going back
#     to the first one. The aim is that target node receives immediatly
#     the informations coming from the reached node:
#     driver->race->circuit
#     We want to aggregate the information for making a prediction for 
#     the driver. By using a reverse technique, in the first GNN layer
#     race is going to receive and aggregate the information of the 
#     final destination of the metapath (in this case circuit) and in
#     the second GNN layer driver is going to receive the infromations 
#     from race, already considering circuit.

#     Also consider that we decided to use a RELU function, while the 
#     paper used a sigmoid function.
#     """
#     def __init__(self,
#                  metapath: List[Tuple[str, str, str]],
#                  hidden_channels: int,  #dimension of the hidden state, 
#                  #after each aggregation
#                  out_channels: int #final dimension of the 
#                  #embeddings produced by the GNN
#         ):
#         super().__init__()
#         self.metapath = metapath
#         self.convs = nn.ModuleList()

#         for _ in metapath:
#             #for each relation in the metapath we consider 
#             #a SAGEConv layer
#             conv = SAGEConv((-1, -1), hidden_channels)   #----> tune
#             self.convs.append(conv)

#         self.out_proj = nn.Linear(hidden_channels, out_channels)

#     def forward(self, x_dict, edge_index_dict, edge_type_dict = None):
#         h_dict = x_dict.copy()
#         for i, (src, rel, dst) in enumerate(reversed(self.metapath)): #reversed
#             conv_idx = len(self.metapath) - 1 - i #obtaining the correct index
#             edge_index = edge_index_dict[(src, rel, dst)]
#             h_dst = self.convs[conv_idx]((h_dict[src], h_dict[dst]), edge_index)
#             h_dict[dst] = F.relu(h_dst)
#         start_type = self.metapath[0][0]
#         return self.out_proj(h_dict[start_type])






class MetaPathSelfAttention(nn.Module):
    """
    This module apply self attention between the different metapaths. 
    It is mostly used as a source of explainability, in orfer to assess
    the relevance contribution of every metapath to the final result.
    It was not present in the original paper.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)  #final prediction
        )

    def forward(self, metapath_embeddings):  # [N, M, D]
        #self attention requires an input of shape [batch, seq_len, embed_dim]
        #print("metapath_embeddings.shape:", metapath_embeddings.shape)
        assert not torch.isnan(metapath_embeddings).any(), "NaN detected"
        assert not torch.isinf(metapath_embeddings).any(), "Inf detected"

        attn_output, _ = self.attn(metapath_embeddings, metapath_embeddings, metapath_embeddings)  # [N, M, D]
        pooled = attn_output.mean(dim=1)  #matapaths mean -> [N, D]
        return self.output_proj(pooled).squeeze(-1)  # output: [N]




class MPSGNN(nn.Module):
    """
    This is the complete Multi META Path model.
    It aggregates multiple metaPathGNN, each of which is dedicated
    to a single MetaPath (and, so produces an embeddings based
    on that metapath). 
    We then compute all the embeddings produced by each metapath, 
    through MetaPathGNN Model to make a final prediction, which in 
    our is a regression task (driver position only for now).

    We use a different GNN model for each distinct metapath 
    making the aggregation only considering that metapath

    We use HeteroEncoder in order to get intial enbeddings for nodes, we also use 
    the TemporalHeteroEncoder provided by Relbench, in order to model in a 
    valid way the temporal information.

    Finally, we employ a nn (a Regressor) to combine the results 
    of the aggregation of the different metapaths.

    This follows the formula indicated in section 4.3 at pag 8 (final lines).

    We decided to change a little bit this structure adding a self attention 
    mechanism for the metapaths. We do not only hope to slightly improve the 
    performances of the model, but we also desires to improve the 
    explainability of the model, having a score value for each of the metapaths
    we can easily indicate how much attention is given to every metapath, 
    allowing us to gain more explainability.

    Finally, we also added the statistic counts in order to indicate how many
    times each metapath is really employed in the graph.
    """
    def __init__(self,
                 data: HeteroData,
                 col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 metapaths: List[List[int]],  # rel_indices
                 metapath_counts: Dict[Tuple, int], #statistics of each metapath
                 hidden_channels: int = 64,
                 out_channels: int = 64,
                 final_out_channels: int = 1):
        super().__init__()
        
        self.metapath_models = nn.ModuleList([
            MetaPathGNN(mp, hidden_channels, out_channels)
            for mp in metapaths
        ]) # we construct a MetaPathGNN for each metapath

        weights = torch.tensor(
            [metapath_counts.get(tuple(mp), 1) for mp in metapaths], dtype=torch.float
        )
        weights = weights/weights.sum() #normalization of count
        self.register_buffer("metapath_weights_tensor", weights) 

        self.regressor = MetaPathSelfAttention(out_channels, num_heads=4)

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

    def forward(self, batch: HeteroData, entity_table=None):

      seed_time = batch[entity_table].seed_time
      
      x_dict = self.encoder(batch.tf_dict)

      rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
      )

      for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
      #print(f"edge_index_dict è {batch.edge_index_dict}")
      #print(f"edge_types è {batch.edge_types}") 
      embeddings = [
          model(x_dict, batch.edge_index_dict, batch.edge_types)
          for model in self.metapath_models 
      ] #create a list of the embeddings, one for each metapath
      concat = torch.stack(embeddings, dim=1) #concatenate the embeddings 
      weighted = concat * self.metapath_weights_tensor.view(1, -1, 1)
      
      return self.regressor(weighted) #finally apply regression

