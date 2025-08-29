"""
Using paper's metapaths concatenating approach.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, SAGEConv
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
from torch_geometric.data import HeteroData
from torch_frame.data.stats import StatType
from typing import Any, Dict, List, Tuple
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer


"""
In this implementation we are solving one major problem related to the 
previous version (which is referred to as MPSGNN_Model_old, please check git history),
which is that, when we used the full x_dict[nodetype]
tensors without checking which nodes were actually connected by the 
relation. This means we were doing message passing over all nodes, 
including nodes that are completely disconnected from the current relation.

The solution should be that instead considering all the dst e src nodes
for the message passing, we focus only on the ones for which we have at
leat an edge between src to dst.

In other words, this code:
def forward(self, x_dict, edge_index_dict):
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
        h_dict[dst] = F.relu(h_dst)
    start_type = self.metapath[0][0]
    return self.out_proj(h_dict[start_type])

was wrong because simply considered x=h_dict[dst], without exclude all the 
dst nodes that are not reached from src, adding them to the aggregation phase.

A solution could be to exclude all the nodes that are not reached from the 
relation, but this would generate a new problem to be managed:
Given edge_index = [[3, 4, 5], [6, 2, 3]], this means there's an edge from
x_dict["races"][3] to x_dict["drivers"][6], and so on. Suppose 
x_dict["races"] originally has shape [128, D] and x_dict["drivers"]
is [200, D]. If you do x_dict["races"] = x_dict["races"][[3,4,5]], then 
x_dict["races"] now has shape [3, D] where index 0 corresponds to global
node 3, index 1 to 4, and index 2 to 5. But edge_index still uses global
indices [3, 4, 5], so when your GNN tries to index x[3], it will go out
of bounds, because x only has indices [0,1,2] now. So after filtering nodes,
you must remap edge_index to the new local indices. If global → local is 
{3:0, 4:1, 5:2}, then remap edge_index[0] = [0,1,2].
"""

class MetaPathGNNLayerOriginal(MessagePassing):  
    """
    MetaPathGNNLayer implements equation 7 from the MPS-GNN paper (https://arxiv.org/abs/2412.00521).

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




# class MetaPathGNNLayer(MessagePassing):  
#     """
#     My update: follow original paper, but instead of applying a sum
#     aggregation, normalize the message in order to delete the 
#     bias given by the degree of the node.

#     We also implemented a temporal decading so that "old" messages
#     will weight less than the recent ones.
#     """
#     def __init__(self, in_channels: int, out_channels: int, relation_index: int):
#         super().__init__(aggr='add', flow='target_to_source')
#         self.relation_index = relation_index

#         # Linear layers for each component of equation 7
#         self.w_l = nn.Linear(in_channels, out_channels)  # for neighbor aggregation
#         self.w_0 = nn.Linear(in_channels, out_channels)  # for current hidden state
#         self.w_1 = nn.Linear(in_channels, out_channels)  # for original input features

#         self.gate = nn.Parameter(torch.tensor(0.5))

#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor, h: torch.Tensor, edge_weight: torch.Tensor = None ) -> torch.Tensor:
#         """
#         Args:
#             x: Original input features of the node (x_v), shape [N_dst, in_channels]
#             edge_index: Edge index for this relation, shape [2, num_edges]
#             h: Current hidden representation of the node (h_v), shape [N_dst, in_channels]
#             edge_weight: weight considering temporal proximity.
#         Returns:
#             Updated node representation after applying the layer, shape [N_dst, out_channels]
#         """

#         agg = self.propagate(edge_index, x=h, edge_weight = edge_weight)

#         row = edge_index[1] #dst-s
#         if edge_weight is None:
#             deg = torch.bincount(row, minlength=agg.size(0)).clamp(min=1).float().unsqueeze(-1)
#         else:
#             deg = torch.bincount(row, weights=edge_weight, minlength=agg.size(0)).clamp(min=1e-6).float().unsqueeze(-1)

#         agg = agg/deg
#         g = torch.sigmoid(self.gate)

#         return self.w_l(agg) + (1 - g) * self.w_0(h) + g * self.w_1(x)
        

#         #return self.w_l(agg) + self.w_0(h) + self.w_1(x)

#     def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor = None ) -> torch.Tensor:
#         if edge_weight is None:
#             return x_j
#         return x_j * edge_weight.unsqueeze(-1)








from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class MetaPathGNNLayer(MessagePassing):
    """
    Eq. (7) stile paper:
      h'_v = W_l * agg_{u in N_r(v)} h_u  + (1-g) * W_0 * h_v + g * W_1 * x_v
    Aggrego da src -> dst (source_to_target) e normalizzo su dst.
    """
    def __init__(self, hidden_channels: int):
        super().__init__(aggr="add", flow="source_to_target")
        self.w_l = nn.Linear(hidden_channels, hidden_channels)
        self.w_0 = nn.Linear(hidden_channels, hidden_channels)
        self.w_1 = nn.Linear(hidden_channels, hidden_channels)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        h_src: torch.Tensor,                 # [#src_loc, D]
        h_dst: torch.Tensor,                 # [#dst_loc, D]
        edge_index: torch.Tensor,            # [2, E] con indici locali (src_loc -> dst_loc)
        x_dst_orig: torch.Tensor,            # [#dst_loc, D]
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.propagate(
            edge_index,
            x=(h_src, h_dst),                # bipartito: x_j = h_src, x_i = h_dst
            edge_weight=edge_weight,
            size=(h_src.size(0), h_dst.size(0))
        )                                    # -> [#dst_loc, D]

        row = edge_index[1]                  # dst locali
        if edge_weight is None:
            deg = torch.bincount(row, minlength=h_dst.size(0)).clamp(min=1).float().unsqueeze(-1)
        else:
            deg = torch.bincount(row, weights=edge_weight, minlength=h_dst.size(0)).clamp(min=1e-6).float().unsqueeze(-1)
        out = out / deg

        g = torch.sigmoid(self.gate)
        return self.w_l(out) + (1. - g) * self.w_0(h_dst) + g * self.w_1(x_dst_orig)

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x_j if edge_weight is None else x_j * edge_weight.unsqueeze(-1)




#version 2, also the one really used!
class MetaPathGNN(nn.Module):
    """
    This is the network that express the GNN operations over a meta path.
    We create a GNN layer for each relation in the metapath. Then, we 
    propagate over the metapath using convolutions.
    Finally we apply a final projection to the initial node embeddings.

    So, we generate embeddings considering the metapath "metapath".
    A metapath is passed, and is a list of tuple (src, rel, dst).

    Here, we use MetaPathGNNLayer as GNN layer, which follows the paper 
    implementation.

    We also fixed a bug that was present in the previous versions: 

    self.convs[conv_idx](
                x=x_dst,
                h=x_dst,
                edge_index=edge_index_remapped
    )

    Here, we were passing to h the current state x, but it is redudant
    since we were passing it already. This was a minor mistake I forgot.


    We also aply a temporal decading weighting for the messages.
    """
    def __init__(self, metapath, hidden_channels, out_channels,
                 dropout_p: float = 0.1,
                 use_time_decay: bool = True,
                 init_lambda: float = 0.1,
                 time_scale: float = 1.0):
        super().__init__()
        self.metapath = metapath
        self.convs = nn.ModuleList()
        for i in range(len(metapath)):
            #self.convs.append(MetaPathGNNLayer(hidden_channels, hidden_channels, relation_index=i))
            self.convs.append(MetaPathGNNLayer(hidden_channels))

        
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(len(metapath))])
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout_p) for _ in range(len(metapath))])
        
        #UPDATE:
        self.use_time_decay = use_time_decay
        self.time_scale = float(time_scale)
        self.raw_lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(init_lambda))) for _ in range(len(metapath))
        ])

        self.out_proj = nn.Linear(hidden_channels, out_channels)


    def forward(self, x_dict, edge_index_dict,
                node_time_dict: dict = None):
        #edge_type_dict is the list of edge types
        #edge_index_dict contains for each edge_type the edges

        #update, instead of this:
        #h_dict = x_dict.copy()
        #we store x0_dict for x and h_dict that will be updated path by path:
        #x0_dict = {k: v.detach() for k, v in x_dict.items()}   # freezed original features
        x0_dict = {k: v for k, v in x_dict.items()} 
        h_dict  = {k: v.clone()  for k, v in x_dict.items()}   # current state: to update

        def pos_lambda(raw):  # λ > 0
            return F.softplus(raw) + 1e-8

        for i, (src, rel, dst) in enumerate(reversed(self.metapath)): #reversed: follow metapath starting from last path!
            conv_idx = len(self.metapath) - 1 - i
            edge_index = edge_index_dict[(src, rel, dst)]

            #Store the list of the nodes that are used in the 
            #relation "(src, rel, dst)":
            #this was not mentioned in original paper
            src_nodes = edge_index[0].unique()
            dst_nodes = edge_index[1].unique()
            """
            Example
            src_nodes = [3, 4, 5]
            dst_nodes = [6, 2, 3]
            """

            #To solve the problem mentioend at the beginning of this file, we use a global->
            #to local mapping:
            #src_map = {int(i.item()): i for i, n in enumerate(src_nodes)}
            src_map = {int(n.item()): i for i, n in enumerate(src_nodes)}
            dst_map = {int(n.item()): i for i, n in enumerate(dst_nodes)}
            """
            Example
            if:
            src_nodes = [3, 4, 5]
            dst_nodes = [6, 2, 3]

            then:
            src_map = {3: 0, 4: 1, 5: 2}
            dst_map = {2: 0, 3: 1, 6: 2}
            """

            #Filter: consider only the nodes in the relation
            x_src = h_dict[src][src_nodes]
            x_dst = h_dict[dst][dst_nodes]
            """
            Example
            x_src = [emb(3), emb(4), emb(5)]
            x_dst = [emb(6), emb(2), emb(3)]
            """

            edge_index_remapped = torch.stack([
                torch.tensor([src_map[int(x)] for x in edge_index[0].tolist()], device=edge_index.device, dtype=torch.long),
                torch.tensor([dst_map[int(x)] for x in edge_index[1].tolist()], device=edge_index.device, dtype=torch.long)
            ])
            """
            Example
            if:
            src_map = {3: 0, 4: 1, 5: 2}
            dst_map = {2: 0, 3: 1, 6: 2}

            then:
            edge_index_remapped = tensor([[0, 1, 2],
                                         [0, 1, 2]])
            """

            #UPDATE: take the original x and update h representation:
            x_dst_orig = x0_dict[dst][dst_nodes]   # ORIGINAL
            h_dst_curr = h_dict[dst][dst_nodes]    # CURRENT

            #Δt for edge and weight: exp(-λ Δt)
            edge_weight = None
            if self.use_time_decay and (node_time_dict is not None) and (src in node_time_dict) and (dst in node_time_dict):
                t_src_all = node_time_dict[src].float()
                t_dst_all = node_time_dict[dst].float()
                t_src_e = t_src_all[edge_index[0]]  # [E_rel]
                t_dst_e = t_dst_all[edge_index[1]]  # [E_rel]
                delta = (t_dst_e - t_src_e) / float(self.time_scale)
                delta = delta.clamp(min=0.0)

                lam = pos_lambda(self.raw_lambdas[conv_idx])      # scalare > 0
                z = (-lam * delta).clamp(min=-60.0)               # stabilità numerica
                edge_weight = torch.exp(z)    



            h_src_curr = h_dict[src][src_nodes]   

            h_dst = self.convs[conv_idx](
                h_src=h_src_curr,
                h_dst=h_dst_curr,
                edge_index=edge_index_remapped,
                x_dst_orig=x_dst_orig,
                edge_weight=edge_weight
            )
            h_dst = F.relu(h_dst)
            h_dst = self.norms[conv_idx](h_dst)
            h_dst = self.dropouts[conv_idx](h_dst)
            h_dict[dst].index_copy_(0, dst_nodes, h_dst)

           
        target_type = self.metapath[-1][2]      #last dst (== 'drivers')
        return self.out_proj(h_dict[target_type])





#Version 1, using SAGEConv:
class MetaPathGNN_SAGEConv(nn.Module):
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
            conv = SAGEConv((-1, -1), hidden_channels)   #----> tune
            self.convs.append(conv)

        self.out_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_type_dict = None):
        h_dict = x_dict.copy()
        for i, (src, rel, dst) in enumerate(reversed(self.metapath)): #reversed
            conv_idx = len(self.metapath) - 1 - i #obtaining the correct index
            edge_index = edge_index_dict[(src, rel, dst)]
            h_dst = self.convs[conv_idx]((h_dict[src], h_dict[dst]), edge_index)
            h_dict[dst] = F.relu(h_dst)
        start_type = self.metapath[0][0]
        return self.out_proj(h_dict[start_type])



class MetaPathSelfAttention(nn.Module):
    """
    This module applies Transformer-based self-attention between the different metapaths.
    It uses a TransformerEncoder. This module apply self attention between the different
    metapaths. It is mostly used as a source of explainability, in orfer to assess the 
    relevance contribution of every metapath to the final result.
    It was not present in the original paper.
    """
    def __init__(self, dim, num_heads=4, out_dim=1, num_layers=4):
        super().__init__()
        self.out_dim = out_dim
        self.attn_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )

        #UPDATE:
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, out_dim)
        )

    def forward(self, metapath_embeddings: torch.Tensor, return_attention: bool=False):  # [N, M, D]
        """
        metapath_embeddings: [N, M, D]
        return_attention: if True, returns intermediate attention embeddings as well
        """
        assert not torch.isnan(metapath_embeddings).any(), "NaN detected"
        assert not torch.isinf(metapath_embeddings).any(), "Inf detected"

        #UPDATE:
        ctx = self.attn_encoder(metapath_embeddings)
        _, A = self.self_attn(ctx, ctx, ctx, need_weights=True, average_attn_weights=True)
        w = A.mean(dim=-1)  
        w = torch.softmax(w, dim=1) 
        gated = ctx * w.unsqueeze(-1)
        pooled = gated.sum(dim=1) 
        out = self.output_proj(pooled).squeeze(-1) 
        if return_attention:
            return out, w
        return out


        # attn_out = self.attn_encoder(metapath_embeddings)  # [N, M, D]
        # pooled = attn_out.mean(dim=1)                      # [N, D]
        # out = self.output_proj(pooled).squeeze(-1)        # [N]
        # if return_attention:
        #     return out, attn_out
        # return out







class XMetaPath2(nn.Module):
    def __init__(self,
                 data: HeteroData,
                 col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
                 metapaths: List[List[int]],  # rel_indices
                 #metapath_counts: Dict[Tuple, int], #statistics of each metapath
                 hidden_channels: int = 64,
                 out_channels: int = 64,
                 num_heads: int = 8,
                 final_out_channels: int = 1,
                 num_layers: int = 4,
                 dropout_p: float = 0.1,
                 time_decay: bool = False,
                 init_lambda: float = 0.1,
                 time_scale: float = 1.0):
        super().__init__()

        self.metapath_models = nn.ModuleList([
            MetaPathGNN(mp, hidden_channels, out_channels,
                        dropout_p=dropout_p,
                        use_time_decay=time_decay,
                        init_lambda=init_lambda,
                        time_scale=time_scale)
            for mp in metapaths
        ]) # we construct a specific MetaPathGNN for each metapath

        self.regressor = MetaPathSelfAttention(out_channels, num_heads=num_heads, out_dim=final_out_channels, num_layers=num_layers)

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
        
        
        embeddings = [#x_dict, edge_index_dict
            model(
                x_dict, 
                batch.edge_index_dict,
                node_time_dict=batch.time_dict
            )
            for model in self.metapath_models 
        ] #create a list of the embeddings, one for each metapath
        concat = torch.stack(embeddings, dim=1) #concatenate the embeddings 
        #weighted = concat * self.metapath_weights_tensor.view(1, -1, 1) #to consider to add statisitcs
        
        return self.regressor(concat) #finally apply regression; just put weighted instead of concat if statistics
     

    

    
