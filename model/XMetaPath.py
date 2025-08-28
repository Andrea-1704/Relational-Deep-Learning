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




        #     #we apply the MetaPAthGNNLayer for this specific path obtaining an embedding h_dst specific for that path:
        #     h_dst = self.convs[conv_idx](
        #         x=x_dst_orig, 
        #         edge_index=edge_index_remapped,
        #         h=h_dst_curr, 
        #         edge_weight=edge_weight
        #     )

        #     #since MetaPathGNNLayer is linear, here we apply the activation function:
        #     h_dst = F.relu(h_dst)

        #     #normalization + dropout:
        #     h_dst = self.norms[conv_idx](h_dst)
        #     h_dst = self.dropouts[conv_idx](h_dst)
            
            
        #     #UPDATE: update the h_dict embeddings with just computed ones
        #     h_dict[dst].index_copy_(0, dst_nodes, h_dst)

        #     """
        #     Change the embeddings of original nodes :
        #     src_nodes = [3, 4, 5]
        #     dst_nodes = [6, 2, 3]
        #     """

        # start_type = self.metapath[0][0] #start from the first node type
        # return self.out_proj(h_dict[start_type])




#Version 1, using SAGEConv:
class MetaPathGNN_SAGEConv(nn.Module):
    """
    This is the network that express the GNN operations over a meta path.
    We create a GNN layer for each relation in the metapath. Then, we 
    propagate over the metapath using convolutions.
    Finally we apply a final prejection to the initial node embeddings.

    So, we generate embeddings considering the metapath "metapath".
    A metapath is passed, and is a list of tuple (src, rel, dst).

    Here, we use SAGEConv as GNN layer, but we can change this choice.

    In Section 4.2 of the aforementioned paper, is indicated that they
    use apply GNN layers by starting from the last layer, going back
    to the first one. The aim is that target node receives immediatly
    the informations coming from the reached node:
    driver->race->circuit
    We want to aggregate the information for making a prediction for 
    the driver. By using a reverse technique, in the first GNN layer
    race is going to receive and aggregate the information of the 
    final destination of the metapath (in this case circuit) and in
    the second GNN layer driver is going to receive the infromations 
    from race, already considering circuit.

    Also consider that we decided to use a RELU function, while the 
    paper used a sigmoid function.
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



class XMetaPath(nn.Module):
    """
    This is the complete Multi META Path model.
    It aggregates multiple metaPathGNN, each of which is dedicated
    to a single MetaPath (and, so produces embeddings fir target nodes based
    on that metapath). 
    We then compute all the embeddings produced by each metapath, 
    through MetaPathGNN Model to make a final prediction.

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
                 metapaths: List[List[int]],  # rel_indices
                 metapath_counts: Dict[Tuple, int], #statistics of each metapath
                 hidden_channels: int = 64,
                 out_channels: int = 64,
                 num_heads: int = 8,
                 final_out_channels: int = 1,
                 num_layers: int = 4,
                 dropout_p: float = 0.1,
                 time_decay: bool = True,
                 init_lambda: float = 0.1,
                 time_scale: float = 1.0):
        super().__init__()

        #####LOCAL INTERPRETABILITY ONLY FOR F1, TO BE CHANGED TIME TO TIME 
        self.data = data
        self.pretty_fields = {
            "drivers": ["forename", "surname", "code", "nationality"],
            "races": ["year", "name", "round"],
            "constructors": ["name", "nationality"],
            "circuits": ["name", "location", "country"],
        }
        #####LOCAL INTERPRETABILITY

        
        self.metapath_models = nn.ModuleList([
            MetaPathGNN(mp, hidden_channels, out_channels,
                        dropout_p=dropout_p,
                        use_time_decay=time_decay,
                        init_lambda=init_lambda,
                        time_scale=time_scale)
            for mp in metapaths
        ]) # we construct a specific MetaPathGNN for each metapath

        weights = torch.tensor(
            [metapath_counts.get(tuple(mp), 1) for mp in metapaths], dtype=torch.float
        )
        weights = weights/weights.sum() #normalization of count
        self.register_buffer("metapath_weights_tensor", weights) 

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
        weighted = concat * self.metapath_weights_tensor.view(1, -1, 1)
        
        return self.regressor(weighted) #finally apply regression
     
    

    def _encode_features_prev(self, batch, entity_table):
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        return x_dict
    

    def _encode_features(self, batch, entity_table):
        device = next(self.parameters()).device

        tf_dict = {nt: tf.to(device) for nt, tf in batch.tf_dict.items()}

        x_dict = self.encoder(tf_dict)

        seed_time = batch[entity_table].seed_time
        if torch.is_tensor(seed_time):
            seed_time = seed_time.to(device)

        time_dict  = {nt: (t.to(device) if torch.is_tensor(t) else t)
                    for nt, t in batch.time_dict.items()}
        batch_dict = {nt: b.to(device) for nt, b in batch.batch_dict.items()}

        rel_time_dict = self.temporal_encoder(seed_time, time_dict, batch_dict)
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        return x_dict


    @torch.no_grad()
    def predict_with_details(self, batch, entity_table):
        device = next(self.parameters()).device
        self.eval()

        x_dict = self._encode_features(batch, entity_table)

        edge_index_dict = {k: v.to(device) for k, v in batch.edge_index_dict.items()}
        time_dict = {nt: (t.to(device) if torch.is_tensor(t) else t)
                    for nt, t in batch.time_dict.items()}

        per_path_emb = [m(x_dict, edge_index_dict, node_time_dict=time_dict)
                        for m in self.metapath_models]
        per_path_emb = torch.stack(per_path_emb, dim=1)
        weighted = per_path_emb * self.metapath_weights_tensor.view(1, -1, 1)
        out, attn_weights = self.regressor(weighted, return_attention=True)
        return out, attn_weights, weighted, per_path_emb

    
    @torch.no_grad()
    def _per_metapath_delta(self, weighted, baseline_out, target_idx):
        """
        Compute the Δ (drop of logit) removing a metapath at a time.
        Returns a list of float of length equal to the number of metapaths.
        """
        N, M, D = weighted.shape
        deltas = []
        for m in range(M):
            masked = weighted.clone()
            masked[:, m, :] = 0
            out_m = self.regressor(masked)                 # [N]
            delta = (baseline_out[target_idx] - out_m[target_idx]).item()
            deltas.append(delta)
        return deltas
    

    def _find_last_hop_to_target(self, metapath, entity_table):
        """Trova l'hop del metapath che aggiorna i nodi di tipo entity_table (quello con dst == entity_table)."""
        for (src, rel, dst) in metapath:
            if dst == entity_table:
                return (src, rel, dst)
        return None

    def _pretty_node(self, node_type: str, idx: int):
        """Ritorna un dict leggibile per il nodo (usa le colonne della tabella TorchFrame)."""
        try:
            df = self.data[node_type].tf.df
            row = df.iloc[int(idx)]
            fields = [c for c in self.pretty_fields.get(node_type, []) if c in df.columns]
            if not fields:
                # fallback: prova a pescare qualcosa di parlante
                candidates = [c for c in df.columns if any(k in c.lower() for k in ["name","title","forename","surname","code","country","nationality","year","round","location"])]
                fields = candidates[:3]
            pretty = {c: row[c] for c in fields}
            # costruiamo un display_name se possibile
            if {"forename","surname"}.issubset(set(fields)):
                pretty["display_name"] = f"{row['forename']} {row['surname']}"
            elif "name" in fields:
                pretty["display_name"] = str(row["name"])
            else:
                pretty["display_name"] = f"{node_type}#{int(idx)}"
            pretty["index"] = int(idx)  # teniamolo comunque
            return pretty
        except Exception:
            return {"display_name": f"{node_type}#{int(idx)}", "index": int(idx)}

    @torch.no_grad()
    def _neighbor_deltas_for_metapath(self, batch, entity_table, target_idx, mp_index, per_path_emb, weighted, baseline_out, top_k_nodes=5):
        """
        Per il meta-path mp_index, calcola Δ per i vicini (edge ablation) dell'hop che aggrega verso entity_table.
        Ritorna lista di dict {node_type, node_idx, delta, pretty:{...}} ordinata per delta desc.
        """
        # 1) individua hop che aggiorna i target
        metapath = self.metapath_models[mp_index].metapath
        hop = self._find_last_hop_to_target(metapath, entity_table)
        if hop is None:
            return []  # questo meta-path non aggiorna direttamente i target
        src, rel, dst = hop
        edge_index = batch.edge_index_dict[(src, rel, dst)]  # [2, E]
        # vicini (global idx) che puntano al target
        mask_dst = (edge_index[1] == int(target_idx))
        src_neighbors = edge_index[0][mask_dst]
        if src_neighbors.numel() == 0:
            return []

        # 2) pre-selezione veloce (se tanti vicini): prendi i più "attivi" per frequenza (qui: tutti)
        unique_src = src_neighbors.unique()

        results = []
        # 3) per ogni vicino, rimuovi solo quell'arco e misura Δ sul logit
        for u in unique_src.tolist():
            # copia edge_index_dict e filtra l'arco (u -> target_idx) solo per questo hop
            new_ei = edge_index[:, ~((edge_index[0] == u) & (edge_index[1] == int(target_idx)))]
            new_edge_index_dict = dict(batch.edge_index_dict)
            new_edge_index_dict[(src, rel, dst)] = new_ei

            # ricalcola SOLO l'embedding del meta-path mp_index
            x_dict = self._encode_features(batch, entity_table)
            new_emb_mp = self.metapath_models[mp_index](x_dict, new_edge_index_dict, node_time_dict=batch.time_dict)  # [N, D]

            # ricomponi il tensore [N, M, D] riusando gli altri canali
            new_per_path = per_path_emb.clone()
            new_per_path[:, mp_index, :] = new_emb_mp
            new_weighted = new_per_path * self.metapath_weights_tensor.view(1, -1, 1)

            new_out = self.regressor(new_weighted)  # [N]
            delta = (baseline_out[int(target_idx)] - new_out[int(target_idx)]).item()

            results.append({
                "node_type": src,
                "node_idx": int(u),
                "delta": float(delta),
                "pretty": self._pretty_node(src, int(u)),
            })

        # ordina e taglia
        results.sort(key=lambda r: r["delta"], reverse=True)
        return results[:top_k_nodes]


    @torch.no_grad()
    def explain_instance(self,
                        batch: HeteroData,
                        entity_table: str,
                        target_idx: int,
                        metapath_names: list,
                        top_k_metapaths: int = 3,
                        top_k_nodes: int = 5):
        """
        Spiegazione locale e self-explainable per una singola istanza (nodo target).
        """
        self.eval()

        # 1) pred + pesi di attenzione + embeddings
        out, w, weighted, per_path_emb = self.predict_with_details(batch, entity_table)
        pred = float(torch.sigmoid(out[int(target_idx)]).item() if out.dim() == 1 else out[int(target_idx)].item())

        # 2) Δ per meta-path (ablation)
        deltas = self._per_metapath_delta(weighted, out, int(target_idx))  # lista di float
        ranked = sorted(list(enumerate(deltas)), key=lambda x: x[1], reverse=True)

        # 3) top meta-path + nodi più influenti per ciascuno
        top_entries = []
        for m, delta_m in ranked[:top_k_metapaths]:
            node_contribs = self._neighbor_deltas_for_metapath(
                batch, entity_table, int(target_idx), m, per_path_emb, weighted, out, top_k_nodes=top_k_nodes
            )
            top_entries.append({
                "metapath_index": int(m),
                "metapath_name": metapath_names[m] if m < len(metapath_names) else f"mp#{m}",
                "attention_weight": float(w[int(target_idx), m].item()),
                "delta_logit": float(delta_m),
                "top_nodes": node_contribs,  # ciascuno con pretty info
            })

        # 4) pacchetto finale
        return {
            "entity_table": entity_table,
            "target_index": int(target_idx),
            "prediction": pred,
            "attention_per_metapath": {
                (metapath_names[i] if i < len(metapath_names) else f"mp#{i}"): float(w[int(target_idx), i].item())
                for i in range(w.size(1))
            },
            "contribution_per_metapath": {
                (metapath_names[i] if i < len(metapath_names) else f"mp#{i}"): float(deltas[i])
                for i in range(len(deltas))
            },
            "top_metapaths": top_entries
        }




def interpret_attention(
    model,
    batch: HeteroData,
    metapath_names: List[str],
    entity_table: str,
    visualize: bool = True,
    figsize: tuple = (12, 6)
) -> List[Dict[str, Any]]:
    """
    Interpret metapath self-attention scores for each target node in a batch.

    Args:
        model: A trained instance of XMetapath.
        batch: A torch_geometric HeteroData batch containing the current data.
        metapath_names: List of human-readable names of metapaths.
        entity_table: The target node type (e.g., "drivers").
        visualize: If True, display a heatmap of attention scores.
        figsize: Tuple for figure size in the visualization.

    Returns:
        List of dictionaries containing node indices and attention weights over metapaths.
    """
    model.eval()
    with torch.no_grad():
        #Encode static and temporal features
        seed_time = batch[entity_table].seed_time
        x_dict = model.encoder(batch.tf_dict)
        rel_time_dict = model.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)

        for node_type in rel_time_dict:
            x_dict[node_type] = x_dict[node_type] + rel_time_dict[node_type]

        #Compute node embeddings via each metapath-specific GNN
        embeddings = [
            m(x_dict, batch.edge_index_dict) for m in model.metapath_models
        ]  # List of [N, D]
        concat = torch.stack(embeddings, dim=1)  # [N, M, D]
        weighted = concat * model.metapath_weights_tensor.view(1, -1, 1)

        #Retrieve output + attention representations
        # _, attn_repr = model.regressor(weighted, return_attention=True)  # [N, M, D]
        # attn_scores = attn_repr.mean(dim=-1).cpu()  # [N, M]
        _, w_true = model.regressor(weighted, return_attention=True)    # [N, M]
        attn_scores = w_true.cpu()

        #Structure the interpretation output
        results = []
        for node_idx in range(attn_scores.size(0)):
            node_attn = attn_scores[node_idx].tolist()
            results.append({
                "node_index": node_idx,
                "attention_per_metapath": dict(zip(metapath_names, node_attn))
            })

        #visualize using a heatmap
        if visualize:
            df = pd.DataFrame(attn_scores.numpy(), columns=metapath_names)
            plt.figure(figsize=figsize)
            sns.heatmap(df, annot=True, fmt=".2f", cmap="YlOrBr", linewidths=0.5,
                        cbar_kws={"label": "Metapath Attention Weight"})
            plt.title(f"Metapath Contribution Heatmap for Nodes in '{entity_table}'", fontsize=14)
            plt.xlabel("Metapath")
            plt.ylabel("Node Index")
            plt.tight_layout()
            plt.show()

        return results



 



"""
This is my version of meta path model, if you encounter any problem, or mistake please 
do not hesitate to reach me out! :)



Main differences with respect to the original paper work:
1. HeteroEncoder
2. TemporalHeteroEncoder->the original paper was not designed for temporal graphs
3. Self attention between metapaths
4. MetapathGNN only consider nodes that are considered for a certain path. This 
   not only make sense logically: when considering a certain path we should not
   change the node embeddings of nodes that are of the type "src", but never 
   connected through the path to "dst"; but it also did show improvement in the 
   performance. The original paper did not discuss about this point, but in the 
   official documentation on GitHub we did not find this check.
5. Interpret_attention, which is possible only thanks to point 3, is implemented 
   above the model to have a fully transparent model.
6. Normalization in MetaPathGNNLayer.forward
7. Weighted messages based on the temporal proximity: the aggregation of the 
   messages that reach a certain destination should be weighted considering, not
   only the degree of nodes and their attention, but also how "recent" is the 
   message.
"""


"""
Short summary:
The model now builds node representations along each meta-path by iterating the 
relations in reverse order and, at every hop, extracting the local subgraph, 
remapping its global node ids to local indices, and updating only the destination
nodes that are actually touched. For each hop it keeps two streams: the original
per-node features frozen at the start and the current hidden state that evolves
through the layers. Messages from neighbors are aggregated on the hidden state 
and are degree-normalized; when node timestamps are available, each edge message
is additionally reweighted by an exponential time decay w = exp(−λ Δt) with 
Δt = max(0, t_dst − t_src), where λ is a learnable, strictly positive per-hop
parameter obtained via softplus and the exponent is clamped for numerical stability.
The hop update then combines the normalized neighbor signal with a learnable gated
residual between the current state and the original features, and applies ReLU, 
LayerNorm, and Dropout before writing the updated vectors back into the global 
tensor at the touched indices. After the last hop the representation for the 
meta-path’s start node type is projected; repeating this for all chosen meta-paths
yields a tensor [batch, num_paths, dim]. These per-path embeddings can be pre-scaled
by your static prior (e.g., usage counts) and are then fused by a self-attention 
regressor that first contextually encodes the path tokens and then applies a final
multi-head self-attention with need_weights enabled; the resulting true attention
weights over meta-paths act as a data-dependent gating to weight path embeddings 
before pooling and passing through an MLP to produce the prediction. The same true
attention weights are also returned by the interpretation routine so the heatmaps 
now reflect exactly the weights used in the forward pass. If node times are absent 
the decay weights are simply not used and the layer reduces to the previous 
behavior, so training and inference pipelines remain unchanged apart from the added
stability from degree normalization, the explicit gating between original features 
and evolving state, and the per-hop regularization (ReLU, LayerNorm, Dropout).
"""