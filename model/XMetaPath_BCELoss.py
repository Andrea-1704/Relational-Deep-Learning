"""
Using BCE focal loss
"""
"""
Test using the edge weight decay
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

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing



def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
    reduction: str = "mean",
):
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    # Probabilità della classe corretta
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)  # [N]
    # Modulazione Focal
    mod = (1.0 - p_t).pow(gamma)
    loss = mod * bce
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss  # 'none'

def make_criterion_from_train_labels(
    y_train: torch.Tensor,
    device: torch.device,
    mode: str = "auto",         # "auto" | "bce" | "focal"
    focal_gamma: float = 2.0,
    focal_thresh: float = 0.10,
):
    # y_train è {0,1}
    num_pos = (y_train == 1).sum().item()
    num_neg = (y_train == 0).sum().item()
    # Evita div/0
    if num_pos == 0:
        ratio = 1.0
    else:
        ratio = num_neg / max(1, num_pos)
    pos_weight = torch.tensor([float(ratio)], device=device)

    pos_rate = (num_pos / max(1, (num_pos + num_neg)))
    use_focal = (mode == "focal") or (mode == "auto" and pos_rate < focal_thresh)

    if use_focal:
        def crit(logits, targets):
            return focal_bce_with_logits(
                logits, targets, gamma=focal_gamma, pos_weight=pos_weight, reduction="mean"
            )
        info = {"loss": "focal_bce_with_logits", "gamma": focal_gamma, "pos_rate": pos_rate, "pos_weight": ratio}
    else:
        bcel = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        def crit(logits, targets):
            return bcel(logits, targets)
        info = {"loss": "bce_with_logits", "pos_rate": pos_rate, "pos_weight": ratio}

    return crit, info





class MetaPathGNNLayer(MessagePassing):
    def __init__(self, hidden_channels: int):
        super().__init__(aggr="add", flow="source_to_target")
        self.w_l = nn.Linear(hidden_channels, hidden_channels)
        self.w_0 = nn.Linear(hidden_channels, hidden_channels)
        self.w_1 = nn.Linear(hidden_channels, hidden_channels)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        h_src: torch.Tensor,                 
        h_dst: torch.Tensor,                 
        edge_index: torch.Tensor,            
        x_dst_orig: torch.Tensor,         
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.propagate(
            edge_index,
            x=(h_src, h_dst),                
            edge_weight=edge_weight,
            size=(h_src.size(0), h_dst.size(0))
        )                                  

        row = edge_index[1]               
        if edge_weight is None:
            deg = torch.bincount(row, minlength=h_dst.size(0)).clamp(min=1).float().unsqueeze(-1)
        else:
            deg = torch.bincount(row, weights=edge_weight, minlength=h_dst.size(0)).clamp(min=1e-6).float().unsqueeze(-1)
        out = out / deg

        g = torch.sigmoid(self.gate)
        return self.w_l(out) + (1. - g) * self.w_0(h_dst) + g * self.w_1(x_dst_orig)

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x_j if edge_weight is None else x_j * edge_weight.unsqueeze(-1)




class MetaPathGNN(nn.Module):
    def __init__(self, metapath, hidden_channels, out_channels,
                 dropout_p: float = 0.1,
                 use_time_decay: bool = True,
                 init_lambda: float = 0.1,
                 time_scale: float = 1.0):
        super().__init__()
        self.metapath = metapath
        self.convs = nn.ModuleList()
        for i in range(len(metapath)):
            self.convs.append(MetaPathGNNLayer(hidden_channels))

        
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(len(metapath))])
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout_p) for _ in range(len(metapath))])
        
        
        self.use_time_decay = use_time_decay
        self.time_scale = float(time_scale)
        self.raw_lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(init_lambda))) for _ in range(len(metapath))
        ])

        self.out_proj = nn.Linear(hidden_channels, out_channels)


    def forward(self, x_dict, edge_index_dict,
                node_time_dict: dict = None):
        x0_dict = {k: v for k, v in x_dict.items()} 
        h_dict  = {k: v.clone()  for k, v in x_dict.items()}   # current state: to update

        def pos_lambda(raw):  # λ > 0
            return F.softplus(raw) + 1e-8

        for i, (src, rel, dst) in enumerate(reversed(self.metapath)): 
            conv_idx = len(self.metapath) - 1 - i
            edge_index = edge_index_dict[(src, rel, dst)]

            src_nodes = edge_index[0].unique()
            dst_nodes = edge_index[1].unique()
            
            src_map = {int(n.item()): i for i, n in enumerate(src_nodes)}
            dst_map = {int(n.item()): i for i, n in enumerate(dst_nodes)}
            
            x_src = h_dict[src][src_nodes]
            x_dst = h_dict[dst][dst_nodes]
            

            edge_index_remapped = torch.stack([
                torch.tensor([src_map[int(x)] for x in edge_index[0].tolist()], device=edge_index.device, dtype=torch.long),
                torch.tensor([dst_map[int(x)] for x in edge_index[1].tolist()], device=edge_index.device, dtype=torch.long)
            ])
            
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









class MetaPathSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, out_dim=1, num_layers=4):
        super().__init__()
        self.out_dim = out_dim
        self.attn_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
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


        ctx = self.attn_encoder(metapath_embeddings)
        _, A = self.self_attn(ctx, ctx, ctx, need_weights=True, average_attn_weights=True)
        w = A.mean(dim=1)  
        w = torch.softmax(w, dim=1) 
        gated = ctx * w.unsqueeze(-1)
        pooled = gated.sum(dim=1) 
        out = self.output_proj(pooled).squeeze(-1) 
        if return_attention:
            return out, w
        return out




class XMetaPath2(nn.Module):
    def __init__(self,
                 data: HeteroData,
                 col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
                 metapaths: List[List[int]],  
                 hidden_channels: int = 64,
                 out_channels: int = 64,
                 num_heads: int = 8,
                 final_out_channels: int = 1,
                 num_layers: int = 4,
                 dropout_p: float = 0.1,
                 time_decay: bool = True, ######CHANGEE
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
        ]) 

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
     
     