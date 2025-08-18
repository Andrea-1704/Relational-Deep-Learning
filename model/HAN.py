"""
In this implementation we are going to try implementing the Heterogeneous GAT model 
according to 'https://arxiv.org/abs/1903.07293'.
To provide the metapaths we are going to use the proposed method which is an estension
of 'arxiv.org/abs/2412.00521'.

Here we are going to propose the model to use, once we have given the metapaths.
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import os
import torch
import relbench
import numpy as np
from torch.nn import BCEWithLogitsLoss, L1Loss
from relbench.datasets import get_dataset
from relbench.tasks import get_task
import math
from tqdm import tqdm
import torch_geometric
import torch_frame
from torch_geometric.seed import seed_everything
from relbench.modeling.utils import get_stype_proposal
from collections import defaultdict
import requests
from io import StringIO
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph
from torch.nn import BCEWithLogitsLoss
import copy
from typing import Any, Dict, List
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from torch_geometric.loader import NeighborLoader
import pyg_lib
from sklearn.metrics import mean_squared_error
#per lo scheduler
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import networkx as nx
#import os
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import torch
from torch import nn
from torch_geometric.nn import Linear
from torch_geometric.utils import softmax, degree
from collections import defaultdict
import networkx as nx

NodeType  = str
EdgeType  = Tuple[str, str, str]           # (src_type, rel_type, dst_type)
MetaPath  = List[EdgeType]


def compose_two_edges(edge_index1: Tensor, edge_index2: Tensor) -> Tensor:
    """
    edge_index1: [2, E1] con (src0 -> mid)
    edge_index2: [2, E2] con (mid -> dst2)
    ritorna:    [2, E]   con (src0 -> dst2)
    """
    # map: mid -> list(dst2)
    mid_to_dsts = {}
    src1, mid1 = edge_index1
    mid2, dst2 = edge_index2
    for m, d in zip(mid2.tolist(), dst2.tolist()):
        mid_to_dsts.setdefault(m, []).append(d)

    src_out = []
    dst_out = []
    for s, m in zip(src1.tolist(), mid1.tolist()):
        if m in mid_to_dsts:
            ds = mid_to_dsts[m]
            src_out.extend([s] * len(ds))
            dst_out.extend(ds)
    if len(src_out) == 0:
        return torch.empty(2, 0, dtype=edge_index1.dtype, device=edge_index1.device)
    ei = torch.tensor([src_out, dst_out], device=edge_index1.device, dtype=edge_index1.dtype)
    # coalesce (rimuovi duplicati) via sort+unique
    if ei.numel() > 0:
        key = ei[0] * (ei[1].max() + 1 if ei[1].numel() > 0 else 1) + ei[1]
        _, unique_idx = torch.unique(key, sorted=False, return_inverse=False, return_counts=False, return_index=True)
        ei = ei[:, unique_idx]
    return ei

def compose_metapath_on_batch(
    edge_index_dict: Dict[EdgeType, Tensor],
    meta_path: MetaPath,
    target_num_nodes: int,
    device=None,
) -> Tensor:
    """
    Costruisce l'edge_index omogeneo tra nodi del tipo target per il meta-path dato.
    Restituisce edge_index [2, E] con convenzione (src -> dst) sul tipo target.
    Aggiunge self-loops (i -> i).
    """
    assert len(meta_path) >= 1
    if device is None:
        # prendi un device da uno degli edge_index presenti
        for et, ei in edge_index_dict.items():
            device = ei.device
            break

    # Prendi i primi due edge come base di composizione
    e0 = edge_index_dict[meta_path[0]]  # (start -> n1)
    # Convertiamo a (src->dst) già coerente con PyG
    comp = e0
    # Componi in catena
    for et in meta_path[1:]:
        e_next = edge_index_dict[et]     # (nk -> nk+1)
        comp = compose_two_edges(comp, e_next)
        if comp.size(1) == 0:
            break

    # comp ora è (start -> end). Per HAN servono archi tra nodi target (start=end=target).
    # Aggiungi self-loops (i -> i)
    loops = torch.arange(target_num_nodes, device=device, dtype=comp.dtype).unsqueeze(0).repeat(2, 1)
    comp = torch.cat([comp, loops], dim=1) if comp.numel() > 0 else loops

    return comp  # (src -> dst) sul target







class TypeSpecificLinear(nn.Module):
    def __init__(self, in_channels_by_ntype: Dict[NodeType, int], out_dim: int):
        super().__init__()
        self.proj = nn.ModuleDict({
            ntype: nn.Linear(in_ch, out_dim, bias=False)
            for ntype, in_ch in in_channels_by_ntype.items()
        })

    def reset_parameters(self):
        for lin in self.proj.values():
            nn.init.xavier_uniform_(lin.weight)

    def forward(self, x_dict: Dict[NodeType, Tensor]) -> Dict[NodeType, Tensor]:
        return {ntype: self.proj[ntype](x) for ntype, x in x_dict.items()}





from torch_geometric.utils import softmax

class MetaPathGATLayer(nn.Module):
    def __init__(self, in_dim: int, heads: int = 8, dropout: float = 0.6, negative_slope: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        # vettori a_ϕ per head: [H, 2*in_dim]
        self.att = nn.Parameter(torch.empty(heads, 2 * in_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att)

    def forward(self, h_t: Tensor, edge_index: Tensor) -> Tensor:
        """
        h_t:        [N, d]  (features proiettate del tipo target)
        edge_index: [2, E]  (src -> dst) per il meta-path, con self-loops inclusi
        ritorna:    [N, H*d] (concat sui head) = Z^(ϕ)
        """
        N, d = h_t.size()
        src, dst = edge_index
        # prendi h_i/h_j per ogni arco (dst/src)
        h_i = h_t[dst].unsqueeze(1).expand(-1, self.heads, -1)  # [E, H, d]
        h_j = h_t[src].unsqueeze(1).expand(-1, self.heads, -1)  # [E, H, d]
        cat = torch.cat([h_i, h_j], dim=-1)                     # [E, H, 2d]
        e = self.leaky_relu(torch.einsum('ehd,hd->eh', cat, self.att))  # [E, H]
        alpha = softmax(e, dst, num_nodes=N)                    # masked softmax sui vicini di i=dst
        alpha = self.dropout(alpha)
        m = h_j * alpha.unsqueeze(-1)                           # [E, H, d]
        out = h_t.new_zeros((N, self.heads, d))                 # [N, H, d]
        out.index_add_(0, dst, m)                               # somma sui vicini
        out = F.elu(out)
        out = out.reshape(N, self.heads * d)                    # concat heads
        return out





class SemanticAttention(nn.Module):
    def __init__(self, in_dim_concat: int, hidden_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(in_dim_concat, hidden_dim)
        self.q  = nn.Parameter(torch.empty(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.q.unsqueeze(0))

    def forward(self, Z_list: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Z_list: liste di Z^(ϕ) [N, D]
        ritorna: Z_fused [N, D], V (pesi per meta-path) [P]
        """
        scores = []
        for Zϕ in Z_list:
            U = torch.tanh(self.fc(Zϕ))         # [N, H]
            s = U @ self.q                      # [N]
            Fϕ = s.mean(dim=0, keepdim=True)    # media sui nodi
            scores.append(Fϕ)
        scores = torch.cat(scores, dim=0).squeeze(-1)  # [P]
        V = torch.softmax(scores, dim=0)               # [P]
        Z = torch.stack(Z_list, dim=0)                 # [P, N, D]
        Z = (V.view(-1, 1, 1) * Z).sum(dim=0)         # [N, D]
        return Z, V







class HeteroHAN(nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        heads: int,
        meta_paths_dict: Dict[NodeType, List[MetaPath]],   # tutti i meta-path per ogni tipo target
        dropout: float = 0.6,
        sem_hidden: int = 128,
    ):
        super().__init__()
        assert channels % heads == 0, "channels deve essere multiplo di heads (concat)."
        self.heads = heads
        self.channels = channels
        self.proj_dim = channels // heads
        self.meta_paths = meta_paths_dict

        # Proiezione type-specific M_q: dalle tue feature (channels) a d = channels//heads
        in_dims_by_ntype = {nt: channels for nt in node_types}
        self.proj = TypeSpecificLinear(in_dims_by_ntype, out_dim=self.proj_dim)

        # Un layer di node-level attention per ciascun meta-path
        self.gat_by_target: nn.ModuleDict = nn.ModuleDict()
        for ntype, paths in meta_paths_dict.items():
            self.gat_by_target[ntype] = nn.ModuleList([
                MetaPathGATLayer(in_dim=self.proj_dim, heads=heads, dropout=dropout)
                for _ in paths
            ])

        # Semantic-level attention (condiviso)
        self.semantic_att = SemanticAttention(in_dim_concat=self.proj_dim * heads, hidden_dim=sem_hidden)

        self.reset_parameters()

    def reset_parameters(self):
        self.proj.reset_parameters()
        for ml in self.gat_by_target.values():
            for layer in ml:
                layer.reset_parameters()
        self.semantic_att.reset_parameters()

    @torch.no_grad()
    def _build_edge_index_for_target(
        self,
        target_type: NodeType,
        edge_index_dict: Dict[EdgeType, Tensor],
        num_nodes_dict: Dict[NodeType, int],
    ) -> List[Tensor]:
        """Crea gli edge_index omogenei per tutti i meta-path del target (sul batch)."""
        edge_indices = []
        paths = self.meta_paths.get(target_type, [])
        N_t = num_nodes_dict[target_type]
        for meta_path in paths:
            # Safety: controlla che ϕ parta e finisca sul tipo target
            if len(meta_path) == 0:
                continue
            start_ntype = meta_path[0][0]
            end_ntype   = meta_path[-1][2]
            if not (start_ntype == target_type and end_ntype == target_type):
                # se non combacia, salto
                continue
            # compone (src->dst) sul target e aggiunge self-loops
            ei = compose_metapath_on_batch(edge_index_dict, meta_path, target_num_nodes=N_t)
            edge_indices.append(ei)
        return edge_indices

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, int]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, int]] = None,
    ) -> Dict[NodeType, Tensor]:
        """
        Ritorna un nuovo x_dict con i tipi target aggiornati via HAN.
        Gli altri tipi sono pass-through (rimangono proiettati M_q o invariati).
        """
        # 1) Proiezione type-specific: x -> x' (dimensione d = channels//heads)
        x_proj = self.proj(x_dict)

        # 2) Per ogni tipo target con meta-path definiti: node-level + semantic-level
        out_dict: Dict[NodeType, Tensor] = {**x_dict}  # default: lascia inalterati
        # per passare dimensioni attuali dei nodi nel batch
        num_nodes_dict = {nt: x.size(0) for nt, x in x_dict.items()}

        for target_type, paths in self.meta_paths.items():
            if len(paths) == 0:
                continue
            # edge_index omogenei per ciascun ϕ
            edge_index_list = self._build_edge_index_for_target(target_type, edge_index_dict, num_nodes_dict)
            if len(edge_index_list) == 0:
                # nessun arco composito (batch troppo piccolo)? fallback: solo self-feat
                out_dict[target_type] = x_dict[target_type]
                continue

            h_t = x_proj[target_type]            # [N_t, d]
            Z_list = []
            for layer, ei in zip(self.gat_by_target[target_type], edge_index_list):
                Zϕ = layer(h_t, ei)              # [N_t, heads*d] == [N_t, channels]
                Z_list.append(Zϕ)

            # 3) Semantic-level attention (condivisa)
            Z_fused, V = self.semantic_att(Z_list)  # [N_t, channels], [P]
            out_dict[target_type] = Z_fused

        return out_dict





class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict["StatType", Any]]],
        num_layers: int,               
        channels: int,
        out_channels: int,
        aggr: str,                     
        norm: str,
        shallow_list: List[NodeType] = [],
        id_awareness: bool = False,
        predictor_n_layers : int = 1,
        meta_paths_dict: Dict[NodeType, List[MetaPath]] = None,   # <--- AGGIUNTO
        heads: int = 8,                                         
        sem_hidden: int = 128,                                  
        dropout_att: float = 0.6,                               
    ):
        super().__init__()

        if meta_paths_dict is None:
            meta_paths_dict = {}

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]],
            channels=channels,
        )


        self.gnn = HeteroHAN(
            node_types=list(data.node_types),
            edge_types=list(data.edge_types),
            channels=channels,
            heads=heads,
            meta_paths_dict=meta_paths_dict,
            dropout=dropout_att,
            sem_hidden=sem_hidden,
        )

        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=predictor_n_layers,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = torch.nn.Embedding(1, channels) if id_awareness else None
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(self, batch: HeteroData, entity_table: NodeType) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for nt, rel_time in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rel_time
        for nt, embedding in self.embedding_dict.items():
            x_dict[nt] = x_dict[nt] + embedding(batch[nt].n_id)

        # HAN aggiorna solo i tipi per cui hai fornito meta-path; gli altri passano invariati
        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            getattr(batch, "num_sampled_nodes_dict", None),
            getattr(batch, "num_sampled_edges_dict", None),
        )
        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(self, batch: HeteroData, entity_table: NodeType, dst_table: NodeType) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError("id_awareness must be set True to use forward_dst_readout")
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for nt, rel_time in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rel_time
        for nt, embedding in self.embedding_dict.items():
            x_dict[nt] = x_dict[nt] + embedding(batch[nt].n_id)

        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        return self.head(x_dict[dst_table])
