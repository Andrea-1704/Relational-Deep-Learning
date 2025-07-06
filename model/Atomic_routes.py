
#####
# Implementing a Composite Message Passing with Atomic routes
# This code is based on the paper https://arxiv.org/abs/2502.06784.
####



import numpy as np
import math
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath("."))


import torch_geometric
import torch_frame
from torch_geometric.seed import seed_everything
from relbench.modeling.utils import get_stype_proposal
from collections import defaultdict
import requests
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph
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
from torch.nn import ModuleDict
import torch.nn.functional as F
from torch import nn
import random

from typing import Any, Dict, List, Optional

import torch
import torch_frame
from torch import Tensor
from torch_frame.nn.models import ResNet
from torch_geometric.nn import HeteroConv, LayerNorm, PositionalEncoding, SAGEConv
from torch_geometric.typing import EdgeType, NodeType

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from typing import Dict, Optional


#definiamo una funzione che estrae tutte le atomic routes:
from typing import List, Tuple, Optional
from torch_geometric.data import HeteroData

from collections import defaultdict

def get_atomic_routes(edge_type_list):
    """
    This is the Relbench official code.
    """
    src_to_tuples = defaultdict(list)
    for src, rel, dst in edge_type_list:
        if rel.startswith('f2p'):
            if src == dst:
                src = src + '--' + rel
            src_to_tuples[src].append((src, rel, dst))

    atomic_routes_list = []
    get_rev_edge = lambda edge: (edge[2], 'rev_' + edge[1], edge[0])
    for src, tuples in src_to_tuples.items():
        if '--' in src:
            src = src.split('--')[0]
        if len(tuples) == 1:
            _, rel, dst = tuples[0]
            edge = (src, rel, dst)
            atomic_routes_list.append(('dim-dim',) + edge)
            atomic_routes_list.append(('dim-dim',) + get_rev_edge(edge))
        else:
            for _, rel_q, dst_q in tuples:
                for _, rel_v, dst_v in tuples:
                    if rel_q != rel_v:
                        edge_q = (src, rel_q, dst_q)
                        edge_v = (src, rel_v, dst_v)                   
                        atomic_routes_list.append(('dim-fact-dim',) + edge_q + get_rev_edge(edge_v))

    return atomic_routes_list

def extract_atomic_routes(data: HeteroData) -> List[Tuple[str, Optional[str], str]]:
    """
    Estrae tutte le atomic routes dallo schema del grafo:
    - route dirette (1 FK)
    - route via bridge node (2 FK)
    """
    atomic_routes = []
    node_types, edge_types = data.metadata()

    # 1. Route semplici: ogni arco diretto diventa un atomic route diretto
    for edge_type in edge_types:
        src, _, dst = edge_type
        atomic_routes.append((src, None, dst))

    # 2. Route composite: bridge nodes
    for mid_type in node_types:
        in_edges = [e for e in edge_types if e[2] == mid_type]
        out_edges = [e for e in edge_types if e[0] == mid_type]

        for e_in in in_edges:
            for e_out in out_edges:
                src = e_in[0]
                dst = e_out[2]
                if src != dst:
                    atomic_routes.append((src, mid_type, dst))
    return atomic_routes

#modello di convoluzione per le atomic routes:


class AtomicRouteConv(nn.Module):
    def __init__(self, src: str, mid: Optional[str], dst: str, channels: int):
        """
        Modulo per un'atomic route:
        - Se mid is None: route diretta (src → dst) → usa SAGEConv standard
        - Se mid esiste: route composita (src → mid → dst) → usa attenzione
        """
        super().__init__()
        self.src = src
        self.mid = mid
        self.dst = dst
        self.channels = channels

        if mid is None:
            # Route semplice: usa una GNN standard tipo GraphSAGE
            self.conv = SAGEConv((channels, channels), channels)
        else:
            # Route con nodo intermedio: usa FUSE + attenzione
            self.W1 = nn.Linear(channels, channels)
            self.W2 = nn.Linear(channels, channels)
            self.att_q = nn.Linear(channels, channels)
            self.att_k = nn.Linear(channels, channels)
            self.att_v = nn.Linear(channels, channels)

    def reset_parameters(self):
        if self.mid is None:
            self.conv.reset_parameters()
        else:
            self.W1.reset_parameters()
            self.W2.reset_parameters()
            self.att_q.reset_parameters()
            self.att_k.reset_parameters()
            self.att_v.reset_parameters()

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]):
      out = {}

      # Caso 1: atomic route semplice (src → dst)
      if self.mid is None:
          # Cerchiamo la chiave edge (src, edge_type, dst)
          edge_key = self._find_edge_key(edge_index_dict, self.src, self.dst)
          if edge_key is None:
              print("errore con none in mid")
              return {}
          #print(f"trovato {edge_index_dict[edge_key]}")
          edge_index = edge_index_dict[edge_key]
          out[self.dst] = self.conv((x_dict[self.src], x_dict[self.dst]), edge_index)
          return out

      # Caso 2: atomic route con nodo intermedio (src → mid → dst)
      edge_key_1 = self._find_edge_key(edge_index_dict, self.src, self.mid)
      edge_key_2 = self._find_edge_key(edge_index_dict, self.mid, self.dst)

      if edge_key_1 is None or edge_key_2 is None:
          print(f"errore!")
          return {}

      edge_index_1 = edge_index_dict[edge_key_1]  # src → mid
      edge_index_2 = edge_index_dict[edge_key_2]  # mid → dst

      if edge_index_1.size(1) == 0 or edge_index_2.size(1) == 0:
          #print(f"errore size vuota")------------------------------------------------------------------------------------------------------------------------DA INDAGARE PERCHE', SE E QUANDO SUCCEDE
          return {}
      #print(f"size non vuota")
      # Step 1: src → mid: creazione dei messaggi h_fuse
      src_idx_1, mid_idx_1 = edge_index_1
      h_src = x_dict[self.src][src_idx_1]        # [E1, C]
      h_mid = x_dict[self.mid][mid_idx_1]        # [E1, C]
      h_fuse = self.W1(h_mid) + self.W2(h_src)   # [E1, C]

      # Step 2: aggregazione dei messaggi h_fuse su ciascun nodo mid
      num_mid_nodes = x_dict[self.mid].size(0)
      h_mid_agg = torch.zeros((num_mid_nodes, self.channels), device=h_fuse.device)
      h_mid_agg.index_add_(0, mid_idx_1, h_fuse)

      # Step 3: mid → dst con attenzione
      mid_idx_2, dst_idx_2 = edge_index_2
      h_dst = x_dict[self.dst]  # [N_dst, C]
      k = self.att_k(h_mid_agg[mid_idx_2])  # [E2, C]
      v = self.att_v(h_mid_agg[mid_idx_2])  # [E2, C]
      q = self.att_q(h_dst[dst_idx_2])      # [E2, C]

      alpha = torch.softmax((q * k).sum(-1) / (self.channels ** 0.5), dim=0)  # [E2]
      msg = v * alpha.unsqueeze(-1)  # [E2, C]

      # Step 4: aggregazione dei messaggi sul nodo dst
      num_dst_nodes = x_dict[self.dst].size(0)
      msg_dst = torch.zeros((num_dst_nodes, self.channels), device=msg.device)
      msg_dst.index_add_(0, dst_idx_2, msg)

      out[self.dst] = msg_dst
      return out


    #@staticmethod
    def _find_edge_key(self, edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor], src: str, dst: str):
        """
        Cerca nel dizionario degli edge_index una chiave (src, *, dst).
        Non si fa affidamento sul nome della relazione.
        """
        for key in edge_index_dict:
            if key[0] == src and key[2] == dst:
                return key
        return None



# definiamo il modello effettivo:
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

class RelGNNEncoder(nn.Module):
    def __init__(
        self,
        node_types: List[str],
        atomic_routes: List[Tuple[str, Optional[str], str]],
        channels: int,
        num_layers: int = 2,
    ):
        """
        Encoder GNN che applica message passing lungo atomic routes,
        su più layer GNN.
        """
        super().__init__()
        self.node_types = node_types
        self.atomic_routes = atomic_routes
        self.num_layers = num_layers

        # Per ogni layer, creiamo un dizionario di conv per ogni route
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict()
            for src, mid, dst in atomic_routes:
                route_key = f"{src}__{mid or 'none'}__{dst}"
                layer[route_key] = AtomicRouteConv(src, mid, dst, channels)
            self.layers.append(layer)

    def reset_parameters(self):
        for layer in self.layers:
            for conv in layer.values():
                conv.reset_parameters()

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
        num_sampled_nodes_dict=None,
        num_sampled_edges_dict=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Applica il message passing composito su tutti i layer.
        Per ogni nodo, somma tutti i messaggi provenienti da tutte le atomic routes.
        """
        for layer in self.layers:
            agg_dict = {ntype: [] for ntype in x_dict}
            for route_key, conv in layer.items():
                msg_dict = conv(x_dict, edge_index_dict)
                for dst, msg in msg_dict.items():
                    agg_dict[dst].append(msg)

            # Somma i messaggi ricevuti da ogni route
            x_dict = {
                ntype: sum(msgs) if msgs else x_dict[ntype]
                for ntype, msgs in agg_dict.items()
            }
        return x_dict



class AtomicRouteModel(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        shallow_list: List[NodeType] = [],
        id_awareness: bool = False,
        predictor_n_layers : int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )

        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )

        # self.gnn = HeteroGraphSAGE(
        #     node_types=data.node_types,
        #     edge_types=data.edge_types,
        #     channels=channels,
        #     aggr=aggr,
        #     num_layers=num_layers,
        # )
        atomic_routes = extract_atomic_routes(data)
        #print(f"le atomic routes sono le seguenti: {atomic_routes}")
        self.gnn = RelGNNEncoder(
            node_types=data.node_types,
            atomic_routes=atomic_routes,
            channels=channels,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            hidden_channels=channels,
            norm=norm,
            num_layers=predictor_n_layers,
        )

        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    #add a method to indicate to the pre training strategy the 
    #parameters to "change":
    def encoder_parameters(self):
        params = list(self.encoder.parameters()) + list(self.temporal_encoder.parameters()) + list(self.gnn.parameters())
        return params


    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
            x_dict[node_type] = self.dropout(x_dict[node_type])

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
            x_dict[node_type] = self.dropout(x_dict[node_type])

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def encode_node_types(self, batch: HeteroData, node_types: List[str], entity_table) -> Dict[str, Tensor]:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)

        for node_type in node_types:
            if node_type in rel_time_dict:
                x_dict[node_type] = x_dict[node_type] + rel_time_dict[node_type]
            if node_type in self.embedding_dict:
                x_dict[node_type] = x_dict[node_type] + self.embedding_dict[node_type](batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return {ntype: x_dict[ntype] for ntype in node_types if ntype in x_dict}



    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])