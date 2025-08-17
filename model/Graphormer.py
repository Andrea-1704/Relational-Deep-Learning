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

class HeteroGraphormerLayerComplete(nn.Module):
    def __init__(self, channels, edge_types, device, num_heads=4, dropout=0.1):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads

        assert self.channels % num_heads == 0, "channels must be divisible by num_heads"

        self.q_lin = Linear(channels, channels)
        self.k_lin = Linear(channels, channels)
        self.v_lin = Linear(channels, channels)
        self.out_lin = Linear(channels, channels)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

        self.edge_type_bias = nn.ParameterDict({
            "__".join(edge_type): nn.Parameter(torch.randn(1))
            for edge_type in edge_types
        })

    def compute_total_degrees(self, x_dict, edge_index_dict):
        device = self.device
        in_deg = defaultdict(lambda: torch.zeros(0, device=device))
        out_deg = defaultdict(lambda: torch.zeros(0, device=device))
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            src, dst = edge_index

            num_src = x_dict[src_type].size(0)
            num_dst = x_dict[dst_type].size(0)

            if out_deg[src_type].numel() == 0:
                out_deg[src_type] = torch.zeros(num_src, device=device)
            if in_deg[dst_type].numel() == 0:
                in_deg[dst_type] = torch.zeros(num_dst, device=device)

            out_deg[src_type] += degree(src, num_nodes=num_src)
            in_deg[dst_type]  += degree(dst, num_nodes=num_dst)

        return {
            node_type: in_deg[node_type] + out_deg[node_type]
            for node_type in x_dict
        }

    def compute_batch_spatial_bias(self, edge_index, num_nodes):
        # Costruisci grafo da edge_index del batch corrente
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        src, dst = edge_index
        for s, d in zip(src.tolist(), dst.tolist()):
            G.add_edge(s, d)

        spatial_bias = {}
        for node in G.nodes():
            lengths = nx.single_source_dijkstra_path_length(G, node)
            for target, dist in lengths.items():
                spatial_bias[(node, target)] = dist

        # Costruzione tensor di bias dallo spatial_bias
        bias_vals = [spatial_bias.get((d, s), -1.0) for s, d in zip(src.tolist(), dst.tolist())]
        return torch.tensor(bias_vals, dtype=torch.float32, device=self.device)

    def forward(self, x_dict, edge_index_dict):
        out_dict = {k: torch.zeros_like(v) for k, v in x_dict.items()}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            x_src, x_dst = x_dict[src_type], x_dict[dst_type]
            src, dst = edge_index

            Q = self.q_lin(x_dst).view(-1, self.num_heads, self.head_dim)
            K = self.k_lin(x_src).view(-1, self.num_heads, self.head_dim)
            V = self.v_lin(x_src).view(-1, self.num_heads, self.head_dim)

            attn_scores = (Q[dst] * K[src]).sum(dim=-1) / self.head_dim**0.5

            # Nuovo spatial bias (batch-local)
            spatial_bias_tensor = self.compute_batch_spatial_bias(edge_index, x_dst.size(0))
            attn_scores = attn_scores + spatial_bias_tensor.unsqueeze(-1)

            bias_name = "__".join(edge_type)
            attn_scores = attn_scores + self.edge_type_bias[bias_name]

            attn_weights = softmax(attn_scores, dst)
            attn_weights = self.dropout(attn_weights)

            out = V[src] * attn_weights.unsqueeze(-1)
            out = out.view(-1, self.channels)

            out_dict[dst_type].index_add_(0, dst, out)

        # Aggiunta degree centrality
        total_deg = self.compute_total_degrees(x_dict, edge_index_dict)
        for node_type in out_dict:
            deg_embed = total_deg[node_type].view(-1, 1).expand(-1, self.channels)
            out_dict[node_type] += deg_embed

        # Residual + layer norm
        for node_type in out_dict:
            out_dict[node_type] = self.norm(out_dict[node_type] + x_dict[node_type])

        return out_dict





class HeteroGraphormer(torch.nn.Module):
    def __init__(self, node_types, edge_types, channels, num_layers=2, device="cuda"):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            HeteroGraphormerLayerComplete(channels, edge_types, device) for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict, *args, **kwargs):
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        return x_dict

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()





class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData, #notice that "data2 is the graph we created with function make_pkey_fkey_graph
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        shallow_list: List[NodeType] = [],
        id_awareness: bool = False,
        predictor_n_layers : int = 1,
    ):
        super().__init__()

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
        self.gnn = HeteroGraphormer(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            num_layers=num_layers,
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

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
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

    def forward(
        self,
        batch: HeteroData, 
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        #this creates a dictionar for all the nodes: each nodes has its
        #embedding

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )
        #this add the temporal information to the node using the 
        #HeteroTemporalEncoder

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        #add some other shallow embedder

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,#feature of nodes
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )#apply the gnn

        return self.head(x_dict[entity_table][: seed_time.size(0)])#final prediction

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
        # Add ID-awareness to the root node
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

