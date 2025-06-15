import torch
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
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_geometric.nn import HeteroConv, LayerNorm, PositionalEncoding, SAGEConv
from torch_geometric.typing import EdgeType, NodeType


import torch
from torch import nn
from torch.nn import ModuleDict
from typing import Any, Dict, List
from torch import Tensor
from torch_frame import TensorFrame


class HeteroEncoderWithAttention(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder, {}),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
        },
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.channels = channels

        self.encoders = torch.nn.ModuleDict()
        self.attn_modules = torch.nn.ModuleDict()
        self.projections = torch.nn.ModuleDict()

        for node_type, col_dict in node_to_col_names_dict.items():
            self.encoders[node_type] = torch.nn.ModuleDict()
            for stype, cols in col_dict.items():
                encoder_cls, encoder_kwargs = default_stype_encoder_cls_kwargs[stype]
                encoder = encoder_cls(**encoder_kwargs)
                for col in cols:
                    self.encoders[node_type][col] = encoder

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, batch_first=True)
            self.attn_modules[node_type] = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.projections[node_type] = nn.Linear(hidden_dim, channels)

    def reset_parameters(self):
        for node_dict in self.encoders.values():
            for enc in node_dict.values():
                enc.reset_parameters()
        for proj in self.projections.values():
            proj.reset_parameters()
        for attn in self.attn_modules.values():
            for layer in attn.layers:
                layer._reset_parameters()

    def forward(self, tf_dict: Dict[NodeType, torch_frame.TensorFrame]) -> Dict[NodeType, Tensor]:
        out = {}
        for node_type, tf in tf_dict.items():
            col_embeddings = []
            for col, encoder in self.encoders[node_type].items():
                col_tensor = tf[col]  # [batch_size]
                emb = encoder(col_tensor)  # [batch_size, hidden_dim]
                col_embeddings.append(emb)
            x = torch.stack(col_embeddings, dim=1)  # [batch_size, num_cols, hidden_dim]
            x = self.attn_modules[node_type](x)  # same shape
            x = x.mean(dim=1)  # [batch_size, hidden_dim]
            out[node_type] = self.projections[node_type](x)  # [batch_size, channels]
        return out
