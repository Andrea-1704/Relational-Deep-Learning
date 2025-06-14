
import torch
import numpy as np
import math
from tqdm import tqdm
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
from matplotlib import pyplot as plt
from itertools import product
import torch
import numpy as np
import copy
import pandas as pd
import EarlyStopping

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, input_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []

        self.input_dim = input_dim

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z_dict: Dict[str, Tuple[Tensor, Tensor, Tensor, Tensor]],
        edge_index: Tensor,
        src_type: str,
        dst_type: str,
    ) -> Tensor:
        src_global, dst_global = edge_index

        z_src, _, _, n_id_src = z_dict[src_type]
        z_dst, _, _, n_id_dst = z_dict[dst_type]

        id2idx_src = {int(nid): i for i, nid in enumerate(n_id_src.tolist())}
        id2idx_dst = {int(nid): i for i, nid in enumerate(n_id_dst.tolist())}

        src_idx = torch.tensor(
                [id2idx_src[int(s)] for s in src_global.tolist()],
                device=z_src.device,
                dtype=torch.long,
        )
        dst_idx = torch.tensor(
                [id2idx_dst[int(d)] for d in dst_global.tolist()],
                device=z_dst.device,
                dtype=torch.long,
        )

        z_i = z_src[src_idx]
        z_j = z_dst[dst_idx]

        out = torch.cat([z_i, z_j, torch.abs(z_i - z_j), z_i * z_j], dim=-1)
        return self.mlp(out).squeeze(-1)




