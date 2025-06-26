
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

import sys
import os
sys.path.append(os.path.abspath("."))


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
from utils.EarlyStopping import EarlyStopping

from typing import Tuple

class VGAEWrapper(nn.Module):
    def __init__(self, full_model, encoder_out_dim, latent_dim, entity_table):
        super().__init__()
        self.encoder = full_model
        self.proj_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.proj_logvar = nn.Linear(encoder_out_dim, latent_dim)
        self.entity_table = entity_table

    def forward(self, batch: HeteroData, node_types: List[str]) -> Dict[str, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        z_dict = {}

        full_z = self.encoder.encode_node_types(batch, node_types, self.entity_table)

        for ntype in node_types:
            h = full_z[ntype]  # [num_nodes, encoder_out_dim]
            mu = self.proj_mu(h)
            logvar = self.proj_logvar(h)
            z = self.reparameterize(mu, logvar)
            n_id = batch[ntype].n_id  # global node IDs
            z_dict[ntype] = (z, mu, logvar, n_id)

        return z_dict

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu


