import torch
from typing import Dict, List
import sys
import os
sys.path.append(os.path.abspath("."))
from torch import Tensor
from torch_geometric.data import HeteroData
from torch import nn
import torch
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


