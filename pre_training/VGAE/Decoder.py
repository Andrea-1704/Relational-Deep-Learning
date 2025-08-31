import torch
import sys
import os
sys.path.append(os.path.abspath("."))
from typing import Dict, Tuple
from torch import Tensor
from torch import nn
import torch
import torch.nn as nn

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []

        self.input_dim = 4 * latent_dim # because we concatenate z_i, z_j, |z_i - z_j|, and z_i * z_j

        layers.append(nn.Linear(self.input_dim, hidden_dim))
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




