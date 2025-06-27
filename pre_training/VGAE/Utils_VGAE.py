
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

import sys
import os
sys.path.append(os.path.abspath("."))


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
from utils.EarlyStopping import EarlyStopping
from pre_training.VGAE.Encoder import VGAEWrapper
from pre_training.VGAE.Decoder import MLPDecoder
from torch_geometric.data import HeteroData
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple

from typing import Tuple
from torch import Tensor
import random

def get_pos_neg_edges(batch: HeteroData, edge_type: Tuple[str, str, str], num_neg_samples: int = None) -> Tuple[Tensor, Tensor]:
    src_type, _, dst_type = edge_type
    edge_index = batch.edge_index_dict[edge_type]  # (2, E)
    src_ids, dst_ids = edge_index[0], edge_index[1]

    # Solo archi tra nodi effettivamente presenti nel batch
    src_n_id_set = set(batch[src_type].n_id.tolist())
    dst_n_id_set = set(batch[dst_type].n_id.tolist())

    mask = torch.tensor([
        int(s) in src_n_id_set and int(d) in dst_n_id_set
        for s, d in zip(src_ids.tolist(), dst_ids.tolist())
    ], dtype=torch.bool, device=edge_index.device)

    pos_edges = edge_index[:, mask]
    if pos_edges.size(1) == 0:
        #print("non abbiamo generato i negativi parte 1")
        return None  # nessun arco positivo → skip batch

    # Set di archi positivi (per evitare duplicati nei negativi)
    pos_edge_set = set(zip(pos_edges[0].tolist(), pos_edges[1].tolist()))

    src = pos_edges[0]
    num_samples = num_neg_samples or src.size(0)
    dst_candidates = list(dst_n_id_set)

    neg_edges = []
    attempts = 0
    max_attempts = num_samples * 10  # safety limit

    while len(neg_edges) < num_samples and attempts < max_attempts:
        s = src[random.randint(0, src.size(0) - 1)].item()
        d = dst_candidates[random.randint(0, len(dst_candidates) - 1)]
        if (s, d) not in pos_edge_set:
            neg_edges.append((s, d))
        attempts += 1

    if not neg_edges:
        #print("non abbiamo generato i negativi parte 2")
        return None

    neg_edges = torch.tensor(neg_edges, dtype=torch.long, device=src.device).T  # shape (2, N)

    return pos_edges, neg_edges


def vgae_loss(
    pos_logits: Tensor,
    neg_logits: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0
) -> Tuple[Tensor, Tensor, Tensor]:
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
    recon = pos_loss + neg_loss


    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


    return recon + beta * kl, recon, kl


def train_vgae(
    model: nn.Module,
    loader_dict: Dict[str, NeighborLoader],
    edge_types: List[Tuple[str, str, str]],
    encoder_out_dim: int,
    entity_table,
    latent_dim: int = 64,
    hidden_dim: int = 128,
    epochs: int = 30,
    device: str = "cuda"
) -> nn.Module:



    wrapper = VGAEWrapper(
        full_model=model,
        encoder_out_dim=encoder_out_dim,
        latent_dim=latent_dim,
        entity_table=entity_table
    ).to(device)


    decoder = MLPDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, input_dim=encoder_out_dim).to(device)

    #the parameter to be optimed should not include the head etc, 
    #but only the ones we need to modify during the pre training:
    optimizer = torch.optim.Adam(
        list(model.encoder_parameters()) + 
        list(wrapper.proj_mu.parameters()) + 
        list(wrapper.proj_logvar.parameters()) + 
        list(decoder.parameters()),
        lr=1e-3
    )

    # optimizer = torch.optim.Adam(
    #     list(wrapper.parameters()) + list(decoder.parameters()),
    #     lr=1e-3
    # )
    # #for edge_type in edge_types:


    for epoch in range(1, epochs + 1):
          wrapper.train()
          decoder.train()

          total_loss = total_recon = total_kl = 0

          for batch in loader_dict["train"]:
              #scelgo casualmente un arco:
              archi_presenti = False
              archi_presenti_effettivi = False

              max_trials = 50
              trial = 0

              while (not archi_presenti or not archi_presenti_effettivi) and trial < max_trials:
                edge_type = random.choice(edge_types)
                edge_index_trial = batch.edge_index_dict[edge_type]
                if len(edge_index_trial[0])!=0:
                  archi_presenti = True

                src_type = edge_type[0]
                dst_type = edge_type[2]

                batch = batch.to(device)

                z_dict = wrapper(batch, node_types=[edge_type[0], edge_type[2]])

                res = get_pos_neg_edges(batch, edge_type)
                #pos_edges, neg_edges = get_pos_neg_edges(batch, edge_type)
                if res is not None:
                  pos_edges, neg_edges = res
                  archi_presenti_effettivi = True
                trial += 1

              if trial == max_trials:
                  print(f"⚠️ Skip batch: nessun edge_type valido trovato dopo {max_trials} tentativi.")
                  continue

              #print(f"per arco {edge_type} abbiamo {len(pos_edges[0])} archi positivi, e {len(neg_edges[0])} archi negativi, ma ne potremmo avere: {len(batch.edge_index_dict[edge_type][0])}")

              pos_logits = decoder(z_dict, pos_edges, src_type, dst_type)
              neg_logits = decoder(z_dict, neg_edges, src_type, dst_type)

              if pos_edges.numel() == 0 or neg_edges.numel() == 0:
                  print("Batch saltato: nessun arco seed-seed.")
                  continue

              beta = min(1.0, epoch / 10)  # warm-up
              mu_src, logvar_src = z_dict[src_type][1], z_dict[src_type][2]
              mu_dst, logvar_dst = z_dict[dst_type][1], z_dict[dst_type][2]
              mu = torch.cat([mu_src, mu_dst], dim=0)
              logvar = torch.cat([logvar_src, logvar_dst], dim=0)
              loss, recon, kl = vgae_loss(pos_logits, neg_logits, mu, logvar, beta)


              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

              total_loss += loss.item()
              total_recon += recon.item()
              total_kl += kl.item()

          print(f"[VGAE] Epoch {epoch:02d} | Loss: {total_loss:.4f} | Recon: {total_recon:.4f} | KL: {total_kl:.4f}")

    print("Pretraining VGAE completato.\n")
    return model
