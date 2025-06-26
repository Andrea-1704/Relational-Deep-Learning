####
#In this file we are going to implement the main function to 
#apply the relation level pre-training task on the subgraph level.
#in particular, we will need to implement the following functions:
#1. get_negative_samples_from_inconsistent_relations
#2. get_negative_samples_from_unrelated_nodes
####
# Standard libraries


import math
import os
import sys
import random
import copy
import requests
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
sys.path.append(os.path.abspath("."))
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import (
    Module,
    ModuleDict,
    Sequential,
    Linear,
    ReLU,
    Dropout,
    BatchNorm1d,
    LayerNorm,
)
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import (
    HeteroConv,
    LayerNorm as GeoLayerNorm,
    PositionalEncoding,
    SAGEConv,
    MLP,
)
from torch_geometric.seed import seed_everything
from torch_geometric.typing import NodeType, EdgeType
import pyg_lib
import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    FeatureEncoder,
)
from torch_frame.nn.encoder.stype_encoder import StypeEncoder
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from relbench.modeling.nn import (
    HeteroEncoder,
    HeteroGraphSAGE,
    HeteroTemporalEncoder,
)



###Part 1: get_negative_samples_from_inconsistent_relations
#Idea:
#Given the positive triple ⟨𝑢, 𝑅, 𝑣⟩ ∈ 𝑃_rel, there exists a node 𝑤
#connected to 𝑢 under a relation type 𝑅⁻ that is inconsistent with 
#𝑅. Thus, the triple ⟨𝑢, 𝑅⁻, 𝑤⟩ represents a different semantic 
#context from ⟨𝑢, 𝑅, 𝑣⟩.
def sample_unrelated_node_negatives(
    data: HeteroData,
    edge_type: Tuple[str, str, str],
    k: int = 5,
    num_negatives: int = 5
) -> List[Tuple[int, int]]:
    src_type, rel_type, dst_type = edge_type
    edge_index = data[edge_type].edge_index
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]

    from collections import defaultdict
    negatives = []

    # Precostruisci mappa delle connessioni dirette
    connected = defaultdict(set)
    for u, v in zip(src_nodes.tolist(), dst_nodes.tolist()):
        connected[u].add(v)

    for u in src_nodes.unique().tolist():
        # Trova subgrafo k-hop da u (considerando tutto il grafo)
        sub_nodes, _, _, _ = k_hop_subgraph(
            node_idx=u,
            num_hops=k,
            edge_index=data.to_homogeneous().edge_index,
            relabel_nodes=False,
        )

        # Filtra per tipo di nodo (solo quelli del tipo di destinazione)
        dst_mask = data[dst_type].node_mask if hasattr(data[dst_type], "node_mask") else \
                   torch.ones(data[dst_type].num_nodes, dtype=torch.bool)
        candidate_v = [v for v in sub_nodes.tolist()
                       if v in data[dst_type].node_ids.tolist() and v not in connected[u]]

        # Campiona negativi
        v_neg = random.sample(candidate_v, min(num_negatives, len(candidate_v)))
        negatives += [(u, v) for v in v_neg]

    return negatives
