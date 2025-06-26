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

#pratically we are going to take a positive relation, such as 
#(p1, PA, a1) which means that paper p1 has been written by author a1.
#this relation really exists in the graph. Then, we take a relation that
#really exists in the graph and that has the same source node p1, but 
#is another kind of relation: imagine something like (p1, PC, c4), which 
#means that p1 has been presented in the conference c4. This relation 
#still really exists in the graph, but is negative with respect to the 
#first one.

#we aim with this pre training that the model could understand to 
#distinguish an embedding space that takes into account the different
#semantic kind of relations.


def get_negative_samples_from_inconsistent_relations(
    data: HeteroData,
    target_edge_type: Tuple[str, str, str],#src, type of relation, dst
    max_negatives_per_node: int = 5 #maximum number of negative samples we want
) -> List[Tuple[int, Tuple[str, str, str], int]]:
    src_type, _, _ = target_edge_type
    negatives = []

    #positive extraction:
    edge_index_pos = data[target_edge_type].edge_index
    u_nodes = edge_index_pos[0].tolist()
    #all node index of type u connected through the relation "target_edge_type"

    #negatives extraction:
    for u in set(u_nodes):#iterate over all u (index) edges (paper edges for ex)
        neg_for_u = []

        for etype in data.edge_types:
            if etype == target_edge_type:
                continue#only consider different relation type as negatives
            
            if etype[0] != src_type:
                continue#only consider edges where the source node u is the same

            edge_index = data[etype].edge_index
            #take all the edges of type etype
            mask = edge_index[0] == u
            #take only the edges where source node is the same as u (the one 
            #of the positive edge)
            w_candidates = edge_index[1][mask].tolist()

            #limit number of negatives per node
            sampled_ws = random.sample(w_candidates, min(len(w_candidates), max_negatives_per_node))
            neg_for_u += [(u, etype, w) for w in sampled_ws]

        negatives += neg_for_u
        #do this for (eventually) many different relations

    return negatives
