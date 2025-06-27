####
#In this file we are going to implement the main function to 
#apply the relation level pre-training task on the subgraph level.
#in particular, we will need to implement the following functions:
#1. get_negative_samples_from_inconsistent_relations
#2. get_negative_samples_from_unrelated_nodes
####



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
from torch_geometric.utils import k_hop_subgraph #useful function to remember



###Part 1: get_negative_samples_from_inconsistent_relations
#Idea:
#Given the positive triple ⟨𝑢, 𝑅, 𝑣⟩ ∈ 𝑃_rel, there exists a node 𝑤
#connected to 𝑢 under a relation type 𝑅⁻ that is inconsistent with 
#𝑅. Thus, the triple ⟨𝑢, 𝑅⁻, 𝑤⟩ represents a different semantic 
#context from ⟨𝑢, 𝑅, 𝑣⟩.
#Pratically we are going to take a positive relation, such as 
#(p1, PA, a1) which means that paper p1 has been written by author a1.
#this relation really exists in the graph. Then, we take a relation that
#really exists in the graph and that has the same source node p1, but 
#is another kind of relation: imagine something like (p1, PC, c4), which 
#means that p1 has been presented in the conference c4. This relation 
#still really exists in the graph, but is negative with respect to the 
#first one.
#We aim with this pre training that the model could understand to 
#distinguish an embedding space that takes into account the different
#semantic kind of relations.
###


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









###Part2: get_negative_samples_from_unrelated_nodes
#Idea:
#We take a source edge (u, R, w) and we consider nodes 'wi' that are of the 
#same node type as the one of 'w', but that are not connected to u.
#to do so, we can simply consider nodes that are not in the graph, or we can 
#randomly sample nodes that are not connected to u. 
#Nevertheless, these approaches would be pretty easy for the model to be 
#solved. 
#The idea we tried to implement is instead to sample negative nodes that are
#not directly connected to u, but that are k-hops away from u and that are
#of the same type of w. In this way
#the two nodes are still pretty correleted, but the model should be able 
#to detect that they are negatives anyway.
###

def get_negative_samples_from_unrelated_nodes(
    data: HeteroData,
    target_edge_type: Tuple[str, str, str],
    k: int = 5,
    num_negatives_per_node: int = 5
) -> List[Tuple[int, Tuple[str, str, str], int]]:
    src_type, rel_type, dst_type = target_edge_type
    edge_index = data[target_edge_type].edge_index

    u_nodes = edge_index[0].tolist()
    v_nodes = edge_index[1].tolist()

    positives = set(zip(u_nodes, v_nodes))
    negatives = []

    #convert to homogeneous graph in order to apply the k_hop_subgraph
    homo_data = data.to_homogeneous()#this function does not change "data"
    edge_index_homo = homo_data.edge_index#all edges in homoGraph
    type_vec = homo_data.node_type#[N], all types of nodes.


    node_type_map = {t: i for i, t in enumerate(data.node_types)}
    #numericID:type of node (for ex driver:0, races:1 etc)
    dst_type_id = node_type_map[dst_type]#id of the dst node (0 for ex, if dst_type==driver)

    for u in set(u_nodes):
        # Step 1: map u from type-specific index to global ID
        u_homo_id = data[src_type].node_id_to_global[u]

        # Step 2: k-hop neighborhood
        sub_nodes, _, _, _ = k_hop_subgraph(
            node_idx=u_homo_id,
            num_hops=k,
            edge_index=edge_index_homo,
            relabel_nodes=False,
        )

        # Step 3: filter only nodes of dst_type (e.g., authors)
        v_minus = [nid.item() for nid in sub_nodes
                   if type_vec[nid] == dst_type_id]

        # Step 4: map back to dst_type local indices
        local_v_minus = [data[dst_type].global_to_node_id[v]
                         for v in v_minus
                         if (u, v) not in positives]

        # Step 5: sample
        sampled = random.sample(local_v_minus, min(len(local_v_minus), num_negatives_per_node))
        negatives += [(u, target_edge_type, v) for v in sampled]

    return negatives



###Part 3 (not strictly needed)
#Just to see how Relation level pretraining performs alone, we
#can decide to implement a function that only uses it 
#(without considering subgraph pre training level task).
###

#loss function 1:

def relation_contrastive_loss_rel1(
    h_dict: Dict[str, torch.Tensor],
    W_R: torch.nn.Parameter,
    target_edge_type: Tuple[str, str, str],
    pos_edges: List[Tuple[int, int]],
    neg_dict: Dict[int, List[int]],
    temperature: float = 0.07
) -> torch.Tensor:
    src_type, _, dst_type = target_edge_type
    h_src = h_dict[src_type]
    h_dst = h_dict[dst_type]

    device = h_src.device
    total_loss = torch.tensor(0.0, device=device)
    N = len(pos_edges)

    for u, v in pos_edges:
        h_u = h_src[u].unsqueeze(0)                      # (1, d)
        h_v = torch.matmul(W_R, h_dst[v])                # (d,)
        h_v = h_v.unsqueeze(0)                           # (1, d)

        # Positivo score
        sim_pos = (h_u @ h_v.T) / temperature            # (1, 1)

        # Negativi
        neg_ws = neg_dict.get(u, [])
        if not neg_ws:
            continue  # Skip if no negatives

        h_w = torch.stack([h_dst[w] for w in neg_ws])    # (n_neg, d)
        h_w = torch.matmul(h_w, W_R.T)                   # (n_neg, d)
        sim_neg = (h_u @ h_w.T) / temperature            # (1, n_neg)

        logits = torch.cat([sim_pos, sim_neg], dim=1)    # (1, 1 + n_neg)
        labels = torch.zeros(1, dtype=torch.long, device=device)

        loss = F.cross_entropy(logits, labels)
        total_loss += loss

    return total_loss / max(N, 1)



#Loss function 2:
def relation_contrastive_loss_rel2(
    h_dict: Dict[str, torch.Tensor],
    target_edge_type: Tuple[str, str, str],
    pos_edges: List[Tuple[int, int]],
    neg_dict: Dict[int, List[int]],
    temperature: float = 0.07
) -> torch.Tensor:
    src_type, _, dst_type = target_edge_type
    h_src = h_dict[src_type]
    h_dst = h_dict[dst_type]

    #total_loss = 0.0
    total_loss = torch.tensor(0.0, device=h_src.device)

    N = len(pos_edges)

    for u, v in pos_edges:
        h_u = h_src[u]  # (d,)
        h_v = h_dst[v]  # (d,)

        sim_pos = torch.exp(torch.dot(h_u, h_v) / temperature)

        #negatives from not connected nodes (but <= k hops away)
        neg_vs = neg_dict.get(u, [])
        sim_negs = []
        for v_minus in neg_vs:
            h_vm = h_dst[v_minus]
            sim_neg = torch.exp(torch.dot(h_u, h_vm) / temperature)
            sim_negs.append(sim_neg)

        if sim_negs:
            denom = sim_pos + torch.stack(sim_negs).sum()
            loss = -torch.log(sim_pos / denom)
            total_loss += loss

    return total_loss / max(N, 1)



def pretrain_relation_level_full_rel(
    #data: HeteroData,
    loader_dict,
    model: torch.nn.Module,
    W_R: torch.nn.Parameter,
    target_edge_type: Tuple[str, str, str],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task,
    num_epochs: int = 100,
    num_neg_per_node: int = 5,
    k: int = 5,
    lambda_rel2: float = 1.0
):
    model = model.to(device)
    W_R = W_R.to(device)
    #data = data.to(device)

    for epoch in range(num_epochs):
        for batch in tqdm(loader_dict["train"]):
            batch = batch.to(device)
            model.train()
            optimizer.zero_grad()

            #forward pass
            #h_dict = model(data.x_dict, data.edge_index_dict)
            h_dict = model.encode_node_types(
                batch,
                batch.node_types,
                task.entity_table,
            )

            #log to be removed:
            for ntype, h in h_dict.items():
                print(f"{ntype} requires_grad: {h.requires_grad}")

            #print(f"questo è h_dict: {h_dict}")

            #positives
            edge_index = batch[target_edge_type].edge_index
            u_pos = edge_index[0].tolist()
            v_pos = edge_index[1].tolist()
            pos_edges = list(zip(u_pos, v_pos))

            #first kind of negatives
            neg_dict_1 = get_negative_samples_from_inconsistent_relations(
                batch, target_edge_type, max_negatives_per_node=num_neg_per_node
            )

            #second kind of negatives
            neg_dict_2 = get_negative_samples_from_unrelated_nodes(
                batch, target_edge_type, k=k, num_negatives_per_node=num_neg_per_node
            )

            #compute the two loss components
            loss_rel1 = relation_contrastive_loss_rel1(
                h_dict=h_dict,
                W_R=W_R,
                target_edge_type=target_edge_type,
                pos_edges=pos_edges,
                neg_dict=neg_dict_1
            )

            # loss_rel2 = relation_contrastive_loss_rel2(
            #     h_dict=h_dict,
            #     target_edge_type=target_edge_type,
            #     pos_edges=pos_edges,
            #     neg_dict=neg_dict_2
            # )

            #add the two loss contribution to obtain the final one
            loss = loss_rel1 #+ lambda_rel2 * loss_rel2
            loss.backward()
            optimizer.step()

            #remeber to change pronit:
            #print(f"[Epoch {epoch+1}] L_rel1: {loss_rel1.item():.4f} | L_rel2: {loss_rel2.item():.4f} | Total: {loss.item():.4f}")
            print(f"[Epoch {epoch+1}] L_rel1: {loss_rel1.item():.4f} | Total: {loss.item():.4f}")


    return model, W_R
