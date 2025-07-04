"""
Refinement of version one with the aim of trying to implement a version 
very close to the original one.
Possibly trying to mimic the result of the authors of the paper.
"""

import torch
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
from relbench.modeling.nn import HeteroEncoder
from collections import defaultdict




def binarize_targets(y: torch.Tensor, threshold: float = 11) -> torch.Tensor:
    """
    This function trasforms a regression task (like the one of driver position)
    into a binary classification problem. 
    To incorporate the original https://arxiv.org/abs/2412.00521 paper, which 
    was only for binary classification we decided to trnasform our task into a 
    binary classfication task, where the driver position gets converted into a 
    binary label:
    1 if position < threshold;
    o otherwise.
    """
    return (y < threshold).long()




def get_candidate_relations(metadata, current_node_type: str) -> List[Tuple[str, str, str]]:
    """
    This function takes the "metadata" of the grafo (which are basicly all the 
    relevant informations about the graph, such as edge types, node types, etc.)
    and returns all the edges (in tuple format "(src_type, name of relation, 
    dst_type)") that starts from "current_node_type" as "src_type".
    """
    return [rel for rel in metadata[1] if rel[0] == current_node_type]




def build_global_to_local_from_edge_index(data: HeteroData) -> Dict[str, Dict[int, int]]:
    mapping = {}
    for ntype in data.node_types:
        node_ids = set()
        for (src, _, dst), edge_index in data.edge_index_dict.items():
            if src == ntype:
                node_ids.update(edge_index[0].tolist())
            if dst == ntype:
                node_ids.update(edge_index[1].tolist())
        sorted_ids = sorted(node_ids)
        mapping[ntype] = {global_id: i for i, global_id in enumerate(sorted_ids)}
    return mapping




def train_theta_for_relation(
    bags: List[List[int]],
    labels: List[int],
    node_embeddings: torch.Tensor,
    alpha_prev: Dict[int, float],
    epochs: int = 100,
    lr: float = 0.01,
) -> nn.Linear:
    device = node_embeddings.device
    theta = nn.Linear(node_embeddings.size(-1), 1, bias=False).to(device)
    optimizer = torch.optim.Adam(theta.parameters(), lr=lr)

    bag_embeddings = []
    alpha_values = []
    binary_labels = torch.tensor(labels, device=device)

    for bag in bags:
        if not bag:
            continue
        emb = node_embeddings[torch.tensor(bag, device=device)]
        alpha = torch.tensor([alpha_prev.get(v, 1.0) for v in bag], device=device)
        bag_embeddings.append(emb)
        alpha_values.append(alpha)

    for _ in range(epochs):
        optimizer.zero_grad()
        preds = []
        for emb, alpha in zip(bag_embeddings, alpha_values):
            scores = theta(emb).squeeze(-1)
            weights = torch.softmax(scores * alpha, dim=0)
            weighted_avg = torch.sum(weights.unsqueeze(-1) * emb, dim=0)
            pred = weighted_avg.mean()
            preds.append(pred)
        preds = torch.stack(preds)

        pos = preds[binary_labels == 1]
        neg = preds[binary_labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        loss = torch.sum(torch.sigmoid(neg.unsqueeze(0) - pos.unsqueeze(1)))
        loss.backward()
        optimizer.step()

    return theta




def evaluate_relation_learned(
    bags: List[List[int]],
    labels: List[float],
    node_embeddings: torch.Tensor,
    alpha_prev: Dict[int, float],
    global_to_local: Dict[int, int],  # nuovo
    epochs: int = 100,
    lr: float = 1e-2,
) -> Tuple[float, nn.Module]:

    device = node_embeddings.device
    binary_labels = torch.tensor(labels, device=device)

    # Allena theta come prima
    theta = train_theta_for_relation(
        bags=bags,
        labels=labels,
        node_embeddings=node_embeddings,
        alpha_prev=alpha_prev,
        epochs=epochs,
        lr=lr
    )

    preds = []
    for bag in bags:
        # Mappa gli ID globali nei bag a ID locali (solo quelli presenti nel mapping)
        local_ids = [global_to_local[v] for v in bag if v in global_to_local]
        if not local_ids:
            preds.append(torch.tensor(0.0, device=device))
            continue

        emb = node_embeddings[torch.tensor(local_ids, device=device)]
        scores = theta(emb).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        weighted_avg = torch.sum(weights.unsqueeze(-1) * emb, dim=0)
        pred = weighted_avg.mean()
        preds.append(pred)

    preds_tensor = torch.stack(preds)
    mae = F.l1_loss(preds_tensor, binary_labels).item()
    return mae, theta





def construct_bags_with_alpha(
    data,
    previous_bags: List[List[int]],
    previous_labels: List[float],
    alpha_prev: Dict[int, float],
    rel: Tuple[str, str, str],
    node_embeddings: torch.Tensor,
    theta: nn.Module,
    src_embeddings: torch.Tensor,
    global_to_local: Dict[int, int],  # nuovo
) -> Tuple[List[List[int]], List[float], Dict[int, float]]:
    edge_index = data.edge_index_dict.get(rel)
    if edge_index is None:
        print(f"this should not have happened, but the relation was not found.")
        return [], [], {}

    edge_src, edge_dst = edge_index
    bags = []
    labels = []
    alpha_next = {}

    for bag_v, label in zip(previous_bags, previous_labels):
        bag_u = []
        for v in bag_v:
            if v not in global_to_local:
                continue
            local_v = global_to_local[v]
            neighbors_u = edge_dst[edge_src == v]
            if len(neighbors_u) == 0:
                continue
            x_v = src_embeddings[local_v]
            theta_xv = theta(x_v).item()
            alpha_v = alpha_prev.get(v, 1.0)
            for u in neighbors_u.tolist():
                alpha_u = theta_xv * alpha_v 
                alpha_next[u] = alpha_next.get(u, 0.0) + alpha_u
                bag_u.append(u)

        if len(bag_u) > 0:
            bags.append(bag_u)
            labels.append(label)

    return bags, labels, alpha_next




def beam_metapath_search_with_bags_learned(
    data: HeteroData,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    node_type: str,
    col_stats_dict: Dict[str, Dict[str, Dict]],
    L_max: int = 3,
    max_rels: int = 10,
    channels: int = 64,
    beam_width: int = 5,
) -> Tuple[List[List[Tuple[str, str, str]]], Dict[Tuple, int]]:
    """
    This function provides more than one metapaths by applying a beam search over the 
    metapaths.
    This implementation also do not require to use a hard cutoff value to stop the 
    algorithm from creating sub-optimal and un-usefull long metapaths: 
    it simply considers all the metapaths (also the intermediate ones and their score
    value to be able to consider also intermediate metapaths).
    We also added a statistics count that takes into account the counts of how many 
    times each metapath has been use in the path (for example assuming to have the 
    metapath A->B->C, we count how many A nodes are linked to C nodes throught this
    set of relations).    
    """
    device = y.device
    all_path_info = []  # (score, path, bags, labels, alpha)
    metapath_counts = defaultdict(int)

    current_paths = [[]]
    current_bags = [[int(i)] for i in torch.where(train_mask)[0]]
    current_labels = [y[i].item() for i in torch.where(train_mask)[0]]
    alpha = {int(i): 1.0 for i in torch.where(train_mask)[0]}

    global_to_local = build_global_to_local_from_edge_index(data)


    for level in range(L_max):
        print(f"LEVEL {level}")
        next_path_info = []

        for path in current_paths:
            last_ntype = node_type if not path else path[-1][2]
            print(f"Expanding from node type: {last_ntype}")

            with torch.no_grad():
                encoder = HeteroEncoder(
                    channels=channels,
                    node_to_col_names_dict={
                        ntype: data[ntype].tf.col_names_dict for ntype in data.node_types
                    },
                    node_to_col_stats=col_stats_dict,
                ).to(device)

                for module in encoder.modules():
                    for name, buf in module._buffers.items():
                        if buf is not None:
                            module._buffers[name] = buf.to(device)

                tf_dict = {
                    ntype: data[ntype].tf.to(device)
                    for ntype in data.node_types if 'tf' in data[ntype]
                }
                node_embeddings_dict = encoder(tf_dict)

            candidate_rels = [
                (src, rel, dst)
                for (src, rel, dst) in data.edge_index_dict.keys()
                if src == last_ntype
            ][:max_rels]

            for rel in candidate_rels:
                print(f"considering relation {rel}")
                src, _, dst = rel
                if dst in [step[0] for step in path] or dst == node_type:
                    continue# avoid loops in met, avoid to return to source node

                src_emb = node_embeddings_dict.get(src)
                dst_emb = node_embeddings_dict.get(dst)
                if src_emb is None or dst_emb is None:
                    print(f"error: embedding of node {dst} not found")
                    continue

                #train θ with ranking loss
                score, theta = evaluate_relation_learned(
                    bags=current_bags,
                    labels=current_labels,
                    node_embeddings=src_emb,
                    alpha_prev=alpha,
                    global_to_local=global_to_local[src],  # passa mapping per tipo src
                )

                print(f"Obtained score {score} for relation {rel}")

                
                bags, labels, alpha_next = construct_bags_with_alpha(
                    data=data,
                    previous_bags=current_bags,
                    previous_labels=current_labels,
                    alpha_prev=alpha,
                    rel=rel,
                    node_embeddings=dst_emb,
                    theta=theta,
                    src_embeddings=src_emb,
                    global_to_local=global_to_local[src],  # mapping corretto
                )


                if len(bags) < 5:
                    continue

                new_path = path + [rel]
                next_path_info.append((score, new_path, bags, labels, alpha_next))

        
        next_path_info.sort(key=lambda x: x[0])
        selected = next_path_info[:beam_width]

        # Prepara per livello successivo
        current_paths = []
        current_bags = []
        current_labels = []
        alpha = {}

        for score, path, bags, labels, alpha_next in selected:
            current_paths.append(path)
            current_bags.extend(bags)
            current_labels.extend(labels)
            alpha.update(alpha_next)
            all_path_info.append((score, path, bags))

    # Seleziona i migliori beam_width finali tra tutti i livelli
    all_path_info.sort(key=lambda x: x[0])
    final_selected = all_path_info[:beam_width]
    metapaths = []

    for _, path, bags in final_selected:
        metapaths.append(path)
        metapath_counts[tuple(path)] = len(bags)

    print(f"FINAL METAPATHS (TOP {beam_width}):")
    for path in metapaths:
        print("  ", path)

    return metapaths, metapath_counts



