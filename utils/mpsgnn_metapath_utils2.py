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

    bag_embeddings = [
        node_embeddings[torch.tensor(bag, device=device)] for bag in bags
    ]

    alpha_values = []
    binary_labels = torch.tensor(labels, device=device)

    for bag in bags:
        #emb = node_embeddings[torch.tensor(bag, device=device)]#####################################
        alpha = torch.tensor([alpha_prev.get(v, 1.0) for v in bag], device=device)
        #bag_embeddings.append(emb)
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
    epochs: int = 100,
    lr: float = 1e-2,
) -> Tuple[float, nn.Module]:

    device = node_embeddings.device
    binary_labels = torch.tensor(labels, device=device)

    #Train theta neural network
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
        emb = node_embeddings[torch.tensor(bag, device=device)]
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
            neighbors_u = edge_dst[edge_src == v]
            if len(neighbors_u) == 0:
                continue
            x_v = src_embeddings[v]
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
    data: HeteroData, #the result of make_pkey_fkey_graph
    db,   #Object that was passed to make_pkey_fkey_graph to build data
    node_id: str, #ex driverId
    train_mask: torch.Tensor,
    node_type: str, 
    col_stats_dict: Dict[str, Dict[str, Dict]], 
    L_max: int = 3,
    max_rels: int = 10,
    channels : int = 64,
    beam_width: int = 5,  #number of metapaths to look for
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metapaths = []
    metapath_counts = {} #for each metapath counts how many bags are presents, so how many istances of that metapath are present
    current_paths = [[]]
    driver_ids_df = db.table_dict[node_type].df[node_id].to_numpy()
    current_bags =  [[int(i)] for i in driver_ids_df if train_mask[i]]
    #current_bags contains the id of the drivers node 
    old_y = data[node_type].y.int().tolist() #ordered as current bags
    print(f"initial y: {old_y}")
    current_labels = []
    for i in range(0, len(old_y)):
        if train_mask[i]:
            current_labels.append(old_y[i])
    print(current_bags)

    
    assert len(current_bags) == len(current_labels)
    
    alpha = {int(i): 1.0 for i in torch.where(train_mask)[0]}
    all_path_info = [] #memorize all the metapaths with scores, in order
    #to select only the best beam_width at the end

    with torch.no_grad():
        encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                ntype: data[ntype].tf.col_names_dict
                for ntype in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        ).to(device)
        for module in encoder.modules():
            for name, buf in module._buffers.items():
                if buf is not None:
                    module._buffers[name] = buf.to(device)
        tf_dict = {
            ntype: data[ntype].tf.to(device) for ntype in data.node_types
        } #each node type as a tf.
        node_embeddings_dict = encoder(tf_dict)


    for level in range(L_max):
        print(f"we are at level {level}")
        
        next_paths_info = []

        for path in current_paths:
            last_ntype = node_type if not path else path[-1][2]
            print(f"current source node is {last_ntype}")

            candidate_rels = [
                (src, rel, dst)
                for (src, rel, dst) in data.edge_index_dict.keys()
                if src == last_ntype
            ][:max_rels] 

            for rel in candidate_rels: 
                print(f"considering relation {rel}")
                src, _, dst = rel
                if dst in [step[0] for step in path] or dst == node_type:  # avoid loops in met and avoid to return to the source node
                  continue

                src_emb = node_embeddings_dict.get(src)
                dst_emb = node_embeddings_dict.get(dst)

                if src_emb is None or dst_emb is None:
                    print(f"error: embedding of node {dst} not found")
                    continue
                
                #node_embeddings = node_embeddings_dict.get(dst) #Tensor[num_node_of_kind_dst, embedding_dim]

                #train θ with ranking loss
                score, theta = evaluate_relation_learned(
                    bags=current_bags,
                    labels=current_labels,
                    #node_embeddings=src_emb,
                    node_embeddings = node_embeddings_dict[src],
                    alpha_prev=alpha,
                    #global_to_local=global_to_local[src],  # passa mapping per tipo src
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
                    src_embeddings = node_embeddings_dict[src]
                )

                if len(bags) < 5:
                    continue

                new_path = path + [rel]

                next_paths_info.append((score, new_path, bags, labels, alpha_next))

        current_paths = []
        current_bags = []
        current_labels = []
        alpha = {}

        for info in next_paths_info:
          _, path, bags, labels, alpha_next = info
          current_paths.append(path)
          current_bags.extend(bags)
          current_labels.extend(labels)
          alpha.update(alpha_next)

        all_path_info.extend(next_paths_info)

    #final selection of the best beamwodth paths:
    all_path_info.sort(key=lambda x:x[0])
    selected = all_path_info[:beam_width]
    for _, path, bags, _, _ in selected:
      metapaths.append(path)
      metapath_counts[tuple(path)] = len(bags)
    print(f"final metapaths are: {metapaths}")
    print(f"metapath counts are: {metapath_counts}")
    return metapaths, metapath_counts