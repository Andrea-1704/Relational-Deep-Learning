"""
In this version we are going to include only the version of MPS GNN 
extended to every kind of task.

So this implementation, differently from what done in the original 
paper, is not limited to the binary classifcation problems, but could
be used for any kind of problems such as regression and multi class
classification.

The first version I am going to propose simply do not need the 
computation of the surrogate task, but considers the test results
obtained from including a possible relation and takes the one that 
minimize the loss or maximise the accuracy.

A XMetapath model is trained for each possible relation extension.

"""

import torch
import math
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict
from relbench.modeling.nn import HeteroEncoder
from collections import defaultdict
from model.XMetaPath import XMetaPath
from utils.utils import evaluate_performance, test, train
import collections 


def flip_rel(rel_name: str) -> str:
    """
    Converts the name of the relation to its reversed version.
    """
    return rel_name[4:] if rel_name.startswith("rev_") else f"rev_{rel_name}"



def to_canonical(mp_outward):
    """
    This function takes as input the metapath found by the model and converts
    it to a metapath with reversed paths, meaning that the source of each 
    path becomes destination, the destination the source and the name of the 
    relation gets converted (flip_rel).
    
    It should be verified that mp[-1][2] == target node. 
    E.g.:
    [('drivers', 'drives', 'cars'), ('cars', 'drives', 'passengers')] to [('passengers', 'rev_drives', 'cars'), ('cars', 'rev_drives', 'drivers')]
    """
    mp = [(dst, flip_rel(rel), src) for (src, rel, dst) in mp_outward[::-1]]
    # assert mp[-1][2] == "drivers"
    return tuple(mp)



def get_candidate_relations(metadata, current_node_type: str) -> List[Tuple[str, str, str]]:
    """
    This function takes the "metadata" of the graph (which are basicly all the 
    relevant informations about the graph, such as edge types, node types, etc.)
    and returns all the edges (in tuple format "(src_type, name of relation, 
    dst_type)") that starts from "current_node_type" as "src_type".
    """
    return [rel for rel in metadata[1] if rel[0] == current_node_type]



def construct_bags(
    data,
    previous_bags: List[List[int]], 
    previous_labels: List[float],     # list of the "v" nodes
    rel: Tuple[str, str, str],
    previous_seed_ids: List[int]
) -> Tuple[List[List[int]], List[float], List[int]]:
    """
    Estend the bags through relation "rel"
    Returns:
    - new bag 
    - labels associated to nodes v
    """

    edge_index = data.edge_index_dict.get(rel)
    if edge_index is None:
        print(f"this should not have happened, but the relation was not found.")
        return [], [], {}

    edge_src, edge_dst = edge_index #tensor [2, #edges]
    bags = [] #the new bags, one for each "v" node.
    labels = [] #for each bag we consider its label, given by the one of the src in relation r.
    seed_ids = []

    for bag_v, label, seed_id in zip(previous_bags, previous_labels, previous_seed_ids):
        #the previous bag now becomes a "v" node

        bag_u = [] #new bag for the node (bag) "bag_v"

        for v in bag_v: #for each node in the previous bag 

            neighbors_u = edge_dst[edge_src == v] #correct for the first step
            #we consider all the edge indexes of destination type that are linked to the 
            #src type through relation "rel", for which the source was exactly the node "v".
            #  Pratically, here we are going through a 
            #relation rel, for example the "patient->prescription" relation and we are 
            # consideringall the prescription that "father" 
            #node of kind patient had.
            if len(neighbors_u) == 0:
                #could be zero even just because that node simply do not have any of such relations
                #test to understand if we are managing correctly the global and local mapping:
                continue

            for u in neighbors_u.tolist():  #consider all the "sons" of node "v" through relation "rel"
                bag_u.append(u)

        if len(bag_u) > 0:
            #new:
            bag_u = list(dict.fromkeys(bag_u))
            bags.append(bag_u) #updates the new list of bags
            labels.append(label) #the label of the current bag is the same 
            #as the one that the father bag had.
            seed_ids.append(seed_id)#consider same seed id if and only if we have a bag for it

    return bags, labels, seed_ids


def _make_metapath_weights(selected_outward_paths, stats_map,
                           alpha=0.5, beta=0.0, gamma=1.0, eps=1e-6):
    """
    Combina coverage^alpha * support^beta * quality^gamma su medie raccolte.
    Ritorna: (keys_canoniche, torch.tensor pesi normalizzati)
    """
    import torch
    scores, keys = [], []
    for mp_out in selected_outward_paths:
        key = to_canonical(mp_out)
        st  = stats_map.get(key, None)
        if st is None or st["cov_seen"] == 0:
            cov = 0.0; sup = 0.0; q = 0.0
        else:
            cov = st["cov_sum"] / st["cov_seen"]
            sup = st["sup_sum"] / st["cov_seen"]
            q   = st["q_sum"]   / max(st["q_seen"], 1)
        cov_s = (cov + eps) ** alpha
        sup_s = (sup + eps) ** beta
        q_s   = max(q, 0.0) ** gamma
        scores.append(cov_s * sup_s * q_s)
        keys.append(key)
    w = torch.tensor(scores, dtype=torch.float)
    w = w / w.sum().clamp_min(eps)
    return keys, w



def greedy_metapath_search(
    data: HeteroData, #the result of make_pkey_fkey_graph
    db,   #Object that was passed to make_pkey_fkey_graph to build data
    node_id: str, #ex. driverId
    loader_dict,
    task, 
    loss_fn,
    tune_metric : str,
    higher_is_better: str,
    train_mask: torch.Tensor,
    node_type: str, 
    col_stats_dict: Dict[str, Dict[str, Dict]], 
    L_max: int = 3,
    channels : int = 64,
    number_of_metapaths: int = 5,  #number of metapaths to look for
    out_channels: int = 128,
    hidden_channels: int = 128,
    lr : float = 0.0001,
    wd: float = 0,
    epochs: int = 100,
    max_rels: int = 10, 
    final_out_channels: int = 1 #this depends on the task (multi class, regression etc)
) -> Tuple[List[List[Tuple[str, str, str]]], Dict[Tuple, int]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            ntype: data[ntype].tf.to(device) for ntype in data.node_types if 'tf' in data[ntype]
        }
        node_embeddings_dict = encoder(tf_dict)

    

    assert data[node_type].y.numel() == data[node_type].num_nodes, f"y size mismatch for {node_type}"
    assert train_mask.numel() == data[node_type].num_nodes, f"train_mask size mismatch for {node_type}"
    # (optional) verify edge index ranges once
    for (src, rel, dst), (edge_src, edge_dst) in data.edge_index_dict.items():
        assert edge_src.min().item() >= 0 and edge_src.max().item() < data[src].num_nodes, f"edge_src out of range for {(src,rel,dst)}"
        assert edge_dst.min().item() >= 0 and edge_dst.max().item() < data[dst].num_nodes, f"edge_dst out of range for {(src,rel,dst)}"
    

    metapath_counts = defaultdict(int) 
    driver_ids_df = db.table_dict[node_type].df[node_id].to_numpy()
    
    current_bags =  [[int(i)] for i in driver_ids_df if train_mask[i]]
    print(f"driver ids df: {current_bags}")
    old_y = data[node_type].y.int().tolist()    #be carefull: in this version we are going to consider only 
    #regression or in general numerical labels!
    #this depends on the task.
    #print(f"we got to change this old_y: {data[node_type].y}")
    
    idxs2 = torch.where(train_mask)[0].tolist()             # internal indices 0..num_nodes-1
    current_bags2   = [[int(i)] for i in idxs2]              # seed one-node bag per training node
    current_labels2 = [float(data[node_type].y[i]) for i in idxs2]  # keep float to support regression
    print(f"isxs2: {current_bags2}")
    #by running this text you'll find out it is just the sae thing, so no index error!

    current_labels = []
    for i in range(0, len(old_y)):
        if train_mask[i]:
            current_labels.append(old_y[i])
    assert len(current_bags) == len(current_labels)
    all_path_info = [] 
    local_path = []

    idxs2 = torch.where(train_mask)[0].tolist()
    current_bags    = [[int(i)] for i in idxs2]
    current_labels  = [float(data[node_type].y[i]) for i in idxs2]
    current_seed_ids = [int(i) for i in idxs2]    # NEW|||||||||||||||
    #seed idx are the target's node id: the one we start to build the 
    #bags from.
    N_seed_batch    = len(current_seed_ids)
    #This number represents the maximum number of nodes in the bag list
    

    metapath_counts = collections.defaultdict(int) #not used anymore. 
    metapath_stats  = collections.defaultdict(lambda: {   # NEW statistics!
        "cov_sum": 0.0, "sup_sum": 0.0, "q_sum": 0.0,
        "cov_seen": 0, "q_seen": 0,
    })

    
    current_paths = [[]] 
    for level in range(L_max):
        print(f"level {level}")
        
        next_paths_info = []

        for path in current_paths: 
            last_ntype = node_type if not path else path[2]
            print(f"current source node is {last_ntype}")
           
            candidate_rels = [ 
                (src, rel, dst)
                for (src, rel, dst) in data.edge_index_dict.keys()
                if src == last_ntype
            ][:max_rels]

            best_rel = None
            best_score = -math.inf if higher_is_better else math.inf  
            best_bags = None
            best_labels = None

            for rel in candidate_rels: 
                print(f"considering relation {rel}")
                src, _, dst = rel
                if dst in [step[0] for step in path] or dst == node_type:  # avoid loops in met, avoid to return to the source node
                  continue
                if rel == ('races', 'rev_f2p_raceId', 'standings') or rel == ('races', 'rev_f2p_raceId', 'qualifying'): # for some reasons it provokes side assertions
                  continue
                
                bags, labels, seed_ids = construct_bags(
                    data=data,
                    previous_bags=current_bags,
                    previous_labels=current_labels,
                    rel=rel,
                    previous_seed_ids=current_seed_ids,
                )

                if len(bags) < 5:
                    continue

                local_path2 = local_path.copy()
                #even if it is not the best one we memorize it because maybe will be selected from beam search:

                local_path2.append(rel)
                loc = [local_path2.copy()]
                model = XMetaPath(
                    data=data,
                    col_stats_dict=col_stats_dict,
                    metapath_counts = metapath_counts,
                    metapaths=loc,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    final_out_channels=final_out_channels,
                ).to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
                #test_table = task.get_table("test", mask_input_cols=False)
                val_table = task.get_table("val")
                best_val_metrics = -math.inf if higher_is_better else math.inf
                for _ in range(0, epochs):
                    train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)
                    # test_pred = test(model, loader_dict["test"], device=device, task=task)
                    # test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)
                    val_pred = test(model, loader_dict["val"], device=device, task=task)
                    val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)
                    if val_metrics[tune_metric] > best_val_metrics and higher_is_better:
                        best_val_metrics = val_metrics[tune_metric]
                    if val_metrics[tune_metric] < best_val_metrics and not higher_is_better:
                        best_val_metrics = val_metrics[tune_metric]
                
                # candidated path "outward"
                mp_candidate = local_path2.copy()
                mp_candidate.append(rel)

                # tranform the metapath in the canonical format:
                mp_key = to_canonical(mp_candidate)

                # COVERAGE: how many unique seed nodes are covered
                # with respects to the total number of seed nodes
                # seen at the beginning. 
                #during the bag expansion process some seed node
                #may become uncoverd making decreasing this score.
                unique_seeds = set(seed_ids)
                coverage_frac = len(unique_seeds) / max(N_seed_batch, 1)

                #suport measurement: 
                #num = number of leaf nodes reached 
                #den = number of seed nodes
                #If 3 driver reach 2, 1 e 2 leaf nodes, then
                #sup_mean = (2+1+2) / 3 = 1.67
                leaves_per_seed = defaultdict(set)
                for b, s in zip(bags, seed_ids):
                    for u in b:
                        leaves_per_seed[s].add(int(u))
                sup_mean = (sum(len(leaves_per_seed[s]) for s in unique_seeds) / max(len(unique_seeds), 1)) if unique_seeds else 0.0

                # Quality: F1 val of candidate
                st = metapath_stats[mp_key]
                st["cov_sum"]  += float(coverage_frac)
                st["sup_sum"]  += float(sup_mean)
                st["q_sum"]    += float(best_val_metrics)
                st["cov_seen"] += 1
                st["q_seen"]   += 1

                
                print(f"For the partial metapath {local_path2.copy()} we obtain F1 test loss equal to {best_val_metrics}")
                all_path_info.append((best_val_metrics, local_path2.copy()))
                score = best_val_metrics #score now is directly the F1 score returneb by training the model on that metapath

                if score > best_score: #higher is better
                    best_rel = rel
                    best_score = score
                    best_bags = bags
                    best_labels = labels

            #set best_rel:
            if best_rel:
                local_path.append(best_rel)
                print(f"Best relation is {best_rel} and now local path is {local_path}")
                next_paths_info.append((best_score, local_path, best_bags, best_labels))
                metapath_counts[tuple(local_path)] += 1
        current_paths = [best_rel] 
        current_bags, current_labels = best_bags, best_labels
    
    best_score_per_path = {}
    for score, path in all_path_info:
        path_tuple = tuple(path)
        if path_tuple not in best_score_per_path:
            best_score_per_path[path_tuple] = score
    sorted_unique_paths = sorted(best_score_per_path.items(), key=lambda x: x[1], reverse=True)#higher is better
    selected_metapaths = [list(path_tuple) for path_tuple, _ in sorted_unique_paths[:number_of_metapaths]]
    
    keys, weights = _make_metapath_weights(
        selected_outward_paths=selected_metapaths,
        stats_map=metapath_stats,
        alpha=0.5, beta=0.0, gamma=1.0,   #  coverage^0.5 * quality
    )

    final_mps_for_model   = [list(k) for k in keys]
    metapath_weight_dict  = {k: float(w) for k, w in zip(keys, weights.tolist())}

    return final_mps_for_model, metapath_weight_dict


