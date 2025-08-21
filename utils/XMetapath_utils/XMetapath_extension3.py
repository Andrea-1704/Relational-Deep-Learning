"""
In this implementation we are going to update the algorithm 
described in the 'mpsgnn_extension1' by going at each
iteration considering the relation r*, choosing it by 
computing the loss obtained by the LLM using the current 
metapath + current relation. r* is the relation that minimizes 
this loss.
"""

import torch
import math
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict
from relbench.modeling.nn import HeteroEncoder
from collections import defaultdict
from utils.XMetapath_utils.XMetapath_extension2 import evaluate_metapath_with_llm


def get_candidate_relations(metadata, current_node_type: str) -> List[Tuple[str, str, str]]:
    """
    This function takes the "metadata" of the grafo (which are basicly all the 
    relevant informations about the graph, such as edge types, node types, etc.)
    and returns all the edges (in tuple format "(src_type, name of relation, 
    dst_type)") that starts from "current_node_type" as "src_type".
    """
    return [rel for rel in metadata[1] if rel[0] == current_node_type]



def greedy_metapath_search(
    data: HeteroData, #the result of make_pkey_fkey_graph
    db,   #Object that was passed to make_pkey_fkey_graph to build data
    node_id: str, #ex. driverId
    task, 
    task_name:str,
    higher_is_better: str,
    train_mask: torch.Tensor,
    node_type: str, 
    col_stats_dict: Dict[str, Dict[str, Dict]], 
    L_max: int = 3,
    channels : int = 64,
    number_of_metapaths: int = 5,  #number of metapaths to look for
    max_rels: int = 10, 
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

    
    metapath_counts = defaultdict(int) 
    driver_ids_df = db.table_dict[node_type].df[node_id].to_numpy()
    current_bags =  [[int(i)] for i in driver_ids_df if train_mask[i]]
    old_y = data[node_type].y.int().tolist()    #be carefull: in this version we are going to consider only 
    #regression or in general numerical labels!
    #this depends on the task.
    #print(f"we got to change this old_y: {data[node_type].y}")
    

    current_labels = []
    for i in range(0, len(old_y)):
        if train_mask[i]:
            current_labels.append(old_y[i])
    assert len(current_bags) == len(current_labels)
    all_path_info = [] 
    local_path = []
    
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

            for rel in candidate_rels: 
                print(f"considering relation {rel}")
                src, _, dst = rel
                #I think we should delte this if in all of these versions:
                # if dst in [step[0] for step in path] or dst == node_type:  # avoid loops in met, avoid to return to the source node
                #   continue
                cur_meta = local_path.copy()
                cur_meta.append(rel)
                score = evaluate_metapath_with_llm(
                    metapath=cur_meta,
                    data=data,
                    db=db,
                    task_name=task_name,
                    task=task,
                    train_mask=train_mask,
                )
                all_path_info.append((score, cur_meta.copy()))
                if score > best_score: #higher is better
                    best_rel = rel
                    best_score = score

            #set best_rel:
            if best_rel:
                local_path.append(best_rel)
                print(f"Best relation is {best_rel} and now local path is {local_path}")
                next_paths_info.append((best_score, local_path))
                metapath_counts[tuple(local_path)] += 1
        current_paths = [best_rel] 
    
    best_score_per_path = {}
    for score, path in all_path_info:
        path_tuple = tuple(path)
        if path_tuple not in best_score_per_path:
            best_score_per_path[path_tuple] = score
    sorted_unique_paths = sorted(best_score_per_path.items(), key=lambda x: x[1], reverse=True)#higher is better
    selected_metapaths = [list(path_tuple) for path_tuple, _ in sorted_unique_paths[:number_of_metapaths]]
 
    return selected_metapaths, metapath_counts

