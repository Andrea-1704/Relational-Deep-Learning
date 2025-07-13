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

The second version is very different from the one presented in the 
work of https://arxiv.org/abs/2412.00521, but tries to summer up the
first implementation I proposed (greedy_metapath_search) and takes
inspiration from the work https://arxiv.org/abs/2411.11829v1.

In particular, the second implementation (name of the implementation) 
tries to choose in a similar way of the first version the metapaths 
but at each iteration it simply considers all the possible relation 
and for all of them build a JSON file that contains the graph 
extracted considering the current relation and passes this file to a 
pre-trained LLM to try to estimate the probability of the next token
(the prediction) following https://arxiv.org/abs/2411.11829v1.
After computing all the metapaths, we have two possibile implementations:
1. Train the MPS GNN model we implemented 
2. Pass the subgraph extracted by the metapath to the LLM
This double implementation could help us to answer to the foundamental 
question raised by https://arxiv.org/abs/2411.11829v1, which is 
whether a pre trained-LLM could improve the performances obtained by 
a GNN architecture.
"""

import json
import torch
import math
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
from relbench.modeling.nn import HeteroEncoder
from collections import defaultdict
from model.MPSGNN_Model import MPSGNN
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train


def get_candidate_relations(metadata, current_node_type: str) -> List[Tuple[str, str, str]]:
    """
    This function takes the "metadata" of the grafo (which are basicly all the 
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
    src_embeddings,
) -> Tuple[List[List[int]], List[float], Dict[int, float]]:
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

    for bag_v, label in zip(previous_bags, previous_labels):
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
            
            x_v = src_embeddings[v]

            for u in neighbors_u.tolist():  #consider all the "sons" of node "v" through relation "rel"
                bag_u.append(u)

        if len(bag_u) > 0:
            bags.append(bag_u) #updates the new list of bags
            labels.append(label) #the label of the current bag is the same 
            #as the one that the father bag had.

    return bags, labels



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
            best_bags = None
            best_labels = None

            for rel in candidate_rels: 
                print(f"considering relation {rel}")
                src, _, dst = rel
                if dst in [step[0] for step in path] or dst == node_type:  # avoid loops in met, avoid to return to the source node
                  continue
                if rel == ('races', 'rev_f2p_raceId', 'standings'): # for some reasons it provokes side assertions
                  continue
                
                bags, labels = construct_bags(
                    data=data,
                    previous_bags=current_bags,
                    previous_labels=current_labels,
                    rel=rel,
                    src_embeddings = node_embeddings_dict[src]
                )
                if len(bags) < 5:
                    continue

                local_path2 = local_path.copy()
                #even if it is not the best one we memorize it because maybe will be selected from beam search:

                local_path2.append(rel)
                loc = [local_path2.copy()]
                model = MPSGNN(
                    data=data,
                    col_stats_dict=col_stats_dict,
                    metadata=data.metadata(),
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
    #print(f"\nfinal metapaths are {selected_metapaths}\n")

    return selected_metapaths, metapath_counts



#VERSION TWO:
def build_json_for_entity(entity_id: int,
                          path: List[Tuple[str, str, str]],
                          db,
                          data: HeteroData,
                          max_depth: int = 3,
                          max_per_hop: int = 5) -> Dict:
    """
    Build a JSON document describing the neighborhood of a node along a metapath.
    """
    document = {
        "source_id": entity_id,
        "source_table": path[0][0] if path else None
    }

    current_nodes = [entity_id]

    for level, (src, rel, dst) in enumerate(path):
        next_nodes = []
        hop_key = f"hop_{level+1}"
        document[hop_key] = []

        edge_index = data.edge_index_dict[(src, rel, dst)]
        src_tensor, dst_tensor = edge_index

        for v in current_nodes:
            matched_indices = (src_tensor == v).nonzero(as_tuple=True)[0]
            dst_nodes = dst_tensor[matched_indices].tolist()[:max_per_hop]
            #we are only following the metapath given by the path, ignoring the 
            #rest of the graph

            for u in dst_nodes:#for all of the reached node following path and 
                #starting from the entity id node
                row = db.table_dict[dst].df.iloc[u].to_dict()
                #we are assuming to use a global node id for u.
                row["table"] = dst
                document[hop_key].append(row)

            next_nodes.extend(dst_nodes)

        current_nodes = next_nodes

    return document


def build_prompt(json_obj: Dict, task_type: str, label_desc: str = "") -> str:
    """
    task_type: one of {"binary", "multiclass", "regression"}
    """
    task_desc = {
        "binary": "Output 1 if you think the entity satisfies the condition, 0 otherwise.",
        "multiclass": "Output one of the following class labels: 0, 1, 2, ..., N.",
        "regression": "Output a real-valued prediction (e.g., a number between 0 and 100)."
    }

    prompt = (
        "You are a data analyst.\n\n"
        "Below is a JSON describing an entity and its context extracted from a relational database.\n"
        f"Task: {task_type.upper()}\n"
        f"Target meaning: {label_desc}\n"
        f"{task_desc[task_type]}\n\n"
        f"JSON:\n{json.dumps(json_obj, indent=2)}\n\n"
        "Answer:\n"
    )

    return prompt



def score_metapath_with_llm(path: List[Tuple[str, str, str]],
                            val_ids: List[int],
                            val_labels: List,
                            db,
                            task_type: str,
                            llm_call_fn,
                            label_desc: str = "",
                            metric_fn=None,
                            max_docs: int = 50) -> float:
    """
    path: the current candidate metapath
    val_ids: source node ids, equal to the initial bags
    val_labels: corresponding ground-truth labels for each of the bags
    llm_call_fn: function that sends prompts and returns predictions
    metric_fn: evaluation function (e.g., f1_score, roc_auc, etc.)
    """
    preds, labels = [], []

    for entity_id, label in zip(val_ids, val_labels):
        json_doc = build_json_for_entity(entity_id, path, db)
        prompt = build_prompt(json_doc, task_type, label_desc)

        prediction = llm_call_fn(prompt)  # string or float
        preds.append(prediction)
        labels.append(label)

        if len(preds) >= max_docs:
            break

    # parse & compute metric
    if metric_fn:
        return metric_fn(labels, preds)
    else:
        return default_metric(task_type, labels, preds)
