
"""
In this version we are going to include only the version of MPS GNN 
extended to every kind of task.

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
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict, Any
import torch
import random
from typing import Dict, List, Tuple, Sequence
import pandas as pd
from torch_geometric.data import HeteroData
from utils.task_cache import get_task_metric, get_task_description
import openai  #pip install openai==0.28
import pandas as pd

def convert_timestamps(obj):
    if isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(i) for i in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    else:
        return obj


def call_llm(prompt: str, model="llama3-70b-8192") -> str:
    """
    Function that sends the prompt to the LLM and gets as an 
    answer the results obtained by the LLM using a certain
    metapath.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("LLM call failed:", e)
        return ""


def _row_from_df(df: pd.DataFrame,
                 node_id: int | str,
                 ntype: str,
                 id_map: Dict[str, pd.Index] | None) -> Dict[str, Any]:
    """Return a dict of the row for `node_id` in table `ntype`."""
    if id_map and ntype in id_map:
        row_pos = id_map[ntype].get_loc(node_id)
    else:
        row_pos = node_id
    row = df.iloc[row_pos].to_dict()
    row["table"] = ntype
    return row


def _merge_child_into_parent(parent: Dict[str, Any],
                             child_list: List[Dict[str, Any]],
                             dst_ntype: str) -> None:
    """
    Add child rows under key `dst_ntype`.

    We keep children under a dedicated list (JSON nesting) instead of flat-merge
    to avoid key collisions.  This is faithful to the Figure 1 example in the
    paper, where the Transaction row owns a list of Products.

    This function is only used to manage the nesting of the fields in the json 
    format.
    """
    parent.setdefault(dst_ntype, []).extend(child_list)


def _build_row_recursive(curr_id: int,
                         path_remaining: Sequence[Tuple[str, str, str]],
                         data: HeteroData,
                         db,
                         id_map: Dict[str, pd.Index] | None,
                         max_per_hop: int,
                         curr_ntype: str,
                         y_value: Any | None) -> Dict[str, Any]:
    """
    Recursive helper that returns a JSON dict for `curr_id`
    and denormalises along the remaining path.

    In practice this function retrieves the informations of 
    'curr_id' node and if an y_value is provided it adds a 
    'target' field containing such value; is the path is 
    finished it simply returns the node, otherwise it 
    follows the path (src->dst) and reaches all the 
    dst nodes linked and build their json recursively.
    """
    row = _row_from_df(db.table_dict[curr_ntype].df,
                       curr_id, curr_ntype, id_map)
    #take the informations of the node "curr_id"

    # attach target (only for labelled examples)
    if y_value is not None:
        row["Target"] = y_value
    #target value should be present only if src type is target 
    #node and we are considering example samples, so the ones
    #coming from the training dataset.

    if not path_remaining:          # reached end of metapath: return the node.
        return row

    src, _, dst = path_remaining[0] #take next path in the metapath
    
    assert curr_ntype == src, f"path mismatch: expected src={src}, got {curr_ntype}" # sanity check
    #print(f"edges: {data.edge_index_dict}")
    src_ids, dst_ids = data.edge_index_dict[path_remaining[0]] #take all the edges of the current rel type

    # neighbours of curr_id via the current relation
    mask = (src_ids == curr_id).nonzero(as_tuple=True)[0]
    neigh = dst_ids[mask].tolist()[:max_per_hop]

    child_jsons: List[Dict] = []
    for nxt_id in neigh: #recursive construction
        child_json = _build_row_recursive(
            nxt_id,
            path_remaining[1:],   # drop first hop, which has already been considered
            data,
            db,
            id_map,
            max_per_hop,
            dst,
            y_value=None          # children are always *unlabelled* rows
        )
        child_jsons.append(child_json)

    # add children under a dedicated key
    _merge_child_into_parent(row, child_jsons, dst_ntype=dst)
    return row


def build_json_for_entity_path(entity_id: int | str,
                               path: List[Tuple[str, str, str]],
                               data: HeteroData,
                               db,
                               task_name : str,
                               max_per_hop: int = 5,
                               y : Any = None,
                               id_map: Dict[str, pd.Index] | None = None) -> Dict:
    """
    Parameters
    ----------
    entity_id : global id of the source node
    path      : ordered list of (src, rel, dst) tuples (the metapath)
    data      : HeteroData with `edge_index_dict`
    db        : RelBench DB object (gives `.table_dict[ntype].df`)
    max_per_hop : cap neighbours per hop to control JSON size
    id_map      : optional {node_type: pandas.Index} for global→row mapping
    y           : optional, is the target value for entity_id

    Returns
    -------
    document : Dict  (ready to dump via json.dumps)
    """
    if not path:
        raise ValueError("Metapath cannot be empty")
    
    source_ntype = path[0][0] #source node of the first relation in 'path'

    doc_root = _build_row_recursive(
        entity_id,
        path_remaining=path,
        data=data,
        db=db,
        id_map=id_map,
        max_per_hop=max_per_hop,
        curr_ntype=source_ntype,
        y_value= y
    )

    # prepend metadata fields expected by the paper
    document = {
        "source_id": int(entity_id),
        "source_table": source_ntype,
        "TaskDescription": get_task_description(task_name),
        "Metric": get_task_metric(task_name),
        **doc_root
    }
    return document



def build_llm_prompt(
    metapath: List[Tuple[str, str, str]],
    target_id: int,
    example_ids: List[int],
    task_name: str,
    db,
    data: HeteroData,
    train_mask: torch.Tensor,
    task,
    max_per_hop: int = 5,
    num_of_examples: int =5, 
    seed: int =42
) -> str:
    
    """
    
    """

    random.seed(seed)
    source_ntype = metapath[0][0]
    task_desc = get_task_description(task_name) #info of task

    # --- select example nodes from training mask ---
    candidate_ids = torch.where(train_mask)[0].tolist()
    example_ids = random.sample(candidate_ids, num_of_examples)

    #json for examples (with the labels)
    example_docs = []
    for eid in example_ids:
        y_val = data[source_ntype].y[eid].item()
        doc = build_json_for_entity_path(
            entity_id=eid,
            path=metapath,
            data=data,
            db=db,
            task_name=task,
            max_per_hop=max_per_hop,
            y=y_val
        )
        example_docs.append(convert_timestamps(doc))

    # --- build target node ---
    target_doc = build_json_for_entity_path(
        entity_id=target_id,
        path=metapath,
        data=data,
        db=db,
        task_name=task_name,
        max_per_hop=max_per_hop,
        y=None
    )
    target_doc = convert_timestamps(target_doc)

    #build final prompt
    prompt = "You are a data analyst.\n\n"
    prompt += f"Task: {task_desc}\n\n"
    prompt += "Here are some examples:\n"

    for doc in example_docs:
        prompt += json.dumps(doc, indent=2, ensure_ascii=False) + "\n\n"

    prompt += "Now predict the label for this example:\n"
    prompt += json.dumps(target_doc, indent=2, ensure_ascii=False)

    return prompt



