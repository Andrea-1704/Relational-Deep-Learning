
"""
In this version we are going to include only the version of MPS GNN 
extended to every kind of task.

The second version is very different from the one presented in the 
work of https://arxiv.org/abs/2412.00521, but tries to summer up the
first implementation I proposed (greedy_metapath_search) and takes
inspiration from the work https://arxiv.org/abs/2411.11829v1.

In particular, the second implementation  
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


In the previous version I forgot to consider a possible form of data 
leakage: 
For every in-context example v, we should always garantee that t_v < t_p.
Where t_p is the one for which we are trackling the prediction.
This is something we have to manually garantee since we are not using 
the NeighbourLoader, but the original graph, without any real 
data leakage check!
"""

import json
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict, Any
import torch
import random
from typing import Dict, List, Tuple, Sequence
import pandas as pd
from torch_geometric.data import HeteroData
from utils.XMetapath_utils.task_cache import get_task_metric, get_task_description
import openai  #pip install openai==0.28
import pandas as pd
import re
from sklearn.metrics import mean_absolute_error, roc_auc_score
from relbench.base.task_base import TaskType
import numpy as np
import time

#put here openai key:
#

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



def call_llm(prompt: str, model="llama3-70b-8192", retries=5, wait=30) -> str:
    """
    Function that sends the prompt to the LLM and gets as an 
    answer the results obtained by the LLM using a certain
    metapath.
    """
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"LLM call failed (attempt {attempt+1}):", e)
            if attempt < retries - 1:
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
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
    num_of_examples: int =2, 
    seed: int =42
) -> str:
    
    """
    Build an LLM prompt for a given metapath and target node, including a few labeled examples.

    This function selects a few example nodes from the training set (using the provided train_mask),
    extracts their local neighborhood along the specified metapath, attaches their ground-truth labels,
    and constructs a prompt string that includes:
      - a task description
      - example graphs in JSON format (with labels)
      - the target graph (without label) to be predicted

    Parameters
    ----------
    metapath : list of (src, rel, dst) tuples
        Ordered list of relations forming the metapath to follow when building neighborhoods.

    target_id : int
        The global ID of the target node whose label should be predicted by the LLM.

    task_name : str
        The name of the RelBench task (e.g. "driver-top3"). Used to retrieve description and metric.

    db : RelBench DB
        A database object with `.table_dict` used to retrieve tabular node features.

    data : HeteroData
        A PyG `HeteroData` object containing `edge_index_dict`, features, and node labels.

    task : Task
        The current task object, required by `build_json_for_entity_path`.

    train_mask : torch.Tensor
        A boolean mask indicating which nodes belong to the training set. Used to sample examples.

    max_per_hop : int, optional
        Maximum number of neighbors to include at each hop (default is 5).

    num_examples : int, optional
        Number of labeled examples to include in the prompt (default is 2).

    seed : int, optional
        Random seed used to select example nodes reproducibly (default is 42).

    Returns
    -------
    prompt : str
        A formatted string to be sent to the LLM, including task description, example graphs,
        and the target graph without label.
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
    prompt += "Your answer must be a single line like: Target: 1"
    prompt += json.dumps(target_doc, indent=2, ensure_ascii=False)

    return prompt





def parse_prediction(pred: str, task_type: str):
    """
    Parses an LLM prediction string based on task type.

    Supports responses like:
    - 'Target: 1'
    - 'target = 0'
    - 'The answer is: 3.5' (fallback)
    """
    try:
        # estrai con regex un numero (intero o float) dopo 'target', ':', '='
        match = re.search(r"target\s*[:=]?\s*(-?\d+(?:\.\d+)?)", pred, re.IGNORECASE)
        if not match:
            # fallback: prova a estrarre primo numero
            match = re.search(r"(-?\d+(?:\.\d+)?)", pred)

        if match:
            value = match.group(1)
            if task_type == TaskType.BINARY_CLASSIFICATION:
                return int(float(value))  # 1.0 -> 1
            elif task_type == TaskType.REGRESSION:
                return float(value)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        else:
            return None  # no number found
    except Exception as e:
        print(f"Failed to parse prediction: {e}")
        return None

    



def sample_val_ids_balanced(val_mask, labels, num_val_samples, seed=42):
    """
    Sample validation node indices ensuring at least one example per label class.

    Parameters
    ----------
    val_mask : torch.Tensor (bool)
        Boolean mask indicating validation nodes.
    labels : torch.Tensor
        Tensor of labels (e.g., data[node_type].y)
    num_val_samples : int
        Total number of validation nodes to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sampled_val_ids : List[int]
        List of sampled node indices covering all label classes.
    """
    import random
    from collections import defaultdict

    random.seed(seed)

    val_indices = torch.where(val_mask)[0].tolist()
    label_values = labels[val_mask].tolist()

    label_to_indices = defaultdict(list)
    for idx, label in zip(val_indices, label_values):
        if not torch.isnan(labels[idx]):  # skip NaNs
            label_to_indices[label].append(idx)

    sampled = []
    # Ensure at least one sample per class
    for indices in label_to_indices.values():
        if indices:
            sampled.append(random.choice(indices))

    # Fill remaining slots randomly
    remaining = list(set(val_indices) - set(sampled))
    random.shuffle(remaining)
    sampled.extend(remaining[:max(0, num_val_samples - len(sampled))])

    return sampled



def evaluate_metapath_with_llm(
    metapath,
    data,
    db,
    task_name,
    task,
    train_mask,
    llm_model="llama3-70b-8192",
    max_per_hop=1,
    num_val_samples=10,
    num_examples_per_prompt=3,
    seed=42
):
    """
    Evaluate a metapath by prompting an LLM with N validation samples
    and computing task-specific performance (e.g. AUROC or MAE).

    Parameters
    ----------
    metapath : list of (src, rel, dst) tuples
        The current metapath being evaluated.
    data : HeteroData
        Heterogeneous graph object (from TorchFrame).
    db : RelBench DB
        Database object.
    task_name : str
        Name of the task (e.g. "driver-top3").
    task : RelBench Task object
        Task object from relbench.get_task(...).
    val_mask : torch.Tensor
        Boolean mask of validation nodes.
    train_mask : torch.Tensor
        Boolean mask of training nodes.
    llm_model : str
        Model name to use with the LLM.
    max_per_hop : int
        Max neighbors per hop when building JSON.
    num_val_samples : int
        Number of validation examples to use.
    num_examples_per_prompt : int
        Number of labeled examples per prompt (in-context).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    score : float
        AUROC for classification, -MAE for regression.
    """

    random.seed(seed)

    target_ntype = metapath[0][0]
    all_labels = data[target_ntype].y
    task_type = task.task_type

    # val_indices = torch.where(val_mask)[0].tolist()
    #train_indices = torch.where(train_mask)[0].tolist()
    #we want to select always at least one class for all of the possible labels
    #otherwise metrics as roc auc would create errors:
    val_indices = sample_val_ids_balanced(
        val_mask=data[target_ntype].val_mask,
        labels=data[target_ntype].y,
        num_val_samples=num_val_samples,
        seed=42,
    )

    train_indices = sample_val_ids_balanced(
        val_mask=train_mask,
        labels=data[target_ntype].y,
        num_val_samples=num_examples_per_prompt,
        seed=seed,
    )

    predictions = []
    ground_truths = []

    sampled_val_ids = random.sample(val_indices, min(num_val_samples, len(val_indices)))
    print(f"len of val indices is {len(val_indices)}")
    print(f"length of sampled val ids is {len(sampled_val_ids)}")
    for target_id in sampled_val_ids:
        if torch.isnan(all_labels[target_id]):
            print(f"nan detected")
            continue  # skip unlabelled

        # Sample examples with valid label
        example_ids = []
        while len(example_ids) < num_examples_per_prompt and train_indices:
            candidate = random.choice(train_indices)
            if not torch.isnan(all_labels[candidate]):
                example_ids.append(candidate)

        if len(example_ids) < num_examples_per_prompt:
            print(f"Skipping target {target_id}: not enough training examples")
            continue

        prompt = build_llm_prompt(
            metapath=metapath,
            target_id=target_id,
            example_ids=example_ids,
            task_name=task_name,
            db=db,
            data=data,
            train_mask=train_mask,
            task=task,
            max_per_hop=max_per_hop,
            num_of_examples=num_examples_per_prompt
        )

        raw_response = call_llm(prompt, model=llm_model)
        print(f"obtained raw response: {raw_response}")
        pred = parse_prediction(raw_response, task_type)
        print(f"obtained pred {pred}")

        if pred is not None:
            y_true = all_labels[target_id].item()
            print(f"the correct label was {y_true}")
            predictions.append(pred)
            ground_truths.append(y_true)
        else:
            print(f"Could not parse prediction for target_id={target_id} → {raw_response}")

    # filter out any accidental NaNs
    pairs = [(y, p) for y, p in zip(ground_truths, predictions) if not np.isnan(y)]

    if not pairs:
        print("No valid predictions to evaluate.")
        return None

    clean_y, clean_preds = zip(*pairs)
    print(f"clean y is {clean_y}")
    print(f"clean preds is {clean_preds}")

    if task_type == TaskType.BINARY_CLASSIFICATION:
        return roc_auc_score(clean_y, clean_preds)
    elif task_type == TaskType.REGRESSION:
        return -mean_absolute_error(clean_y, clean_preds)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")





def build_metapath(
    data,
    db,
    task,
    task_name: str,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    target_node:str,
    max_hops: int = 3,
    epsilon: float = 1e-3,
    max_per_hop: int = 2,
    num_val_samples: int = 2,
    num_examples_per_prompt: int = 10,
    llm_model: str = "llama3-70b-8192",
    seed: int = 42
) -> List[Tuple[str, str, str]]:
    """
    Greedy LLM-guided metapath construction.

    At each hop, the algorithm tries all possible outgoing relations from the
    current node type and selects the one that yields the best LLM score.
    Stops when the improvement in score is less than epsilon.

    Parameters
    ----------
    data: HeteroData
        The heterogeneous graph.
    db: RelBench database
        The relational database object (used to extract features and rows).
    task: RelBench task
        Task object with metadata.
    task_name: str
        Name of the task (e.g. "driver-top3").
    train_mask, val_mask: torch.Tensor
        Masks for training and validation nodes.
    max_hops: int
        Maximum number of hops in the metapath.
    epsilon: float
        Minimum performance improvement required to continue expanding.
    max_per_hop: int
        Max neighbors per hop when building JSON.
    num_val_samples: int
        Number of validation nodes to sample for scoring.
    num_examples_per_prompt: int
        Number of training examples in each prompt.
    llm_model: str
        Name of the LLM model (e.g. llama3).
    seed: int
        Random seed for reproducibility.

    Returns
    -------
    List of (src, rel, dst) tuples representing the greedy metapath.
    """

    current_path = []
    current_score = -float("inf")
    history = []

    source_ntype = target_node
    for hop in range(max_hops):
        print(f"\n--- Hop {hop+1} ---")
        candidates = []

        #Finds all the outgoing relationships from current node
        last_type = source_ntype if not current_path else current_path[-1][2]
        candidate_rels = [
            rel for rel in data.edge_index_dict.keys()
            if rel[0] == last_type and rel not in current_path
        ]

        best_rel = None
        best_score = -float("inf")

        for rel in candidate_rels:
            temp_path = current_path + [rel]
            score = evaluate_metapath_with_llm(
                metapath=temp_path,
                data=data,
                db=db,
                task_name=task_name,
                task=task,
                train_mask=train_mask,
                llm_model=llm_model,
                max_per_hop=max_per_hop,
                num_val_samples=num_val_samples,
                num_examples_per_prompt=num_examples_per_prompt,
                seed=seed
            )
            print(f"Candidate relation {rel} → score: {score}")
            candidates.append((rel, score))

            if score is not None and score > best_score:
                best_score = score
                best_rel = rel

        #interrumpt if improvement is below a threshold
        if best_rel is None or best_score - current_score < epsilon:
            print("Stopping criteria met.")
            break

        current_path.append(best_rel)
        current_score = best_score
        history.append((best_rel, best_score))
        print(f"Selected relation {best_rel} → cumulative path: {current_path} → score: {current_score:.4f}")

    return current_path