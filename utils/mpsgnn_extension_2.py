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
from pathlib import Path
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
import json
from typing import Dict, List, Tuple, Sequence
import pandas as pd
from torch_geometric.data import HeteroData



def build_json_for_entity(entity_id: int,
                          path: Sequence[Tuple[str, str, str]],
                          data: HeteroData,
                          db,
                          max_per_hop: int = 5,
                          id_map: Dict[str, pd.Index] | None = None) -> Dict:
    """
    Build a JSON document describing `entity_id` and all neighbours
    reached by `path`.

    Each hop k is stored under key "hop_k".
    Follows the layout used in 'Tackling prediction tasks in RDL with LLMs'
    (Alg. 1): flat JSON with explicit "table" tags.

    Parameters
    ----------
    entity_id      : global node id of the source entity
    path           : list of (src, rel, dst) tuples
    data           : HeteroData containing edge_index_dict
    db             : RelBench DB object (table_dict gives DataFrames)
    max_per_hop    : sample at most this many neighbours per source node
    id_map         : optional {node_type: pandas.Index}.  If provided,
                     we translate global_id -> dataframe row via id_map[ntype].get_loc()

    Returns
    -------
    document : dict
    """
    if not path:
        raise ValueError("Path must contain at least one relation")

    document = {
        "source_id": int(entity_id),
        "source_table": path[0][0],
    }

    current_nodes = [int(entity_id)]
    current_ntype = path[0][0]

    for lvl, (src, rel, dst) in enumerate(path, start=1):
        hop_key = f"hop_{lvl}"
        document[hop_key] = []
        next_nodes: list[int] = []

        # edge_index shape [2, E]
        src_idx, dst_idx = data.edge_index_dict[(src, rel, dst)]

        # translate global->local for fast masking if needed
        if id_map and src in id_map:
            src_id2row = id_map[src].get_indexer
        else:
            src_id2row = lambda ids: ids  # identity

        if id_map and dst in id_map:
            dst_row_of = lambda nid: id_map[dst].get_loc(nid)
        else:
            dst_row_of = lambda nid: nid

        for v in current_nodes:
            mask = (src_idx == v).nonzero(as_tuple=True)[0]
            neigh = dst_idx[mask].tolist()[:max_per_hop]

            for u in neigh:
                row_dict = (
                    db.table_dict[dst]
                      .df.iloc[dst_row_of(u)]
                      .to_dict()
                )
                row_dict["table"] = dst  # disambiguate for the LLM
                document[hop_key].append(row_dict)

            next_nodes.extend(neigh)

        current_nodes = next_nodes
        current_ntype = dst

    return document





# === tiny cache of task descriptions downloaded once from relbench.stanford.edu =============
_TASK_CACHE: Dict[str, Dict[str, str]] = {}


def _download_task_descriptions() -> None:
    """Fetch (dataset, task) -> description from RelBench website once."""
    import requests
    from bs4 import BeautifulSoup

    root = "https://relbench.stanford.edu/databases"
    datasets = ["rel-f1", "rel-trial", "rel-amazon", "rel-hm", "rel-avito", "rel-stack", "rel-event"]
    for ds in datasets:
        html = requests.get(f"{root}/{ds}/").text
        soup = BeautifulSoup(html, "html.parser")
        headers = soup.find_all("h3")
        for h in headers:
            task_id = h.text.strip("`")
            desc = h.find_next("p").text
            _TASK_CACHE[task_id] = {"dataset": ds, "description": desc}


def get_task_description(task_name: str) -> str:
    if not _TASK_CACHE:
        _download_task_descriptions()
    return _TASK_CACHE.get(task_name, {}).get("description", "")


def build_prompt(json_doc: Dict,
                 task_name: str,
                 task_type: str) -> str:
    """
    task_type ∈ {"binary", "multiclass", "regression"}
    task_name e.g. "driver-top3"
    """
    task_desc = get_task_description(task_name)

    requirement = {
        "binary":    "Return **only** 0 or 1.",
        "multiclass": "Return **only** an integer class label (e.g., 0,1,2...).",
        "regression": "Return **only** one real number (no units)."
    }[task_type]

    prompt = (
        "You are an expert sports data analyst.\n\n"
        f"**Task ({task_name})**: {task_desc}\n"
        f"{requirement}\n\n"
        "Below is a JSON dump of the entity and its neighbourhood in the database.\n"
        "Use ONLY the information in the JSON.\n\n"
        "<JSON>\n"
        f"{json.dumps(json_doc, ensure_ascii=False)}\n"
        "</JSON>\n\n"
        "Answer:\n"
    )
    return prompt
