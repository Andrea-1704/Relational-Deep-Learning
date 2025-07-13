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
# _TASK_CACHE: Dict[str, Dict[str, str]] = {}
_TASK_CACHE = {
    # === rel-f1 (Formula 1) ===
    "driver-top3": {
        "dataset": "rel-f1",
        "description": (
            "For each driver predict if they will qualify in the top‑3 "
            "for a race in the next 1 month."
        ),
        "metric": "AUROC"
    },
    "driver-dnf": {
        "dataset": "rel-f1",
        "description": (
            "For each driver predict if they will DNF (did not finish) "
            "a race in the next 1 month."
        ),
        "metric": "AUROC"
    },
    "driver-position": {
        "dataset": "rel-f1",
        "description": (
            "Predict the average finishing position of each driver in all "
            "races in the next 2 months."
        ),
        "metric": "MAE"
    },

    # === rel-trial (Clinical Trials) ===
    "study-outcome": {
        "dataset": "rel-trial",
        "description": (
            "Predict if the clinical trial will achieve its primary outcome "
            "(defined as p‑value < 0.05)."
        ),
        "metric": "AUROC"
    },
    "study-adverse": {
        "dataset": "rel-trial",
        "description": (
            "Predict the number of affected patients with severe adverse events or "
            "death for the trial."
        ),
        "metric": "MAE"
    },
    "site-success": {
        "dataset": "rel-trial",
        "description": (
            "Predict the success rate of a trial site in the next 1 year."
        ),
        "metric": "MAE"
    },
    "condition-sponsor-run": {
        "dataset": "rel-trial",
        "description": (
            "Predict whether the sponsor (pharma/hospital) will run clinical trials "
            "for the condition in the next year."
        ),
        "metric": "MAP"
    },
    "site-sponsor-run": {
        "dataset": "rel-trial",
        "description": (
            "Predict whether this sponsor (pharma/hospital) will have a trial in "
            "the facility in the next year."
        ),
        "metric": "MAP"
    },

    # === rel-event (Event recommendation) ===
    "user-repeat": {
        "dataset": "rel-event",
        "description": (
            "Predict whether a user will attend an event (respond yes or maybe) "
            "in the next 7 days, after attending an event in the last 14 days."
        ),
        "metric": "AUROC"
    },
    "user-ignore": {
        "dataset": "rel-event",
        "description": (
            "Predict whether a user will ignore more than 2 event invitations "
            "in the next 7 days."
        ),
        "metric": "AUROC"
    },
    "user-attendance": {
        "dataset": "rel-event",
        "description": (
            "Predict how many events each user will respond yes or maybe "
            "to in the next 7 days."
        ),
        "metric": "MAE"
    },

    # === rel-amazon (E-commerce) ===
    "user-churn": {
        "dataset": "rel-amazon",
        "description": (
            "For each user, predict 1 if the customer does not review any product in "
            "the next 3 months, and 0 otherwise."
        ),
        "metric": "AUROC"
    },
    "item-churn": {
        "dataset": "rel-amazon",
        "description": (
            "For each product, predict 1 if the product does not receive any reviews in "
            "the next 3 months."
        ),
        "metric": "AUROC"
    },
    "user-ltv": {
        "dataset": "rel-amazon",
        "description": (
            "For each user, predict the $ value of the total number of products they buy "
            "and review in the next 3 months."
        ),
        "metric": "MAE"
    },
    "item-ltv": {
        "dataset": "rel-amazon",
        "description": (
            "For each product, predict the $ value of the total number of purchases and "
            "reviews it receives in the next 3 months."
        ),
        "metric": "MAE"
    },
    "user-item-purchase": {
        "dataset": "rel-amazon",
        "description": (
            "Predict the list of distinct items each customer will purchase in the next 3 months."
        ),
        "metric": "MAP"
    },
    "user-item-rate": {
        "dataset": "rel-amazon",
        "description": (
            "Predict the list of distinct items each customer will purchase and give a 5‑star review "
            "in the next 3 months."
        ),
        "metric": "MAP"
    },
    "user-item-review": {
        "dataset": "rel-amazon",
        "description": (
            "Predict the list of distinct items each customer will purchase and give a detailed review "
            "in the next 3 months."
        ),
        "metric": "MAP"
    },

    # === rel-stack (Stack‑Exchange) ===
    "user-engagement": {
        "dataset": "rel-stack",
        "description": (
            "For each user, predict if they will make any votes, posts, or comments in the next 3 months."
        ),
        "metric": "AUROC"
    },
    "user-badge": {
        "dataset": "rel-stack",
        "description": (
            "For each user, predict if they will receive a new badge in the next 3 months."
        ),
        "metric": "AUROC"
    },
    "post-votes": {
        "dataset": "rel-stack",
        "description": (
            "For each post, predict how many votes it will receive in the next 3 months."
        ),
        "metric": "MAE"
    },
    "user-post-comment": {
        "dataset": "rel-stack",
        "description": (
            "Predict a list of existing posts that a user will comment on in the next 2 years."
        ),
        "metric": "MAP"
    },
    "user-post-related": {
        "dataset": "rel-stack",
        "description": (
            "Predict a list of existing posts that users will link a given post to in the next 2 years."
        ),
        "metric": "MAP"
    },

    # === rel-hm (H&M) ===
    "user-churn": {
        "dataset": "rel-hm",
        "description": (
            "Predict the churn for a customer (no transactions) in the next week."
        ),
        "metric": "AUROC"
    },
    "item-sales": {
        "dataset": "rel-hm",
        "description": (
            "Predict the total sales for an article (the sum of prices of the associated transactions) in the next week."
        ),
        "metric": "MAE"
    },

    # === rel-avito (Avito ads) ===
    "user-visits": {
        "dataset": "rel-avito",
        "description": (
            "Predict whether each customer will visit more than one ad in the next 4 days."
        ),
        "metric": "AUROC"
    },
    "user-clicks": {
        "dataset": "rel-avito",
        "description": (
            "Predict whether each customer will click on more than one ad in the next 4 days."
        ),
        "metric": "AUROC"
    },
    "ad-ctr": {
        "dataset": "rel-avito",
        "description": (
            "Assuming the ad will be clicked in the next 4 days, predict the click-through-rate (CTR) for each ad."
        ),
        "metric": "MAE"
    },
    "user-ad-visit": {
        "dataset": "rel-avito",
        "description": (
            "Predict the list of ads a user will visit in the next 4 days."
        ),
        "metric": "MAP"
    },
}



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
