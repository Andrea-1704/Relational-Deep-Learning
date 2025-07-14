import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import relbench
import numpy as np
from relbench.datasets import get_dataset
from relbench.tasks import get_task
import math
from tqdm import tqdm
import torch_geometric
import torch_frame
from torch_geometric.seed import seed_everything
from relbench.modeling.utils import get_stype_proposal
from collections import defaultdict
import requests
from io import StringIO
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph
import copy
from typing import Any, Dict, List
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_frame.data.stats import StatType
import torch
import math
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
from relbench.modeling.nn import HeteroEncoder
from collections import defaultdict


import sys
import os
sys.path.append(os.path.abspath("."))

from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
from utils.EarlyStopping import EarlyStopping
from utils.mpsgnn_extention2_v2 import build_json_for_entity_path
from model.MPSGNN_Model import MPSGNN
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
from utils.task_cache import get_task_description, get_task_metric  # se non l’hai ancora fatto


dataset = get_dataset("rel-f1", download=True)
task = get_task("rel-f1", "driver-top3", download=True)

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")

out_channels = 1

#loss_fn = nn.BCEWithLogitsLoss()
tune_metric = "f1"
higher_is_better = True #is referred to the tune metric

seed_everything(42) #We should remember to try results 5 times with
#different seed values to provide a confidence interval over results.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "./data"

db = dataset.get_db()
col_to_stype_dict = get_stype_proposal(db)
db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)

data_official, col_stats_dict_official = make_pkey_fkey_graph(
    db_nuovo,
    col_to_stype_dict=col_to_stype_dict_nuovo,
    text_embedder_cfg = None,
    cache_dir=None
)
#do not use the textual information: this db is mostly not textual

graph_driver_ids = db_nuovo.table_dict["drivers"].df["driverId"].to_numpy()
id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}


train_df_raw = train_table.df
driver_ids_raw = train_df_raw["driverId"].to_numpy()
qualifying_positions = train_df_raw["qualifying"].to_numpy() #labels (train)


binary_top3_labels_raw = qualifying_positions #do not need to binarize 
#since the task is already a binary classification task


target_vector_official = torch.full((len(graph_driver_ids),), float("nan")) #inizialize a vector with all "nan" elements
for i, driver_id in enumerate(driver_ids_raw):
    if driver_id in id_to_idx:#if the driver is in the training
        target_vector_official[id_to_idx[driver_id]] = binary_top3_labels_raw[i]


data_official['drivers'].y = target_vector_official.float()
data_official['drivers'].train_mask = ~torch.isnan(target_vector_official)
y_full = data_official['drivers'].y.float()
train_mask_full = data_official['drivers'].train_mask
num_pos = (y_full[train_mask_full] == 1).sum()
num_neg = (y_full[train_mask_full] == 0).sum()
pos_weight = torch.tensor([num_neg / num_pos], device=device)
data_official['drivers'].y = target_vector_official

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

hidden_channels = 128
out_channels = 128

loader_dict = loader_dict_fn(
    batch_size=1024,
    num_neighbours=512,
    data=data_official,
    task=task,
    train_table=train_table,
    val_table=val_table,
    test_table=test_table
)
lr=1e-02
wd=0
node_type="drivers"


#execution:
task_name = "driver-top3"
path = [('drivers', 'rev_f2p_driverId', 'results')]

# In-context example 1
example_1 = build_json_for_entity_path(
    entity_id=18,
    path=path,
    data=data_official,
    db=db,
    task_name=task_name,
    y=1  # inserisce il campo "Target"
)

# In-context example 2
example_2 = build_json_for_entity_path(
    entity_id=27,
    path=path,
    data=data_official,
    db=db,
    task_name=task_name,
    y=0
)

# Target node (senza y)
target_node = build_json_for_entity_path(
    entity_id=42,
    path=path,
    data=data_official,
    db=db,
    task_name=task_name,
    y=None  #  non inserisce "Target"
)

docs = [example_1, example_2, target_node]
import json
prompt = "You are a data analyst.\n\n"
prompt += f"Task: {get_task_description(task_name)}\n\n"
prompt += "Here are some examples:\n"
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


safe_docs = [convert_timestamps(doc) for doc in docs]

for doc in safe_docs[:-1]:  # solo gli in-context con Target
    prompt += json.dumps(doc, indent=2, ensure_ascii=False) + "\n\n"

prompt += "Now predict the label for this example:\n"
prompt += json.dumps(safe_docs[-1], indent=2, ensure_ascii=False)

print(prompt)