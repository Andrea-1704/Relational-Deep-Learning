from enum import Enum
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
from utils.mpsgnn_extension2 import build_json_for_entity_path
from model.MPSGNN_Model import MPSGNN
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
from utils.task_cache import get_task_description, get_task_metric  
from utils.mpsgnn_extension3 import greedy_metapath_search
from utils.mpsgnn_extension4 import RLAgent, warmup_rl_agent, final_metapath_search_with_rl
from relbench.base.task_base import TaskType

task_name = "driver-top3"

dataset = get_dataset("rel-f1", download=True)
task = get_task("rel-f1", task_name, download=True)
task_type = task.task_type

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

# Ricava gli ID dei driver nella validation table
val_df_raw = val_table.df
val_driver_ids = val_df_raw["driverId"].to_numpy()

# Costruisci la mask come boolean mask sul vettore completo
val_mask = torch.tensor([driver_id in val_driver_ids for driver_id in graph_driver_ids])
data_official["drivers"].val_mask = val_mask


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

agent = RLAgent(tau=1.0, alpha=0.5)

# Warm-up: breve ma ripetuto
warmup_rl_agent(
    agent=agent,
    data=data_official,
    db=db_nuovo,
    node_id='driverId',
    loader_dict=loader_dict,
    task=task,
    loss_fn=loss_fn,
    tune_metric=tune_metric,
    higher_is_better=higher_is_better,
    train_mask=train_mask_full,
    node_type='drivers',
    col_stats_dict=col_stats_dict_official,
    num_episodes=10,
    L_max=2,
    epochs=5
)

# Chiamata finale
metapath = final_metapath_search_with_rl(
    agent=agent,
    data=data_official,
    db=db_nuovo,
    node_id='driverId',
    loader_dict=loader_dict,
    task=task,
    loss_fn=loss_fn,
    tune_metric=tune_metric,
    higher_is_better=higher_is_better,
    train_mask=train_mask_full,
    node_type='drivers',
    col_stats_dict=col_stats_dict_official,
    L_max=3,
    epochs=20,
    number_of_metapaths=5
)

print(f"The final metapath is {metapath}")