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

import sys
import os
sys.path.append(os.path.abspath("."))

from model.MPSGNN_Model import MPSGNN
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.mpsgnn_metapath_utils import binarize_targets # binarize_targets sarà usata qui
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
from utils.EarlyStopping import EarlyStopping
from utils.mpsgnn_metapath_utils import greedy_metapath_search_with_bags_learned, beam_metapath_search_with_bags_learned



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


train_df = train_table.df

assert "driverId" in train_df.columns, "driverId not in train_table"

# Conta quanti driver unici ci sono nel training set
unique_driver_ids = train_df["driverId"].nunique()
print(f"Numero di driver unici nel training set: {unique_driver_ids}")
print("Totale driver nel grafo:", db_nuovo.table_dict["drivers"].df["driverId"].nunique())




#do not use the textual information: this db is mostly not textual

graph_driver_ids = db_nuovo.table_dict["drivers"].df["driverId"].to_numpy()
id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}

train_df_raw = train_table.df
driver_ids_raw = train_df_raw["driverId"].to_numpy()
qualifying_positions = train_df_raw["qualifying"].to_numpy() #labels (train)

binary_top3_labels_raw = qualifying_positions 

target_vector_official = torch.full((len(graph_driver_ids),), float("nan"))
for i, driver_id in enumerate(driver_ids_raw):
  if driver_id in id_to_idx:#if the driver is in the training
      target_vector_official[id_to_idx[driver_id]] = binary_top3_labels_raw[i]


data_official['drivers'].y = target_vector_official.float()
data_official['drivers'].train_mask = ~torch.isnan(target_vector_official)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_full, col_stats_dict_full = make_pkey_fkey_graph(
  db_nuovo,
  col_to_stype_dict=col_to_stype_dict_nuovo,
  text_embedder_cfg=None,
  cache_dir=None
)
data_full = data_full.to(device)


data_full['drivers'].y = target_vector_official.float()
data_full['drivers'].train_mask = ~torch.isnan(target_vector_official)


y_full = data_full['drivers'].y.float()
train_mask_full = data_full['drivers'].train_mask
num_pos = (y_full[train_mask_full] == 1).sum()
num_neg = (y_full[train_mask_full] == 0).sum()
pos_weight = torch.tensor([num_neg / num_pos], device=device)

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

hidden_channels = 128
out_channels = 128

y = data_official['drivers'].y
train_mask = data_official['drivers'].train_mask

print("Num training targets:", train_mask.sum().item())
print("Class balance in training set:")
print("  Class 0:", (y[train_mask] == 0).sum().item())
print("  Class 1:", (y[train_mask] == 1).sum().item())

loader_dict = loader_dict_fn(
  batch_size=1024,
  num_neighbours=512,
  data=data_official,
  task=task,
  train_table=train_table,
  val_table=val_table,
  test_table=test_table
)

