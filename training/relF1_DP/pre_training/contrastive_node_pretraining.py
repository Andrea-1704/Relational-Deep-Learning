
import os
import torch
import relbench
import numpy as np
from torch.nn import BCEWithLogitsLoss, L1Loss
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
from torch.nn import BCEWithLogitsLoss
import copy
from typing import Any, Dict, List
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType

import sys
import os
sys.path.append(os.path.abspath("."))

from pre_training.contrastive_node_pretraining import utility

from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from torch_geometric.loader import NeighborLoader
import pyg_lib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import ModuleDict, Linear
import torch.nn.functional as F
from torch import nn
import random
from model.HGraphSAGE import Model
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from pre_training.VGAE.Utils_VGAE import train_vgae
from utils.EarlyStopping import EarlyStopping
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train



dataset = get_dataset("rel-f1", download=True)
task = get_task("rel-f1", "driver-position", download=True)

train_table = task.get_table("train") #date  driverId  qualifying
val_table = task.get_table("val") #date  driverId  qualifying
test_table = task.get_table("test") # date  driverId

out_channels = 1
loss_fn = L1Loss()
# this is the mae loss and is used when have regressions tasks.
tune_metric = "mae"
higher_is_better = False

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "./data"

db = dataset.get_db() #get all tables
col_to_stype_dict = get_stype_proposal(db)
#this is used to get the stype of the columns

#let's use the merge categorical values:
db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)

# Create the graph
data, col_stats_dict = make_pkey_fkey_graph(
    db_nuovo,
    col_to_stype_dict=col_to_stype_dict_nuovo,
    #text_embedder_cfg=text_embedder_cfg,
    text_embedder_cfg = None,
    cache_dir=None  # disabled
)


# pre training phase with the VGAE
channels = 128

model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=2,
    channels=channels,
    out_channels=1,
    aggr="max",
    norm="batch_norm",
).to(device)



optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=0
)

scheduler = CosineAnnealingLR(optimizer, T_max=100)


early_stopping = EarlyStopping(
    patience=30,
    delta=0.0,
    verbose=True,
    path="best_basic_model.pt"
)

loader_dict = loader_dict_fn(
    batch_size=512, 
    num_neighbours=256, 
    data=data, 
    task=task,
    train_table=train_table, 
    val_table=val_table, 
    test_table=test_table
)

num_neighbours = 256
batch_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim = channels
entity_table = "drivers"

#costruzione loader di pre trai

# Recupera i seed del TRAIN, i tempi dei seed e la transform dal task
input_nodes_tr, input_time_tr, transform_tr = get_node_train_table_input(train_table, task)  
assert isinstance(entity_table, str), f"entity_table must be str, got {type(entity_table)}"

from torch_geometric.loader import NeighborLoader

train_loader_pretrain = NeighborLoader(
    data,                                        # HeteroData materializzato
    input_nodes=(entity_table, input_nodes_tr),  # (node_type, tensor di seed ids)
    input_time=input_time_tr,                    # <-- CRITICO: abilita batch.seed_time
    transform=transform_tr,                      # <-- CRITICO: annota mapping nodo→seed nel batch
    time_attr="time",                            # attributo temporale nei NodeStorage
    temporal_strategy="uniform",                 # sampler time-aware
    num_neighbors=[num_neighbours, num_neighbours], 
    batch_size=batch_size,
    shuffle=True,
)


#per debug:
batch = next(iter(train_loader_pretrain))
print("has seed_time:", hasattr(batch, "seed_time"))
print("seed_time shape:", getattr(batch, "seed_time", None).shape if hasattr(batch, "seed_time") else None)
print("entity time present:", hasattr(batch[entity_table], "time"))
# Alcune pipeline aggiungono anche un indice per nodo→seed:
print("has per-type seed mapping (common names):",
      hasattr(batch[entity_table], "seed_time_index"),
      hasattr(batch[entity_table], "batch"))



# 1) PRETRAIN DGI (in-domain, solo train)
model = utility.pretrain_dgi(
    model=model,
    data=data,
    loader=train_loader_pretrain,        
    entity_table=entity_table,
    hidden_dim=channels,
    device=device,
    epochs=20,
    lr=1e-3,
    weight_decay=0.0,
    use_shallow=True,                   # abilita shallow per rendere efficace la corruzione
    log_every=50
)

# 2) LINEAR PROBE (opzionale ma raccomandata)
val_mae_lp, test_mae_lp = utility.run_linear_probe(
    model=model,
    train_loader=loader_dict["train_eval"],  # loader "eval" senza shuffle, o riusa train con shuffle=False
    val_loader=loader_dict["val"],
    test_loader=loader_dict["test"],
    entity_table=entity_table,
    hidden_dim=hidden_dim,
    device=device,
    lr=1e-3,
    weight_decay=0.0,
    epochs=50
)

# 3) FINE-TUNING supervisionato
val_mae_ft, test_mae_ft = utility.fine_tune_supervised(
    model=model,
    train_loader=loader_dict["train"],
    val_loader=loader_dict["val"],
    test_loader=loader_dict["test"],
    entity_table=entity_table,
    device=device,
    lr=5e-4,
    weight_decay=0.0,
    epochs=100,
    early_stopping_patience=10
)

print(f"[RESULTS] Linear probe  -> val {val_mae_lp:.4f} | test {test_mae_lp:.4f}")
print(f"[RESULTS] Fine-tuning   -> val {val_mae_ft:.4f} | test {test_mae_ft:.4f}")