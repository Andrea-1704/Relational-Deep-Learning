
#####
# Training Heterogeneous Graph SAGE 
# This code is designed to work with the RelBench framework and PyTorch Geometric.
# It includes data loading, model training, and evaluation.
####


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
from torch_frame.config.text_embedder import TextEmbedderConfig

import sys
import os
sys.path.append(os.path.abspath("."))


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
from model.others.HGraphSAGE import Model
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from pre_training.VGAE.Utils_VGAE import train_vgae
from utils.EarlyStopping import EarlyStopping
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train


from torch_frame.config.text_embedder import TextEmbedderConfig
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

class SBERTTextEmbedding:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
    def __call__(self, sentences):
        # ritorna torch.FloatTensor [N, D]
        arr = self.model.encode(sentences, convert_to_numpy=True, normalize_embeddings=False)
        return torch.from_numpy(np.asarray(arr)).to(self.device)

device = "cuda" if torch.cuda.is_available() else "cpu"
text_embedder_cfg = TextEmbedderConfig(
    text_embedder=SBERTTextEmbedding(device=device),
    batch_size=256,
)



dataset = get_dataset("rel-trial", download=True)
task = get_task("rel-trial", "study-adverse", download=True)

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

db_nuovo = dataset.get_db() #get all tables
col_to_stype_dict_nuovo = get_stype_proposal(db_nuovo)
#this is used to get the stype of the columns

# Create the graph
data, col_stats_dict = make_pkey_fkey_graph(
    db_nuovo,
    col_to_stype_dict=col_to_stype_dict_nuovo,
    text_embedder_cfg=text_embedder_cfg,
    cache_dir=None,
)


# pre training phase with the VGAE
channels = 128

model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=2,
    channels=channels,
    out_channels=1,
    aggr="sum",
    norm="batch_norm",
).to(device)



optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0)

#scheduler = CosineAnnealingLR(optimizer, T_max=100)


# early_stopping = EarlyStopping(
#     patience=30,
#     delta=0.0,
#     verbose=True,
#     path="best_basic_model.pt"
# )

loader_dict = loader_dict_fn(
    batch_size=512, 
    num_neighbours=128, 
    data=data, 
    task=task,
    train_table=train_table, 
    val_table=val_table, 
    test_table=test_table
)

# from relbench.modeling.graph import get_node_train_table_input
# from utils.utils import alignment_check

# # === Ottieni i node id attesi (ordinati come val/test table) ===
# val_input = get_node_train_table_input(
#     data=data,
#     task=task,
#     split="val",              # oppure table=val_table in alcune versioni
#     mask_input_cols=False
# )
# expected_val_ids = val_input["input_nodes"][task.entity_table].cpu()

# test_input = get_node_train_table_input(
#     data=data,
#     task=task,
#     split="test",             # oppure table=test_table in alcune versioni
#     mask_input_cols=False
# )
# expected_test_ids = test_input["input_nodes"][task.entity_table].cpu()

# # === Check: i batch del loader devono rispettare esattamente quell’ordine ===
# alignment_check(loader_dict["val"],  expected_val_ids,  device=device, task=task)
# alignment_check(loader_dict["test"], expected_test_ids, device=device, task=task)



# Training loop
epochs = 50

state_dict = None
test_table = task.get_table("test", mask_input_cols=False)
best_val_metric = -math.inf if higher_is_better else math.inf
best_test_metric = -math.inf if higher_is_better else math.inf
for epoch in range(1, epochs + 1):
    train_loss = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

    train_pred = test(model, loader_dict["train"], device=device, task=task)
    train_metrics = evaluate_performance(train_pred, train_table, task.metrics, task=task)
    train_mae_preciso = evaluate_on_full_train(model, loader_dict["train"], device=device, task=task)

    val_pred = test(model, loader_dict["val"], device=device, task=task)
    val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)

    test_pred = test(model, loader_dict["test"], device=device, task=task)
    test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)

    #scheduler.step(val_metrics[tune_metric])

    if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
            not higher_is_better and val_metrics[tune_metric] < best_val_metric
    ):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())

    #test:
    if (higher_is_better and test_metrics[tune_metric] > best_test_metric) or (
            not higher_is_better and test_metrics[tune_metric] < best_test_metric
    ):
        best_test_metric = test_metrics[tune_metric]
        state_dict_test = copy.deepcopy(model.state_dict())

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch: {epoch:02d}, Train {tune_metric}: {train_mae_preciso:.2f}, Validation {tune_metric}: {val_metrics[tune_metric]:.2f}, Test {tune_metric}: {test_metrics[tune_metric]:.2f}, LR: {current_lr:.6f}")

    # early_stopping(val_metrics[tune_metric], model)

    # if early_stopping.early_stop:
    #     print(f"Early stopping triggered at epoch {epoch}")
    #     break
print(f"best validation results: {best_val_metric}")
print(f"best test results: {best_test_metric}")
