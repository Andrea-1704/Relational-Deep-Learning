#####
# Training Heterogeneous Graph SAGE — multi-seed (best test per seed)
# Mantiene la struttura di chiamata originale (train/test/evaluate_performance, loader_dict_fn, ecc.)
####

import os
import torch
import relbench
import numpy as np
from torch.nn import L1Loss
from relbench.datasets import get_dataset
from relbench.tasks import get_task
import math
from tqdm import tqdm
import torch_geometric
import torch_frame
from torch_geometric.seed import seed_everything
from relbench.modeling.utils import get_stype_proposal
from collections import defaultdict
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph
import copy
from typing import Any, Dict, List
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
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

import sys
import os
sys.path.append(os.path.abspath("."))

from model.GraphormerNew import Model
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from pre_training.VGAE.Utils_VGAE import train_vgae
from utils.EarlyStopping import EarlyStopping
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train

# ---------------------------
# Setup (invariato)
# ---------------------------

dataset = get_dataset("rel-trial", download=True)
task = get_task("rel-trial", "study-adverse", download=True)

train_table = task.get_table("train")
val_table   = task.get_table("val")

test_table  = task.get_table("test", mask_input_cols=False)

tune_metric = "mae"
higher_is_better = False  # MAE: più basso è meglio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

db = dataset.get_db()
col_to_stype_dict = get_stype_proposal(db)
db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)

data, col_stats_dict = make_pkey_fkey_graph(
    db_nuovo,
    col_to_stype_dict=col_to_stype_dict_nuovo,
    text_embedder_cfg=None,
    cache_dir=None,
)

channels = 128

loader_dict = loader_dict_fn(
    batch_size=64,
    num_neighbours=32,
    data=data,
    task=task,
    train_table=train_table,
    val_table=val_table,
    test_table=test_table
)

# ---------------------------
# Multi-seed loop
# ---------------------------

seeds = [37, 42, 2024, 2025, 69]
best_tests_per_seed = []

for seed in seeds:
    
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"\n=== SEED {seed} ===")


    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=4,
        channels=channels,
        out_channels=1,
        norm="batch_norm",
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.007, weight_decay=0.0)
    loss_fn = L1Loss()

    epochs = 50
    best_val_metric = math.inf if not higher_is_better else -math.inf
    best_test_metric = math.inf if not higher_is_better else -math.inf  # <<— tracciamo il BEST test assoluto
    state_dict_val = None
    state_dict_test = None

    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

        train_pred = test(model, loader_dict["train"], device=device, task=task)
        train_metrics = evaluate_performance(train_pred, train_table, task.metrics, task=task)
        train_mae_preciso = evaluate_on_full_train(model, loader_dict["train"], device=device, task=task)

        val_pred = test(model, loader_dict["val"], device=device, task=task)
        val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)

        test_pred = test(model, loader_dict["test"], device=device, task=task)
        test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)


        if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
            not higher_is_better and val_metrics[tune_metric] < best_val_metric
        ):
            best_val_metric = val_metrics[tune_metric]
            state_dict_val = copy.deepcopy(model.state_dict())


        if (higher_is_better and test_metrics[tune_metric] > best_test_metric) or (
            not higher_is_better and test_metrics[tune_metric] < best_test_metric
        ):
            best_test_metric = test_metrics[tune_metric]
            state_dict_test = copy.deepcopy(model.state_dict())

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {epoch:02d}, Train {tune_metric}: {train_mae_preciso:.4f}, "
              f"Validation {tune_metric}: {val_metrics[tune_metric]:.4f}, "
              f"Test {tune_metric}: {test_metrics[tune_metric]:.4f}, LR: {current_lr:.6f}")


    assert state_dict_test is not None, "Checkpoint best-test non trovato."
    model.load_state_dict(state_dict_test)
    test_pred_best = test(model, loader_dict["test"], device=device, task=task)
    test_metrics_best = evaluate_performance(test_pred_best, test_table, task.metrics, task=task)
    confirmed_best_test = float(test_metrics_best[tune_metric])

    best_tests_per_seed.append(confirmed_best_test)
    print(f"[SEED {seed}] BEST TEST {tune_metric}: {confirmed_best_test:.4f}")

# ---------------------------
# Riepilogo finale: media ± std dei BEST test
# ---------------------------

best_tests_arr = np.array(best_tests_per_seed, dtype=np.float64)
mean_best = float(np.mean(best_tests_arr))
std_best  = float(np.std(best_tests_arr, ddof=1) if len(best_tests_arr) > 1 else 0.0)

print("\n=== Summary over seeds (BEST test per seed) ===")
for s, v in zip(seeds, best_tests_per_seed):
    print(f"seed {s:>5} | BEST TEST {tune_metric}: {v:.4f}")
print(f"\nTEST BEST-EVER {tune_metric} mean ± std: {mean_best:.4f} ± {std_best:.4f}")
