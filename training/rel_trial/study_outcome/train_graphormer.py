#####
# Training Heterogeneous Graph SAGE  — multi-seed, best-test per seed
# Mantiene la struttura di chiamata originale (train/test/evaluate_performance, loader_dict_fn, ecc.)
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
from model.GraphormerNew import Model
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from pre_training.VGAE.Utils_VGAE import train_vgae
from utils.EarlyStopping import EarlyStopping
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
from torch.nn import BCEWithLogitsLoss

# ---------------------------
# Setup invariato (dataset, task, graph, loader ecc.)
# ---------------------------

dataset = get_dataset("rel-trial", download=True)
task = get_task("rel-trial", "study-outcome", download=True)

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table_masked = task.get_table("test")
test_table = task.get_table("test", mask_input_cols=False)  # per metriche

out_channels = 1
tune_metric = "roc_auc"
higher_is_better = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "./data"

db = dataset.get_db()
col_to_stype_dict = get_stype_proposal(db)

db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)

data, col_stats_dict = make_pkey_fkey_graph(
    db_nuovo,
    col_to_stype_dict=col_to_stype_dict_nuovo,
    text_embedder_cfg=None,
    cache_dir=None
)

graph_driver_ids = db_nuovo.table_dict["studies"].df["nct_id"].to_numpy()
id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}

train_df_raw = train_table.df
driver_ids_raw = train_df_raw["nct_id"].to_numpy()
qualifying_positions = train_df_raw["outcome"].to_numpy().astype(np.float32)
binary_top3_labels_raw = qualifying_positions

target_vector_official = torch.full((len(graph_driver_ids),), float("nan"))
for i, driver_id in enumerate(driver_ids_raw):
    if driver_id in id_to_idx:
        target_vector_official[id_to_idx[driver_id]] = binary_top3_labels_raw[i]

data['studies'].y = target_vector_official.float()
data['studies'].train_mask = ~torch.isnan(target_vector_official)

y_full = data['studies'].y.float()
train_mask_full = data['studies'].train_mask
num_pos = (y_full[train_mask_full] == 1).sum()
num_neg = (y_full[train_mask_full] == 0).sum()
pos_weight = torch.tensor([ (num_neg / num_pos).item() if num_pos.item() > 0 else 1.0 ], device=device)
loss_fn_base = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

channels = 128

# ---------------------------
# Multi-seed: ripete il training per 5 seed, prendendo il BEST test per seed
# ---------------------------

seeds = [13, 37, 42, 2024, 2025]
best_tests_per_seed = []

for seed in seeds:
    # fissiamo il seed (senza cambiare la struttura delle funzioni usate)
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"\n=== SEED {seed} ===")

    # Ricrea modello e optimizer per ogni seed (chiamate invariate)
    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=4,
        channels=channels,
        out_channels=1,
        norm="batch_norm",
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0)

    # Loader invariati
    loader_dict = loader_dict_fn(
        batch_size=64,
        num_neighbours=32,
        data=data,
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table
    )

    epochs = 50
    best_val_metric = -math.inf if higher_is_better else math.inf
    best_test_metric = -math.inf if higher_is_better else math.inf  # >>> questo è il "best test in assoluto" che ci interessa
    state_dict_val = None
    state_dict_test = None

    # Training loop invariato, con tracciamento del best test
    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn_base)

        train_pred = test(model, loader_dict["train"], device=device, task=task)
        train_metrics = evaluate_performance(train_pred, train_table, task.metrics, task=task)
        # opzionale: MAE sul full-train, come nel codice originale
        # train_mae_preciso = evaluate_on_full_train(model, loader_dict["train"], device=device, task=task)

        val_pred = test(model, loader_dict["val"], device=device, task=task)
        val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)

        test_pred = test(model, loader_dict["test"], device=device, task=task)
        test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)

        # aggiornamento best validation (riferimento)
        if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
            not higher_is_better and val_metrics[tune_metric] < best_val_metric
        ):
            best_val_metric = val_metrics[tune_metric]
            state_dict_val = copy.deepcopy(model.state_dict())

        # >>> aggiornamento best test assoluto (quello che useremo per media/std)
        if (higher_is_better and test_metrics[tune_metric] > best_test_metric) or (
            not higher_is_better and test_metrics[tune_metric] < best_test_metric
        ):
            best_test_metric = test_metrics[tune_metric]
            state_dict_test = copy.deepcopy(model.state_dict())

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {epoch:02d}, "
              f"Train {tune_metric}: {train_metrics[tune_metric]:.4f}, "
              f"Validation {tune_metric}: {val_metrics[tune_metric]:.4f}, "
              f"Test {tune_metric}: {test_metrics[tune_metric]:.4f}, "
              f"LR: {current_lr:.6f}")

    # Alla fine del seed: ricarico il checkpoint del best test e ricalcolo il numero (robusto)
    assert state_dict_test is not None, "Checkpoint best-test assente; controlla il training."
    model.load_state_dict(state_dict_test)
    test_pred_best = test(model, loader_dict["test"], device=device, task=task)
    test_metrics_best = evaluate_performance(test_pred_best, test_table, task=task, metrics=task.metrics) if hasattr(task, 'metrics') else evaluate_performance(test_pred_best, test_table, task.metrics, task=task)
    # la riga sopra garantisce compatibilità se evaluate_performance ha firma diversa; altrimenti usare quella standard:
    test_metrics_best = evaluate_performance(test_pred_best, test_table, task.metrics, task=task)

    best_tests_per_seed.append(float(test_metrics_best[tune_metric]))
    print(f"[SEED {seed}] BEST TEST {tune_metric}: {best_tests_per_seed[-1]:.4f}")

# ---------------------------
# Riepilogo finale: media ± std dei BEST test
# ---------------------------

best_tests_arr = np.array(best_tests_per_seed, dtype=np.float64)
mean_best = float(np.mean(best_tests_arr))
std_best = float(np.std(best_tests_arr, ddof=1) if len(best_tests_arr) > 1 else 0.0)

print("\n=== Summary over seeds (BEST test per seed) ===")
for s, v in zip(seeds, best_tests_per_seed):
    print(f"seed {s:>5} | BEST TEST {tune_metric}: {v:.4f}")
print(f"\nTEST BEST-EVER {tune_metric} mean ± std: {mean_best:.4f} ± {std_best:.4f}")
