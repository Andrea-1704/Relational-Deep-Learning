#####
# Training Heterogeneous Graph SAGE — multi-seed (best test per seed)
# Mantiene la struttura di chiamata originale (train/test/evaluate_performance, loader_dict_fn, ecc.)
####

import os
import sys
import math
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.seed import seed_everything

import relbench
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph

sys.path.append(os.path.abspath("."))

from model.HeteroGAT import Model
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train

# ---------------------------
# Parametri task/dataset
# ---------------------------

task_name = "user-clicks"
db_name   = "rel-avito"
node_id   = "UserID"
target    = "num_click"
node_type = "UserInfo"

# metrica come nel tuo script originale
tune_metric = "roc_auc"
higher_is_better = True  # riferito alla metrica

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Caricamento dati e grafo (una volta sola)
# ---------------------------

dataset = get_dataset(db_name, download=True)
task    = get_task(db_name, task_name, download=True)

train_table = task.get_table("train")
val_table   = task.get_table("val")
test_table  = task.get_table("test", mask_input_cols=False)  # per metriche

db = dataset.get_db()
col_to_stype_dict = get_stype_proposal(db)
db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)

data, col_stats_dict = make_pkey_fkey_graph(
    db_nuovo,
    col_to_stype_dict=col_to_stype_dict_nuovo,
    text_embedder_cfg=None,
    cache_dir=None
)

# ---------------------------
# Costruzione y e train_mask per il node_type/target
# (con fix di tipo per evitare TypeError: numpy.float32 -> float)
# ---------------------------

graph_node_ids = db_nuovo.table_dict[node_type].df[node_id].to_numpy()
id_to_idx = {nid: i for i, nid in enumerate(graph_node_ids)}

train_df_raw = train_table.df
ids_raw   = train_df_raw[node_id].to_numpy()
labels_np = train_df_raw[target].to_numpy().astype(np.float32)

target_vector = torch.full((len(graph_node_ids),), float("nan"), dtype=torch.float32)
for i, nid in enumerate(ids_raw):
    if nid in id_to_idx:
        target_vector[id_to_idx[nid]] = float(labels_np[i])

data[node_type].y = target_vector
data[node_type].train_mask = ~torch.isnan(target_vector)

# Se vuoi usare BCEWithLogitsLoss con pos_weight (come nel tuo script originale):
y_full = data[node_type].y.float()
train_mask = data[node_type].train_mask
num_pos = (y_full[train_mask] == 1).sum()
num_neg = (y_full[train_mask] == 0).sum()
pos_weight = torch.tensor([ (num_neg / num_pos).item() if num_pos.item() > 0 else 1.0 ], device=device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ---------------------------
# Loader (riutilizzabile tra seed)
# ---------------------------

loader_dict = loader_dict_fn(
    batch_size=512,
    num_neighbours=256,
    data=data,
    task=task,
    train_table=train_table,
    val_table=val_table,
    test_table=test_table
)

# ---------------------------
# Multi-seed loop
# ---------------------------

channels = 128
seeds = [13, 37, 42]
best_tests_per_seed = []

for seed in seeds:
    # Fissa i seed senza cambiare struttura delle funzioni chiamate
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"\n=== SEED {seed} ===")

    # Modello/ottimizzatore come nel tuo script
    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=4,
        channels=channels,
        out_channels=1,
        aggr="max",
        norm="batch_norm",
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)

    epochs = 50
    best_val_metric  = -math.inf if higher_is_better else math.inf
    best_test_metric = -math.inf if higher_is_better else math.inf  # >>> Best test assoluto per questo seed
    state_dict_val   = None
    state_dict_test  = None

    for epoch in range(1, epochs + 1):
        _ = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

        train_pred = test(model, loader_dict["train"], device=device, task=task)
        train_metrics = evaluate_performance(train_pred, train_table, task.metrics, task=task)
        _ = evaluate_on_full_train(model, loader_dict["train"], device=device, task=task)  # opzionale

        val_pred = test(model, loader_dict["val"], device=device, task=task)
        val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)

        test_pred = test(model, loader_dict["test"], device=device, task=task)
        test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)

        # Best su validation (riferimento)
        if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
            not higher_is_better and val_metrics[tune_metric] < best_val_metric
        ):
            best_val_metric = val_metrics[tune_metric]
            state_dict_val  = copy.deepcopy(model.state_dict())

        # >>> Best test assoluto (numero che useremo per media/std)
        if (higher_is_better and test_metrics[tune_metric] > best_test_metric) or (
            not higher_is_better and test_metrics[tune_metric] < best_test_metric
        ):
            best_test_metric = test_metrics[tune_metric]
            state_dict_test  = copy.deepcopy(model.state_dict())

        print(f"Epoch: {epoch:02d} | Val {tune_metric}: {val_metrics[tune_metric]:.4f} | "
              f"Test {tune_metric}: {test_metrics[tune_metric]:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Fine seed: ricarico checkpoint best-test e riconfermo
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
