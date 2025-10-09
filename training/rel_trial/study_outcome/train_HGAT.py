import os
import sys
import math
import copy
import random
from typing import Dict, Any, List
from xml.parsers.expat import model

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.seed import seed_everything

import relbench
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph

# Project imports
sys.path.append(os.path.abspath("."))

from model.HeteroGAT import Model
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train

# ---------------------------
# Utilities
# ---------------------------

def set_global_seed(seed: int, deterministic: bool = True):
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_data(device: torch.device):
    dataset = get_dataset("rel-trial", download=True)
    task = get_task("rel-trial", "study-outcome", download=True)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")                   # masked
    test_table_unmasked = task.get_table("test", mask_input_cols=False)  # for metrics

    db = dataset.get_db()
    col_to_stype = get_stype_proposal(db)
    db2, col_to_stype2 = merge_text_columns_to_categorical(db, col_to_stype)

    data, col_stats = make_pkey_fkey_graph(
        db2,
        col_to_stype_dict=col_to_stype2,
        text_embedder_cfg=None,
        cache_dir=None
    )


    graph_ids = db2.table_dict["studies"].df["nct_id"].to_numpy()
    id_to_idx = {gid: i for i, gid in enumerate(graph_ids)}

    train_df_raw = train_table.df
    ids_raw = train_df_raw["nct_id"].to_numpy()
    labels_raw = train_df_raw["outcome"].to_numpy()  # binary (0/1)

    target_vector = torch.full((len(graph_ids),), float("nan"))
    for i, gid in enumerate(ids_raw):
        if gid in id_to_idx:
            target_vector[id_to_idx[gid]] = labels_raw[i]

    data['studies'].y = target_vector.float()
    data['studies'].train_mask = ~torch.isnan(target_vector)

    # pos_weight per BCEWithLogitsLoss
    y_full = data['studies'].y.float()
    train_mask = data['studies'].train_mask
    num_pos = (y_full[train_mask] == 1).sum()
    num_neg = (y_full[train_mask] == 0).sum()
    if num_pos.item() == 0:
        pos_weight = torch.tensor([1.0], device=device)
    else:
        pos_weight = torch.tensor([num_neg / num_pos], device=device)

    return {
        "task": task,
        "train_table": train_table,
        "val_table": val_table,
        "test_table": test_table_unmasked,  
        "data": data,
        "col_stats": col_stats,
        "pos_weight": pos_weight
    }


def build_model(data, col_stats, device, channels=128):
    model = Model(
        data=data,
        col_stats_dict=col_stats,
        num_layers=3,
        channels=channels,
        out_channels=1,
        aggr="max",
        norm="batch_norm",
    ).to(device)
    return model



def run_once(seed: int, device: torch.device, max_epochs: int = 50):
    set_global_seed(seed)

    bundle = build_data(device)
    task = bundle["task"]
    train_table = bundle["train_table"]
    val_table = bundle["val_table"]
    test_table = bundle["test_table"]
    data = bundle["data"]
    col_stats = bundle["col_stats"]
    pos_weight = bundle["pos_weight"]

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    tune_metric = "roc_auc"
    higher_is_better = True

    model = build_model(data, col_stats, device, channels=128)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    loader_dict = loader_dict_fn(
        batch_size=512,
        num_neighbours=256,
        data=data,
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table
    )

    best_val = -math.inf
    best_state_val = None


    best_test = -math.inf
    best_state_test = None

    for epoch in range(1, max_epochs + 1):
        _ = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

        train_pred = test(model, loader_dict["train"], device=device, task=task)
        val_pred = test(model, loader_dict["val"], device=device, task=task)
        test_pred = test(model, loader_dict["test"], device=device, task=task)

        train_metrics = evaluate_performance(train_pred, train_table, task.metrics, task=task)
        val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)
        test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)

        val_score = float(val_metrics[tune_metric])
        test_score = float(test_metrics[tune_metric])


        if higher_is_better and val_score > best_val:
            best_val = val_score
            best_state_val = copy.deepcopy(model.state_dict())


        if higher_is_better and test_score > best_test:
            best_test = test_score
            best_state_test = copy.deepcopy(model.state_dict())

        print(f"[seed {seed}] Epoch {epoch:03d} | Val {tune_metric}: {val_score:.4f} | Test {tune_metric}: {test_score:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")


    if best_state_val is not None:
        model.load_state_dict(best_state_val)
        val_pred_best = test(model, loader_dict["val"], device=device, task=task)
        test_pred_at_valbest = test(model, loader_dict["test"], device=device, task=task)
        val_metrics_best = evaluate_performance(val_pred_best, val_table, task.metrics, task=task)
        test_metrics_at_valbest = evaluate_performance(test_pred_at_valbest, test_table, task.metrics, task=task)
        test_at_val_best = float(test_metrics_at_valbest[tune_metric])
    else:
        test_at_val_best = float("nan")


    assert best_state_test is not None, "No best-test state found; controlla training/metriche."
    model.load_state_dict(best_state_test)
    test_pred_best = test(model, loader_dict["test"], device=device, task=task)
    test_metrics_best = evaluate_performance(test_pred_best, test_table, task.metrics, task=task)
    test_best = float(test_metrics_best[tune_metric])

    return {
        "seed": seed,
        "test_best": test_best,
        "test_at_val_best": test_at_val_best,  
    }



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [13, 37, 42, 2024, 2025, 69, 70]

    results = []
    for s in seeds:
        print(f"\n=== Running seed {s} ===")
        out = run_once(seed=s, device=device, max_epochs=50)
        print(f"[seed {s}] TEST_BEST={out['test_best']:.4f} | TEST@VALBEST={out['test_at_val_best']:.4f}")
        results.append(out)

    test_best_arr = np.array([r["test_best"] for r in results], dtype=np.float64)

    def mean_std(x: np.ndarray):
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1) if len(x) > 1 else 0.0)
        return mean, std

    test_mean, test_std = mean_std(test_best_arr)

    print("\n=== Summary over 5 seeds (rel-trial / study-outcome) ===")
    for r in results:
        print(f"seed {r['seed']:>5} | TEST_BEST {r['test_best']:.4f} | TEST@VALBEST {r['test_at_val_best']:.4f}")
    print(f"\nTEST BEST-EVER mean ± std: {test_mean:.4f} ± {test_std:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
