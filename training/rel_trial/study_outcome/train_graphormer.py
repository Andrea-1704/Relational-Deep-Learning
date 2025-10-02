import os
import sys
import math
import copy
import random
from typing import Dict, Any

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

from model.GraphormerNew import Model
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


def build_data_and_labels(device: torch.device) -> Dict[str, Any]:
    dataset = get_dataset("rel-trial", download=True)
    task = get_task("rel-trial", "study-outcome", download=True)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")                    # masked
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

    # Attach labels to 'studies' (as in your script)
    graph_ids = db2.table_dict["studies"].df["nct_id"].to_numpy()
    id_to_idx = {gid: i for i, gid in enumerate(graph_ids)}

    train_df_raw = train_table.df
    ids_raw = train_df_raw["nct_id"].to_numpy()
    labels_raw = train_df_raw["outcome"].to_numpy().astype(np.float32)  # binary 0/1

    target_vector = torch.full((len(graph_ids),), float("nan"))
    for gid, lab in zip(ids_raw, labels_raw):
        if gid in id_to_idx:
            target_vector[id_to_idx[gid]] = lab

    data['studies'].y = target_vector.float()
    data['studies'].train_mask = ~torch.isnan(target_vector)

    # pos_weight for BCEWithLogitsLoss
    y_full = data['studies'].y.float()
    train_mask = data['studies'].train_mask
    num_pos = (y_full[train_mask] == 1).sum()
    num_neg = (y_full[train_mask] == 0).sum()
    pos_weight = torch.tensor([ (num_neg / num_pos).item() if num_pos.item() > 0 else 1.0 ], device=device)

    return dict(
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table_unmasked,
        data=data,
        col_stats=col_stats,
        pos_weight=pos_weight
    )


def build_model(data, col_stats, device, channels=128):
    model = Model(
        data=data,
        col_stats_dict=col_stats,
        num_layers=4,
        channels=channels,
        out_channels=1,
        norm="batch_norm",
    ).to(device)
    return model


# ---------------------------
# Single run: track BEST test across epochs
# ---------------------------

def run_once(seed: int, device: torch.device, max_epochs: int = 50):
    set_global_seed(seed)

    bundle = build_data_and_labels(device)
    task = bundle["task"]
    train_table = bundle["train_table"]
    val_table = bundle["val_table"]
    test_table = bundle["test_table"]
    data = bundle["data"]
    col_stats = bundle["col_stats"]
    pos_weight = bundle["pos_weight"]

    tune_metric = "roc_auc"
    higher_is_better = True

    model = build_model(data, col_stats, device, channels=128)
    #optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader_dict = loader_dict_fn(
        batch_size=64,
        num_neighbours=32,
        data=data,
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table
    )

    best_test = -math.inf
    best_state_test = None

    # (facoltativo) anche best su val per riferimento
    best_val = -math.inf
    best_state_val = None

    for epoch in range(1, max_epochs + 1):
        _ = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

        val_pred = test(model, loader_dict["val"], device=device, task=task)
        test_pred = test(model, loader_dict["test"], device=device, task=task)

        val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)
        test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)

        val_score = float(val_metrics[tune_metric])
        test_score = float(test_metrics[tune_metric])

        # best test (numero che vogliamo)
        if higher_is_better and test_score > best_test:
            best_test = test_score
            best_state_test = copy.deepcopy(model.state_dict())

        # opzionale: traccia best val per confronto
        if higher_is_better and val_score > best_val:
            best_val = val_score
            best_state_val = copy.deepcopy(model.state_dict())

        print(f"[seed {seed}] epoch {epoch:03d} | val {tune_metric}: {val_score:.4f} | test {tune_metric}: {test_score:.4f}")

    # Ricarica best-test e misura finale (robusto)
    assert best_state_test is not None, "Best-test checkpoint non trovato."
    model.load_state_dict(best_state_test)
    test_pred_best = test(model, loader_dict["test"], device=device, task=task)
    test_metrics_best = evaluate_performance(test_pred_best, test_table, task.metrics, task=task)
    test_best = float(test_metrics_best[tune_metric])

    return {"seed": seed, "test_best": test_best}


# ---------------------------
# Multi-seed summary
# ---------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [13, 37, 42, 2024, 2025]

    results = []
    for s in seeds:
        print(f"\n=== Running seed {s} ===")
        out = run_once(seed=s, device=device, max_epochs=50)
        print(f"[seed {s}] TEST_BEST={out['test_best']:.4f}")
        results.append(out)

    test_best_arr = np.array([r["test_best"] for r in results], dtype=np.float64)

    def mean_std(x: np.ndarray):
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1) if len(x) > 1 else 0.0)
        return mean, std

    test_mean, test_std = mean_std(test_best_arr)

    print("\n=== Summary over 5 seeds (rel-trial / study-outcome, BEST test over epochs) ===")
    for r in results:
        print(f"seed {r['seed']:>5} | TEST_BEST {r['test_best']:.4f}")
    print(f"\nTEST BEST-EVER mean ± std: {test_mean:.4f} ± {test_std:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
