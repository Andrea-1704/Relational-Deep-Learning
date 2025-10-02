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

from model.HeteroGAT import Model
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
# from pre_training.VGAE.Utils_VGAE import train_vgae  # opzionale

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


def build_graph_and_task(device: torch.device) -> Dict[str, Any]:
    dataset = get_dataset("rel-trial", download=True)
    task = get_task("rel-trial", "study-adverse", download=True)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")                      # masked
    test_table_unmasked = task.get_table("test", mask_input_cols=False)  # unmasked for metrics

    db = dataset.get_db()
    col_to_stype = get_stype_proposal(db)
    db2, col_to_stype2 = merge_text_columns_to_categorical(db, col_to_stype)

    data, col_stats = make_pkey_fkey_graph(
        db2,
        col_to_stype_dict=col_to_stype2,
        text_embedder_cfg=None,
        cache_dir=None
    )

    return dict(
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table_unmasked,  # use unmasked for evaluation
        data=data,
        col_stats=col_stats
    )


def build_model(data, col_stats, device, channels=128):
    model = Model(
        data=data,
        col_stats_dict=col_stats,
        num_layers=4,
        channels=channels,
        out_channels=1,
        aggr="max",
        norm="batch_norm",
    ).to(device)
    return model

# ---------------------------
# Single run (one seed)
# ---------------------------

def run_once(seed: int, device: torch.device, max_epochs: int = 50):
    set_global_seed(seed)

    bundle = build_graph_and_task(device)
    task = bundle["task"]
    train_table = bundle["train_table"]
    val_table = bundle["val_table"]
    test_table = bundle["test_table"]  # unmasked
    data = bundle["data"]
    col_stats = bundle["col_stats"]

    loss_fn = nn.L1Loss()  # MAE (lower is better)
    tune_metric = "mae"
    higher_is_better = False

    model = build_model(data, col_stats, device, channels=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0)

    loader_dict = loader_dict_fn(
        batch_size=512,
        num_neighbours=128,
        data=data,
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table
    )

    best_val = math.inf if not higher_is_better else -math.inf
    best_state_val = None

    # >>> traccia anche il miglior test assoluto <<<
    best_test = math.inf  # per MAE, più basso è meglio
    best_state_test = None

    for epoch in range(1, max_epochs + 1):
        _ = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

        train_pred = test(model, loader_dict["train"], device=device, task=task)
        val_pred = test(model, loader_dict["val"], device=device, task=task)
        test_pred = test(model, loader_dict["test"], device=device, task=task)

        train_metrics = evaluate_performance(train_pred, train_table, task.metrics, task=task)
        val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)
        test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)

        # MAE sul train "full" se vuoi loggarlo (come nel tuo stile)
        # train_mae_full = evaluate_on_full_train(model, loader_dict["train"], device=device, task=task)

        val_score = val_metrics[tune_metric]
        test_score = test_metrics[tune_metric]

        # best su validation (standard)
        if (not higher_is_better and val_score < best_val) or (higher_is_better and val_score > best_val):
            best_val = val_score
            best_state_val = copy.deepcopy(model.state_dict())

        # >>> best assoluto su test (richiesta) <<<
        if test_score < best_test:  # lower is better
            best_test = test_score
            best_state_test = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch:03d} | Val MAE: {val_score:.4f} | Test MAE: {test_score:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # ---- valutazioni finali ----
    # (A) Test @ best validation (riferimento)
    assert best_state_val is not None, "No best validation state found"
    model.load_state_dict(best_state_val)
    val_pred_best = test(model, loader_dict["val"], device=device, task=task)
    test_pred_at_valbest = test(model, loader_dict["test"], device=device, task=task)
    val_metrics_best = evaluate_performance(val_pred_best, val_table, task.metrics, task=task)
    test_metrics_at_valbest = evaluate_performance(test_pred_at_valbest, test_table, task.metrics, task=task)
    test_at_val_best_mae = float(test_metrics_at_valbest["mae"])

    # (B) Best test assoluto lungo il training (quello che vuoi)
    assert best_state_test is not None, "No best test state found"
    model.load_state_dict(best_state_test)
    test_pred_best = test(model, loader_dict["test"], device=device, task=task)
    test_metrics_best = evaluate_performance(test_pred_best, test_table, task.metrics, task=task)
    test_best_mae = float(test_metrics_best["mae"])

    return {
        "seed": seed,
        "val_best_mae": float(val_metrics_best["mae"]),
        "test_at_val_best_mae": test_at_val_best_mae,
        "test_best_mae": test_best_mae
    }

# ---------------------------
# Main: 5 seeds summary
# ---------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [13, 37, 42, 2024, 2025]

    results = []
    for s in seeds:
        print(f"\n=== Running seed {s} ===")
        out = run_once(seed=s, device=device, max_epochs=50)
        print(f"[seed {s}] val_best_mae={out['val_best_mae']:.4f} | "
              f"test_at_val_best_mae={out['test_at_val_best_mae']:.4f} | "
              f"test_best_mae={out['test_best_mae']:.4f}")
        results.append(out)

    # Media/std sui BEST test (quello che vuoi riportare)
    test_best_arr = np.array([r["test_best_mae"] for r in results], dtype=np.float64)

    def mean_std(x):
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1) if len(x) > 1 else 0.0)
        return mean, std

    test_best_mean, test_best_std = mean_std(test_best_arr)

    print("\n=== Summary over 5 seeds (rel-trial / study-adverse, MAE lower is better) ===")
    for r in results:
        print(f"seed {r['seed']:>5} | VAL_best {r['val_best_mae']:.4f} | "
              f"TEST@VALbest {r['test_at_val_best_mae']:.4f} | TEST_best {r['test_best_mae']:.4f}")

    print(f"\nTEST BEST-EVER MAE mean ± std: {test_best_mean:.4f} ± {test_best_std:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
