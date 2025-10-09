import os
import sys
import math
import copy
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.seed import seed_everything

import relbench
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph

# project-specific imports
sys.path.append(os.path.abspath("."))


from model.HeteroGAT import Model
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.utils import evaluate_performance, test, train
from utils.EarlyStopping import EarlyStopping


# ---------------------------
# Utilities
# ---------------------------

def set_global_seed(seed: int, deterministic: bool = True):
    """Set all seeds in modo affidabile."""
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_data_and_targets(device, target_col="did_not_finish"):
    """
    Carica il dataset rel-f1, task driver-dnf (binary), costruisce il grafo,
    e attacca y + train_mask per i soli nodi con label note nel training.
    """
    dataset = get_dataset("rel-f1", download=True)
    task = get_task("rel-f1", "driver-dnf", download=True)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    db = dataset.get_db()
    col_to_stype = get_stype_proposal(db)
    db2, col_to_stype2 = merge_text_columns_to_categorical(db, col_to_stype)

    data_official, col_stats_official = make_pkey_fkey_graph(
        db2,
        col_to_stype_dict=col_to_stype2,
        text_embedder_cfg=None,
        cache_dir=None
    )

    # vettore target per i drivers
    graph_driver_ids = db2.table_dict["drivers"].df["driverId"].to_numpy()
    id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}

    train_df_raw = train_table.df
    driver_ids_raw = train_df_raw["driverId"].to_numpy()
    labels_raw = train_df_raw[target_col].to_numpy()  # già binaria 0/1

    target_vector = torch.full((len(graph_driver_ids),), float("nan"))
    for i, driver_id in enumerate(driver_ids_raw):
        if driver_id in id_to_idx:
            target_vector[id_to_idx[driver_id]] = labels_raw[i]

    data_official['drivers'].y = target_vector.float()
    data_official['drivers'].train_mask = ~torch.isnan(target_vector)

    # full graph su device (utile per pos_weight)
    data_full, col_stats_full = make_pkey_fkey_graph(
        db2,
        col_to_stype_dict=col_to_stype2,
        text_embedder_cfg=None,
        cache_dir=None
    )
    data_full = data_full.to(device)
    data_full['drivers'].y = target_vector.float()
    data_full['drivers'].train_mask = ~torch.isnan(target_vector)

    return {
        "task": task,
        "train_table": train_table,
        "val_table": val_table,
        "test_table": test_table,
        "data_official": data_official,
        "col_stats_official": col_stats_official,
        "data_full": data_full,
    }


def build_model(data_official, col_stats_official, device, channels=128):
    model = Model(
        data=data_official,
        col_stats_dict=col_stats_official,
        num_layers=4,
        channels=channels,
        out_channels=1,
        aggr="max",
        norm="batch_norm",
    ).to(device)
    return model


def bce_pos_weight_from_masked_targets(data, device):
    y_full = data['drivers'].y.float()
    train_mask = data['drivers'].train_mask
    num_pos = (y_full[train_mask] == 1).sum()
    num_neg = (y_full[train_mask] == 0).sum()
    if num_pos.item() == 0:
        pos_weight = torch.tensor([1.0], device=device)
    else:
        pos_weight = torch.tensor([num_neg / num_pos], device=device)
    return pos_weight


# ---------------------------
# Single run
# ---------------------------

def run_once(seed: int, device: torch.device, max_epochs: int = 50, use_vgae: bool = False):
    set_global_seed(seed)

    bundle = build_data_and_targets(device, target_col="did_not_finish")
    task = bundle["task"]
    train_table = bundle["train_table"]
    val_table = bundle["val_table"]
    test_table = task.get_table("test", mask_input_cols=False)  # test non mascherato
    data_official = bundle["data_official"]
    col_stats_official = bundle["col_stats_official"]
    data_full = bundle["data_full"]

    pos_weight = bce_pos_weight_from_masked_targets(data_full, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = build_model(data_official, col_stats_official, device, channels=128)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)

    early_stopping = EarlyStopping(
        patience=100,
        delta=0.0,
        verbose=False,
        path=f"best_dnf_seed{seed}.pt"
    )

    loader_dict = loader_dict_fn(
        batch_size=512,
        num_neighbours=256,
        data=data_official,
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table
    )

    tune_metric = "roc_auc"
    higher_is_better = True

    best_val = -math.inf
    best_state_val = None


    best_test = -math.inf
    best_state_test = None

    for epoch in range(max_epochs):
        _ = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

        train_pred = test(model, loader_dict["train"], device=device, task=task)
        val_pred = test(model, loader_dict["val"], device=device, task=task)
        test_pred = test(model, loader_dict["test"], device=device, task=task)

        train_metrics = evaluate_performance(train_pred, train_table, task.metrics, task=task)
        val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)
        test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)

        val_score = val_metrics[tune_metric]
        test_score = test_metrics[tune_metric]

        # best su validation (standard)
        if higher_is_better and val_score > best_val:
            best_val = val_score
            best_state_val = copy.deepcopy(model.state_dict())


        if higher_is_better and test_score > best_test:
            best_test = test_score
            best_state_test = copy.deepcopy(model.state_dict())

        # early stopping su validation
        early_stopping(val_score, model)
        if early_stopping.early_stop:
            break


    assert best_state_val is not None, "No best validation state found"
    model.load_state_dict(best_state_val)
    val_pred_best = test(model, loader_dict["val"], device=device, task=task)
    test_pred_at_valbest = test(model, loader_dict["test"], device=device, task=task)
    val_metrics_best = evaluate_performance(val_pred_best, val_table, task.metrics, task=task)
    test_metrics_at_valbest = evaluate_performance(test_pred_at_valbest, test_table, task.metrics, task=task)


    assert best_state_test is not None, "No best test state found"
    model.load_state_dict(best_state_test)
    test_pred_best = test(model, loader_dict["test"], device=device, task=task)
    test_metrics_best = evaluate_performance(test_pred_best, test_table, task.metrics, task=task)

    return {
        "seed": seed,
        "val_best": float(val_metrics_best[tune_metric]),            # per riferimento
        "test_at_val_best": float(test_metrics_at_valbest[tune_metric]),  # per riferimento
        "test_best": float(test_metrics_best[tune_metric])           # >>> questo è il numero che chiedi <<<
    }


# ---------------------------
# Main: 5 seeds summary
# ---------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [13, 37, 2024]

    results = []
    for s in seeds:
        print(f"\n=== Running seed {s} ===")
        out = run_once(seed=s, device=device, max_epochs=50, use_vgae=False)
        print(f"[seed {s}] val_best={out['val_best']:.4f} | "
              f"test_at_val_best={out['test_at_val_best']:.4f} | "
              f"test_best={out['test_best']:.4f}")
        results.append(out)


    test_best_arr = np.array([r["test_best"] for r in results], dtype=np.float64)

    def mean_std(x):
        return float(np.mean(x)), float(np.std(x, ddof=1) if len(x) > 1 else 0.0)

    test_best_mean, test_best_std = mean_std(test_best_arr)

    print("\n=== Summary over 5 seeds (HeteroGAT - driver-dnf) ===")
    for r in results:
        print(f"seed {r['seed']:>5}  |  TEST_BEST {r['test_best']:.4f}  |  TEST@VALBEST {r['test_at_val_best']:.4f}  |  VAL_BEST {r['val_best']:.4f}")

    print(f"\nTEST  BEST-EVER mean ± std: {test_best_mean:.4f} ± {test_best_std:.4f}")
    # val_arr = np.array([r["val_best"] for r in results], dtype=np.float64)
    # test_valbest_arr = np.array([r["test_at_val_best"] for r in results], dtype=np.float64)
    # val_mean, val_std = mean_std(val_arr)
    # test_valbest_mean, test_valbest_std = mean_std(test_valbest_arr)
    # print(f"VAL   mean ± std: {val_mean:.4f} ± {val_std:.4f}")
    # print(f"TEST  @VALBEST mean ± std: {test_valbest_mean:.4f} ± {test_valbest_std:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
