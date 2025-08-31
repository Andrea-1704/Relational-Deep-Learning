# tune_XMetaPath.py
import os
import sys
import json
import math
import copy
import random
from itertools import product
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.seed import seed_everything

# Assicura import locali come nel tuo benchmark
sys.path.append(os.path.abspath("."))

# ---- Import dal tuo progetto ----
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph

from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.EarlyStopping import EarlyStopping
from utils.utils import evaluate_performance, test, train

from model.XMetaPath2 import XMetaPath2


# ---------------------------- Utils varie ----------------------------
def flip_rel(rel_name: str) -> str:
    return rel_name[4:] if rel_name.startswith("rev_") else f"rev_{rel_name}"

def to_canonical(mp_outward, target="drivers"):
    mp = [(dst, flip_rel(rel), src) for (src, rel, dst) in mp_outward[::-1]]
    assert mp[-1][2] == target
    return tuple(mp)

def canonicalize_metapaths(metapaths_raw, target="drivers"):
    out = []
    for mp in metapaths_raw:
        mp = mp.copy()
        out.append(to_canonical(mp, target))
    return out


# ---------------------------- Config tuning ----------------------------
# Imposta qui semi-random e sampling dal grid
GLOBAL_SEED = 5
MAX_TRIALS = None   # None = prova tutte le combinazioni; oppure un intero (es. 40)
N_EPOCHS = 300
EARLY_STOP_PATIENCE = 60
TUNE_METRIC = "f1"
HIGHER_IS_BETTER = True
TARGET_NODE = "drivers"
CHECKPOINT_DIR = "checkpoints"
LOG_PATH = "tuning_log.csv"
TRIED_PATH = "tuning_tried.json"

# Griglia degli iperparametri (scegli valori sensati ma non enormi)
PARAM_GRID = {
    # Ottimizzatore e relativi iperparametri
    "optimizer": ["AdamW", "Adam", "SGD"],
    "lr": [1e-3, 3e-3, 1e-2, 3e-2],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "momentum": [0.8, 0.9],                       # usato solo se SGD
    "betas": [(0.9, 0.999), (0.9, 0.99)],         # usato solo se Adam/AdamW

    # Modello
    "hidden_channels": [64, 128, 256],
    "out_channels": [64, 128, 256],               # spesso = hidden_channels, ma lasciamo liberi
    "dropout_p": [0.0, 0.1, 0.2, 0.3],
    "num_heads": [4, 8],
    "num_layers": [2, 4],

    # Loader
    "batch_size": [256, 512, 1024],
    "num_neighbours": [64, 128, 256],

    # Scheduler
    "scheduler": ["none", "cosine", "plateau"],
    "cosine_Tmax": [10, 25],                      # usato solo se cosine
    "plateau_patience": [10, 20],                 # usato solo se plateau
    "plateau_factor": [0.5, 0.3],                 # usato solo se plateau
}


# ---------------------------- Dataset pipeline (come nel tuo benchmark) ----------------------------
def prepare_data_and_labels(device):
    dataset = get_dataset("rel-f1", download=True)
    task = get_task("rel-f1", "driver-top3", download=True)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)
    db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)

    data_official, col_stats_dict_official = make_pkey_fkey_graph(
        db_nuovo,
        col_to_stype_dict=col_to_stype_dict_nuovo,
        text_embedder_cfg=None,
        cache_dir=None
    )

    graph_driver_ids = db_nuovo.table_dict["drivers"].df["driverId"].to_numpy()
    id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}

    # Label train (già binarie nel tuo benchmark)
    train_df_raw = train_table.df
    driver_ids_raw = train_df_raw["driverId"].to_numpy()
    qualifying_positions = train_df_raw["qualifying"].to_numpy()

    target_vector_official = torch.full((len(graph_driver_ids),), float("nan"))
    for i, driver_id in enumerate(driver_ids_raw):
        if driver_id in id_to_idx:
            target_vector_official[id_to_idx[driver_id]] = qualifying_positions[i]

    data_official['drivers'].y = target_vector_official.float()
    data_official['drivers'].train_mask = ~torch.isnan(target_vector_official)

    # Mask validation
    val_df_raw = val_table.df
    val_driver_ids = val_df_raw["driverId"].to_numpy()
    val_mask = torch.tensor([driver_id in val_driver_ids for driver_id in graph_driver_ids])
    data_official["drivers"].val_mask = val_mask

    # Pos weight (come nel tuo benchmark)
    y_full = data_official['drivers'].y.float()
    train_mask_full = data_official['drivers'].train_mask
    num_pos = (y_full[train_mask_full] == 1).sum()
    num_neg = (y_full[train_mask_full] == 0).sum()
    ratio = (num_neg / num_pos) if num_pos > 0 else 1.0
    pos_weight = torch.tensor([ratio], device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return (data_official, col_stats_dict_official,
            task, train_table, val_table, test_table, loss_fn)


def build_metapaths_canonical():
    # stessi meta-path del tuo benchmark
    metapaths_raw = [
        [('drivers', 'rev_f2p_driverId', 'standings'),
         ('standings', 'f2p_raceId', 'races')],
        [('drivers', 'rev_f2p_driverId', 'qualifying'),
         ('qualifying', 'f2p_constructorId', 'constructors'),
         ('constructors', 'rev_f2p_constructorId', 'constructor_results'),
         ('constructor_results', 'f2p_raceId', 'races')],
        [('drivers', 'rev_f2p_driverId', 'results'),
         ('results', 'f2p_constructorId', 'constructors'),
         ('constructors', 'rev_f2p_constructorId', 'constructor_standings'),
         ('constructor_standings', 'f2p_raceId', 'races')],
    ]
    return canonicalize_metapaths(metapaths_raw, target=TARGET_NODE)


def make_loader_dict(data_official, task, train_table, val_table, test_table,
                     batch_size, num_neighbours):
    loader_dict = loader_dict_fn(
        batch_size=batch_size,
        num_neighbours=num_neighbours,
        data=data_official,
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table
    )
    return loader_dict


# ---------------------------- Build model/optimizer/scheduler ----------------------------
def build_model(data_official, col_stats_dict, params, device):
    model = XMetaPath2(
        data=data_official,
        col_stats_dict=col_stats_dict,
        metapaths=build_metapaths_canonical(),
        hidden_channels=params["hidden_channels"],
        out_channels=params["out_channels"],
        final_out_channels=1,
        num_heads=params["num_heads"],
        num_layers=params["num_layers"],
        dropout_p=params["dropout_p"],
    ).to(device)
    return model


def build_optimizer(model, params):
    opt_name = params["optimizer"].lower()
    lr = params["lr"]
    wd = params["weight_decay"]

    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                               momentum=params["momentum"], nesterov=True)
    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd,
                                betas=params["betas"])
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                 betas=params["betas"])
    else:
        raise ValueError(f"Optimizer {params['optimizer']} not supported.")


def build_scheduler(optimizer, params):
    name = params["scheduler"].lower()
    if name == "none":
        return None
    elif name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=params["cosine_Tmax"])
    elif name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=params["plateau_patience"],
            factor=params["plateau_factor"],
            verbose=False
        )
    else:
        raise ValueError(f"Scheduler {params['scheduler']} not supported.")


# ---------------------------- Tuning core ----------------------------
def dict_to_key(d):
    """Chiave hashable per evitare ripetizioni (ordina items)."""
    return tuple(sorted((k, str(v)) for k, v in d.items()))

# def load_tried():
#     if os.path.exists(TRIED_PATH):
#         with open(TRIED_PATH, "r") as f:
#             return set(json.load(f))
#     return set()

# def save_tried(tried_set):
#     with open(TRIED_PATH, "w") as f:
#         json.dump(sorted(list(tried_set)), f, indent=2)
from pathlib import Path
HERE = Path(__file__).resolve().parent
LOG_PATH = str((HERE / "tuning_log.csv").resolve())
TRIED_PATH = str((HERE / "tuning_tried.json").resolve())


def load_tried():
    """Legge tuning_tried.json e ricrea le chiavi hashable (tuple di coppie)."""
    if os.path.exists(TRIED_PATH):
        with open(TRIED_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)  # lista di liste [[k, v], ...]
        tried = set()
        for item in raw:
            # item è una lista di coppie [k, v] -> riconverti in tuple((k,v), ...)
            tried.add(tuple((str(k), str(v)) for k, v in item))
        return tried
    return set()

def save_tried(tried_set):
    """Salvataggio atomico del set su JSON (evita file troncati in caso di crash)."""
    tmp = TRIED_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted(list(tried_set)), f, indent=2, ensure_ascii=False)
    os.replace(tmp, TRIED_PATH)


def append_log(row_dict):
    header = not os.path.exists(LOG_PATH)
    import csv
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if header:
            writer.writeheader()
        writer.writerow(row_dict)

def iter_param_grid(param_grid, max_trials=None, seed=GLOBAL_SEED):
    keys = list(param_grid.keys())
    all_vals = [param_grid[k] for k in keys]
    all_combos = [dict(zip(keys, vals)) for vals in product(*all_vals)]

    # Filtra combo non sensate (scheduler-specific params non usati non creano problemi, ma ok tenerli)
    # Esempio: nulla da filtrare qui, lasciamo tutto.

    if max_trials is not None and max_trials < len(all_combos):
        random.Random(seed).shuffle(all_combos)
        all_combos = all_combos[:max_trials]
    return all_combos


def run_single_trial(device, base_seed, params):
    # seed
    seed_everything(base_seed)

    # Carica dati + loss
    (data_official, col_stats,
     task, train_table, val_table, test_table, loss_fn) = prepare_data_and_labels(device)

    # Loader con parametri dal grid
    loader_dict = make_loader_dict(
        data_official, task, train_table, val_table, test_table,
        batch_size=params["batch_size"],
        num_neighbours=params["num_neighbours"]
    )

    # Modello / Optim / Scheduler
    model = build_model(data_official, col_stats, params, device)
    optimizer = build_optimizer(model, params)
    scheduler = build_scheduler(optimizer, params)

    # Early stopping + checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_val = os.path.join(CHECKPOINT_DIR, f"best_val_{run_tag}.pt")
    ckpt_test = os.path.join(CHECKPOINT_DIR, f"best_test_{run_tag}.pt")

    early_stopping = EarlyStopping(
        patience=EARLY_STOP_PATIENCE,
        delta=0.0,
        verbose=True,
        higher_is_better=HIGHER_IS_BETTER,
        path=ckpt_val
    )

    best_val = -math.inf if HIGHER_IS_BETTER else math.inf
    best_test = -math.inf if HIGHER_IS_BETTER else math.inf

    # Training
    for epoch in range(N_EPOCHS):
        _ = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

        train_pred = test(model, loader_dict["train"], device=device, task=task)
        val_pred = test(model, loader_dict["val"], device=device, task=task)
        test_pred = test(model, loader_dict["test"], device=device, task=task)

        train_metrics = evaluate_performance(train_pred, train_table, task.metrics, task=task)
        val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)
        test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)

        # Step scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_metrics[TUNE_METRIC])
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {epoch:02d}, Train {TUNE_METRIC}: {train_metrics[TUNE_METRIC]:.2f}, "
              f"Val {TUNE_METRIC}: {val_metrics[TUNE_METRIC]:.2f}, "
              f"Test {TUNE_METRIC}: {test_metrics[TUNE_METRIC]:.2f}, LR: {current_lr:.6f}")

        # Track & save bests
        cur_val = val_metrics[TUNE_METRIC]
        cur_test = test_metrics[TUNE_METRIC]

        improved_val = (cur_val > best_val) if HIGHER_IS_BETTER else (cur_val < best_val)
        if improved_val:
            best_val = cur_val
            state_dict = copy.deepcopy(model.state_dict())
            torch.save(state_dict, ckpt_val)

        improved_test = (cur_test > best_test) if HIGHER_IS_BETTER else (cur_test < best_test)
        if improved_test:
            best_test = cur_test
            state_dict_test = copy.deepcopy(model.state_dict())
            torch.save(state_dict_test, ckpt_test)

        # Early stopping sulla validation
        early_stopping(cur_val, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return {
        "best_val": float(best_val),
        "best_test": float(best_test),
        "ckpt_val": ckpt_val,
        "ckpt_test": ckpt_test,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Genera tutte le combinazioni (o un sottoinsieme casuale)
    combos = iter_param_grid(PARAM_GRID, MAX_TRIALS, seed=GLOBAL_SEED)

    # Carica combinazioni già provate (persistenti tra run)
    tried = load_tried()

    best_overall_val = -math.inf if HIGHER_IS_BETTER else math.inf
    best_overall = None

    for params in combos:
        # Normalizza i param per evitare conflitti inutili
        # (es. se optimizer=SGD, betas non influiscono — li lasciamo comunque nella chiave per semplicità)
        key = dict_to_key(params)
        if key in tried:
            print("[SKIP] already tried:", params)
            continue

        tried.add(key)
        save_tried(tried)

        print("\n================= NEW TRIAL =================")
        print(json.dumps(params, indent=2))
        print("=============================================\n")


        try:
            # Trial
            result = run_single_trial(device, GLOBAL_SEED, params)

            # Log dei risultati
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "optimizer": params["optimizer"],
                "lr": params["lr"],
                "weight_decay": params["weight_decay"],
                "momentum": params["momentum"],
                "betas": str(params["betas"]),
                "hidden_channels": params["hidden_channels"],
                "out_channels": params["out_channels"],
                "dropout_p": params["dropout_p"],
                "num_heads": params["num_heads"],
                "num_layers": params["num_layers"],
                "batch_size": params["batch_size"],
                "num_neighbours": params["num_neighbours"],
                "scheduler": params["scheduler"],
                "cosine_Tmax": params["cosine_Tmax"],
                "plateau_patience": params["plateau_patience"],
                "plateau_factor": params["plateau_factor"],
                "best_val": result["best_val"],
                "best_test": result["best_test"],
                "ckpt_val": result["ckpt_val"],
                "ckpt_test": result["ckpt_test"],
            }
            append_log(row)

            # Aggiorna best globale
            if (result["best_val"] > best_overall_val) if HIGHER_IS_BETTER else (result["best_val"] < best_overall_val):
                best_overall_val = result["best_val"]
                best_overall = (params, result)
                print("\n*** NEW BEST (validation) ***")
                print("Params:", json.dumps(params, indent=2))
                print("Best Val:", result["best_val"], "Best Test:", result["best_test"])
                print("Checkpoints:", result["ckpt_val"], "|", result["ckpt_test"])
                print("*****************************\n")

        except Exception as e:
            #  Per non riprovare automaticamente la stessa combo, NON rimuovere la chiave:
            print(f"[ERROR] Trial failed and will be skipped next runs: {e}")
            # Se invece vuoi poterla riprovare in futuro, scommenta le due righe sotto:
            # tried.remove(key)
            # save_tried(tried)
            continue

    if best_overall is not None:
        params, res = best_overall
        print("\n========== SUMMARY (BEST ON VAL) ==========")
        print(json.dumps(params, indent=2))
        print(f"Best Val: {res['best_val']:.4f} | Best Test: {res['best_test']:.4f}")
        print(f"Checkpoints: {res['ckpt_val']} | {res['ckpt_test']}")
        print("===========================================\n")
    else:
        print("No trials executed.")


if __name__ == "__main__":
    main()
