#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tuner per RelBench rel-trial / study-adverse con HGraphSAGE
— allineato ai tuoi file locali e compatibile col tuo loader (nessun text embedding).

- Costruzione grafo come nel tuo train:
  merge_text_columns_to_categorical + make_pkey_fkey_graph(..., text_embedder_cfg=None)
- Usa le tue funzioni: loader_dict_fn, train, test, evaluate_performance
- Scheduler CosineAnnealingLR **corretto** (step() senza metrica); ReduceLROnPlateau usa una metrica di riferimento
- Early stopping + early pruning
- Griglia prioritaria su optimizer/lr/wd/layers/channels/aggr/sampler/batch_size/loss
- Log CSV + salvataggio checkpoint
- **Sampler sempre intero** per non rompere il tuo loader (se riceve "20,10", prende il primo numero)
- **Valutazione safe**: se lo split non ha la colonna target (es. test), le metriche sono NaN senza crash
- **Compatibile** con task.metrics sia come dict sia come list

Esempi:
  python tune_study_adverse_fixed.py --max_trials 20 --preset relbench_paper
  python tune_study_adverse_fixed.py --max_trials 12 --preset fast
  python tune_study_adverse_fixed.py --epochs 120 --sampler "128"
"""

import os
import math
import json
import time
import copy
import argparse
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import numpy as np
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal

import sys
sys.path.append(os.path.abspath("."))

from model.others.HGraphSAGE import Model
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.EarlyStopping import EarlyStopping
from utils.utils import evaluate_performance, test, train


# ------------------------- Utilities -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrialConfig:
    seed: int = 42
    # modello
    channels: int = 256
    num_layers: int = 3
    aggr: str = "mean"         # "mean" | "max" | "sum"
    norm: str = "batch_norm"   # come nel tuo HGraphSAGE
    # training
    optimizer: str = "AdamW"   # "Adam" | "AdamW"
    lr: float = 1e-3
    weight_decay: float = 5e-5
    epochs: int = 180
    batch_size: int = 1024
    sampler: str = "128"       # <-- SEMPRE INTERO (stringa), es. "256", "128", "64"
    # loss / scheduler
    loss: str = "smoothl1"     # "l1" | "smoothl1" | "mse"
    scheduler: str = "cosine"  # "cosine" | "plateau"


def get_loss_fn(cfg: TrialConfig):
    if cfg.loss == "l1":
        return nn.L1Loss()
    if cfg.loss == "mse":
        return nn.MSELoss()
    return nn.SmoothL1Loss(beta=1.0)


def get_optimizer(cfg: TrialConfig, model: nn.Module):
    if cfg.optimizer.lower() == "adam":
        return Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def get_scheduler(cfg: TrialConfig, optimizer):
    if cfg.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, threshold=0.0)
    # cosine default (step() senza metrica)
    return CosineAnnealingLR(optimizer, T_max=cfg.epochs)


def parse_sampler_to_int(s: str) -> int:
    """
    Il tuo loader duplica *così com'è* num_neighbours per layer.
    Per evitare liste-di-liste, forziamo un **intero**.
    Se arriva una stringa 'a,b', prendiamo solo 'a'.
    """
    try:
        if "," in s:
            first = s.split(",")[0].strip()
            val = int(first)
            print(f"[WARN] sampler '{s}' non supportato dal loader; uso {val} (replicato per layer).")
            return val
        return int(s.strip())
    except Exception:
        print(f"[WARN] sampler '{s}' non parsabile; uso 256.")
        return 256


# ------------------------- Metrics helpers -------------------------
def _metric_names(metrics) -> List[str]:
    # Supporta sia dict che list/tuple o stringhe/callable/classi Metric
    if isinstance(metrics, dict):
        return list(metrics.keys())
    names: List[str] = []
    if isinstance(metrics, (list, tuple)):
        for m in metrics:
            if isinstance(m, str):
                names.append(m)
            elif hasattr(m, "name") and isinstance(getattr(m, "name"), str):
                names.append(m.name)
            elif hasattr(m, "__name__"):
                names.append(m.__name__)
            else:
                names.append("metric")
    if not names:
        names = ["metric"]
    return names


def safe_metrics(pred: Dict[str, Any], table, metrics, task) -> Dict[str, float]:
    """Calcola le metriche solo se il target è disponibile nello split."""
    target_col = getattr(task, "target_col", None)
    has_target = (
        (target_col is not None)
        and hasattr(table, "df")
        and (target_col in table.df.columns)
    )
    if has_target:
        return evaluate_performance(pred, table, metrics, task=task)
    # target assente (tipicamente test): restituisci dizionario con NaN
    names = _metric_names(metrics)
    return {name: float("nan") for name in names}


def pick_ref_value(metrics_dict: Dict[str, float]) -> float:
    """Restituisce la metrica di riferimento per scheduler/early stop (preferisce MAE)."""
    if not isinstance(metrics_dict, dict) or not metrics_dict:
        return float("inf")
    if "mae" in metrics_dict and isinstance(metrics_dict["mae"], (int, float)):
        return metrics_dict["mae"]
    if "MAE" in metrics_dict and isinstance(metrics_dict["MAE"], (int, float)):
        return metrics_dict["MAE"]
    # prima metrica numerica disponibile
    for v in metrics_dict.values():
        if isinstance(v, (int, float)):
            return v
    return float("inf")


# ------------------------- Pipeline -------------------------
def build_graph_and_loaders(cfg: TrialConfig, dataset_name="rel-trial", task_name="study-adverse"):
    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)

    # Esattamente come nel tuo train: merge testo -> categorico, nessun text embedder
    db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)

    data, col_stats_dict = make_pkey_fkey_graph(
        db_nuovo,
        col_to_stype_dict=col_to_stype_dict_nuovo,
        text_embedder_cfg=None,
        cache_dir=None
    )

    # Loader identico allo stile del tuo script
    fanout_int = parse_sampler_to_int(cfg.sampler)  # <-- QUI forziamo **int**
    loader_dict = loader_dict_fn(
        data=data,
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table,
        batch_size=cfg.batch_size,
        num_neighbours=fanout_int,  # <-- passa INT, il tuo loader farà [int, int]
    )
    return task, train_table, val_table, test_table, loader_dict, data, col_stats_dict


def build_model(cfg: TrialConfig, data, col_stats_dict, device="cuda"):
    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=cfg.num_layers,
        channels=cfg.channels,
        out_channels=1,            # regression
        aggr=cfg.aggr,
        norm=cfg.norm,
        shallow_list=[],
        id_awareness=False,
        predictor_n_layers=1,
    ).to(device)
    return model


def evaluate_loop(model, task, train_table, val_table, test_table, loader_dict, device):
    train_pred = test(model, loader_dict["train"], device=device, task=task)
    train_metrics = safe_metrics(train_pred, train_table, task.metrics, task)

    val_pred = test(model, loader_dict["val"], device=device, task=task)
    val_metrics = safe_metrics(val_pred, val_table, task.metrics, task)

    test_pred = test(model, loader_dict["test"], device=device, task=task)
    test_metrics = safe_metrics(test_pred, test_table, task.metrics, task)

    return train_metrics, val_metrics, test_metrics


def train_one_trial(cfg: TrialConfig, device="cuda", log_dir="tune_logs"):
    seed_everything(cfg.seed)
    os.makedirs(log_dir, exist_ok=True)

    task, train_table, val_table, test_table, loader_dict, data, col_stats_dict = build_graph_and_loaders(cfg)

    loss_fn = get_loss_fn(cfg)
    model = build_model(cfg, data, col_stats_dict, device=device)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    higher_is_better = False
    early_stopping = EarlyStopping(
        patience=12, delta=0.0, verbose=False,
        higher_is_better=higher_is_better,
        path=str(Path(log_dir) / "tmp-best.pt")
    )

    best_val = math.inf
    best_test = float("nan")
    state_dict_val = None
    state_dict_test = None

    # pruning: se dopo 25 epoche sei >1.0 peggio del best, stoppa il trial
    worse_by_threshold = 1.0
    check_epoch = 25

    for epoch in range(1, cfg.epochs + 1):
        # ALLINEATO al tuo train: chiamiamo la tua train() una volta per epoca
        _ = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

        # Eval (safe su test senza target)
        train_metrics, val_metrics, test_metrics = evaluate_loop(
            model, task, train_table, val_table, test_table, loader_dict, device
        )

        # ---- Scheduler + tracking robusti anche se 'mae' non c'è ----
        val_ref = pick_ref_value(val_metrics)

        # Scheduler step
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()  # CORRETTO: nessuna metrica qui
        else:
            scheduler.step(val_ref)

        # Best tracking (usa la metrica di riferimento)
        if val_ref < best_val:
            best_val = val_ref
            state_dict_val = copy.deepcopy(model.state_dict())

        # Aggiorna "best_test" solo se esiste una metrica numerica su test
        test_ref = pick_ref_value(test_metrics)
        if not math.isnan(test_ref) and math.isfinite(test_ref):
            best_test = test_ref
            state_dict_test = copy.deepcopy(model.state_dict())

        # Early stopping e pruning
        early_stopping(val_ref, model)
        if early_stopping.early_stop:
            break
        if epoch >= check_epoch and (val_ref - best_val) > worse_by_threshold:
            break

    # Save checkpoints
    exp_name = (
        f"{cfg.optimizer}_lr{cfg.lr}_wd{cfg.weight_decay}_h{cfg.channels}"
        f"_L{cfg.num_layers}_{cfg.aggr}_sam{cfg.sampler}"
        f"_bs{cfg.batch_size}_loss{cfg.loss}"
    )
    ckpt_dir = Path(log_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if state_dict_val is not None:
        torch.save(state_dict_val, ckpt_dir / f"best_val_{exp_name}.pt")
    if state_dict_test is not None:
        torch.save(state_dict_test, ckpt_dir / f"best_test_{exp_name}.pt")

    return {"val_mae": float(best_val), "test_mae": float(best_test), "exp_name": exp_name, **asdict(cfg)}


def prioritized_grid(preset: str, max_trials: int):
    base = TrialConfig()
    if preset == "relbench_paper":
        base.optimizer = "AdamW"
        base.lr = 1e-3
        base.weight_decay = 5e-5
        base.aggr = "mean"
        base.epochs = 180
        base.sampler = "128"   # più stabile del 256 pieno e rapido
        base.batch_size = 1024
    elif preset == "fast":
        base.optimizer = "AdamW"
        base.lr = 8e-4
        base.weight_decay = 1e-5
        base.epochs = 140
        base.sampler = "64"
        base.batch_size = 2048

    seeds = [42]  # un seed per velocità
    lrs = [base.lr, 5e-4, 2e-4]
    wds = [base.weight_decay, 1e-5, 0.0]
    chans = [256, 512]
    layers = [3, 2]
    aggrs = [base.aggr, "max"]
    samplers = [base.sampler, "256", "128", "64"]   # <-- SOLO INTERI
    batches = [base.batch_size, 2048]
    losses = ["smoothl1", "l1"]  # MAE target: SmoothL1 di solito è più stabile

    grid = []
    for sd in seeds:
        for lr in lrs:
            for wd in wds:
                for ch in chans:
                    for L in layers:
                        for ag in aggrs:
                            for sam in samplers:
                                for bs in batches:
                                    for ls in losses:
                                        cfg = copy.deepcopy(base)
                                        cfg.seed = sd
                                        cfg.lr = lr
                                        cfg.weight_decay = wd
                                        cfg.channels = ch
                                        cfg.num_layers = L
                                        cfg.aggr = ag
                                        cfg.sampler = sam
                                        cfg.batch_size = bs
                                        cfg.loss = ls
                                        grid.append(cfg)
    # Priorità: batch grande, aggr="mean", 3 layer
    grid.sort(key=lambda c: (-c.batch_size, c.aggr != "mean", -c.num_layers, c.lr))
    return grid[:max_trials]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_trials", type=int, default=20)
    parser.add_argument("--preset", type=str, default="relbench_paper", choices=["relbench_paper", "fast", "none"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_dir", type=str, default="tune_logs")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per trial.")
    parser.add_argument("--sampler", type=str, default=None, help='Override fanout (es. "128" o "256")')
    args = parser.parse_args()

    # Build grid
    grid = prioritized_grid(args.preset, args.max_trials)
    for cfg in grid:
        if args.epochs is not None:
            cfg.epochs = int(args.epochs)
        if args.sampler is not None:
            # Forziamo int (in stringa) per compatibilità col loader
            cfg.sampler = str(parse_sampler_to_int(args.sampler))

    # Run
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = []
    best = math.inf
    best_row = None

    csv_path = Path(args.log_dir) / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() and len(grid) > 0:
        with open(csv_path, "w") as f:
            f.write(",".join(list(asdict(grid[0]).keys()) + ["val_mae", "test_mae", "exp_name"]) + "\n")

    start = time.time()
    for i, cfg in enumerate(grid, 1):
        print(f"\n=== Trial {i}/{len(grid)} ===\n{cfg}\n")
        row = train_one_trial(cfg, device=device, log_dir=args.log_dir)
        results.append(row)
        with open(csv_path, "a") as f:
            values = [str(v) for v in asdict(cfg).values()] + [str(row["val_mae"]), str(row["test_mae"]), row["exp_name"]]
            f.write(",".join(values) + "\n")

        if row["val_mae"] < best:
            best = row["val_mae"]
            best_row = row
            print(f"[BEST so far] val_mae={best:.4f}  test_mae={row['test_mae']:.4f}  exp={row['exp_name']}")

    dur = time.time() - start
    print("\n=== DONE ===")
    print(f"Ran {len(results)} trials in {dur/60:.1f} min")
    print("Best row:", json.dumps(best_row, indent=2))


if __name__ == "__main__":
    main()
