
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast tuner for RelBench rel-trial / study-adverse with HGraphSAGE.

- Prioritized grid of (optimizer, lr, weight_decay, hidden, layers, aggr, dropout, sampler, batch_size)
- Early pruning (patience + "worse-than-best-by" threshold)
- AMP + grad clipping for speed/stability
- CosineAnnealingLR **fixed** (step() without metric)
- Optional preset "relbench_paper" (AdamW, lr=1e-3, wd=5e-5, aggr=mean) as a strong baseline
- Logs each trial to CSV and saves best checkpoint

Usage:
    python tune_study_adverse.py --max_trials 20 --preset relbench_paper
    python tune_study_adverse.py --max_trials 30 --no-text-embeds   # if you really need to disable text embeds

This script expects your repo structure to expose:
    - model.others.HGraphSAGE.Model
    - data_management.data.loader_dict_fn
    - utils.{EarlyStopping, utils:{evaluate_performance, evaluate_on_full_train, test, train}}
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
from typing import Dict, Any, List, Tuple
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import numpy as np
from relbench.datasets import get_dataset
from relbench.tasks import get_task

import sys
import os
sys.path.append(os.path.abspath("."))


from model.others.HGraphSAGE import Model
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.EarlyStopping import EarlyStopping
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train


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
    # model
    channels: int = 256
    num_layers: int = 3
    aggr: str = "mean"      # "mean" | "max" | "sum"
    dropout: float = 0.5
    # training
    optimizer: str = "AdamW"  # "Adam" | "AdamW"
    lr: float = 1e-3
    weight_decay: float = 5e-5
    epochs: int = 40
    batch_size: int = 1024
    sampler: str = "20,10"    # fanout per layer as "a,b"
    # loss/schedule
    loss: str = "smoothl1"    # "l1" | "smoothl1" | "mse"
    scheduler: str = "cosine" # "cosine" | "plateau"
    # misc
    grad_clip: float = 1.0
    amp: bool = True
    # data
    use_text_embeds: bool = True  # if False, we merge text columns -> categorical (faster but usually worse)


def build_model(cfg: TrialConfig, in_channels_dict, out_channels=1, device="cuda"):
    model = Model(
        in_channels_dict=in_channels_dict,
        hidden_channels=cfg.channels,
        out_channels=out_channels,
        num_layers=cfg.num_layers,
        aggr=cfg.aggr,
        dropout=cfg.dropout,
    ).to(device)
    return model


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
    # cosine as default
    return CosineAnnealingLR(optimizer, T_max=cfg.epochs)


def parse_sampler(s: str) -> List[int]:
    return [int(x) for x in s.split(",")]


def maybe_merge_text_columns(db):
    # lightweight wrapper in case user wants to disable text embeds for speed
    return merge_text_columns_to_categorical(db)


def get_data_and_loaders(cfg: TrialConfig, dataset_name="rel-trial", task_name="study-adverse", device="cuda"):
    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    db = dataset.get_db()
    if not cfg.use_text_embeds:
        db = maybe_merge_text_columns(db)
    col_to_stype_dict = get_stype_proposal(db)
    #this is used to get the stype of the columns

    #let's use the merge categorical values:
    db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)

    
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict_nuovo,
        #text_embedder_cfg=text_embedder_cfg,
        text_embedder_cfg = None,
        cache_dir=None  # disabled
    )

    # Build loaders via your project's helper
    fanouts = parse_sampler(cfg.sampler)
    loader_dict, in_channels_dict = loader_dict_fn(
        #db=db,
        data=data,
        task= task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table,
        batch_size=cfg.batch_size,
        num_neighbours=fanouts,
        #device=device,
    )
    return task, train_table, val_table, test_table, loader_dict, in_channels_dict


def evaluate_loop(model, task, train_table, val_table, test_table, loader_dict, loss_fn, device, amp):
    train_pred = test(model, loader_dict["train"], device=device, task=task)
    train_metrics = evaluate_performance(train_pred, train_table, task.metrics, task=task)
    val_pred = test(model, loader_dict["val"], device=device, task=task)
    val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)
    test_pred = test(model, loader_dict["test"], device=device, task=task)
    test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)
    return train_metrics, val_metrics, test_metrics


def train_one_trial(cfg: TrialConfig, device="cuda", log_dir="tune_logs"):
    seed_everything(cfg.seed)
    os.makedirs(log_dir, exist_ok=True)

    task, train_table, val_table, test_table, loader_dict, in_channels_dict = get_data_and_loaders(cfg, device=device)

    out_channels = 1
    loss_fn = get_loss_fn(cfg)
    model = build_model(cfg, in_channels_dict, out_channels=out_channels, device=device)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    higher_is_better = False
    tune_metric = "mae"
    early_stopping = EarlyStopping(patience=12, delta=0.0, verbose=False, higher_is_better=higher_is_better,
                                   path=str(Path(log_dir) / "tmp-best.pt"))

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val = math.inf
    best_test = math.inf
    state_dict_val = None
    state_dict_test = None

    # Early-pruning threshold: if val is worse than best by >1.0 after 25 epochs, stop to save time
    worse_by_threshold = 1.0
    check_epoch = 25

    for epoch in range(1, cfg.epochs + 1):
        # single-epoch train
        model.train()
        running_loss = 0.0
        for batch in loader_dict["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                loss = train(model, optimizer=None, loader_dict=None, device=device, task=task, loss_fn=loss_fn, batch=batch, return_loss=True)
            scaler.scale(loss).backward()
            # grad clip
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())

        # Eval
        train_metrics, val_metrics, test_metrics = evaluate_loop(model, task, train_table, val_table, test_table, loader_dict, loss_fn, device, cfg.amp)

        # Scheduler step (cosine: step() w/out metric; plateau: step(val))
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        else:
            scheduler.step(val_metrics[tune_metric])

        # Track best
        if val_metrics[tune_metric] < best_val:
            best_val = val_metrics[tune_metric]
            state_dict_val = copy.deepcopy(model.state_dict())
        if val_metrics[tune_metric] <= best_val + 1e-9:
            best_test = test_metrics[tune_metric]
            state_dict_test = copy.deepcopy(model.state_dict())

        # Early stop (patience-based)
        early_stopping(val_metrics[tune_metric], model)
        if early_stopping.early_stop:
            break

        # Early prune if far worse after check_epoch
        if epoch >= check_epoch and (val_metrics[tune_metric] - best_val) > worse_by_threshold:
            break

    # Save checkpoints
    exp_name = f"{cfg.optimizer}_lr{cfg.lr}_wd{cfg.weight_decay}_h{cfg.channels}_L{cfg.num_layers}_{cfg.aggr}_do{cfg.dropout}_sam{cfg.sampler}_bs{cfg.batch_size}_loss{cfg.loss}"
    ckpt_dir = Path(log_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if state_dict_val is not None:
        torch.save(state_dict_val, ckpt_dir / f"best_val_{exp_name}.pt")
    if state_dict_test is not None:
        torch.save(state_dict_test, ckpt_dir / f"best_test_{exp_name}.pt")

    return {
        "val_mae": float(best_val),
        "test_mae": float(best_test),
        "exp_name": exp_name,
        **asdict(cfg),
    }


def prioritized_grid(preset: str, max_trials: int) -> List[TrialConfig]:
    # A small, prioritized grid focusing on combinations that tend to work for rel-trial
    # Preset "relbench_paper" biases to AdamW, lr=1e-3, wd=5e-5, aggr=mean.
    base = TrialConfig()
    if preset == "relbench_paper":
        base.optimizer = "AdamW"
        base.lr = 1e-3
        base.weight_decay = 5e-5
        base.aggr = "mean"
        base.epochs = 180
    elif preset == "fast":
        base.optimizer = "AdamW"
        base.lr = 8e-4
        base.weight_decay = 1e-5
        base.epochs = 140

    seeds = [42]  # keep single seed for speed; can extend to [42, 1, 2]
    lrs = [base.lr, 5e-4, 2e-4]
    wds = [base.weight_decay, 1e-5, 0.0]
    chans = [256, 512]
    layers = [3, 2]
    aggrs = [base.aggr, "max"]
    dropouts = [0.5, 0.3]
    samplers = ["20,10", "15,10"]
    batches = [1024, 2048]
    losses = ["smoothl1", "l1"]

    grid = []
    for sd in seeds:
        for lr in lrs:
            for wd in wds:
                for ch in chans:
                    for L in layers:
                        for ag in aggrs:
                            for do in dropouts:
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
                                            cfg.dropout = do
                                            cfg.sampler = sam
                                            cfg.batch_size = bs
                                            cfg.loss = ls
                                            grid.append(cfg)
    # Prioritize by heuristic: higher bs, mean aggr, 3 layers first
    grid.sort(key=lambda c: (-c.batch_size, c.aggr != "mean", -c.num_layers, c.lr))
    return grid[:max_trials]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_trials", type=int, default=20)
    parser.add_argument("--preset", type=str, default="relbench_paper", choices=["relbench_paper", "fast", "none"])
    parser.add_argument("--no-text-embeds", action="store_true", help="Disable text embeddings (faster, but usually worse).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_dir", type=str, default="tune_logs")
    args = parser.parse_args()

    # Build grid
    grid = prioritized_grid(args.preset, args.max_trials)
    for cfg in grid:
        cfg.use_text_embeds = not args.no_text_embeds

    # Run
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = []
    best = math.inf
    best_row = None

    csv_path = Path(args.log_dir) / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
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
