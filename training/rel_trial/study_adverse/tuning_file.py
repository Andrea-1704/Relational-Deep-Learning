#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast tuner for RelBench rel-trial / study-adverse with HGraphSAGE (compatibile con il tuo codice).

- Griglia prioritaria (optimizer, lr, weight_decay, hidden, layers, aggr, dropout, sampler, batch_size)
- AMP + grad clipping, Cosine LR **corretto** (step() senza metrica)
- Supporto MiniLM text embeddings (consigliato) o fallback categorico (--no-text-embeds)
- Early stopping + early pruning; log su CSV; salvataggio checkpoint best

Esempi:
  python tune_study_adverse_fixed.py --max_trials 20 --preset relbench_paper
  python tune_study_adverse_fixed.py --max_trials 12 --preset fast --no-text-embeds
  python tune_study_adverse_fixed.py --epochs 120 --sampler "15,10"

Richiede nel repo:
  - model.others.HGraphSAGE.Model  (signature: Model(data, col_stats_dict, num_layers, channels, out_channels, aggr, norm, ...))
  - data_management.data.loader_dict_fn  (signature: loader_dict_fn(data=..., task=..., ...))
  - utils.{EarlyStopping} e utils.utils.{evaluate_performance, test, train}
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
from typing import List

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import numpy as np
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal

# text embedder (se manca, gestiamo il fallback in runtime)
try:
    from torch_frame.config.text_embedder import TextEmbedderConfig
except Exception:
    TextEmbedderConfig = None  # type: ignore

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
    norm: str = "batch_norm"   # passa alla head MLP
    dropout: float = 0.5
    # training
    optimizer: str = "AdamW"   # "Adam" | "AdamW"
    lr: float = 1e-3
    weight_decay: float = 5e-5
    epochs: int = 180
    batch_size: int = 1024
    sampler: str = "20,10"     # fanout per layer come "a,b"
    # loss / scheduler
    loss: str = "smoothl1"     # "l1" | "smoothl1" | "mse"
    scheduler: str = "cosine"  # "cosine" | "plateau"
    # misc
    grad_clip: float = 1.0
    amp: bool = True
    # data
    use_text_embeds: bool = True  # se False: merge testo->categorico (più veloce, di solito peggiore)


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
    # cosine default
    return CosineAnnealingLR(optimizer, T_max=cfg.epochs)


def parse_sampler(s: str) -> List[int]:
    return [int(x) for x in s.split(",")]


def build_graph_and_loaders(cfg: TrialConfig, dataset_name="rel-trial", task_name="study-adverse"):
    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    db = dataset.get_db()
    col_to_stype = get_stype_proposal(db)

    # text embeddings (MiniLM) oppure merge a categorico
    text_cfg = None
    db_used = db
    col_to_stype_used = col_to_stype
    if cfg.use_text_embeds and TextEmbedderConfig is not None:
        text_cfg = TextEmbedderConfig(name="minilm", pooling="mean", max_length=64)
    else:
        db_used, col_to_stype_used = merge_text_columns_to_categorical(db, col_to_stype)

    data, col_stats_dict = make_pkey_fkey_graph(
        db_used,
        col_to_stype_dict=col_to_stype_used,
        text_embedder_cfg=text_cfg,
        cache_dir=None  # metti una cartella per cachare gli embed testuali tra trial
    )

    fanouts = parse_sampler(cfg.sampler)
    loader_dict = loader_dict_fn(
        data=data,
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table,
        batch_size=cfg.batch_size,
        num_neighbours=fanouts,
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
    train_metrics = evaluate_performance(train_pred, train_table, task.metrics, task=task)
    val_pred = test(model, loader_dict["val"], device=device, task=task)
    val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)
    test_pred = test(model, loader_dict["test"], device=device, task=task)
    test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)
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
    tune_metric = "mae"
    early_stopping = EarlyStopping(patience=12, delta=0.0, verbose=False,
                                   higher_is_better=higher_is_better,
                                   path=str(Path(log_dir) / "tmp-best.pt"))

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val = math.inf
    best_test = math.inf
    state_dict_val = None
    state_dict_test = None

    # pruning: se dopo 25 epoche sei >1.0 peggio del best, stoppa il trial
    worse_by_threshold = 1.0
    check_epoch = 25

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in loader_dict["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                # train() del tuo progetto può restituire la loss dato un batch singolo
                loss = train(model, optimizer=None, loader_dict=None, device=device,
                             task=task, loss_fn=loss_fn, batch=batch, return_loss=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        # Eval
        train_metrics, val_metrics, test_metrics = evaluate_loop(
            model, task, train_table, val_table, test_table, loader_dict, device
        )

        # Scheduler step
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()  # <-- NIENTE metrica qui
        else:
            scheduler.step(val_metrics[tune_metric])

        # Best tracking
        if val_metrics[tune_metric] < best_val:
            best_val = val_metrics[tune_metric]
            state_dict_val = copy.deepcopy(model.state_dict())
        if val_metrics[tune_metric] <= best_val + 1e-9:
            best_test = test_metrics[tune_metric]
            state_dict_test = copy.deepcopy(model.state_dict())

        # Early stopping
        early_stopping(val_metrics[tune_metric], model)
        if early_stopping.early_stop:
            break

        # Early pruning
        if epoch >= check_epoch and (val_metrics[tune_metric] - best_val) > worse_by_threshold:
            break

    # Save checkpoints
    exp_name = (
        f"{cfg.optimizer}_lr{cfg.lr}_wd{cfg.weight_decay}_h{cfg.channels}"
        f"_L{cfg.num_layers}_{cfg.aggr}_do{cfg.dropout}_sam{cfg.sampler}"
        f"_bs{cfg.batch_size}_loss{cfg.loss}"
    )
    ckpt_dir = Path(log_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if state_dict_val is not None:
        torch.save(state_dict_val, ckpt_dir / f"best_val_{exp_name}.pt")
    if state_dict_test is not None:
        torch.save(state_dict_test, ckpt_dir / f"best_test_{exp_name}.pt")

    return {"val_mae": float(best_val), "test_mae": float(best_test), "exp_name": exp_name, **asdict(cfg)}


def prioritized_grid(preset: str, max_trials: int) -> List[TrialConfig]:
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

    seeds = [42]  # per velocità; eventualmente aggiungi altri seed
    lrs = [base.lr, 5e-4, 2e-4]
    wds = [base.weight_decay, 1e-5, 0.0]
    chans = [256, 512]
    layers = [3, 2]
    aggrs = [base.aggr, "max"]
    dropouts = [0.5, 0.3]
    samplers = ["20,10", "15,10"]
    batches = [1024, 2048]
    losses = ["smoothl1", "l1"]

    grid: List[TrialConfig] = []
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
    # priorità: batch più grande, aggr="mean", 3 layer
    grid.sort(key=lambda c: (-c.batch_size, c.aggr != "mean", -c.num_layers, c.lr))
    return grid[:max_trials]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_trials", type=int, default=20)
    parser.add_argument("--preset", type=str, default="relbench_paper", choices=["relbench_paper", "fast", "none"])
    parser.add_argument("--no-text-embeds", action="store_true", help="Disabilita text embeddings (più veloce, di solito peggio).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_dir", type=str, default="tune_logs")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per trial.")
    parser.add_argument("--sampler", type=str, default=None, help='Override fanout string, es. "15,10"')
    args = parser.parse_args()

    # Build grid
    grid = prioritized_grid(args.preset, args.max_trials)
    for cfg in grid:
        cfg.use_text_embeds = not args.no_text_embeds
        if args.epochs is not None:
            cfg.epochs = int(args.epochs)
        if args.sampler is not None:
            cfg.sampler = args.sampler

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
