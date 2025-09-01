# pre_training/VGAE/tune_vgae.py
import os
import math
import copy
import itertools
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import torch
from torch.nn import L1Loss
from torch.optim.lr_scheduler import CosineAnnealingLR

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_geometric.seed import seed_everything

import sys
import os
sys.path.append(os.path.abspath("."))


from model.others.HGraphSAGE import Model
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from pre_training.VGAE.Utils_VGAE import train_vgae
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train


@dataclass
class PretrainConfig:
    # pre-training (VGAE)
    latent_dim: int = 32
    hidden_dim: int = 128
    vgae_epochs: int = 80
    # data loader per pre-training
    batch_size_pt: int = 128
    num_neighbours_pt: int = 64
    # modello backbone
    channels: int = 128          # dimensione dell'encoder/backbone
    num_layers: int = 2
    aggr: str = "max"
    norm: str = "batch_norm"
    # fine-tuning "breve" per scoring
    ft_epochs: int = 25
    ft_lr: float = 5e-4
    ft_weight_decay: float = 0.0
    ft_batch_size: int = 512
    ft_num_neighbours: int = 256
    # sampling casuale / grid search
    seed: int = 42
    trial_id: str = ""           # popolato a runtime


def set_seed(seed: int):
    seed_everything(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_data_and_graph():
    # Carica dataset e task
    dataset = get_dataset("rel-f1", download=True)
    task = get_task("rel-f1", "driver-position", download=True)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    db = dataset.get_db()
    col_to_stype = get_stype_proposal(db)
    db2, col_to_stype2 = merge_text_columns_to_categorical(db, col_to_stype)

    data, col_stats_dict = make_pkey_fkey_graph(
        db2,
        col_to_stype_dict=col_to_stype2,
        text_embedder_cfg=None,
        cache_dir=None,
    )
    return data, col_stats_dict, task, train_table, val_table, test_table


def make_model(cfg: PretrainConfig, data, col_stats_dict, device):
    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=cfg.num_layers,
        channels=cfg.channels,
        out_channels=1,
        aggr=cfg.aggr,
        norm=cfg.norm,
    ).to(device)
    return model


def get_edge_types_from_first_batch(loader_dict) -> List[Tuple[str, str, str]]:
    et = []
    for b in loader_dict["train"]:
        et = b.edge_types
        break
    return et


def build_loaders_for(cfg: PretrainConfig, data, task, train_table, val_table, test_table,
                      batch_size: int, num_neighbours: int):
    return loader_dict_fn(
        batch_size=batch_size,
        num_neighbours=num_neighbours,
        data=data,
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table
    )


def score_config_after_pretraining(model, cfg: PretrainConfig, device, task, train_table, val_table,
                                   data, ft_batch_size: int, ft_num_neighbours: int) -> float:
    """Breve fine-tuning supervisionato per stimare la qualità del pretraining.
    Ritorna la MAE di validazione (più bassa = meglio)."""
    loss_fn = L1Loss()

    # loader per fine-tuning veloce
    loader_dict = build_loaders_for(
        cfg, data, task, train_table, val_table, task.get_table("test"),
        batch_size=ft_batch_size, num_neighbours=ft_num_neighbours
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.ft_lr,
        weight_decay=cfg.ft_weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(10, cfg.ft_epochs))

    best_val = math.inf
    for epoch in range(1, cfg.ft_epochs + 1):
        _ = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)
        val_pred = test(model, loader_dict["val"], device=device, task=task)
        val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)
        best_val = min(best_val, val_metrics["mae"])
        scheduler.step()
    return best_val


def run_single_trial(cfg: PretrainConfig, device: str = None) -> Dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # 1) dati + grafo
    data, col_stats_dict, task, train_table, val_table, test_table = build_data_and_graph()

    # 2) modello
    model = make_model(cfg, data, col_stats_dict, device)

    # 3) loader per pre-training
    loader_pt = build_loaders_for(
        cfg, data, task, train_table, val_table, test_table,
        batch_size=cfg.batch_size_pt, num_neighbours=cfg.num_neighbours_pt
    )

    edge_types = get_edge_types_from_first_batch(loader_pt)

    # 4) PRE-TRAIN VGAE (usa la tua funzione)
    model = train_vgae(
        model=model,
        loader_dict=loader_pt,
        edge_types=edge_types,
        encoder_out_dim=cfg.channels,
        entity_table=task.entity_table,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        epochs=cfg.vgae_epochs,
        device=device
    )

    # 5) Breve fine-tuning -> MAE val come punteggio della config
    val_mae = score_config_after_pretraining(
        model, cfg, device, task, train_table, val_table, data,
        ft_batch_size=cfg.ft_batch_size, ft_num_neighbours=cfg.ft_num_neighbours
    )

    result = asdict(cfg)
    result.update({
        "val_mae": float(val_mae),
    })
    return result


def grid_search(space: Dict[str, List[Any]], base_cfg: PretrainConfig, n_trials_cap: int = None) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))
    if n_trials_cap is not None:
        combos = combos[:n_trials_cap]

    results = []
    for i, values in enumerate(combos, 1):
        cfg = copy.deepcopy(base_cfg)
        for k, v in zip(keys, values):
            setattr(cfg, k, v)
        cfg.trial_id = f"grid_{i:03d}"
        print(f"\n=== Trial {cfg.trial_id} ===")
        print({k: getattr(cfg, k) for k in keys})
        try:
            res = run_single_trial(cfg)
            results.append(res)
            print(f"→ val_mae: {res['val_mae']:.4f}")
        except Exception as e:
            print(f"[WARN] Trial {cfg.trial_id} failed: {e}")
    return results


def random_search(space: Dict[str, List[Any]], base_cfg: PretrainConfig, n_trials: int = 10) -> List[Dict[str, Any]]:
    results = []
    keys = list(space.keys())
    for i in range(1, n_trials + 1):
        cfg = copy.deepcopy(base_cfg)
        for k in keys:
            setattr(cfg, k, random.choice(space[k]))
        cfg.trial_id = f"rand_{i:03d}"
        print(f"\n=== Trial {cfg.trial_id} ===")
        print({k: getattr(cfg, k) for k in keys})
        try:
            res = run_single_trial(cfg)
            results.append(res)
            print(f"→ val_mae: {res['val_mae']:.4f}")
        except Exception as e:
            print(f"[WARN] Trial {cfg.trial_id} failed: {e}")
    return results


def summarize(results: List[Dict[str, Any]]):
    if not results:
        print("Nessun risultato.")
        return
    # ordina per MAE crescente
    results = sorted(results, key=lambda r: r["val_mae"])
    best = results[0]
    print("\n========== RISULTATI ==========")
    for r in results:
        print(r)
    print("\n========== MIGLIORE ==========")
    print(best)
    print("\nSuggerimento: rilancia un addestramento finale lungo con questa config migliore.")


if __name__ == "__main__":
    base = PretrainConfig()

    # SPAZIO DI RICERCA — modifica liberamente
    search_space = {
        # VGAE
        "latent_dim": [16, 32, 64],
        "hidden_dim": [64, 128, 256],
        "vgae_epochs": [50, 80, 120],
        # Loader pre-training
        "batch_size_pt": [64, 128],
        "num_neighbours_pt": [32, 64],
        # Backbone
        "channels": [128],       
        "num_layers": [2, 3],
        # Fine-tuning veloce
        "ft_epochs": [20],        # breve per rapidità
        "ft_lr": [5e-4, 1e-3],
        "ft_weight_decay": [0.0, 1e-4],
    }

    # SCEGLI: grid search (esaustivo) o random search (più veloce)
    # results = grid_search(search_space, base_cfg=base, n_trials_cap=None)
    results = random_search(search_space, base_cfg=base, n_trials=8)

    summarize(results)
