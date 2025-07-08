import torch
import numpy as np
import math
from tqdm import tqdm
import torch_geometric
import torch_frame
from torch_geometric.seed import seed_everything
from relbench.modeling.utils import get_stype_proposal
from collections import defaultdict
import requests
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph
import copy
from typing import Any, Dict, List
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from torch_geometric.loader import NeighborLoader
import pyg_lib
from torch.nn import ModuleDict
import torch.nn.functional as F
from torch import nn
import random
from matplotlib import pyplot as plt
from itertools import product
import torch
import numpy as np
import copy
import pandas as pd
from utils.EarlyStopping import EarlyStopping


def train(model, optimizer, loader_dict, device, task, loss_fn) -> float:
    model.train()

    loss_accum = count_accum = 0
    for batch in loader_dict["train"]:
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,
        )
        #pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred = pred.view(-1) if pred.dim() == 2 and pred.size(1) == 1 else pred
        loss = loss_fn(pred.float(), batch[task.entity_table].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

    return loss_accum / count_accum


@torch.no_grad()
def test(model, loader: NeighborLoader, device, task) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.dim() == 2 and pred.size(1) == 1 else pred

        #pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()



def tune_hyperparameters(
    data,
    train_table,
    val_table,
    loader_dict_fn,
    model_class,
    evaluate_fn,
    train_fn,
    test_fn,
    device,
    col_stats_dict,
    task,
    tune_metric='mae'
):
    param_grid = {
        'lr': [0.0005, 0.001, 0.005],
        'weight_decay': [0, 1e-4],
        'channels': [128, 256],
        'num_layers': [1, 2, 3],
        'aggr': ['mean', 'sum', 'max'],
        'norm': ["batch_norm", "layer_norm"],
        'batch_size': [512, 1024],
        'num_neighbours': [128, 256]
    }

    combinations = list(product(
        param_grid['lr'],
        param_grid['weight_decay'],
        param_grid['channels'],
        param_grid['num_layers'],
        param_grid['aggr'],
        param_grid['norm'],
        param_grid['batch_size'],
        param_grid['num_neighbours']
    ))

    print(f"Testiamo {len(combinations)} combinazioni di iperparametri...\n")

    best_score = float('inf')
    best_config = None
    results = []

    for i, (lr, wd, ch, nl, aggr, norm, batch_size, num_neighbours) in enumerate(combinations):
        print(f"\nRun {i+1}/{len(combinations)}")
        print(f"Params: lr={lr}, wd={wd}, ch={ch}, nl={nl}, aggr={aggr}, norm={norm}, batch_size={batch_size}, num_neighbours={num_neighbours}")

        loader_dict = loader_dict_fn(batch_size=batch_size, num_neighbours=num_neighbours)

        model = model_class(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=nl,
            channels=ch,
            out_channels=1,
            aggr=aggr,
            norm=norm,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=10, delta=0.0, verbose=False)

        for epoch in range(1, 101):
            train_fn(model, optimizer, loader_dict)
            val_pred = test_fn(model, loader_dict["val"])
            val_metrics = evaluate_fn(val_pred, val_table, task.metrics)
            val_score = val_metrics[tune_metric]

            scheduler.step(val_score)
            early_stopping(val_score, model)

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        model.load_state_dict(torch.load(early_stopping.path))
        final_pred = test_fn(model, loader_dict["val"])
        final_metrics = evaluate_fn(final_pred, val_table, task.metrics)
        final_score = final_metrics[tune_metric]
        results.append((final_score, lr, wd, ch, nl, aggr, norm, batch_size, num_neighbours))



        print(f"Final validation {tune_metric.upper()}: {final_score:.3f}")

        if final_score < best_score:
            best_score = final_score
            best_config = {
                'lr': lr, 'weight_decay': wd, 'channels': ch,
                'num_layers': nl, 'aggr': aggr, 'norm': norm,
                'batch_size': batch_size, 'num_neighbours': num_neighbours,
                'val_score': final_score
            }
            best_model = copy.deepcopy(model.state_dict())

    print("\nBest hyperparameter configuration:")
    for k, v in best_config.items():
        print(f"{k}: {v}")

    # === Analisi importanza parametri ===
    print("\nAnalisi importanza parametri (varianza media delle performance):")

    df = pd.DataFrame(results, columns=[
        'val_score', 'lr', 'weight_decay', 'channels',
        'num_layers', 'aggr', 'norm', 'batch_size', 'num_neighbours'
    ])

    importance = {}
    for col in df.columns[1:]:  # escludi 'val_score'
        grouped = df.groupby(col)['val_score']
        var = grouped.var()
        importance[col] = var.mean()

    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for param, var in sorted_imp:
        print(f"- {param:15s} → varianza media: {var:.5f}")

    print("\nParametri con varianza più alta sono più critici: cambiano le prestazioni quando li modifichi.\n")

    # Crea grafico a barre sull'importanza dei parametri
    params, variances = zip(*sorted_imp)
    plt.figure(figsize=(10, 6))
    plt.barh(params, variances)
    plt.gca().invert_yaxis()
    plt.xlabel("Varianza media del validation score")
    plt.title("Importanza degli iperparametri")
    plt.grid(axis='x')
    plt.tight_layout()

    # Salva su file
    plt.savefig("hyperparameter_importance.png")
    plt.close()

    return best_config, results


@torch.no_grad()
def evaluate_on_full_train(model, loader, device, task) -> float:
    model.eval()
    pred_list, target_list = [], []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch, task.entity_table)
        #pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred = pred.view(-1) if pred.dim() == 2 and pred.size(1) == 1 else pred
        pred_list.append(pred.cpu())
        target_list.append(batch[task.entity_table].y.cpu())

    pred_all = torch.cat(pred_list, dim=0).numpy()
    target_all = torch.cat(target_list, dim=0).numpy()

    mae = np.mean(np.abs(pred_all - target_all))
    return mae


def rmse(true, pred):
    """Calculate the Root Mean Squared Error (RMSE)."""
    return np.sqrt(np.mean((true - pred)**2))


import numpy as np

def evaluate_on_train_during_training(model, loader_dict, device, task) -> float:
    model.eval()
    pred_list, target_list = [], []

    for batch in loader_dict["train"]:
        batch = batch.to(device)
        pred = model(batch, task.entity_table)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
        target_list.append(batch[task.entity_table].y.detach().cpu())

    pred_all = torch.cat(pred_list, dim=0).numpy()
    target_all = torch.cat(target_list, dim=0).numpy()

    mae = np.mean(np.abs(pred_all - target_all))
    return mae


def evaluate_performance(pred: np.ndarray, target_table, metrics, task) -> dict:
    """Custom evaluation function to replace task.evaluate."""
    target = target_table.df[task.target_col].to_numpy()

    if len(pred) != len(target):
        raise ValueError(
            f"The length of pred and target must be the same (got "
            f"{len(pred)} and {len(target)}, respectively)."
        )

    results = {}
    for metric_fn in metrics:
        if metric_fn.__name__ == "rmse":
            results["rmse"] = np.sqrt(np.mean((target - pred)**2))
        else:
            results[metric_fn.__name__] = metric_fn(target, pred)

    return results


@torch.no_grad()
def alignment_check(loader: NeighborLoader, expected_node_ids: torch.Tensor, device, task) -> None:
    node_id_list = []

    for batch in loader:
        batch = batch.to(device)

        node_id_list.append(batch[task.entity_table].n_id.cpu())

    actual_node_ids = torch.cat(node_id_list, dim=0)

    assert len(actual_node_ids) == len(expected_node_ids), "Mismatch nella lunghezza"

    if not torch.equal(actual_node_ids, expected_node_ids):
        raise ValueError("Ordine dei nodi predetti diverso da val_table!")

    return