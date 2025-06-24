import os
import torch
import itertools
import json
from torch_geometric.seed import seed_everything
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph

from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.EarlyStopping import EarlyStopping
from model.Atomic_routes import AtomicRouteModel
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from VGAE.Utils_VGAE import train_vgae
from utils.EarlyStopping import EarlyStopping
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
from torch.optim.lr_scheduler import CosineAnnealingLR


def run_experiment(
    channels=128,
    dropout=0.0,
    learning_rate=0.0005,
    save_model=False,
    run_name="default"
):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset e task ---
    dataset = get_dataset("rel-f1", download=True)
    task = get_task("rel-f1", "driver-position", download=True)
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    # --- DB & feature engineering ---
    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)
    db, col_to_stype_dict = merge_text_columns_to_categorical(db, col_to_stype_dict)

    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=None,
        cache_dir=None
    )

    # --- Model ---
    model = AtomicRouteModel(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=2,
        channels=channels,
        out_channels=1,
        aggr="max",
        norm="batch_norm",
        shallow_list=[],
        id_awareness=False,
        predictor_n_layers=2,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    early_stopping = EarlyStopping(
        patience=30,
        delta=0.0,
        verbose=False,
        path=f"best_model_{run_name}.pt"
    )

    loader_dict = loader_dict_fn(
        batch_size=512,
        num_neighbours=256,
        data=data,
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table
    )

    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False

    # --- Training loop ---
    best_val_metric = float("inf")
    best_test_metric = float("inf")
    state_dict = None

    for epoch in range(1, 101):
        train_loss = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)
        train_mae_preciso = evaluate_on_full_train(model, loader_dict["train"], device=device, task=task)

        val_pred = test(model, loader_dict["val"], device=device, task=task)
        val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)

        test_pred = test(model, loader_dict["test"], device=device, task=task)
        test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)

        scheduler.step(val_metrics[tune_metric])

        if val_metrics[tune_metric] < best_val_metric:
            best_val_metric = val_metrics[tune_metric]
            best_test_metric = test_metrics[tune_metric]
            if save_model:
                state_dict = copy.deepcopy(model.state_dict())
                torch.save(state_dict, f"best_model_{run_name}.pt")

        early_stopping(val_metrics[tune_metric], model)
        if early_stopping.early_stop:
            break

    return best_val_metric, best_test_metric





#Griglia di iperparametri
grid = {
    "channels": [64, 128],
    "dropout": [0.0, 0.2, 0.5],
    "lr": [5e-4, 1e-3]
}

# Genera tutte le combinazioni
grid_list = list(itertools.product(grid["channels"], grid["dropout"], grid["lr"]))

# Per salvare i risultati
results = []
best_val = float("inf")
best_config = None

for i, (channels, dropout, lr) in enumerate(grid_list):
    print(f"\nRun {i+1}/{len(grid_list)} | channels={channels}, dropout={dropout}, lr={lr}")
    
    # Run esperimento
    val_mae, test_mae = run_experiment(
        channels=channels,
        dropout=dropout,
        learning_rate=lr,
        save_model=True,        # salva solo se migliora
        run_name=f"tune_run_{i}"
    )

    results.append({
        "channels": channels,
        "dropout": dropout,
        "lr": lr,
        "val_mae": val_mae,
        "test_mae": test_mae
    })

    # ✅ Teniamo traccia del migliore
    if val_mae < best_val:
        best_val = val_mae
        best_config = results[-1]

# 💾 Salva tutti i risultati
with open("tune_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n🏆 Best config:")
print(best_config)
