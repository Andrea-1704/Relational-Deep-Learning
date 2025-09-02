"""
Tuning leggero: cambia SOLO gli iperparametri.
Chiamate/estrazione dati identiche al file originale:
- merge_text_columns_to_categorical
- make_pkey_fkey_graph
- costruzione y con position/driverId
- RL warmup + ricerca finale
- train / test / evaluate_performance

Nota: lo split 'test' in rel-f1/driver-position può non avere 'position'.
Per evitare crash, la chiamata a evaluate_performance(...) sul test è
inserita in un try/except KeyError: se manca la colonna, stampiamo 'n/a'.
"""

import os, math, copy, time, json, sys
import torch
from torch.nn import L1Loss
from torch_geometric.seed import seed_everything

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph

# === import IDENTICI al tuo file ===
sys.path.append(os.path.abspath("."))
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.EarlyStopping import EarlyStopping
from model.XMetaPath2 import XMetaPath2
from utils.utils import evaluate_performance, test, train
from utils.XMetapath_utils.XMetaPath_extension4 import (
    RLAgent, warmup_rl_agent, final_metapath_search_with_rl
)

# ------------------------ setup ------------------------
seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tune_metric = "mae"           # MAE: minore è meglio
higher_is_better = False
loss_fn = L1Loss()

# griglia piccola e mirata (6 run)
GRID = [
    {"lr": 1e-3,  "wd": 0.0,  "batch": 512, "neighbors": 128, "hidden": 128},
    {"lr": 5e-4,  "wd": 0.0,  "batch": 512, "neighbors": 128, "hidden": 128},
    {"lr": 1e-3,  "wd": 1e-4, "batch": 512, "neighbors": 128, "hidden": 128},
    {"lr": 1e-3,  "wd": 0.0,  "batch": 256, "neighbors": 128, "hidden": 128},
    {"lr": 1e-3,  "wd": 0.0,  "batch": 512, "neighbors": 256, "hidden": 128},
    {"lr": 1e-3,  "wd": 0.0,  "batch": 512, "neighbors": 128, "hidden": 64},
]

MAX_EPOCHS = 15
PATIENCE = 3
MIN_DELTA = 0.0

# ------------------------ dati + grafo (IDENTICO) ------------------------
print("Loading dataset/task...")
dataset = get_dataset("rel-f1", download=True)
task = get_task("rel-f1", "driver-position", download=True)

train_table = task.get_table("train")
val_table   = task.get_table("val")
test_table  = task.get_table("test")

db = dataset.get_db()
col_to_stype_dict = get_stype_proposal(db)
db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)

print("Building PK/FK graph...")
data, col_stats_dict = make_pkey_fkey_graph(
    db_nuovo,
    col_to_stype_dict=col_to_stype_dict_nuovo,
    text_embedder_cfg=None,
    cache_dir=None,
)
node_type = "drivers"

# === costruzione label/maschera IDENTICA al tuo file ===
graph_driver_ids = db_nuovo.table_dict["drivers"].df["driverId"].to_numpy()
id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}

train_df = train_table.df
driver_labels = train_df["position"].to_numpy()
driver_ids    = train_df["driverId"].to_numpy()

target_vector = torch.full((len(graph_driver_ids),), float("nan"))
for i, driver_id in enumerate(driver_ids):
    if driver_id in id_to_idx:
        target_vector[id_to_idx[driver_id]] = driver_labels[i]

data['drivers'].y = target_vector
data['drivers'].train_mask = ~torch.isnan(target_vector)
y_full = data['drivers'].y.float()
train_mask_full = data['drivers'].train_mask
print(f"[info] y full has {int(torch.isfinite(y_full).sum())}/{y_full.numel()} finite labels")

# ------------------------ util canonico (IDENTICO) ------------------------
def flip_rel(rel_name: str) -> str:
    return rel_name[4:] if rel_name.startswith("rev_") else f"rev_{rel_name}"

def to_canonical(mp_outward):
    mp = [(dst, flip_rel(rel), src) for (src, rel, dst) in mp_outward[::-1]]
    assert mp[-1][2] == node_type, f"Expected {node_type}, got {mp[-1][2]}"
    return tuple(mp)

# ------------------------ RL: warmup + ricerca finale (UNA VOLTA) ------------------------
print("\n[RL] Learning metapaths once (will be reused across runs)...")
base_loader_for_rl = loader_dict_fn(
    batch_size=512,              # valori “di servizio” per la fase RL
    num_neighbours=256,
    data=data,
    task=task,
    train_table=train_table,
    val_table=val_table,
    test_table=test_table
)

# agent = RLAgent(tau=1.0, alpha=0.5)
# agent.best_score_by_path_global.clear()

# warmup_rl_agent(
#     agent=agent,
#     data=data,
#     loader_dict=base_loader_for_rl,
#     task=task,
#     loss_fn=loss_fn,
#     tune_metric=tune_metric,
#     higher_is_better=higher_is_better,
#     train_mask=train_mask_full,
#     node_type='drivers',
#     col_stats_dict=col_stats_dict,
#     num_episodes=3,
#     L_max=4,
#     epochs=3
# )

# K = 3
# agent.tau = 0.3
# agent.alpha = 0.2

# metapaths, metapath_count = final_metapath_search_with_rl(
#     agent=agent,
#     data=data,
#     loader_dict=base_loader_for_rl,
#     task=task,
#     loss_fn=loss_fn,
#     tune_metric=tune_metric,
#     higher_is_better=higher_is_better,
#     train_mask=train_mask_full,
#     node_type='drivers',
#     col_stats_dict=col_stats_dict,
#     L_max=4,
#     epochs=10,
#     number_of_metapaths=K
# )

# print(f"[RL] Final metapaths: {metapaths}")

metapaths = [
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

canonical = [to_canonical(mp.copy()) for mp in metapaths]

# ------------------------ runner di UNA configurazione ------------------------
def run_one(cfg):
    # loader (API IDENTICA)
    loader_dict = loader_dict_fn(
        batch_size=cfg["batch"],
        num_neighbours=cfg["neighbors"],
        data=data,
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table
    )

    # modello (IDENTICO; variano solo i numeri)
    model = XMetaPath2(
        data=data,
        col_stats_dict=col_stats_dict,
        metapaths=canonical,
        hidden_channels=cfg["hidden"],
        out_channels=cfg["hidden"],
        final_out_channels=1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    # early = EarlyStopping(
    #     patience=PATIENCE,
    #     delta=MIN_DELTA,
    #     verbose=False,
    #     higher_is_better=higher_is_better,
    #     path=None,
    # )

    best_state = None
    best_val = math.inf if not higher_is_better else -math.inf
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

        train_pred = test(model, loader_dict["train"], device=device, task=task)
        val_pred   = test(model, loader_dict["val"],   device=device, task=task)
        test_pred  = test(model, loader_dict["test"],  device=device, task=task)

        # === chiamate IDENTICHE su train/val ===
        train_m = evaluate_performance(train_pred, train_table, task.metrics, task=task)
        val_m   = evaluate_performance(val_pred,   val_table,   task.metrics, task=task)

        # === stessa chiamata sul test, ma protetta (evita KeyError se 'position' manca) ===
        try:
            test_m = evaluate_performance(test_pred, test_table, task.metrics, task=task)
            test_val = float(test_m[tune_metric])
        except KeyError:
            test_m = {m.__name__: float("nan") for m in task.metrics}
            test_val = float("nan")

        val_score = val_m[tune_metric]
        improved = (val_score < best_val) if not higher_is_better else (val_score > best_val)
        if improved:
            best_val = val_score
            best_state = copy.deepcopy(model.state_dict())

        history.append({
            "epoch": epoch,
            "train_mae": float(train_m[tune_metric]),
            "val_mae": float(val_m[tune_metric]),
            "test_mae": test_val,
            "lr": optimizer.param_groups[0]["lr"],
        })

        #early(val_score, model)
        # if getattr(early, "early_stop", False):
        #     break

    # valuta al best sulla val (con la stessa protezione per il test)
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        val_pred  = test(model, loader_dict["val"],  device=device, task=task)
        test_pred = test(model, loader_dict["test"], device=device, task=task)
        val_m  = evaluate_performance(val_pred,  val_table,  task.metrics, task=task)
        try:
            test_m = evaluate_performance(test_pred, test_table, task.metrics, task=task)
        except KeyError:
            test_m = {m.__name__: float("nan") for m in task.metrics}

    result = {
        "cfg": cfg,
        "best_val_mae": float(val_m[tune_metric]),
        "test_mae_at_best_val": float(test_m[tune_metric]) if not math.isnan(test_m[tune_metric]) else float("nan"),
        "history": history,
    }
    print(f" -> cfg={cfg} | VAL {result['best_val_mae']:.3f} | TEST {('n/a' if math.isnan(result['test_mae_at_best_val']) else f'{result['test_mae_at_best_val']:.3f}')}")

    return result

# ------------------------ loop di tuning ------------------------
def main():
    t0 = time.time()
    results = []
    for i, cfg in enumerate(GRID, 1):
        print(f"\n=== Run {i}/{len(GRID)} ===")
        try:
            out = run_one(cfg)
            results.append(out)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and cfg["batch"] > 256:
                print("OOM: retry with batch=256")
                cfg2 = dict(cfg); cfg2["batch"] = 256
                out = run_one(cfg2)
                results.append(out)
            else:
                print("Errore:", e)

    results = sorted(results, key=lambda r: r["best_val_mae"])
    print("\n===== RANKING (miglior VAL MAE → peggiore) =====")
    for r in results:
        c = r["cfg"]
        tv = r["test_mae_at_best_val"]
        tvs = "n/a" if (tv is None or (isinstance(tv, float) and math.isnan(tv))) else f"{tv:.3f}"
        print(f"VAL {r['best_val_mae']:.3f} | TEST {tvs} || "
              f"lr={c['lr']} wd={c['wd']} batch={c['batch']} neigh={c['neighbors']} hidden={c['hidden']}")

    os.makedirs("tuning_logs", exist_ok=True)
    with open(os.path.join("tuning_logs", "results_XMetaPath2.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTempo totale: {time.time()-t0:.1f}s")
    if results:
        print("\nMigliore configurazione:")
        print(results[0]["cfg"])

if __name__ == "__main__":
    main()
