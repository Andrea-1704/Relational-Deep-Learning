import os, copy, math, time, json
import torch
from torch.nn import L1Loss
from torch_geometric.seed import seed_everything

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph

import sys
import os
sys.path.append(os.path.abspath("."))


from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.utils import evaluate_performance, test, train
from utils.EarlyStopping import EarlyStopping
from model.XMetaPath2 import XMetaPath2

# --------------------------------------------------------------------------------------
# Config generali
# --------------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)
torch.backends.cudnn.benchmark = True

DATASET = "rel-f1"
TASKSET = "driver-position"   # stesso task del tuo file
TUNE_METRIC = "mae"           # MAE: più basso è meglio
HIGHER_IS_BETTER = False

MAX_EPOCHS = 12               # poche epoche per confronto rapido
PATIENCE = 3
MIN_DELTA = 0.0

# Griglia RISTRETTA ma sensata (8 run)
GRID = [
    # forti “default” stile RelBench (veloci + stabili su node regression)
    {"lr": 1e-3,  "wd": 0.0,   "batch": 512, "neighbors": 128, "hidden": 128},
    {"lr": 5e-4,  "wd": 0.0,   "batch": 512, "neighbors": 128, "hidden": 128},
    {"lr": 1e-3,  "wd": 1e-4,  "batch": 512, "neighbors": 128, "hidden": 128},
    {"lr": 5e-4,  "wd": 1e-4,  "batch": 512, "neighbors": 128, "hidden": 128},

    # piccoli cambi mirati
    {"lr": 1e-3,  "wd": 0.0,   "batch": 512, "neighbors": 256, "hidden": 128},
    {"lr": 5e-4,  "wd": 0.0,   "batch": 512, "neighbors": 256, "hidden": 128},
    {"lr": 1e-3,  "wd": 0.0,   "batch": 256, "neighbors": 128, "hidden": 128},
    {"lr": 1e-3,  "wd": 0.0,   "batch": 512, "neighbors": 128, "hidden": 64},
]

# --------------------------------------------------------------------------------------
# Setup dati e grafo (uguale alla tua pipeline)
# --------------------------------------------------------------------------------------
print("Loading dataset/task...")
dataset = get_dataset(DATASET, download=True)
task = get_task(DATASET, TASKSET, download=True)

train_table = task.get_table("train")
val_table   = task.get_table("val")
test_table  = task.get_table("test")

db = dataset.get_db()
col_to_stype_dict = get_stype_proposal(db)
# come nel tuo script: testo a categorico per velocità (puoi rimettere un text_embedder più avanti)
db_cat, col_to_stype_cat = merge_text_columns_to_categorical(db, col_to_stype_dict)

print("Building PK/FK graph...")
data, col_stats_dict = make_pkey_fkey_graph(
    db_cat,
    col_to_stype_dict=col_to_stype_cat,
    text_embedder_cfg=None,   # veloce: nessun embed testuale
    cache_dir=None,
)
node_type = "drivers"

# === target come nel tuo file ===
graph_driver_ids = db_cat.table_dict["drivers"].df["driverId"].to_numpy()
id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}

train_df = train_table.df
driver_labels = train_df[task.target_col].to_numpy()
driver_ids    = train_df["driverId"].to_numpy()

target_vector = torch.full((len(graph_driver_ids),), float("nan"))
for i, driver_id in enumerate(driver_ids):
    if driver_id in id_to_idx:
        target_vector[id_to_idx[driver_id]] = driver_labels[i]

data['drivers'].y = target_vector
data['drivers'].train_mask = ~torch.isnan(target_vector)
y_full = data['drivers'].y.float()

# --------------------------------------------------------------------------------------
# Metapath (come nel tuo file) + canonicalizzazione
# --------------------------------------------------------------------------------------
def flip_rel(rel_name: str) -> str:
    return rel_name[4:] if rel_name.startswith("rev_") else f"rev_{rel_name}"

def to_canonical(mp_outward, expect="drivers"):
    mp = [(dst, flip_rel(rel), src) for (src, rel, dst) in mp_outward[::-1]]
    assert mp[-1][2] == expect, f"Expected {expect}, got {mp[-1][2]}"
    return tuple(mp)

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
canonical = [to_canonical(mp.copy(), expect=node_type) for mp in metapaths]

# --------------------------------------------------------------------------------------
# Runner di una singola configurazione
# --------------------------------------------------------------------------------------
def run_one(cfg):
    seed_everything(42)  # per confronti equi

    # loader (usa le tue API)
    loaders = loader_dict_fn(
        batch_size=cfg["batch"],
        num_neighbours=cfg["neighbors"],
        data=data,
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table,
    )

    # modello
    model = XMetaPath2(
        data=data,
        col_stats_dict=col_stats_dict,
        metapaths=canonical,
        hidden_channels=cfg["hidden"],
        out_channels=cfg["hidden"],
        final_out_channels=1,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = L1Loss()

    early = EarlyStopping(
        patience=PATIENCE,
        delta=MIN_DELTA,
        verbose=False,
        higher_is_better=HIGHER_IS_BETTER,
        path=None,   # non salviamo su disco ad ogni run, teniamo in RAM lo state_dict
    )

    best_state = None
    best_val = math.inf if not HIGHER_IS_BETTER else -math.inf
    hist = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train(model, optimizer, loader_dict=loaders, device=DEVICE, task=task, loss_fn=loss_fn)

        # predizioni
        train_pred = test(model, loaders["train"], device=DEVICE, task=task)
        val_pred   = test(model, loaders["val"],   device=DEVICE, task=task)
        test_pred  = test(model, loaders["test"],  device=DEVICE, task=task)

        # metriche
        train_m = evaluate_performance(train_pred, train_table, task.metrics, task=task)
        val_m   = evaluate_performance(val_pred,   val_table,   task.metrics, task=task)
        test_m  = evaluate_performance(test_pred,  test_table,  task.metrics, task=task)

        val_score = val_m[TUNE_METRIC]
        hist.append({"epoch": epoch, "train_mae": train_m[TUNE_METRIC], "val_mae": val_score, "test_mae": test_m[TUNE_METRIC]})

        # track best
        improved = (val_score < best_val) if not HIGHER_IS_BETTER else (val_score > best_val)
        if improved:
            best_val = val_score
            best_state = copy.deepcopy(model.state_dict())

        # early stopping
        early(val_score, model)
        if getattr(early, "early_stop", False):
            break

    # valuta il best state su val/test
    if best_state is not None:
        model.load_state_dict(best_state)
        with torch.no_grad():
            val_pred  = test(model, loaders["val"],  device=DEVICE, task=task)
            test_pred = test(model, loaders["test"], device=DEVICE, task=task)
            val_m  = evaluate_performance(val_pred,  val_table,  task.metrics, task=task)
            test_m = evaluate_performance(test_pred, test_table, task.metrics, task=task)
    else:
        # fallback: ultimo
        with torch.no_grad():
            val_pred  = test(model, loaders["val"],  device=DEVICE, task=task)
            test_pred = test(model, loaders["test"], device=DEVICE, task=task)
            val_m  = evaluate_performance(val_pred,  val_table,  task.metrics, task=task)
            test_m = evaluate_performance(test_pred, test_table, task.metrics, task=task)

    return {
        "cfg": cfg,
        "best_val_mae": float(val_m[TUNE_METRIC]),
        "test_mae_at_best_val": float(test_m[TUNE_METRIC]),
        "history": hist,
    }

# --------------------------------------------------------------------------------------
# Loop di tuning
# --------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    results = []
    for i, cfg in enumerate(GRID, 1):
        print(f"\n=== Run {i}/{len(GRID)} ===  cfg={cfg}")
        try:
            out = run_one(cfg)
            print(f" -> val MAE: {out['best_val_mae']:.3f} | test MAE (at best val): {out['test_mae_at_best_val']:.3f}")
            results.append(out)
        except RuntimeError as e:
            # gestisci eventuali OOM riducendo batch
            if "out of memory" in str(e).lower() and cfg["batch"] > 256:
                print("OOM: ritento con batch=256")
                cfg2 = dict(cfg); cfg2["batch"] = 256
                out = run_one(cfg2)
                print(f" -> val MAE: {out['best_val_mae']:.3f} | test MAE (at best val): {out['test_mae_at_best_val']:.3f}")
                results.append(out)
            else:
                print("Errore:", e)

    # ordina per val MAE crescente
    results = sorted(results, key=lambda x: x["best_val_mae"])
    print("\n===== RANKING (miglior VAL MAE -> peggiore) =====")
    for r in results:
        c = r["cfg"]
        print(f"VAL {r['best_val_mae']:.3f} | TEST {r['test_mae_at_best_val']:.3f} || lr={c['lr']} wd={c['wd']} batch={c['batch']} neigh={c['neighbors']} hidden={c['hidden']}")

    # salva json riassunto
    os.makedirs("tuning_logs", exist_ok=True)
    with open(os.path.join("tuning_logs", "results_studyAdverse_XMetaPath2.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTempo totale: {time.time()-t0:.1f}s")
    if results:
        print("\nMigliore configurazione:")
        print(results[0]["cfg"])

if __name__ == "__main__":
    main()
