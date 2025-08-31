# ricostruisci tuning_tried.json da tuning_log.csv
import csv, json
from pathlib import Path

HERE = Path(__file__).resolve().parent             
PROJ = HERE.parent.parent                          
TRAINING = HERE.parent 


# Candidati tipici
candidates = [
    HERE / "tuning_log.csv",                       # stessa cartella dello script
    TRAINING / "tuning_log.csv",                   # cartella training (dallo screenshot)
    PROJ / "tuning_log.csv",                       # root repo
]

# Se non trovato nei candidati, cerca dappertutto nel progetto
LOG_PATH = next((p for p in candidates if p.exists()), None)
if LOG_PATH is None:
    found = list(PROJ.rglob("tuning_log.csv"))
    if found:
        LOG_PATH = found[0]                        # prendi il primo trovato

if LOG_PATH is None:
    raise FileNotFoundError(
        "Non trovo 'tuning_log.csv'.\n"
        f"Provati: {[str(p) for p in candidates]}\n"
        f"Working dir: {Path.cwd()}\n"
        f"Script dir:  {HERE}\n"
        "Suggerimenti: 1) rinomina il file esattamente 'tuning_log.csv' "
        "2) evita duplicati tipo 'tuning_log (1).csv' 3) sposta il file in /training."
    )

# Dove salvare/leggere il JSON (scegli tu: training o stessa cartella dello script)
TRIED_PATH = TRAINING / "tuning_tried.json"
# TRIED_PATH = HERE / "tuning_tried.json"   # <-- usa questo se lo vuoi accanto allo script

print(f"[INFO] Script dir: {HERE}")
print(f"[INFO] CSV usato: {LOG_PATH}")
print(f"[INFO] JSON out : {TRIED_PATH}")

def dict_to_key(d):
    return tuple(sorted((k, str(v)) for k, v in d.items()))

tried = set()
with open(LOG_PATH) as f:
    r = csv.DictReader(f)
    for row in r:
        params = {
            "optimizer": row["optimizer"],
            "lr": float(row["lr"]),
            "weight_decay": float(row["weight_decay"]),
            "momentum": float(row["momentum"]),
            "betas": tuple(map(float, row["betas"].strip("()[]").split(","))),
            "hidden_channels": int(row["hidden_channels"]),
            "out_channels": int(row["out_channels"]),
            "dropout_p": float(row["dropout_p"]),
            "num_heads": int(row["num_heads"]),
            "num_layers": int(row["num_layers"]),
            "batch_size": int(row["batch_size"]),
            "num_neighbours": int(row["num_neighbours"]),
            "scheduler": row["scheduler"],
            "cosine_Tmax": int(row["cosine_Tmax"]),
            "plateau_patience": int(row["plateau_patience"]),
            "plateau_factor": float(row["plateau_factor"]),
        }
        tried.add(dict_to_key(params))

with open(TRIED_PATH, "w") as f:
    json.dump(sorted(list(tried)), f, indent=2)
print(f"Ricostruite {len(tried)} combinazioni in {TRIED_PATH}")
