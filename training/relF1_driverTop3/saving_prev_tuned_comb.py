# ricostruisci tuning_tried.json da tuning_log.csv
import csv, json
from pathlib import Path

LOG_PATH = "tuning_log.csv"
TRIED_PATH = "tuning_tried.json"

def dict_to_key(d):  # come nel tuo script
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
