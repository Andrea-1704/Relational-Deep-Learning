"""
This function is used for benchmarking the model against the others and improve
the model performance considering the same metapath.

These metapath were found by extension 4.
"""
# --- TorchFrame + RelBench compatibility shim (put this at the VERY TOP) ---
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # less TF noise

import sys, types, importlib
import torch_frame as _tf

# 1) Expose stype labels at top-level if missing
for _name in ("multicategorical", "categorical", "continuous", "timestamp", "text"):
    if not hasattr(_tf, _name):
        setattr(_tf, _name, _name)  # simple string labels are fine for RelBench

# 2) Provide a torch_frame.stype module so "from torch_frame import stype" works
if not hasattr(_tf, "stype"):
    _stype_mod = types.ModuleType("torch_frame.stype")
    for _name in ("multicategorical", "categorical", "continuous", "timestamp", "text"):
        setattr(_stype_mod, _name, getattr(_tf, _name))
    _tf.stype = _stype_mod
    sys.modules["torch_frame.stype"] = _stype_mod

# 3) Provide torch_frame.utils.infer_df_stype if absent (lightweight heuristic)
try:
    from torch_frame.utils import infer_df_stype as _unused
except Exception:
    import pandas as pd
    _tf_utils = importlib.import_module("torch_frame.utils")

    def infer_df_stype(df):
        out = {}
        for col in df.columns:
            s = df[col]
            try:
                if s.apply(lambda x: isinstance(x, (list, tuple, set))).any():
                    out[col] = _tf.multicategorical
                    continue
            except Exception:
                pass
            if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_bool_dtype(s):
                out[col] = _tf.categorical
            elif pd.api.types.is_datetime64_any_dtype(s):
                out[col] = _tf.timestamp
            elif pd.api.types.is_numeric_dtype(s):
                out[col] = _tf.continuous
            else:
                out[col] = _tf.text
        return out

    setattr(_tf_utils, "infer_df_stype", infer_df_stype)

# 4) Inject a minimal torch_frame.typing module to break circular imports
#    Needed symbols: IndexSelectType, TensorData, TextTokenizationOutputs
if "torch_frame.typing" not in sys.modules:
    tf_typing = types.ModuleType("torch_frame.typing")
    from typing import Any, NamedTuple
    class TextTokenizationOutputs(NamedTuple):
        input_ids: Any
        attention_mask: Any
    IndexSelectType = Any
    TensorData = Any

    setattr(tf_typing, "TextTokenizationOutputs", TextTokenizationOutputs)
    setattr(tf_typing, "IndexSelectType", IndexSelectType)
    setattr(tf_typing, "TensorData", TensorData)

    sys.modules["torch_frame.typing"] = tf_typing

# --- Now it's safe to import RelBench/TorchFrame stuff that assumed the old API ---

import torch
import torch.nn as nn
import os
from relbench.datasets import get_dataset
from relbench.tasks import get_task
import math
from torch_geometric.seed import seed_everything

# Ora è sicuro importare utility RelBench che aspettano torch_frame.stype
from relbench.modeling.utils import get_stype_proposal
#problem:
# from relbench.modeling.utils import get_stype_proposal
# from relbench.modeling.graph import make_pkey_fkey_graph
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import math
import torch.nn as nn
from torch.nn import L1Loss

import sys
import os
sys.path.append(os.path.abspath("."))

from data_new import loader_dict_fn, merge_text_columns_to_categorical
from utils.EarlyStopping import EarlyStopping
from model.XMetaPath2 import XMetaPath2
from utils.utils import evaluate_performance, test, train
from utils.XMetapath_utils.XMetaPath_extension4 import RLAgent, warmup_rl_agent, final_metapath_search_with_rl


from typing import Dict, Tuple
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData

# ---------- 2.1) Stima stype (senza TorchFrame) ----------
def infer_stype_from_db(db) -> Dict[Tuple[str, str], str]:
    """
    Restituisce un dizionario {(table, col) -> stype} con etichette: 
    'categorical' | 'multicategorical' | 'continuous' | 'timestamp' | 'text'.
    """
    out = {}
    for tname, df in db.tables.items():
        for col in df.columns:
            s = df[col]
            # multicategorical se almeno una cella è lista/insieme/tupla
            try:
                if s.apply(lambda x: isinstance(x, (list, tuple, set))).any():
                    out[(tname, col)] = "multicategorical"; continue
            except Exception:
                pass
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_categorical_dtype(s):
                out[(tname, col)] = "categorical"
            elif pd.api.types.is_datetime64_any_dtype(s):
                out[(tname, col)] = "timestamp"
            elif pd.api.types.is_numeric_dtype(s):
                out[(tname, col)] = "continuous"
            else:
                out[(tname, col)] = "text"
    return out

# ---------- 2.2) Utility per chiavi primarie/esterne ----------
def guess_primary_key(df: pd.DataFrame) -> str:
    # euristica robusta: prima colonna *_id o *Id con valori unici
    candidates = [c for c in df.columns if c.lower().endswith("id")]
    for c in candidates:
        if df[c].is_unique:
            return c
    # fallback: prima colonna unica
    for c in df.columns:
        if df[c].is_unique:
            return c
    # se proprio nulla, usa la prima colonna
    return df.columns[0]

def find_foreign_keys(db, table: str, pkeys: Dict[str, str]) -> Dict[str, str]:
    """
    Cerca colonne di 'table' che referenziano pkey di altre tabelle.
    Ritorna dict {fk_col -> referenced_table}
    Heuristica: nome colonna uguale alla pkey dell’altra tabella (es. raceId) 
    oppure termina con lo stesso suffisso della pkey (case-insensitive).
    """
    df = db.tables[table]
    res = {}
    for other_table, pk in pkeys.items():
        if other_table == table:
            continue
        # match diretto (es. 'raceId')
        if pk in df.columns:
            res[pk] = other_table
            continue
        # match per suffisso (es. 'constructorId' ↔ '...constructorid')
        low_cols = {c.lower(): c for c in df.columns}
        if pk.lower() in low_cols:
            res[low_cols[pk.lower()]] = other_table
            continue
    return res

# ---------- 2.3) Encoding feature per nodo ----------
class _CatEncoder:
    def __init__(self):
        self.maps = {}  # {(tname, col) -> {val -> idx}}

    def fit_transform(self, df: pd.DataFrame, tname: str, cols):
        mats = []
        for col in cols:
            vals = df[col].astype("object").fillna("__NA__").values
            key = (tname, col)
            if key not in self.maps:
                uniq = pd.Index(vals).unique().tolist()
                self.maps[key] = {v: i for i, v in enumerate(uniq)}
            idx = np.vectorize(self.maps[key].get)(vals)
            mats.append(torch.from_numpy(idx).long().unsqueeze(1))
        if mats:
            return torch.cat(mats, dim=1)
        return None

def standardize_numeric(x: torch.Tensor, eps=1e-6):
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True)
    return (x - mu) / (sd + eps), {"mean": mu, "std": sd}

# ---------- 2.4) Costruzione HeteroData pkey↔fkey ----------
def build_heterodata_from_db(db, col_to_stype, embed_dim=32, standardize=True):
    """
    Crea un HeteroData PyG con:
      - nodi per tabella (indice = pkey reindexato 0..N-1)
      - edges per ogni fkey rilevata: (table, 'f2p_<fk>', ref_table)
        + l'edge inverso 'rev_f2p_<fk>'
      - feature semplici:
         * continue -> z-score tensor
         * categoriche -> embedding indices (concat) + EmbeddingBag per tipo di colonna
         * timestamp/text -> ignorate qui (puoi gestirle a parte)
    Ritorna (data, col_stats_dict).
    """
    data = HeteroData()
    pkeys = {t: guess_primary_key(df) for t, df in db.tables.items()}
    # map per node ids (valore pkey -> indice 0..N-1)
    idmaps = {}
    # per embeddings categoriche (costruiamo col->embedding per tipo di tabella)
    cat_enc = _CatEncoder()
    cat_embed_tables = {}
    col_stats_dict = {}

    # 1) crea i nodi con feature
    for tname, df in db.tables.items():
        pk = pkeys[tname]
        # indicizzazione nodi
        pvals = df[pk].values
        uniq_vals = pd.Index(pvals).unique().tolist()
        idmaps[tname] = {v: i for i, v in enumerate(uniq_vals)}
        # mask per order: all rows in df corrispondono ad un nodo? 
        # Se la tabella non è "entity" pura (es. facts), potresti voler deduplicare; qui semplifichiamo:
        node_index = np.array([idmaps[tname][v] for v in pvals], dtype=np.int64)
        num_nodes = len(uniq_vals)
        # feature
        cat_cols = [c for c in df.columns if (tname, c) in col_to_stype and col_to_stype[(tname, c)] == "categorical" and c != pk]
        num_cols = [c for c in df.columns if (tname, c) in col_to_stype and col_to_stype[(tname, c)] == "continuous"]
        # categoriche -> indici (concat)
        X_cat_idx = cat_enc.fit_transform(df, tname, cat_cols)  # [rows, n_cat]
        # continue -> tensor float
        X_num = None
        if num_cols:
            X_num = torch.tensor(df[num_cols].to_numpy(dtype=np.float32))
            if standardize:
                X_num, stats = standardize_numeric(X_num)
                col_stats_dict[(tname, "numeric")] = stats
        # aggrega per nodo (se df ha più righe per stesso pk): usiamo media
        def aggregate_by_node(tensor, name):
            if tensor is None:
                return None
            out = torch.zeros((num_nodes, tensor.size(1)), dtype=tensor.dtype)
            counts = torch.zeros(num_nodes, 1, dtype=torch.float32)
            for row_id, nid in enumerate(node_index):
                out[nid] += tensor[row_id].to(out.dtype)
                counts[nid] += 1.0
            out = out / counts.clamp_min(1.0)
            return out

        x_num = aggregate_by_node(X_num, "num")
        x_cat_idx = aggregate_by_node(X_cat_idx, "cat")

        # emb per categoriche (una embedding per colonna)
        if x_cat_idx is not None and x_cat_idx.numel() > 0:
            # per semplicità: lasciamo gli indici come feature integer; 
            # la tua XMetaPath2 può gestire embedding internamente, altrimenti puoi applicare nn.EmbeddingBag fuori.
            x = x_cat_idx.float()
            if x_num is not None:
                x = torch.cat([x, x_num], dim=1)
        else:
            x = x_num if x_num is not None else torch.zeros((num_nodes, 1))
        data[tname].x = x
        data[tname].num_nodes = num_nodes

    # 2) crea gli edge_index da fkey
    for src_table, df in db.tables.items():
        fk_map = find_foreign_keys(db, src_table, pkeys)
        for fk_col, dst_table in fk_map.items():
            src_pk = pkeys[src_table]
            dst_pk = pkeys[dst_table]
            # costruisci edges usando le righe dove fk non è NaN e matcha un id valido
            valid = df[fk_col].notna()
            src_vals = df.loc[valid, src_pk].values
            dst_vals = df.loc[valid, fk_col].values
            # mappa a indici nodo
            src_idx = [idmaps[src_table].get(v, None) for v in src_vals]
            dst_idx = [idmaps[dst_table].get(v, None) for v in dst_vals]
            pairs = [(s, d) for s, d in zip(src_idx, dst_idx) if s is not None and d is not None]
            if not pairs:
                continue
            ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()  # [2, E]
            rel = f"f2p_{fk_col}"
            data[(src_table, rel, dst_table)].edge_index = ei
            # edge inverso
            data[(dst_table, f"rev_{rel}", src_table)].edge_index = ei.flip(0)

    return data, col_stats_dict


# utility functions:
def flip_rel(rel_name: str) -> str:
    return rel_name[4:] if rel_name.startswith("rev_") else f"rev_{rel_name}"

def to_canonical(mp_outward):
    # mp_outward: [(src, rel, dst), ...] dalla costruzione RL (parte da 'drivers')
    mp = [(dst, flip_rel(rel), src) for (src, rel, dst) in mp_outward[::-1]]
    assert mp[-1][2] == "drivers"
    return tuple(mp)






dataset = get_dataset("rel-f1", download=True)
task = get_task("rel-f1", "driver-dnf", download=True)

train_table = task.get_table("train") #date  driverId  qualifying
val_table = task.get_table("val") #date  driverId  qualifying
test_table = task.get_table("test") # date  driverId

out_channels = 1
loss_fn = nn.BCEWithLogitsLoss()
# this is the mae loss and is used when have regressions tasks.
tune_metric = "roc-auc"
higher_is_better = True

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "./data"

db = dataset.get_db() #get all tables
col_to_stype_dict = get_stype_proposal(db)
col_to_stype_dict = infer_stype_from_db(db)                 # << definita nella PATCH 2
db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)
data, col_stats_dict = build_heterodata_from_db(
     db=db_nuovo,
     col_to_stype=col_to_stype_dict_nuovo,
     embed_dim=32,    # emb per categoriche
     standardize=True # z-score per continue
)
#this is used to get the stype of the columns

#let's use the merge categorical values:
#db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)

# Create the graph
# data, col_stats_dict = make_pkey_fkey_graph(
#     db_nuovo,
#     col_to_stype_dict=col_to_stype_dict_nuovo,
#     #text_embedder_cfg=text_embedder_cfg,
#     text_embedder_cfg = None,
#     cache_dir=None  # disabled
# )
node_type="drivers"

metapaths = [[('drivers', 'rev_f2p_driverId', 'standings'), ('standings', 'f2p_raceId', 'races')], [('drivers', 'rev_f2p_driverId', 'qualifying'), ('qualifying', 'f2p_constructorId', 'constructors'), ('constructors', 'rev_f2p_constructorId', 'constructor_results'), ('constructor_results', 'f2p_raceId', 'races')], [('drivers', 'rev_f2p_driverId', 'results'), ('results', 'f2p_constructorId', 'constructors'), ('constructors', 'rev_f2p_constructorId', 'constructor_standings'), ('constructor_standings', 'f2p_raceId', 'races')]]

canonical = []
for mp in metapaths:
    #change to canonical:
    mp = mp.copy()
    mp_key   = to_canonical(mp)         
    assert mp_key[-1][2] == node_type, \
        f"Il meta-path canonico deve terminare su '{node_type}', invece termina su '{mp_key[-1][2]}'"
    canonical.append(mp_key)

hidden_channels = 128
out_channels = 128

model = XMetaPath2(
    data=data,
    col_stats_dict=col_stats_dict,
    #metapath_counts = metapath_count, 
    metapaths=canonical,               
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    final_out_channels=1,
).to(device)

loader_dict = loader_dict_fn(
    batch_size=512,
    num_neighbours=256,
    data=data,
    task=task,
    train_table=train_table,
    val_table=val_table,
    test_table=test_table
)

lr=1e-02
wd = 0

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

scheduler = CosineAnnealingLR(optimizer, T_max=25)

early_stopping = EarlyStopping(
    patience=60,
    delta=0.0,
    verbose=True,
    higher_is_better = True,
    path="best_basic_model.pt"
)

best_val_metric = -math.inf 
test_table = task.get_table("test", mask_input_cols=False)
best_test_metric = -math.inf 
epochs = 150
for epoch in range(0, epochs):
    train_loss = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

    train_pred = test(model, loader_dict["train"], device=device, task=task)
    val_pred = test(model, loader_dict["val"], device=device, task=task)
    test_pred = test(model, loader_dict["test"], device=device, task=task)
    
    train_metrics = evaluate_performance(train_pred, train_table, task.metrics, task=task)
    val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)
    test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)

    #scheduler.step(val_metrics[tune_metric])

    if (higher_is_better and val_metrics[tune_metric] > best_val_metric):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())

    if (higher_is_better and test_metrics[tune_metric] > best_test_metric):
        best_test_metric = test_metrics[tune_metric]
        state_dict_test = copy.deepcopy(model.state_dict())

    current_lr = optimizer.param_groups[0]["lr"]
    
    print(f"Epoch: {epoch:02d}, Train {tune_metric}: {train_metrics[tune_metric]:.2f}, Validation {tune_metric}: {val_metrics[tune_metric]:.2f}, Test {tune_metric}: {test_metrics[tune_metric]:.2f}, LR: {current_lr:.6f}")

    # early_stopping(val_metrics[tune_metric], model)

    # if early_stopping.early_stop:
    #     print(f"Early stopping triggered at epoch {epoch}")
    #     break



print(f"best validation results: {best_val_metric}")
print(f"best test results: {best_test_metric}")