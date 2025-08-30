# data_management/data.py  — versione “core” senza TorchFrame/relbench.modeling
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

# ---------- stima stype senza TorchFrame ----------
def infer_stype_from_db(db) -> Dict[Tuple[str, str], str]:
    """
    Restituisce {(table, col) -> stype} con etichette:
    'categorical' | 'multicategorical' | 'continuous' | 'timestamp' | 'text'.
    """
    out: Dict[Tuple[str, str], str] = {}
    for tname, df in db.tables.items():
        for col in df.columns:
            s = df[col]
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

# ---------- utility pkey/fkey ----------
def _guess_primary_key(df: pd.DataFrame) -> str:
    cands = [c for c in df.columns if c.lower().endswith("id")]
    for c in cands:
        if df[c].is_unique:
            return c
    for c in df.columns:
        if df[c].is_unique:
            return c
    return df.columns[0]

def _find_foreign_keys(db, table: str, pkeys: Dict[str, str]) -> Dict[str, str]:
    df = db.tables[table]
    res: Dict[str, str] = {}
    low_cols = {c.lower(): c for c in df.columns}
    for other_table, pk in pkeys.items():
        if other_table == table:
            continue
        # stesso nome colonna (case-insensitive)
        if pk.lower() in low_cols:
            res[low_cols[pk.lower()]] = other_table
    return res

# ---------- build HeteroData pkey↔fkey ----------
def build_heterodata_from_db(db, col_to_stype: Dict[Tuple[str, str], str],
                             standardize: bool = True) -> tuple[HeteroData, Dict]:
    data = HeteroData()
    pkeys = {t: _guess_primary_key(df) for t, df in db.tables.items()}
    idmaps: Dict[str, Dict[object, int]] = {}
    col_stats_dict: Dict = {}

    # nodi + feature (semplice: numeriche standardizzate; categoriche come indici float)
    for tname, df in db.tables.items():
        pk = pkeys[tname]
        uniq_vals = pd.Index(df[pk].values).unique().tolist()
        idmaps[tname] = {v: i for i, v in enumerate(uniq_vals)}
        num_nodes = len(uniq_vals)

        # selezione colonne
        num_cols = [c for c in df.columns if (tname, c) in col_to_stype and col_to_stype[(tname, c)] == "continuous"]
        cat_cols = [c for c in df.columns if (tname, c) in col_to_stype and col_to_stype[(tname, c)] == "categorical" and c != pk]

        # mappa ogni riga a nodo
        node_index = np.array([idmaps[tname][v] for v in df[pk].values], dtype=np.int64)

        X_num = None
        if num_cols:
            X_num = torch.tensor(df[num_cols].to_numpy(dtype=np.float32))
            if standardize and X_num.numel() > 0:
                mu = X_num.mean(dim=0, keepdim=True)
                sd = X_num.std(dim=0, keepdim=True)
                X_num = (X_num - mu) / (sd + 1e-6)
                col_stats_dict[(tname, "numeric")] = {"mean": mu, "std": sd}

        X_cat = None
        if cat_cols:
            mats = []
            for col in cat_cols:
                idx = pd.factorize(df[col].astype("object").fillna("__NA__"))[0].astype(np.int64)
                mats.append(torch.from_numpy(idx).long().unsqueeze(1))
            if mats:
                X_cat = torch.cat(mats, dim=1).float()  # semplice embedding “one-hot-like” via indici float

        def _agg_by_node(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None: return None
            out = torch.zeros((num_nodes, tensor.size(1)), dtype=tensor.dtype)
            cnt = torch.zeros((num_nodes, 1), dtype=torch.float32)
            for r, nid in enumerate(node_index):
                out[nid] += tensor[r].to(out.dtype)
                cnt[nid] += 1.0
            return out / cnt.clamp_min(1.0)

        x_num = _agg_by_node(X_num)
        x_cat = _agg_by_node(X_cat)

        if x_cat is not None and x_num is not None:
            x = torch.cat([x_cat, x_num], dim=1)
        elif x_cat is not None:
            x = x_cat
        elif x_num is not None:
            x = x_num
        else:
            x = torch.zeros((num_nodes, 1), dtype=torch.float32)

        data[tname].x = x
        data[tname].num_nodes = num_nodes

    # archi pkey↔fkey
    for src_table, df in db.tables.items():
        fk_map = _find_foreign_keys(db, src_table, pkeys)
        for fk_col, dst_table in fk_map.items():
            src_pk = pkeys[src_table]
            valid = df[fk_col].notna()
            src_vals = df.loc[valid, src_pk].values
            dst_vals = df.loc[valid, fk_col].values
            src_idx = [idmaps[src_table].get(v) for v in src_vals]
            dst_idx = [idmaps[dst_table].get(v) for v in dst_vals]
            pairs = [(s, d) for s, d in zip(src_idx, dst_idx) if s is not None and d is not None]
            if not pairs:
                continue
            ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()
            rel = f"f2p_{fk_col}"
            data[(src_table, rel, dst_table)].edge_index = ei
            data[(dst_table, f"rev_{rel}", src_table)].edge_index = ei.flip(0)

    return data, col_stats_dict

# ---------- merge text->categorical (semplice) ----------
def merge_text_columns_to_categorical(db, col_to_stype: Dict[Tuple[str, str], str]):
    """
    Converte colonne di tipo 'text' in categoriche via factorize(), aggiornando lo stype.
    Ritorna (db_modificato, col_to_stype_modificato).
    """
    for tname, df in db.tables.items():
        for col in df.columns:
            key = (tname, col)
            if col_to_stype.get(key) == "text":
                # se cardinalità enorme, puoi filtrare o troncare — qui fattorizziamo tutto
                df[col] = pd.factorize(df[col].astype("object").fillna("__NA__"))[0]
                col_to_stype[key] = "categorical"
    return db, col_to_stype

# ---------- NeighborLoader dict ----------
def loader_dict_fn(batch_size: int,
                   num_neighbours: int,
                   data: HeteroData,
                   task,
                   train_table,
                   val_table,
                   test_table,
                   entity_table: Optional[str] = None):
    """
    Costruisce 3 NeighborLoader per train/val/test senza dipendenze da relbench.modeling.
    entity_table: se None, prova a dedurre; per i task driver-* è 'drivers'.
    """
    if entity_table is None:
        # euristica: usa 'drivers' se presente, altrimenti il primo node type
        entity_table = "drivers" if "drivers" in data.node_types else data.node_types[0]

    seeds = torch.arange(data[entity_table].num_nodes)
    kwargs = dict(num_neighbors=[num_neighbours, num_neighbours], batch_size=batch_size)

    loaders = {
        "train": NeighborLoader(data, input_nodes=(entity_table, seeds), shuffle=True, **kwargs),
        "val":   NeighborLoader(data, input_nodes=(entity_table, seeds), shuffle=False, **kwargs),
        "test":  NeighborLoader(data, input_nodes=(entity_table, seeds), shuffle=False, **kwargs),
    }
    return loaders
