import numpy as np
import torch
from typing import Dict, List, Tuple
from torch_geometric.data import HeteroData

import sys
import os
sys.path.append(os.path.abspath("."))

from pre_training.MaskedAttributePrediction.Decoder import MAPDecoder

# Utils_mask.py

from typing import Dict, List
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_frame.data import TensorFrame

# ---------- helper: ricrea un TensorFrame da un DataFrame pandas ----------
def _rebuild_tensorframe_from_df(df, col_stats_node):
    """
    Ricostruisce un TensorFrame a partire da un pandas.DataFrame.
    Prova più vie per essere compatibile con torch-frame 0.2.5.
    """
    # Alcune build espongono factory diverse: prova in ordine
    for fn_name in ["from_dataframe", "from_pandas", "from_df"]:
        fn = getattr(TensorFrame, fn_name, None)
        if callable(fn):
            return fn(df=df, col_stats=col_stats_node)

    # Fallback: costruttore diretto (alcune versioni accettano (df, col_stats))
    try:
        return TensorFrame(df, col_stats=col_stats_node)
    except TypeError:
        # Variante positional-only
        return TensorFrame(df, col_stats_node)

# ---------- masking che lavora direttamente sul DB.df ----------
def mask_attributes(batch: HeteroData,
                    maskable_attributes: Dict[str, Dict[str, List[str]]],
                    p_mask: float = 0.3,
                    device: str = "cuda",
                    col_stats_dict: Dict = None,
                    db=None):
    """
    Maschera attributi lavorando sul DataFrame sorgente in db.table_dict[node_type].df,
    limitandosi alle colonne realmente usate dal TensorFrame nel batch corrente (col_names_dict)
    e alle righe presenti nel batch (n_id). Poi ricrea il TensorFrame dal DataFrame.
    """
    if col_stats_dict is None:
        raise ValueError("mask_attributes richiede col_stats_dict.")
    if db is None or not hasattr(db, "table_dict"):
        raise ValueError("mask_attributes richiede `db` con `table_dict`.")

    target_values = {}

    for node_type, type_dict in maskable_attributes.items():
        if node_type not in batch.node_types:
            print(f"[mask_attributes] node type {node_type} non presente in questo batch: {batch.node_types}")
            continue
        if node_type not in db.table_dict:
            print(f"[mask_attributes] {node_type} non presente nel DB.")
            continue

        # df completo dal DB
        df_full = db.table_dict[node_type].df

        # colonne effettivamente usate dal TF del batch
        tf = batch[node_type].tf
        cols_used = []
        if hasattr(tf, "col_names_dict"):
            for _, cols_in in tf.col_names_dict.items():
                cols_used.extend(cols_in)
        else:
            cols_used = list(df_full.columns)  # fallback

        cols_used = [c for c in cols_used if c in df_full.columns]
        if not cols_used:
            continue

        # copia profonda solo delle colonne usate
        df = df_full.loc[:, cols_used].copy(deep=True)

        # global ids presenti nel batch
        if hasattr(batch[node_type], "n_id"):
            local_ids = batch[node_type].n_id.detach().cpu().numpy()
        else:
            print(f"[mask_attributes] {node_type} non espone n_id, salto.")
            continue
        if local_ids.size == 0:
            continue

        # masking colonna per colonna
        for attr_type, cols in type_dict.items():
            for col in cols:
                if col not in df.columns:
                    print(f"[mask_attributes] colonna {col} non trovata per {node_type}")
                    continue

                bern = (torch.rand(len(local_ids), device=device) < p_mask).detach().cpu().numpy()
                masked_pos = np.nonzero(bern)[0]
                if masked_pos.size == 0:
                    continue
                masked_global = local_ids[masked_pos]

                # salva ground truth
                target_values[(node_type, col)] = {
                    "indices": masked_pos.tolist(),              # posizioni nel batch
                    "values": df.loc[masked_global, col].tolist() # valori originali
                }

                # applica sentinel
                if attr_type == "categorical":
                    df.loc[masked_global, col] = 0
                elif attr_type == "numerical":
                    df.loc[masked_global, col] = 0.0
                else:
                    raise ValueError(f"Tipo non supportato: {attr_type}")

        # ricostruisci il TensorFrame dal df mascherato e rimontalo nel batch
        batch[node_type].tf = _rebuild_tensorframe_from_df(df, col_stats_dict[node_type])

    return batch, target_values


# ---------- train_map: ricordati di passare anche db ----------
def train_map(model,
              loader_dict,
              maskable_attributes,
              encoder_out_dim: int,
              device: str,
              cat_values,
              epochs: int = 20,
              col_stats_dict=None,
              db=None):
    """
    Addestra MAPDecoder usando il masking batch-local. Richiede db e col_stats_dict.
    """
    if col_stats_dict is None:
        raise ValueError("train_map richiede col_stats_dict.")
    if db is None:
        raise ValueError("train_map richiede db (Database) per il masking.")

    from pre_training.MaskedAttributePrediction.Decoder import MAPDecoder  # import locale per evitare dipendenze circolari
    model.train()
    decoder = MAPDecoder(encoder_out_dim, cat_values=cat_values).to(device)

    opt = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    mse = torch.nn.MSELoss(reduction="mean")

    for epoch in range(1, epochs + 1):
        total = 0.0
        for batch in loader_dict["train"]:
            batch = batch.to(device)

            # MASKING
            batch, mask_info = mask_attributes(
                batch,
                maskable_attributes,
                p_mask=0.3,
                device=device,
                col_stats_dict=col_stats_dict,
                db=db
            )

            # ENCODE solo i node_types effettivamente mascherati
            node_types = list(maskable_attributes.keys())
            z_dict = model.encode_node_types(batch, node_types=node_types)

            # DECODIFICA e LOSS
            loss = 0.0
            for (node_type, col), info in mask_info.items():
                if info["indices"]:
                    z = z_dict[node_type][info["indices"]]
                    out = decoder(node_type, col, z)

                    gt = info["values"]
                    # cat vs num
                    if decoder.is_categorical(node_type, col):
                        # mappa gt al vocab index usando decoder.cat2idx
                        idx = torch.tensor([decoder.cat2idx[node_type][col].get(v, 0) for v in gt],
                                           device=device, dtype=torch.long)
                        loss = loss + loss_fn(out, idx)
                    else:
                        y = torch.tensor(gt, device=device, dtype=torch.float32).view(-1, 1)
                        loss = loss + mse(out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu())

        print(f"[MAP] epoch {epoch}/{epochs} - loss {total:.4f}")







from collections import defaultdict

def extract_categorical_values_from_db(db, maskable_attributes):
    value_dict = defaultdict(lambda: defaultdict(set))

    for node_type, type_dict in maskable_attributes.items():
        if node_type not in db.table_dict:
            print(f"Tabella {node_type} non trovata nel db.")
            continue

        table = db.table_dict[node_type]
        df = table.df

        for col in type_dict.get("categorical", []):
            if col not in df.columns:
                print(f"Colonna {col} non trovata nella tabella {node_type}")
                continue

            unique_vals = df[col].dropna().unique()
            value_dict[node_type][col].update(unique_vals)

    # Converte in liste ordinate per sicurezza
    value_dict_final = {
        node: {col: sorted(list(vals)) for col, vals in col_dict.items()}
        for node, col_dict in value_dict.items()
    }

    return value_dict_final



# def train_map(model, loader_dict, maskable_attributes, encoder_out_dim: int, device: str, cat_values, epochs: int = 20, col_stats_dict=None):
#     if col_stats_dict is None:
#         raise ValueError("train_map richiede col_stats_dict per il masking batch-local.")
#     model.train()
#     decoder = MAPDecoder(encoder_out_dim)

#     #Inizializzazione decoder per ogni colonna
#     for node_type, type_dict in maskable_attributes.items():
#       for col in type_dict.get("categorical", []):
#           try:
#               out_dim = len(cat_values[node_type][col])#qua prendo il numero di possibili valori
#               decoder.add_decoder(f"{node_type}__{col}", out_dim=out_dim, task="classification")
#               #print(f"Aggiunto decoder per {node_type}__{col} con out_dim={out_dim}")
#           except KeyError:
#               print(f"Nessun valore trovato per {node_type}__{col}, decoder non aggiunto")

#       for col in type_dict.get("numerical", []):
#           decoder.add_decoder(f"{node_type}__{col}", out_dim=1, task="regression")


#     decoder.to(device)
#     optimizer = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=1e-3)

#     for epoch in range(1, epochs + 1):
#         total_loss = 0.0
#         for batch in loader_dict["train"]:
#             batch = batch.to(device)
#             batch, mask_info = mask_attributes(batch, maskable_attributes, col_stats_dict=col_stats_dict)
#             z_dict = model.encode_node_types(batch, node_types=list(maskable_attributes.keys()))

#             loss = decoder(z_dict, batch, mask_info)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         print(f"[MAP] Epoch {epoch:02d} | Loss: {total_loss:.4f}")

#     return model