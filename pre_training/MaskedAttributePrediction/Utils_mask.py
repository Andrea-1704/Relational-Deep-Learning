import numpy as np
import torch
from typing import Dict, List, Tuple
from torch_geometric.data import HeteroData

import sys
import os
sys.path.append(os.path.abspath("."))

from pre_training.MaskedAttributePrediction.Decoder import MAPDecoder

# aggiungi import
from torch_frame.data import TensorFrame

from typing import Dict, List, Tuple
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_frame.data import TensorFrame, Table  # assicurati dell'import
# Se il tuo TorchFrame non ha Table, rimuovi l'import di Table e gestisci i casi 1) e 2)

def _tf_get_table(tf: TensorFrame, col_stats_node, db_table=None):
    # 1) versione nuova
    if hasattr(tf, "to_table"):
        try:
            return tf.to_table()
        except Exception:
            pass
    # 2) versione intermedia
    if hasattr(tf, "table") and tf.table is not None:
        return tf.table
    # 3) fallback da DB se disponibile
    if db_table is not None:
        # ricava le colonne realmente usate dal TensorFrame
        cols = []
        for _, cols_in in tf.col_names_dict.items():
            cols.extend(cols_in)
        cols = [c for c in cols if c in db_table.df.columns]
        # copia subset, mantieni indici globali
        df_subset = db_table.df.loc[:, cols].copy()
        # se Table non esiste nella tua versione, commenta le 2 righe sotto e alza errore
        return Table(df=df_subset, col_stats=col_stats_node)
    raise AttributeError("Impossibile ottenere un Table dal TensorFrame: mancano to_table e table, e nessun db_table fornito.")

def _tf_from_table(table_obj, col_stats_node):
    # versione nuova
    if hasattr(TensorFrame, "from_table"):
        return TensorFrame.from_table(table=table_obj, col_stats=col_stats_node)
    # versione legacy
    try:
        return TensorFrame(table_obj, col_stats=col_stats_node)
    except TypeError:
        # alcune versioni vogliono positional args
        return TensorFrame(table_obj, col_stats_node)

def mask_attributes(batch: HeteroData,
                    maskable_attributes: Dict[str, Dict[str, List[str]]],
                    p_mask=0.3,
                    device="cuda",
                    col_stats_dict=None,
                    db=None):
    if col_stats_dict is None:
        raise ValueError("mask_attributes richiede col_stats_dict per ricostruire il TensorFrame.")

    target_values = {}

    for node_type, type_dict in maskable_attributes.items():
        if node_type not in batch.node_types:
            print(f"node type {node_type} non trovato, questo grafo ha nodi {batch.node_types}")
            continue

        db_table = None
        if db is not None and hasattr(db, "table_dict") and node_type in db.table_dict:
            db_table = db.table_dict[node_type]

        # ottieni il Table in modo compatibile
        table = _tf_get_table(batch[node_type].tf, col_stats_dict[node_type], db_table)
        df = table.df.copy(deep=True)

        # colonne effettivamente presenti nel TensorFrame
        table_columns = []
        for _, cols_in in batch[node_type].tf.col_names_dict.items():
            table_columns.extend(cols_in)

        # indici globali presenti nel batch
        if hasattr(batch[node_type], "n_id"):
            local_ids = batch[node_type].n_id.cpu().numpy()
        else:
            # fallback, se la tua versione usa un altro nome
            local_ids = np.arange(df.shape[0])

        if local_ids.size == 0:
            continue

        for attr_type, cols in type_dict.items():
            for col in cols:
                if col not in table_columns:
                    print(f"colonna {col} non trovata nel TensorFrame di {node_type}")
                    continue

                bern = (torch.rand(len(local_ids), device=device) < p_mask).cpu().numpy()
                masked_pos = np.nonzero(bern)[0]
                if masked_pos.size == 0:
                    continue

                masked_global = local_ids[masked_pos]

                target_values[(node_type, col)] = {
                    "indices": masked_pos.tolist(),
                    "values": df.loc[masked_global, col].tolist()
                }

                if attr_type == "categorical":
                    df.loc[masked_global, col] = 0
                elif attr_type == "numerical":
                    df.loc[masked_global, col] = 0.0
                else:
                    raise ValueError(f"Tipo non supportato: {attr_type}")

        # scrivi indietro e ricostruisci un TensorFrame
        table.df = df
        batch[node_type].tf = _tf_from_table(table, col_stats_dict[node_type])

    return batch, target_values









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



def train_map(model, loader_dict, maskable_attributes, encoder_out_dim: int, device: str, cat_values, epochs: int = 20, col_stats_dict=None):
    if col_stats_dict is None:
        raise ValueError("train_map richiede col_stats_dict per il masking batch-local.")
    model.train()
    decoder = MAPDecoder(encoder_out_dim)

    #Inizializzazione decoder per ogni colonna
    for node_type, type_dict in maskable_attributes.items():
      for col in type_dict.get("categorical", []):
          try:
              out_dim = len(cat_values[node_type][col])#qua prendo il numero di possibili valori
              decoder.add_decoder(f"{node_type}__{col}", out_dim=out_dim, task="classification")
              #print(f"Aggiunto decoder per {node_type}__{col} con out_dim={out_dim}")
          except KeyError:
              print(f"Nessun valore trovato per {node_type}__{col}, decoder non aggiunto")

      for col in type_dict.get("numerical", []):
          decoder.add_decoder(f"{node_type}__{col}", out_dim=1, task="regression")


    decoder.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=1e-3)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader_dict["train"]:
            batch = batch.to(device)
            batch, mask_info = mask_attributes(batch, maskable_attributes, col_stats_dict=col_stats_dict)
            z_dict = model.encode_node_types(batch, node_types=list(maskable_attributes.keys()))

            loss = decoder(z_dict, batch, mask_info)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[MAP] Epoch {epoch:02d} | Loss: {total_loss:.4f}")

    return model