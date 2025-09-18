import numpy as np
import torch
from typing import Dict, List, Tuple
from torch_geometric.data import HeteroData

import sys
import os
sys.path.append(os.path.abspath("."))

from pre_training.MaskedAttributePrediction.Decoder import MAPDecoder

# aggiungi import

# --- IMPORT ---
from typing import Dict, List
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_frame.data import TensorFrame  # <-- rimosso Table

# --- HELPERS COMPAT ---
def _get_table_like(tf: TensorFrame):
    """
    Ritorna un oggetto 'table-like' con almeno un attributo .df
    provando prima tf.to_table() poi tf.table.
    Se nessuno dei due esiste, solleva un errore con istruzioni per Colab.
    """
    # 1) API più recente
    if hasattr(tf, "to_table"):
        try:
            tbl = tf.to_table()
            # controllo minimo
            if hasattr(tbl, "df"):
                return tbl
        except Exception:
            pass
    # 2) API intermedia/vecchia
    if hasattr(tf, "table") and tf.table is not None and hasattr(tf.table, "df"):
        return tf.table

    # 3) non c'è modo affidabile: spiega come aggiornare
    raise RuntimeError(
        "Questo TensorFrame non espone né .to_table() né .table; "
        "per usare il masking batch-local serve una versione di TorchFrame compatibile. "
        "Su Colab puoi provare:\n"
        "  !pip install -U 'git+https://github.com/pyg-team/torch-frame.git'\n"
        "oppure reinstallare la versione del tuo progetto che espone almeno uno dei due."
    )

def _rebuild_tensorframe_from_table(table_obj, col_stats_node):
    """
    Ricostruisce un TensorFrame a partire da un oggetto table-like.
    Usa TensorFrame.from_table se esiste; altrimenti prova il costruttore legacy.
    """
    # preferisci API nuova
    if hasattr(TensorFrame, "from_table"):
        return TensorFrame.from_table(table=table_obj, col_stats=col_stats_node)

    # fallback: alcune versioni accettano (table_obj, col_stats) nel costruttore
    try:
        return TensorFrame(table_obj, col_stats=col_stats_node)
    except TypeError:
        # altra variante legacy
        return TensorFrame(table_obj, col_stats_node)

# --- FUNZIONE PRINCIPALE ---
def mask_attributes(batch: HeteroData,
                    maskable_attributes: Dict[str, Dict[str, List[str]]],
                    p_mask=0.3,
                    device="cuda",
                    col_stats_dict=None):
    if col_stats_dict is None:
        raise ValueError("mask_attributes richiede col_stats_dict per ricostruire il TensorFrame.")

    target_values = {}

    for node_type, type_dict in maskable_attributes.items():
        if node_type not in batch.node_types:
            print(f"node type {node_type} non trovato, questo grafo ha nodi {batch.node_types}")
            continue

        # ottieni un oggetto con .df indipendentemente dalla versione
        table = _get_table_like(batch[node_type].tf)
        df = table.df.copy(deep=True)

        # colonne effettive del TF
        table_columns = []
        if hasattr(batch[node_type].tf, "col_names_dict"):
            for _, cols_in in batch[node_type].tf.col_names_dict.items():
                table_columns.extend(cols_in)
        else:
            # fallback conservativo: usa colonne del df
            table_columns = list(df.columns)

        # indici globali presenti nel batch
        if hasattr(batch[node_type], "n_id"):
            local_ids = batch[node_type].n_id.cpu().numpy()
        else:
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

        # scrivi indietro nel table-like e ricostruisci il TensorFrame
        table.df = df
        batch[node_type].tf = _rebuild_tensorframe_from_table(table, col_stats_dict[node_type])

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