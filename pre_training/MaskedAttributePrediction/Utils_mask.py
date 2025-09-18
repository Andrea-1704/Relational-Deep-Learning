import numpy as np
import torch
from typing import Dict, List, Tuple
from torch_geometric.data import HeteroData

import sys
import os
sys.path.append(os.path.abspath("."))

from pre_training.MaskedAttributePrediction.Decoder import MAPDecoder
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






def train_map(model, loader_dict, maskable_attributes, encoder_out_dim: int, device: str, cat_values ,epochs: int = 20):
    model.train()
    decoder = MAPDecoder(encoder_out_dim)

    #Inizializzazione decoder per ogni colonna
    for node_type, type_dict in maskable_attributes.items():
      for col in type_dict.get("categorical", []):
          try:
              out_dim = len(cat_values[node_type][col])#qua prendo il numero di possibili valori
              decoder.add_decoder(f"{node_type}__{col}", out_dim=out_dim, task="classification")
              print(f"Aggiunto decoder per {node_type}__{col} con out_dim={out_dim}")
          except KeyError:
              print(f"Nessun valore trovato per {node_type}__{col}, decoder non aggiunto")

      for col in type_dict.get("numerical", []):
          decoder.add_decoder(f"{node_type}__{col}", out_dim=1, task="regression")
          print(f"Aggiunto decoder per {node_type}__{col} con out_dim=1 (regression)")


    decoder.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=1e-3)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader_dict["train"]:
            batch = batch.to(device)
            # batch, mask_info = mask_attributes(batch, maskable_attributes)
            # Utils_mask.py, funzione train_map: dentro il loop dei batch
            batch, mask_info = mask_attributes(batch, maskable_attributes, db=db, device=device)
            z_dict = model.encode_node_types(
                batch,
                node_types=list(maskable_attributes.keys()),
                mask_info=mask_info 
            )
            loss = decoder(z_dict, batch, mask_info)

            z_dict = model.encode_node_types(batch, node_types=list(maskable_attributes.keys()))

            loss = decoder(z_dict, batch, mask_info)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[MAP] Epoch {epoch:02d} | Loss: {total_loss:.4f}")

    return model





# def mask_attributes(batch: HeteroData,
#                     maskable_attributes: Dict[str, Dict[str, List[str]]],
#                     p_mask=0.3,
#                     device="cuda") -> Tuple[HeteroData, Dict]:
#     """
#     Maschera dinamicamente alcune feature nel batch, e restituisce anche i valori originali mascherati.
#     """
#     target_values = {}  # per il decoder: quali campi/righe sono stati mascherati e con quali valori originali

#     for node_type, type_dict in maskable_attributes.items():
#         if node_type not in batch.node_types:
#             print(f"node type {node_type} non trovato, questo grafo ha nodi {batch.node_types}")
#             continue

#         for attr_type, cols in type_dict.items():
#             for col in cols:

#                 table_columns = []
#                 for stype, cols_in in batch[node_type].tf.col_names_dict.items():
#                     table_columns.extend(cols_in)

#                 if col not in table_columns:
#                     print(f"non abbiamo trovato la colonna {col} nella tabella {node_type}")
#                     continue  # Colonna non disponibile nel batch

#                 # --- batch-only: lavora sul DF del batch, non sul DB globale ---
#                 table = batch[node_type].tf.table
#                 df = table.df.copy(deep=True)

#                 # ID globali presenti nel batch e maschera Bernoulli batch-local
#                 local_ids = batch[node_type].n_id.cpu().numpy()
#                 if local_ids.size == 0:
#                     continue
#                 bern = (torch.rand(len(local_ids), device=device) < p_mask).cpu().numpy()
#                 masked_pos = np.nonzero(bern)[0]  # indici batch-local
#                 if masked_pos.size == 0:
#                     continue
#                 masked_global = local_ids[masked_pos]

#                 # salva indici batch-local e valori originali
#                 target_values[(node_type, col)] = {
#                     "indices": masked_pos.tolist(),
#                     "values": df.loc[masked_global, col].tolist()
#                 }

#                 # applica sentinel SOLO nel batch
#                 if attr_type == "categorical":
#                     df.loc[masked_global, col] = 0
#                 elif attr_type == "numerical":
#                     df.loc[masked_global, col] = 0.0
#                 else:
#                     raise ValueError(f"Tipo non supportato: {attr_type}")

#                 # scrivi indietro SOLO al batch
#                 table.df = df

#     return batch, target_values


# Utils_mask.py

import numpy as np
import torch
from typing import Dict, List, Tuple
from torch_geometric.data import HeteroData

def mask_attributes(
    batch: HeteroData,
    maskable_attributes: Dict[str, Dict[str, List[str]]],
    db,                      # <--- nuovo argomento: DB globale
    p_mask: float = 0.3,
    device: str = "cuda"
) -> Tuple[HeteroData, Dict]:
    """
    Sceglie quali celle mascherare per questo batch e ritorna:
      - batch invariato
      - mask_info: dict con (node_type, col) -> {indices (locali), global_ids, values, task}
    Non tenta di modificare il TensorFrame del batch.
    """
    mask_info = {}

    for node_type, type_dict in maskable_attributes.items():
        if node_type not in batch.node_types:
            continue

        # ID globali presenti nel batch (già li usi in codice attuale)
        if not hasattr(batch[node_type], "n_id"):
            # se usi full-batch senza NeighborLoader potresti non avere n_id:
            # in tal caso gli indici locali coincidono coi globali
            local_ids = torch.arange(batch[node_type].num_nodes, device=device).cpu().numpy()
        else:
            local_ids = batch[node_type].n_id.cpu().numpy()
        if local_ids.size == 0:
            continue

        # colleziona i nomi colonna disponibili nel TensorFrame (solo per filtrare)
        table_columns = []
        if hasattr(batch[node_type], "tf") and hasattr(batch[node_type].tf, "col_names_dict"):
            for _, cols_in in batch[node_type].tf.col_names_dict.items():
                table_columns.extend(cols_in)

        for attr_type, cols in type_dict.items():
            for col in cols:
                if len(table_columns) > 0 and col not in table_columns:
                    # colonna non presente in questo mini-batch
                    continue

                # Bernoulli su posizioni batch-local
                bern = (torch.rand(len(local_ids), device=device) < p_mask).cpu().numpy()
                masked_pos = np.nonzero(bern)[0]
                if masked_pos.size == 0:
                    continue

                masked_global = local_ids[masked_pos]

                # Valori veri dal DB globale (niente .tf.table!)
                df_global = db.table_dict[node_type].df
                true_vals = df_global.loc[masked_global, col].tolist()

                mask_info[(node_type, col)] = {
                    "indices": masked_pos.tolist(),       # indici locali nel batch
                    "global_ids": masked_global.tolist(), # indici globali nel DB
                    "values": true_vals,                  # per la loss nel decoder
                    "task": attr_type                     # "categorical" o "numerical"
                }

    return batch, mask_info
