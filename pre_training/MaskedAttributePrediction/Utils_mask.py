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

# --- Utils_mask.py ---

from typing import Dict, List
import numpy as np
import torch
from torch_geometric.data import HeteroData

def mask_attributes(batch: HeteroData,
                    maskable_attributes: Dict[str, Dict[str, List[str]]],
                    p_mask: float = 0.3,
                    device: str = "cuda"):
    """
    Versione compatibile con torch-frame 0.2.5:
    - NON usa Table né DataFrame
    - maschera direttamente batch[node_type].tf.feat_dict[stype]
    - salva i valori originali:
        * categoriche: indici interi (torch.long)
        * numeriche: float (torch.float)
    - indices sono POSIZIONI NEL BATCH (0..N-1), non id globali
    """
    target_values = {}  # chiave: (node_type, col) -> {"indices": [...], "values": Tensor 1D}

    for node_type, type_dict in maskable_attributes.items():
        if node_type not in batch.node_types:
            print(f"[mask_attributes] node type {node_type} non presente in questo batch: {batch.node_types}")
            continue

        tf = batch[node_type].tf
        # mappa colonna -> (stype_key, col_idx)
        col2loc = {}
        for stype_key, cols in tf.col_names_dict.items():
            for j, col_name in enumerate(cols):
                col2loc[col_name] = (stype_key, j)

        # posizioni riga nel batch (0..num_nodes_batch-1)
        num_rows = None
        # prendi la prima stype per leggere la dimensione righe
        for stype_key, tensor in tf.feat_dict.items():
            num_rows = tensor.size(0)
            break
        if num_rows is None or num_rows == 0:
            continue

        # per ogni colonna da mascherare
        for attr_type, cols in type_dict.items():
            for col in cols:
                if col not in col2loc:
                    print(f"[mask_attributes] colonna {col} non trovata in tf.col_names_dict per {node_type}")
                    continue

                stype_key, j = col2loc[col]
                X = tf.feat_dict[stype_key]  # shape: [N, C_stype]
                # genera maschera Bernoulli su righe del BATCH
                bern_idx = (torch.rand(num_rows, device=X.device) < p_mask).nonzero(as_tuple=False).view(-1)
                orig = X.index_select(0, bern_idx)[:, j]  # 1D

                target_values[(node_type, col)] = {
                    "indices": bern_idx.detach().cpu().tolist(),  # salvi come lista su CPU
                    "values":  orig.detach().cpu()                # GT su CPU
                }

                if attr_type == "categorical":
                    X[bern_idx, j] = 0
                elif attr_type == "numerical":
                    X[bern_idx, j] = 0.0
                # scrivi indietro (X è mutato in place, spesso non serve riassegnare)
                tf.feat_dict[stype_key] = X

        # riassegna tf (non strettamente necessario, ma esplicito)
        batch[node_type].tf = tf

    return batch, target_values


def train_map(model,
              loader_dict,
              maskable_attributes,
              encoder_out_dim: int,
              device: str,
              cat_values,
              epochs: int = 20,
              db=None,
              entity_table=None,
              **kwargs):
    from pre_training.MaskedAttributePrediction.Decoder import MAPDecoder

    model.train()






    # Utils_mask.py  (dentro train_map, subito dopo decoder = MAPDecoder(...).to(device))
    decoder = MAPDecoder(encoder_out_dim).to(device)

    # >>> AGGIUNGI QUESTO BLOCCO <<<
    for nt, type_dict in maskable_attributes.items():
        # categoriche: out_dim = numero di classi
        for col in type_dict.get("categorical", []):
            vocab = cat_values.get(nt, {}).get(col, None)
            if vocab is None:
                print(f"[MAP] Nessun vocab per {nt}__{col}, salto la testa di classificazione")
                continue
            decoder.add_decoder(f"{nt}__{col}", out_dim=len(vocab), task="classification")

        # numeriche: out_dim = 1
        for col in type_dict.get("numerical", []):
            decoder.add_decoder(f"{nt}__{col}", out_dim=1, task="regression")



    #decoder = MAPDecoder(encoder_out_dim).to(device)

    opt = torch.optim.Adam(
        list(model.parameters()) + list(decoder.parameters()),
        lr=1e-3
    )

    for epoch in range(1, epochs + 1):
        total = 0.0
        for batch in loader_dict["train"]:
            batch = batch.to(device)

            # 1) Applica il masking IN-PLACE sui tensori (versione che hai già)
            batch, mask_info = mask_attributes(
                batch,
                maskable_attributes,
                p_mask=0.3,
                device=device,
            )

            # 2) Determina entity_table come NOME del node type (non un dict!)
            if isinstance(entity_table, str):
                ent_tab = entity_table
            else:
                # autodetect: prendi il node type che ha 'seed_time'
                ent_tab = None
                for nt in batch.node_types:
                    if hasattr(batch[nt], "seed_time") or ("seed_time" in getattr(batch[nt], "__dict__", {})):
                        ent_tab = nt
                        break
                if ent_tab is None:
                    # fallback: primo tra quelli mascherati
                    nts = list(maskable_attributes.keys())
                    ent_tab = nts[0] if len(nts) > 0 else batch.node_types[0]

            # 3) Encode embeddings per i node types di interesse
            node_types = list(maskable_attributes.keys())
            try:
                z_dict = model.encode_node_types(batch, ent_tab, node_types=node_types)
            except TypeError:
                z_dict = model.encode_node_types(batch=batch, entity_table=ent_tab, node_types=node_types)

            # 4) LASCIA che il decoder calcoli la LOSS usando (z_dict, batch, mask_info)
            loss = decoder(z_dict, batch, mask_info)

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