import torch
import numpy as np
import math
from tqdm import tqdm
import torch_geometric
import torch_frame
from torch_geometric.seed import seed_everything
from relbench.modeling.utils import get_stype_proposal
from collections import defaultdict
import requests
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph
import copy
from typing import Any, Dict, List
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from torch_geometric.loader import NeighborLoader
import pyg_lib
from torch.nn import ModuleDict
import torch.nn.functional as F
from torch import nn
import random

def loader_dict_fn(batch_size, num_neighbours, data, task, train_table, val_table, test_table):
    loader_dict = {}

    for split, table in [
        ("train", train_table),
        ("val", val_table),
        ("test", test_table),
    ]:
        table_input = get_node_train_table_input(
            table=table,
            task=task,
        )

        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[num_neighbours for _ in range(2)],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=batch_size,
            temporal_strategy="uniform",
            shuffle=split == "train",
            num_workers=0,
            persistent_workers=False,
        )

    return loader_dict



def get_stype_enum_with_value(full_stype_dict, target_value: str):
    for table_entry in full_stype_dict.values():
        if isinstance(table_entry, dict):
            for val in table_entry.values():
                if str(val) == target_value:
                    return val
    raise ValueError(f"Tipo '{target_value}' non trovato nei tipi esistenti")



from torch_frame.data.stats import StatType

def merge_text_columns_to_categorical(db, stype_dict):
    for table_name in db.table_dict:
        table = db.table_dict[table_name]
        new_colname = "merged_text_id"

        # Cerca colonne di tipo test_embedded
        col_to_type = stype_dict[table_name]
        all_text_cols = [
            col for col, stype in col_to_type.items()
            if str(stype) == "text_embedded"
        ]

        if not all_text_cols:
            continue

        # Ordina le colonne alfabeticamente per avere ordine stabile
        sorted_cols = sorted(set(all_text_cols))

        # Combina i valori riga per riga
        merged_col = table.df[sorted_cols].astype(str).apply(lambda row: "_".join(row), axis=1)

        # Elimina i vecchi tipi dallo stype_dict
        for col in sorted_cols:
            if col in stype_dict[table_name]:
                del stype_dict[table_name][col]

        # Elimina le colonne anche dal DataFrame
        table.df.drop(columns=sorted_cols, inplace=True, errors="ignore")

        # Aggiungi la colonna fusa
        table.df[new_colname] = merged_col

        # Assegna il tipo corretto
        #stype_dict[table_name][new_colname] = stype.categorical
        categorical_type = get_stype_enum_with_value(stype_dict, "categorical")
        stype_dict[table_name][new_colname] = categorical_type


    return db, stype_dict
