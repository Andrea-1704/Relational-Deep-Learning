import torch
import torch.nn.functional as F

import os
import torch
import relbench
import numpy as np
from torch.nn import BCEWithLogitsLoss, L1Loss
from relbench.datasets import get_dataset
from relbench.tasks import get_task
import math
from tqdm import tqdm
import torch_geometric
import torch_frame
from torch_geometric.seed import seed_everything
from relbench.modeling.utils import get_stype_proposal
from collections import defaultdict
import requests
from io import StringIO
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph
from torch.nn import BCEWithLogitsLoss
import copy
from typing import Any, Dict, List
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType

import sys
import os
sys.path.append(os.path.abspath("."))

from model.MPSGNN_Model import MPSGNN
from data_management.data import loader_dict_fn
from utils.mpsgnn_metapath_utils import binarize_targets, greedy_metapath_search
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical


dataset = get_dataset("rel-f1", download=True)
task = get_task("rel-f1", "driver-position", download=True)

train_table = task.get_table("train") #date  driverId  qualifying
val_table = task.get_table("val") #date  driverId  qualifying
test_table = task.get_table("test") # date  driverId

out_channels = 1
loss_fn = L1Loss()
# this is the mae loss and is used when have regressions tasks.
tune_metric = "mae"
higher_is_better = False

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "./data"

db = dataset.get_db() #get all tables
col_to_stype_dict = get_stype_proposal(db)
#this is used to get the stype of the columns

#let's use the merge categorical values:
db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)

# Create the graph
data, col_stats_dict = make_pkey_fkey_graph(
    db_nuovo,
    col_to_stype_dict=col_to_stype_dict_nuovo,
    text_embedder_cfg = None,
    cache_dir=None  # disabled
)



def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_full, _ = make_pkey_fkey_graph(
        db_nuovo,
        col_to_stype_dict=col_to_stype_dict_nuovo,
        text_embedder_cfg=None,
        cache_dir=None
    )
    data_full = data_full.to(device)

    #retrieve the id from the driver nodes
    graph_driver_ids = db_nuovo.table_dict["drivers"].df["driverId"].to_numpy()
    id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}

    #get the labels and the ids of the drivers from the table
    train_df = train_table.df
    driver_labels = train_df["position"].to_numpy()
    driver_ids = train_df["driverId"].to_numpy()

    #map the correct labels for all drivers node (which are target ones)
    target_vector = torch.full((len(graph_driver_ids),), float("nan")) #inizial
    for i, driver_id in enumerate(driver_ids):
        if driver_id in id_to_idx:
            target_vector[id_to_idx[driver_id]] = driver_labels[i]

    
    data_full['driver'].y = target_vector
    data_full['driver'].train_mask = ~torch.isnan(target_vector)




    #take y and mask complete for the dataset:
    y_full = data_full['driver'].y.float()
    train_mask_full = data_full['driver'].train_mask
    y_bin_full = binarize_targets(y_full, threshold=10)

    metapaths = greedy_metapath_search(
        data_full,
        y_bin=y_bin_full,
        train_mask=train_mask_full,
        node_type='driver',
        L_max=3
    )

    data, _ = make_pkey_fkey_graph(
        db_nuovo,
        col_to_stype_dict=col_to_stype_dict_nuovo,
        text_embedder_cfg=None,
        cache_dir=None
    )
    data = data.to(device)

    #now we can use the loader dict and batch work SGD
    loader_dict = loader_dict_fn(
        batch_size=512, 
        num_neighbours=256, 
        data=data, 
        task=task,
        train_table=train_table, 
        val_table=val_table, 
        test_table=test_table
    )
    #print("questo è esattamente il codice che stiamo eseguendo")

    model = MPSGNN(
        metadata=data_full.metadata(), #
        metapaths=metapaths,
        hidden_channels=64,
        out_channels=64,
        final_out_channels=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for epoch in range(1, 51):
      for batch in tqdm(loader_dict["train"]):
        data = batch.to(device)
        y = data['driver'].y.float()
        train_mask = data['driver'].train_mask

        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.l1_loss(out[train_mask], y[train_mask])  # MAE per coerenza col tuo setup
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")



if __name__ == '__main__':
    train()
