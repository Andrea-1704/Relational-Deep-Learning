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
from mpsgnn_metapath_utils import binarize_targets, greedy_metapath_search
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

    loader_dict = loader_dict_fn(
        batch_size=512, 
        num_neighbours=256, 
        data=data, 
        task=task,
        train_table=train_table, 
        val_table=val_table, 
        test_table=test_table
    )

    #data = load_relbench_f1(split='train')
    
    # data = data.to(device)

    # y = data['driver'].y.float()
    # train_mask = data['driver'].train_mask

    # y_bin = binarize_targets(y, threshold=10)
    #batch sfould be created once for the entire dataset
    metapaths = greedy_metapath_search(data, y_bin, train_mask, node_type='driver', L_max=2)

    model = MPSGNN(
        metadata=data.metadata(),
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
        y_bin = binarize_targets(y, threshold=10)
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.mse_loss(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")


if __name__ == '__main__':
    train()
