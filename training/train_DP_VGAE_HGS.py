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


from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from torch_geometric.loader import NeighborLoader
import pyg_lib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import ModuleDict, Linear
import torch.nn.functional as F
from torch import nn
import random
from model.HGraphSAGE import Model
from model.SelfAttention_attempt2 import MyModel
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from VGAE.Utils_VGAE import train_vgae
from utils.EarlyStopping import EarlyStopping
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train





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
#db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)
db_nuovo, col_to_stype_dict_nuovo = db, col_to_stype_dict

class LightweightGloveEmbedder:
    def __init__(self, device=None):
        self.device = device
        self.embeddings = defaultdict(lambda: np.zeros(300))
        self._load_embeddings()

    def _load_embeddings(self):
      try:
          path = "glove.6B.300d.txt"
          with open(path, encoding="utf-8") as f:
              for line in f:
                  parts = line.strip().split()
                  word = parts[0]
                  vector = np.array(parts[1:], dtype=np.float32)
                  self.embeddings[word] = vector
          #print(f"Loaded {len(self.embeddings)} GloVe embeddings.")
      except Exception as e:
          print(f"Failed to load GloVe: {e}")

    def __call__(self, sentences):
        results = []
        for text in sentences:
            words = text.lower().split()
            vectors = [self.embeddings[w] for w in words if w in self.embeddings]
            if vectors:
                # print("trovato")
                avg_vector = np.mean(vectors, axis=0)
            else:
                #print("non trovato")
                #print(f"Numero parole in embedding: {len(self.embeddings)}")

                avg_vector = np.zeros(300)
            results.append(avg_vector)

        tensor = torch.tensor(np.array(results), dtype=torch.float32)
        return tensor.to(self.device) if self.device else tensor

text_embedder_cfg = TextEmbedderConfig(
    text_embedder=LightweightGloveEmbedder(device=device), batch_size=256
)

# Create the graph
data, col_stats_dict = make_pkey_fkey_graph(
    db_nuovo,
    col_to_stype_dict=col_to_stype_dict_nuovo,
    text_embedder_cfg=text_embedder_cfg,
    #text_embedder_cfg = None,
    cache_dir=None  # disabled
)

print(f"il valore di col_stats_dict nel main è {col_stats_dict}")


# pre training phase with the VGAE
channels = 64#512

model = MyModel(
    #db=db_nuovo,
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=2,
    channels=channels,
    out_channels=1,
    aggr="max",
    norm="batch_norm",
).to(device)



loader_dict = loader_dict_fn(
    batch_size=64,#512, 
    num_neighbours=32,#256, 
    data=data, 
    task=task,
    train_table=train_table, 
    val_table=val_table, 
    test_table=test_table
)

for batch in loader_dict["train"]:
    edge_types=batch.edge_types
    break


model = train_vgae(
    model=model,
    loader_dict=loader_dict,
    edge_types=edge_types,
    encoder_out_dim=channels,
    entity_table=task.entity_table,
    latent_dim=16,#128,
    hidden_dim=32,#256,
    epochs=500,
    device=device
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=0
)

scheduler = CosineAnnealingLR(optimizer, T_max=100)


early_stopping = EarlyStopping(
    patience=30,
    delta=0.0,
    verbose=True,
    path="best_basic_model.pt"
)

loader_dict = loader_dict_fn(
    batch_size=512, 
    num_neighbours=256, 
    data=data, 
    task=task,
    train_table=train_table, 
    val_table=val_table, 
    test_table=test_table
)




# Training loop
epochs = 100

state_dict = None
test_table = task.get_table("test", mask_input_cols=False)
best_val_metric = -math.inf if higher_is_better else math.inf
best_test_metric = -math.inf if higher_is_better else math.inf
for epoch in range(1, epochs + 1):
    train_loss = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

    train_pred = test(model, loader_dict["train"], device=device, task=task)
    train_metrics = evaluate_performance(train_pred, train_table, task.metrics, task=task)
    train_mae_preciso = evaluate_on_full_train(model, loader_dict["train"], device=device, task=task)

    val_pred = test(model, loader_dict["val"], device=device, task=task)
    val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)

    test_pred = test(model, loader_dict["test"], device=device, task=task)
    test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)

    scheduler.step(val_metrics[tune_metric])

    if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
            not higher_is_better and val_metrics[tune_metric] < best_val_metric
    ):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())

    #test:
    if (higher_is_better and test_metrics[tune_metric] > best_test_metric) or (
            not higher_is_better and test_metrics[tune_metric] < best_test_metric
    ):
        best_test_metric = test_metrics[tune_metric]
        state_dict_test = copy.deepcopy(model.state_dict())

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch: {epoch:02d}, Train {tune_metric}: {train_mae_preciso:.2f}, Validation {tune_metric}: {val_metrics[tune_metric]:.2f}, Test {tune_metric}: {test_metrics[tune_metric]:.2f}, LR: {current_lr:.6f}")

    early_stopping(val_metrics[tune_metric], model)

    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        break
print(f"best validation results: {best_val_metric}")
print(f"best test results: {best_test_metric}")
