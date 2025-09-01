"""
This function is used for benchmarking the model against the others and improve
the model performance considering the same metapath.

In this executor, we are first going to learn meaningful metapaths using reinforcement learning, and 
then using them for driver position task.
"""

import torch
import torch.nn as nn
import os
from relbench.datasets import get_dataset
from relbench.tasks import get_task
import math
from torch_geometric.seed import seed_everything
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import math
import torch.nn as nn
from torch.nn import L1Loss

import sys
import os
sys.path.append(os.path.abspath("."))

from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.EarlyStopping import EarlyStopping
from model.XMetaPath2 import XMetaPath2
from utils.utils import evaluate_performance, test, train
from utils.XMetapath_utils.XMetaPath_extension4 import RLAgent, warmup_rl_agent, final_metapath_search_with_rl

# utility functions:
def flip_rel(rel_name: str) -> str:
    return rel_name[4:] if rel_name.startswith("rev_") else f"rev_{rel_name}"

def to_canonical(mp_outward):
    # mp_outward: [(src, rel, dst), ...] dalla costruzione RL (parte da 'drivers')
    mp = [(dst, flip_rel(rel), src) for (src, rel, dst) in mp_outward[::-1]]
    assert mp[-1][2] == "drivers"
    return tuple(mp)






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
    #text_embedder_cfg=text_embedder_cfg,
    text_embedder_cfg = None,
    cache_dir=None  # disabled
)
node_type="drivers"

loader_dict = loader_dict_fn(
    batch_size=512,
    num_neighbours=256,
    data=data,
    task=task,
    train_table=train_table,
    val_table=val_table,
    test_table=test_table
)

import torch

node_type = "drivers"

# 1) Recupera i pkey dei driver nel train split
train_driver_ids = torch.tensor(
    train_table["driverId"].unique(), dtype=torch.long
)  # 'driverId' è nel tuo train_table :contentReference[oaicite:1]{index=1}

# 2) Costruisci una mappa pkey->indice di nodo per 'drivers'
id_to_idx = None

# (a) Preferibile: se il grafo contiene i pkey in un tensore 'node_id' allineato agli indici 0..num_nodes-1
if hasattr(data[node_type], "node_id"):
    pkeys = data[node_type].node_id  # tensor [num_nodes] di pkey (driverId) in ordine di indice
    # Costruisci mapping pkey -> indice
    id_to_idx = {int(pk): i for i, pk in enumerate(pkeys.tolist())}

# (b) Fallback: usa l'indice della tabella nel Database/TensorFrame
if id_to_idx is None:
    # Prova TensorFrame.index (se hai usato TorchFrame/RelBench standard)
    try:
        index_values = db_nuovo.table_dict[node_type].tf.index.tolist()
    except Exception:
        # Altrimenti usa l'indice del DataFrame sottostante
        index_values = db_nuovo.table_dict[node_type].df.index.tolist()
    id_to_idx = {int(pk): i for i, pk in enumerate(index_values)}

# 3) Mappa i driverId del train agli indici di nodo
train_node_idx = []
missing = []
for pk in train_driver_ids.tolist():
    if pk in id_to_idx:
        train_node_idx.append(id_to_idx[pk])
    else:
        missing.append(pk)

train_node_idx = torch.tensor(train_node_idx, dtype=torch.long)

# 4) Crea la mask sul tipo di nodo corretto ('drivers')
train_mask_full = torch.zeros(data[node_type].num_nodes, dtype=torch.bool)
train_mask_full[train_node_idx] = True

# 5) Sanity checks utili
assert train_mask_full.dtype == torch.bool
assert train_mask_full.numel() == data[node_type].num_nodes
print(f"[Info] Train drivers nella mask: {int(train_mask_full.sum())}/{data[node_type].num_nodes}")
if missing:
    print(f"[Warning] {len(missing)} driverId nel train non mappati a nodi: esempi {missing[:5]}")


#Learning metapaths:
agent = RLAgent(tau=1.0, alpha=0.5)
"""
We build a single agent and perform sequential warmups
"""
agent.best_score_by_path_global.clear() #azzera registro punteggi

warmup_rl_agent(
    agent=agent,
    data=data,
    loader_dict=loader_dict,
    task=task,
    loss_fn=loss_fn,
    tune_metric=tune_metric,
    higher_is_better=higher_is_better,
    train_mask=train_mask_full,
    node_type='drivers',
    col_stats_dict=col_stats_dict,
    num_episodes=5,   
    L_max=7,          
    epochs=5         
)

#Extract the Top-K metapaths found out with warmup:
K = 3
global_best_map = agent.best_score_by_path_global

agent.tau = 0.3   # meno esplorazione
agent.alpha = 0.2 # update più conservativo

metapaths, metapath_count = final_metapath_search_with_rl(
    agent=agent,
    data=data,
    loader_dict=loader_dict,
    task=task,
    loss_fn=loss_fn,
    tune_metric=tune_metric,
    higher_is_better=higher_is_better,
    train_mask=train_mask_full,
    node_type='drivers',
    col_stats_dict=col_stats_dict,
    L_max=7,                 
    epochs=100,
    number_of_metapaths=K    
)



print(f"The final metapath are {metapaths}")




canonical = []
for mp in metapaths:
    #change to canonical:
    mp = mp.copy()
    mp_key   = to_canonical(mp)         
    assert mp_key[-1][2] == node_type, \
        f"Il meta-path canonico deve terminare su '{node_type}', invece termina su '{mp_key[-1][2]}'"
    canonical.append(mp_key)

hidden_channels = 128
out_channels = 128

model = XMetaPath2(
    data=data,
    col_stats_dict=col_stats_dict,
    #metapath_counts = metapath_count, 
    metapaths=canonical,               
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    final_out_channels=1,
).to(device)



lr=1e-02
wd = 0

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

scheduler = CosineAnnealingLR(optimizer, T_max=25)

early_stopping = EarlyStopping(
    patience=60,
    delta=0.0,
    verbose=True,
    higher_is_better = True,
    path="best_basic_model.pt"
)

best_val_metric = -math.inf 
test_table = task.get_table("test", mask_input_cols=False)
best_test_metric = -math.inf 
epochs = 150
for epoch in range(0, epochs):
    train_loss = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

    train_pred = test(model, loader_dict["train"], device=device, task=task)
    val_pred = test(model, loader_dict["val"], device=device, task=task)
    test_pred = test(model, loader_dict["test"], device=device, task=task)
    
    train_metrics = evaluate_performance(train_pred, train_table, task.metrics, task=task)
    val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)
    test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)

    #scheduler.step(val_metrics[tune_metric])

    if (higher_is_better and val_metrics[tune_metric] > best_val_metric):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())

    if (higher_is_better and test_metrics[tune_metric] > best_test_metric):
        best_test_metric = test_metrics[tune_metric]
        state_dict_test = copy.deepcopy(model.state_dict())

    current_lr = optimizer.param_groups[0]["lr"]
    
    print(f"Epoch: {epoch:02d}, Train {tune_metric}: {train_metrics[tune_metric]:.2f}, Validation {tune_metric}: {val_metrics[tune_metric]:.2f}, Test {tune_metric}: {test_metrics[tune_metric]:.2f}, LR: {current_lr:.6f}")

    # early_stopping(val_metrics[tune_metric], model)

    # if early_stopping.early_stop:
    #     print(f"Early stopping triggered at epoch {epoch}")
    #     break



print(f"best validation results: {best_val_metric}")
print(f"best test results: {best_test_metric}")