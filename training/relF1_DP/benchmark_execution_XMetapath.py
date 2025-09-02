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

# utility functions:
def flip_rel(rel_name: str) -> str:
    return rel_name[4:] if rel_name.startswith("rev_") else f"rev_{rel_name}"

def to_canonical(mp_outward):
    # mp_outward: [(src, rel, dst), ...] dalla costruzione RL (parte da 'drivers')
    mp = [(dst, flip_rel(rel), src) for (src, rel, dst) in mp_outward[::-1]]
    assert mp[-1][2] == node_type , f"Expected {node_type}, got {mp[-1][2]}"
    return tuple(mp)

loader_dict = loader_dict_fn(
    batch_size=512,
    num_neighbours=256,
    data=data,
    task=task,
    train_table=train_table,
    val_table=val_table,
    test_table=test_table
)

#way 1:
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


data['drivers'].y = target_vector
data['drivers'].train_mask = ~torch.isnan(target_vector)

#take y and mask complete for the dataset:
y_full = data['drivers'].y.float()
train_mask_full = data['drivers'].train_mask
y_bin_full = y_full

print(f"y full : {y_full}")

#way2:

# --- Build data['drivers'].y from the train_table (RelBench keeps labels in task tables) ---
# import pandas as pd
# import torch

# node_type = "drivers"

# def table_df(t):
#     if hasattr(t, "df"):
#         return t.df
#     if hasattr(t, "to_pandas"):
#         return t.to_pandas()
#     raise TypeError("Unsupported Table type (no .df / .to_pandas)")

# df_train = table_df(train_table).copy()

# # 1) individua la colonna label in modo robusto:
# #    - rimuoviamo timestamp e pkey
# pk_col = node_type    # per rel-f1 driver table
# time_col = "date"
# candidates = [c for c in df_train.columns if c not in {pk_col, time_col}]
# # tieni solo numeric (la label è numerica per 'driver-position' con MAE)
# num_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df_train[c])]
# if "label" in df_train.columns:
#     target_col = "label"
# elif len(num_candidates) == 1:
#     target_col = num_candidates[0]
# else:
#     # euristica: scegli la colonna numerica con varianza maggiore
#     target_col = df_train[num_candidates].var().sort_values(ascending=False).index[0]

# # 2) mappa pkey (driverId) -> indice di nodo 'drivers'
# try:
#     index_values = db_nuovo.table_dict[node_type].tf.index.tolist()
# except Exception:
#     index_values = db_nuovo.table_dict[node_type].df.index.tolist()
# id_to_idx = {int(pk): i for i, pk in enumerate(index_values)}

# # 3) aggrega per driver: media della label nel train
# agg = (
#     df_train[[pk_col, target_col]]
#     .groupby(pk_col, as_index=False)
#     .mean()
# )

# y = torch.full((data[node_type].num_nodes,), float("nan"), dtype=torch.float32)
# for pk, val in zip(agg[pk_col].tolist(), agg[target_col].tolist()):
#     if int(pk) in id_to_idx:
#         y[id_to_idx[int(pk)]] = float(val)

# # 4) opzionale: riempi i NaN con la media globale del train (o lasciali NaN se l’RL li ignora via mask)
# global_mean = torch.tensor(pd.to_numeric(agg[target_col], errors="coerce").mean(), dtype=torch.float32)
# y = torch.where(torch.isnan(y), global_mean, y)

# # 5) attacca al grafo
# data[node_type].y = y
# print(f"[Info] y attached to data['{node_type}']: {int(torch.isfinite(y).sum())}/{y.numel()} finite labels")


# # --- Build train_mask_full dai driverId del train split ---
# train_driver_ids = torch.as_tensor(
#     table_df(train_table)["driverId"].unique(), dtype=torch.long
# )

# mapped = [id_to_idx[pk] for pk in train_driver_ids.tolist() if int(pk) in id_to_idx]
# train_node_idx = torch.tensor(mapped, dtype=torch.long)

# train_mask_full = torch.zeros(data[node_type].num_nodes, dtype=torch.bool)
# train_mask_full[train_node_idx] = True

# print(f"[Info] Train mask: {int(train_mask_full.sum())}/{data[node_type].num_nodes} drivers nel train")


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
    num_episodes=3,   
    L_max=4,          
    epochs=3         
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
    L_max=4,                 
    epochs=10,
    number_of_metapaths=K    
)



print(f"The final metapath are {metapaths}")




canonical = []
for mp in metapaths:
    #change to canonical:
    mp = mp.copy()
    mp_key   = to_canonical(mp)         
    # assert mp_key[-1][2] == node_type, \
    #     f"Il meta-path canonico deve terminare su '{node_type}', invece termina su '{mp_key[-1][2]}'"
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



lr=0.0005
wd = 0

#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=0
)
#scheduler = CosineAnnealingLR(optimizer, T_max=25)

# early_stopping = EarlyStopping(
#     patience=80,
#     delta=0.0,
#     verbose=True,
#     higher_is_better = True,
#     path="best_basic_model.pt"
# )

best_val_metric = -math.inf 
test_table = task.get_table("test", mask_input_cols=False)
best_test_metric = -math.inf 
epochs = 200
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