"""
This function is used for benchmarking the model against the others and improve
the model performance considering the same metapath.
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

import sys
import os
sys.path.append(os.path.abspath("."))

from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from model.XMetaPath2 import XMetaPath2
from utils.utils import evaluate_performance, test, train
from utils.XMetapath_utils.solving_problems_AVITO import RLAgent, warmup_rl_agent, final_metapath_search_with_rl


# utility functions:
#############################################
def flip_rel(rel_name: str) -> str:
    return rel_name[4:] if rel_name.startswith("rev_") else f"rev_{rel_name}"
def to_canonical(mp_outward):
    if mp_outward[-1][2] == node_type:
        return mp_outward #if already canonical leave it as it is.
    mp = [(dst, flip_rel(rel), src) for (src, rel, dst) in mp_outward[::-1]]
    #The assert is done in the caller (see code below)
    return tuple(mp)
#############################################


#Configuration for the task:
#############################################
task_name = "user-visits"
db_name = "rel-avito"
node_id = "UserID"
target = "num_click"
node_type = "UserInfo"
dataset = get_dataset(db_name, download=True)
task = get_task(db_name, task_name, download=True)
task_type = task.task_type
out_channels = 1
tune_metric = "roc_auc"
higher_is_better = True
#############################################


train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")
seed = 42
seed_everything(seed) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "./data"


#Configuration for a binary task: Node (Binary) Classification:
#############################################
db = dataset.get_db()
col_to_stype_dict = get_stype_proposal(db)
db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)
data_official, col_stats_dict_official = make_pkey_fkey_graph(
    db_nuovo,
    col_to_stype_dict=col_to_stype_dict_nuovo,
    text_embedder_cfg = None,
    cache_dir=None
)
df = db_nuovo.table_dict[node_type].df
print(df.columns.tolist())
graph_driver_ids = db_nuovo.table_dict[node_type].df[node_id].to_numpy()
id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}
train_df_raw = train_table.df
driver_ids_raw = train_df_raw[node_id].to_numpy()
print(f"this is train_df_raw  {train_df_raw}")
binary_top3_labels_raw = train_df_raw[target].to_numpy() 
target_vector_official = torch.full((len(graph_driver_ids),), float("nan")) #inizialize a vector with all "nan" elements

for i, driver_id in enumerate(driver_ids_raw):
    if driver_id in id_to_idx:#if the driver is in the training
        target_vector_official[id_to_idx[driver_id]] = float(binary_top3_labels_raw[i])

data_official[node_type].y = target_vector_official.float()
data_official[node_type].train_mask = ~torch.isnan(target_vector_official)
y_full = data_official[node_type].y.float()
train_mask_full = data_official[node_type].train_mask
num_pos = (y_full[train_mask_full] == 1).sum()
num_neg = (y_full[train_mask_full] == 0).sum()
ratio = (num_neg / num_pos) if num_pos > 0 else 1.0
pos_weight = torch.tensor([ratio], device=device)
data_official[node_type].y = target_vector_official
val_df_raw = val_table.df
val_driver_ids = val_df_raw[node_id].to_numpy()
val_mask = torch.tensor([driver_id in val_driver_ids for driver_id in graph_driver_ids])
data_official[node_type].val_mask = val_mask
#############################################



loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
hidden_channels = 128
out_channels = 128
loader_dict = loader_dict_fn(
    batch_size=512,
    num_neighbours=256,
    data=data_official,
    task=task,
    train_table=train_table,
    val_table=val_table,
    test_table=test_table
)



#Learn the most useful metapaths:
#############################################
agent = RLAgent(tau=1.0, alpha=0.5)
agent.best_score_by_path_global.clear() 
warmup_rl_agent(
    agent=agent,
    data=data_official,
    loader_dict=loader_dict,
    task=task,
    loss_fn=loss_fn,
    tune_metric=tune_metric,
    higher_is_better=higher_is_better,
    train_mask=train_mask_full,
    node_type=node_type,
    col_stats_dict=col_stats_dict_official,
    num_episodes=15,   
    L_max=7,          
    epochs=8        
)
K = 3
global_best_map = agent.best_score_by_path_global
agent.tau = 0.3   
agent.alpha = 0.2 
metapaths, metapath_count = final_metapath_search_with_rl(
    agent=agent,
    data=data_official,
    loader_dict=loader_dict,
    task=task,
    loss_fn=loss_fn,
    tune_metric=tune_metric,
    higher_is_better=higher_is_better,
    train_mask=train_mask_full,
    node_type=node_type,
    col_stats_dict=col_stats_dict_official,
    L_max=4,                 
    epochs=5,
    number_of_metapaths=K    
)
canonical = []
for mp in metapaths:
    mp = mp.copy()
    mp_key   = to_canonical(mp)         
    assert mp_key[-1][2] == node_type, \
        f"Il meta-path canonico deve terminare su '{node_type}', invece termina su '{mp_key[-1][2]}'"
    canonical.append(mp_key)
print(f"Canonical metapaths are: {canonical}")
#############################################


#Train the final model with the metapaths found:
#############################################
model = XMetaPath2(
    data=data_official,
    col_stats_dict=col_stats_dict_official,
    metapaths=canonical,               
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    final_out_channels=1,
).to(device)
lr=0.0005
wd = 0
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=wd
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
    if (higher_is_better and val_metrics[tune_metric] > best_val_metric):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())
    if (higher_is_better and test_metrics[tune_metric] > best_test_metric):
        best_test_metric = test_metrics[tune_metric]
        state_dict_test = copy.deepcopy(model.state_dict())
    print(f"Epoch: {epoch:02d}, Train {tune_metric}: {train_metrics[tune_metric]:.2f}, Validation {tune_metric}: {val_metrics[tune_metric]:.2f}, Test {tune_metric}: {test_metrics[tune_metric]:.2f}")
#############################################

#Final results:
print(f"best validation results: {best_val_metric}")
print(f"best test results: {best_test_metric}")