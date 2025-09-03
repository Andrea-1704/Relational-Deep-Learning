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
    
    mp = [(dst, flip_rel(rel), src) for (src, rel, dst) in mp_outward[::-1]]
    
    return tuple(mp)






dataset = get_dataset("rel-trial", download=True)
task = get_task("rel-trial", "study-adverse", download=True)

train_table = task.get_table("train") 
val_table = task.get_table("val") 
test_table = task.get_table("test") 

print(train_table)
target_table = "studies"
target_column = "num_of_adverse_events"
node_type = target_table

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
data_official, col_stats_dict_official = make_pkey_fkey_graph(
        db_nuovo,
        col_to_stype_dict=col_to_stype_dict_nuovo,
        text_embedder_cfg = None,
        cache_dir=None  # disabled
)


graph_driver_ids = db_nuovo.table_dict[target_table].df["nct_id"].to_numpy()
id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}

train_df = train_table.df
driver_labels = train_df[target_column].to_numpy()
driver_ids = train_df["nct_id"].to_numpy()

target_vector = torch.full((len(graph_driver_ids),), float("nan"))
for i, driver_id in enumerate(driver_ids):
    if driver_id in id_to_idx:
        target_vector[id_to_idx[driver_id]] = driver_labels[i]

data_official[target_table].y = target_vector
data_official[target_table].train_mask = ~torch.isnan(target_vector)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_full, col_stats_dict_full = make_pkey_fkey_graph(
    db_nuovo,
    col_to_stype_dict=col_to_stype_dict_nuovo,
    text_embedder_cfg=None,
    cache_dir=None
)
data_full = data_full.to(device)

#retrieve the id from the driver nodes
graph_driver_ids = db_nuovo.table_dict[target_table].df["nct_id"].to_numpy()
id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}

#get the labels and the ids of the drivers from the table
train_df = train_table.df
driver_labels = train_df[target_column].to_numpy()
driver_ids = train_df["nct_id"].to_numpy()

#map the correct labels for all drivers node (which are target ones)
target_vector = torch.full((len(graph_driver_ids),), float("nan")) #inizial
for i, driver_id in enumerate(driver_ids):
    if driver_id in id_to_idx:
        target_vector[id_to_idx[driver_id]] = driver_labels[i]


data_full[target_table].y = target_vector
data_full[target_table].train_mask = ~torch.isnan(target_vector)

#take y and mask complete for the dataset:
y_full = data_full[target_table].y.float()
train_mask_full = data_full[target_table].train_mask
y_bin_full = y_full

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
lr=1e-02
wd=0



#Learning metapaths:
metapaths = [('studies', 'rev_f2p_nct_id', 'reported_event_totals')]


print(f"The final metapath are {metapaths}")




canonical = []
for mp in metapaths:
    #change to canonical:
    mp = mp.copy()
    mp_key   = to_canonical(mp)         
    
    canonical.append(mp_key)

hidden_channels = 128
out_channels = 128

model = XMetaPath2(
    data=data_official,
    col_stats_dict=col_stats_dict_official,
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
    patience=80,
    delta=0.0,
    verbose=True,
    higher_is_better = True,
    path="best_basic_model.pt"
)

best_val_metric = -math.inf 
test_table = task.get_table("test", mask_input_cols=False)
best_test_metric = -math.inf 
epochs = 500
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





