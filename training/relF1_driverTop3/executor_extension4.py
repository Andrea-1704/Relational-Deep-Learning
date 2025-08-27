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
from utils.EarlyStopping import EarlyStopping
from model.XMetaPath import XMetapath, interpret_attention
from utils.utils import evaluate_performance, test, train
from utils.XMetapath_utils.XMetapath_extension4 import RLAgent, warmup_rl_agent, final_metapath_search_with_rl

task_name = "driver-top3"

dataset = get_dataset("rel-f1", download=True)
task = get_task("rel-f1", task_name, download=True)
task_type = task.task_type

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")

out_channels = 1

#loss_fn = nn.BCEWithLogitsLoss()
tune_metric = "f1"
higher_is_better = True #is referred to the tune metric

seed_everything(42) #We should remember to try results 5 times with
#different seed values to provide a confidence interval over results.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "./data"

db = dataset.get_db()
col_to_stype_dict = get_stype_proposal(db)
db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)

data_official, col_stats_dict_official = make_pkey_fkey_graph(
    db_nuovo,
    col_to_stype_dict=col_to_stype_dict_nuovo,
    text_embedder_cfg = None,
    cache_dir=None
)
#do not use the textual information: this db is mostly not textual

graph_driver_ids = db_nuovo.table_dict["drivers"].df["driverId"].to_numpy()
id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}


train_df_raw = train_table.df
driver_ids_raw = train_df_raw["driverId"].to_numpy()
qualifying_positions = train_df_raw["qualifying"].to_numpy() #labels (train)


binary_top3_labels_raw = qualifying_positions #do not need to binarize 
#since the task is already a binary classification task


target_vector_official = torch.full((len(graph_driver_ids),), float("nan")) #inizialize a vector with all "nan" elements
for i, driver_id in enumerate(driver_ids_raw):
    if driver_id in id_to_idx:#if the driver is in the training
        target_vector_official[id_to_idx[driver_id]] = binary_top3_labels_raw[i]


data_official['drivers'].y = target_vector_official.float()
data_official['drivers'].train_mask = ~torch.isnan(target_vector_official)
y_full = data_official['drivers'].y.float()
train_mask_full = data_official['drivers'].train_mask
num_pos = (y_full[train_mask_full] == 1).sum()
num_neg = (y_full[train_mask_full] == 0).sum()
ratio = (num_neg / num_pos) if num_pos > 0 else 1.0
pos_weight = torch.tensor([ratio], device=device)
data_official['drivers'].y = target_vector_official

# --- ONE-TIME index maps
from dataclasses import dataclass

@dataclass
class IndexMaps:
    pk_to_idx: dict            # {PK -> internal index}
    idx_to_pk: list            # [internal index -> PK] as list/array

def build_index_maps(db, data, node_type: str, node_id_col: str) -> IndexMaps:
    # 1) Extract PKs in the SAME ORDER used to build the hetero graph
    pk_series = db.table_dict[node_type].df[node_id_col]
    idx_to_pk = pk_series.to_numpy().tolist()
    # 2) Build reverse map
    pk_to_idx = {int(pk): idx for idx, pk in enumerate(idx_to_pk)}
    # 3) Sanity: sizes match
    assert len(idx_to_pk) == data[node_type].num_nodes, \
        f"Size mismatch for {node_type}: table rows={len(idx_to_pk)} vs num_nodes={data[node_type].num_nodes}"
    # 4) Sanity: PKs unique
    assert len(pk_to_idx) == len(idx_to_pk), "Duplicate PKs detected in table order."
    return IndexMaps(pk_to_idx=pk_to_idx, idx_to_pk=idx_to_pk)


# Ricava gli ID dei driver nella validation table
val_df_raw = val_table.df
val_driver_ids = val_df_raw["driverId"].to_numpy()

# Costruisci la mask come boolean mask sul vettore completo
val_mask = torch.tensor([driver_id in val_driver_ids for driver_id in graph_driver_ids])
data_official["drivers"].val_mask = val_mask


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
lr=1e-02
wd=0
node_type="drivers"

agent = RLAgent(tau=1.0, alpha=0.5)
"""
We build a single agent and perform sequential warmups
"""
agent.best_score_by_path_global.clear() #azzera registro punteggi

warmup_rl_agent(
    agent=agent,
    data=data_official,
    loader_dict=loader_dict,
    task=task,
    loss_fn=loss_fn,
    tune_metric=tune_metric,
    higher_is_better=higher_is_better,
    train_mask=train_mask_full,
    node_type='drivers',
    col_stats_dict=col_stats_dict_official,
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
    data=data_official,
    loader_dict=loader_dict,
    task=task,
    loss_fn=loss_fn,
    tune_metric=tune_metric,
    higher_is_better=higher_is_better,
    train_mask=train_mask_full,
    node_type='drivers',
    col_stats_dict=col_stats_dict_official,
    L_max=7,                 
    epochs=100,
    number_of_metapaths=K    
)



print(f"The final metapath are {metapaths}")



#train the final model on the chosen paths
lr=1e-02
wd=0
    
model = XMetapath(
    data=data_official,
    col_stats_dict=col_stats_dict_official,
    metapath_counts = metapath_count,  # keep your existing counts
    metapaths=metapaths,               # NEW: train with global Top-K
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    final_out_channels=1,
).to(device)

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
epochs = 100
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

    early_stopping(val_metrics[tune_metric], model)

    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        break

#giving interpretability (local interpretability)
meta_names = []
for m in metapaths:
  cur_metapath=m[0][0]
  for metapath in m:
    source = metapath[0]
    dst = metapath[2]
    cur_metapath=cur_metapath+"->"+dst
  meta_names.append(cur_metapath)
for batch in loader_dict["test"]:
    batch.to(device)
    results = interpret_attention(
        model=model,
        batch=batch,
        metapath_names=meta_names,
        entity_table="drivers"
    )
    print(f"result of interpretability are {results}")

print(f"best validation results: {best_val_metric}")
print(f"best test results: {best_test_metric}")
