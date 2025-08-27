import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import relbench
import numpy as np
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from tqdm import tqdm
from torch_geometric.seed import seed_everything
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph
import copy

import sys
import os
sys.path.append(os.path.abspath("."))

from model.XMetaPath import XMetaPath
import math
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
from utils.EarlyStopping import EarlyStopping
from utils.XMetapath_utils.XMetaPath_metapath_utils import greedy_metapath_search_with_bags_learned, greedy_metapath_search_with_bags_learned_2, greedy_metapath_search_with_bags_learned_3, beam_metapath_search_with_bags_learned_2
#from utils.mapping_utils import get_global_to_local_id_map

def train2():
    dataset = get_dataset("rel-f1", download=True)
    task = get_task("rel-f1", "driver-top3", download=True)

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


    target_vector_official = torch.full((len(graph_driver_ids),), float("nan"))
    for i, driver_id in enumerate(driver_ids_raw):
        if driver_id in id_to_idx:#if the driver is in the training
            target_vector_official[id_to_idx[driver_id]] = binary_top3_labels_raw[i]


    data_official['drivers'].y = target_vector_official.float()
    data_official['drivers'].train_mask = ~torch.isnan(target_vector_official)
    y_full = data_official['drivers'].y.float()
    train_mask_full = data_official['drivers'].train_mask
    num_pos = (y_full[train_mask_full] == 1).sum()
    num_neg = (y_full[train_mask_full] == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], device=device)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    hidden_channels = 128
    out_channels = 128

    loader_dict = loader_dict_fn(
        batch_size=1024,
        num_neighbours=512,
        data=data_official,
        task=task,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table
    )
    lr=1e-02
    wd=0
    
    
    metapaths = [[('drivers', 'rev_f2p_driverId', 'standings')]]
    metapath_counts = {(('drivers', 'rev_f2p_driverId', 'standings'),): 1}
    model = XMetaPath(
        data=data_official,
        col_stats_dict=col_stats_dict_official,
        metapath_counts = metapath_counts,
        metapaths=metapaths,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        final_out_channels=1,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    ### EXPLAINABILITY
    # for batch in loader_dict["train"]:
    #     model.eval()
    #     report = model.explain_instance(
    #         batch=batch, 
    #         entity_table="drivers",
    #         target_idx=1,     # l’indice del driver di cui vuoi spiegare la predizione
    #         metapath_names=["drv→race→circ→race→drv", "drv→team→race→drv", "..."],
    #         top_k_metapaths=3,
    #         top_k_nodes=5
    #     )
    #     print(report)
    #     break
    ### EXPLAINABILITY
    
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

    #   early_stopping(val_metrics[tune_metric], model)

    #   if early_stopping.early_stop:
    #       print(f"Early stopping triggered at epoch {epoch}")
    #       break
    print(f"best validation results: {best_val_metric}")
    print(f"best test results: {best_test_metric}")


if __name__ == '__main__':
    train2()