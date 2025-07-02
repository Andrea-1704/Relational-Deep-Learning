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
from torch import Tensor, batch_norm_gather_stats
from torch.nn import Embedding, ModuleDict
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_frame.data.stats import StatType

import sys
import os
sys.path.append(os.path.abspath("."))

from model.MPSGNN_Model import MPSGNN
from data_management.data import loader_dict_fn
from utils.mpsgnn_metapath_utils import binarize_targets# greedy_metapath_search
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
from utils.EarlyStopping import EarlyStopping
from utils.mpsgnn_metapath_utils import greedy_metapath_search_with_bags_learned, beam_metapath_search_with_bags_learned




def train2():
    dataset = get_dataset("rel-f1", download=True)
    task = get_task("rel-f1", "driver-position", download=True)

    train_table = task.get_table("train") #date  driverId  qualifying
    val_table = task.get_table("val") #date  driverId  qualifying
    test_table = task.get_table("test") # date  driverId

    out_channels = 1
    loss_fn = L1Loss()
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

    graph_driver_ids = db_nuovo.table_dict["drivers"].df["driverId"].to_numpy()
    id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}

    train_df = train_table.df
    driver_labels = train_df["position"].to_numpy()
    driver_ids = train_df["driverId"].to_numpy()

    target_vector = torch.full((len(graph_driver_ids),), float("nan"))
    for i, driver_id in enumerate(driver_ids):
        if driver_id in id_to_idx:
            target_vector[id_to_idx[driver_id]] = driver_labels[i]

    data_official['drivers'].y = target_vector
    data_official['drivers'].train_mask = ~torch.isnan(target_vector)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_full, col_stats_dict_full = make_pkey_fkey_graph(
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

    
    data_full['drivers'].y = target_vector
    data_full['drivers'].train_mask = ~torch.isnan(target_vector)

    #take y and mask complete for the dataset:
    y_full = data_full['drivers'].y.float()
    train_mask_full = data_full['drivers'].train_mask
    y_bin_full = binarize_targets(y_full, threshold=10)

    hidden_channels = 128
    out_channels = 64

    metapaths, metapath_counts = greedy_metapath_search_with_bags_learned(
        col_stats_dict = col_stats_dict_full,
        data=data_full,
        y=y_full,
        train_mask=train_mask_full,
        node_type='drivers',
        L_max=2, #----> tune
        channels = hidden_channels, 
        #beam_width = 4
    )  

    #now we can use the loader dict and batch work SGD
    loader_dict = loader_dict_fn(
        batch_size=256, #----> tune
        num_neighbours=128,  #----> tune
        data=data_official, 
        task=task,
        train_table=train_table, 
        val_table=val_table, 
        test_table=test_table
    )

    model = MPSGNN(
        data=data_official,
        col_stats_dict=col_stats_dict_official,
        metadata=data_full.metadata(),
        metapath_counts = metapath_counts,
        metapaths=metapaths,
        hidden_channels=hidden_channels, #----> tune
        out_channels=out_channels,    #----> tune
        final_out_channels=1, 
    ).to(device)


    optimizer = torch.optim.Adam(   #----> tune
      model.parameters(),
      lr=0.001,
      weight_decay=0
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=25)  #---> tune


    early_stopping = EarlyStopping(
        patience=30,
        delta=0.0,
        verbose=True,
        path="best_basic_model.pt"
    )

    test_table = task.get_table("test", mask_input_cols=False)
    best_val_metric = -math.inf if higher_is_better else math.inf
    best_test_metric = -math.inf if higher_is_better else math.inf
    epochs = 150
    for epoch in range(0, epochs):
      train_loss = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)
      train_pred = test(model, loader_dict["train"], device=device, task=task)
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

      





if __name__ == '__main__':
    train2()
