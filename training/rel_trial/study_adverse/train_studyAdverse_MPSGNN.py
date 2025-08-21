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

from model.XMetapath_Model import MPSGNN
from data_management.data import loader_dict_fn
from utils.XMetapath_utils.XMetapath_metapath_utils import binarize_targets# greedy_metapath_search
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
from utils.EarlyStopping import EarlyStopping
from utils.XMetapath_utils.XMetapath_metapath_utils import greedy_metapath_search_with_bags_learned, beam_metapath_search_with_bags_learned
from utils.XMetapath_utils.XMetapath_metapath_utils import greedy_metapath_search_with_bags_learned_3



def train2():

    dataset = get_dataset("rel-trial", download=True)
    task = get_task("rel-trial", "study-adverse", download=True)

    train_table = task.get_table("train") #date  driverId  qualifying
    val_table = task.get_table("val") #date  driverId  qualifying
    test_table = task.get_table("test") # date  driverId

    print(train_table)
    target_table = "studies"
    target_column = "num_of_adverse_events"
    

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
    y_full = data_full[target_column].y.float()
    train_mask_full = data_full[target_column].train_mask
    y_bin_full = binarize_targets(y_full, threshold=10)
    
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

    metapaths, metapath_counts = greedy_metapath_search_with_bags_learned_3(
        col_stats_dict = col_stats_dict_official,
        data=data_official,
        db= db_nuovo,
        node_id='driverId',
        train_mask=train_mask_full,
        node_type='drivers',
        L_max=3,
        channels = hidden_channels,
        number_of_metapaths = 4,     
        out_channels = out_channels,
        hidden_channels = hidden_channels, 
        loader_dict = loader_dict,
        lr = lr,
        wd = wd,
        task = task,
        loss_fn= loss_fn, 
        epochs = 100, 
        tune_metric = tune_metric,
        higher_is_better= higher_is_better
    )

    print(f"\nfinal metapaths are {metapaths}\n")
    print(f"\nmetapaths counts are {metapath_counts}\n")

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

    lr=1e-02
    wd=0

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    

    scheduler = CosineAnnealingLR(optimizer, T_max=25)  #---> tune


    early_stopping = EarlyStopping(
        patience=60,
        delta=0.0,
        verbose=True,
        path="best_basic_model.pt"
    )

    test_table = task.get_table("test", mask_input_cols=False)
    best_test_metrics = -math.inf if higher_is_better else math.inf
    epochs = 200
    for _ in range(0, epochs):
        train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)
        test_pred = test(model, loader_dict["test"], device=device, task=task)
        test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)
        if test_metrics[tune_metric] > best_test_metrics and higher_is_better:
            best_test_metrics = test_metrics[tune_metric]
        if test_metrics[tune_metric] < best_test_metrics and not higher_is_better:
            best_test_metrics = test_metrics[tune_metric]
    print(f"We obtain F1 test loss equal to {best_test_metrics}")

      





if __name__ == '__main__':
    train2()
