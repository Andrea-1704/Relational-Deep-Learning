import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import relbench
import numpy as np
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
import copy
from typing import Any, Dict, List
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_frame.data.stats import StatType

import sys
import os
sys.path.append(os.path.abspath("."))

from model.MPSGNN_Model import MPSGNN
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.mpsgnn_metapath_utils import binarize_targets # binarize_targets sarà usata qui
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
from utils.EarlyStopping import EarlyStopping
from utils.mpsgnn_metapath_utils import greedy_metapath_search_with_bags_learned, beam_metapath_search_with_bags_learned


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
    # lr=1e-02
    #wd=0
    
    
    metapaths = [[('drivers', 'rev_f2p_driverId', 'standings')]]
    metapath_count = {(('drivers', 'rev_f2p_driverId', 'standings'),): 1}





    from utils.EarlyStopping import EarlyStopping

    optimizers_to_try = ['SGD', 'Adam']
    lrs_to_try = [1e-2, 1e-3, 1e-4]
    wds_to_try = [0.0, 1e-4]
    momenta_to_try = [0.0, 0.5, 0.9]  # solo per SGD
    test_table = task.get_table("test", mask_input_cols=False)

    max_epochs = 1000
    early_patience = 50

    best_score = -math.inf
    best_config = None
    best_model_state = None

    results_log = []  # Per salvare i risultati per ogni configurazione

    for opt_name in optimizers_to_try:
        for lr in lrs_to_try:
            for wd in wds_to_try:
                momenta = [None] if opt_name != "SGD" else momenta_to_try
                for momentum in momenta:
                    print(f"\nTrying optimizer={opt_name}, lr={lr}, wd={wd}, momentum={momentum}")

                    model = MPSGNN(
                        data=data_official,
                        col_stats_dict=col_stats_dict_official,
                        metadata=data_official.metadata(),
                        metapath_counts=metapath_count,
                        metapaths=metapaths,
                        hidden_channels=hidden_channels,
                        out_channels=out_channels,
                        final_out_channels=1,
                    ).to(device)

                    if opt_name == 'Adam':
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                    elif opt_name == 'SGD':
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

                    early_stopping = EarlyStopping(
                        patience=early_patience,
                        delta=0.0,
                        verbose=False,
                        higher_is_better=True,
                        path="tmp_model.pt"
                    )

                    for epoch in range(max_epochs):
                        train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)
                        val_pred = test(model, loader_dict["val"], device=device, task=task)
                        val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)
                        val_score = val_metrics[tune_metric]

                        early_stopping(val_score, model)
                        if early_stopping.early_stop:
                            print(f"Early stopping triggered at epoch {epoch}")
                            break

                    # Carica miglior modello
                    model.load_state_dict(torch.load("tmp_model.pt", map_location=device))

                    # Val & Test
                    final_val_pred = test(model, loader_dict["val"], device=device, task=task)
                    final_val_metrics = evaluate_performance(final_val_pred, val_table, task.metrics, task=task)
                    final_f1_val = final_val_metrics[tune_metric]

                    final_test_pred = test(model, loader_dict["test"], device=device, task=task)
                    final_test_metrics = evaluate_performance(final_test_pred, test_table, task.metrics, task=task)
                    final_f1_test = final_test_metrics[tune_metric]

                    print(f"✅ Config completed: optimizer={opt_name}, lr={lr}, wd={wd}, momentum={momentum} → Val F1: {final_f1_val:.4f}, Test F1: {final_f1_test:.4f}")

                    results_log.append({
                        "optimizer": opt_name,
                        "lr": lr,
                        "wd": wd,
                        "momentum": momentum,
                        "val_f1": final_f1_val,
                        "test_f1": final_f1_test
                    })

                    if final_f1_val > best_score:
                        best_score = final_f1_val
                        best_config = (opt_name, lr, wd, momentum)
                        best_model_state = copy.deepcopy(model.state_dict())

    # 🏁 Riepilogo finale
    print("\n=== RIEPILOGO CONFIGURAZIONI PROVATE ===")
    for r in results_log:
        print(f"[{r['optimizer']} | lr={r['lr']} | wd={r['wd']} | momentum={r['momentum']}] → Val F1: {r['val_f1']:.4f} | Test F1: {r['test_f1']:.4f}")

    print(f"\n🏆 Best config: optimizer={best_config[0]}, lr={best_config[1]}, wd={best_config[2]}, momentum={best_config[3]} with val F1={best_score:.4f}")








    # model = MPSGNN(
    #     data=data_official,
    #     col_stats_dict=col_stats_dict_official,
    #     metadata=data_official.metadata(),
    #     metapath_counts = metapath_counts,
    #     metapaths=metapaths,
    #     hidden_channels=hidden_channels,
    #     out_channels=out_channels,
    #     final_out_channels=1,
    # ).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    # #EPOCHS:
    # epochs = 100
    # test_table = task.get_table("test", mask_input_cols=False)
    # best_test_metrics = -math.inf if higher_is_better else math.inf
    # for _ in range(0, epochs):
    #     train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)
    #     test_pred = test(model, loader_dict["test"], device=device, task=task)
    #     test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)
    #     if test_metrics[tune_metric] > best_test_metrics and higher_is_better:
    #         best_test_metrics = test_metrics[tune_metric]
    #     if test_metrics[tune_metric] < best_test_metrics and not higher_is_better:
    #         best_test_metrics = test_metrics[tune_metric]
    # print(f"We obtain F1 test loss equal to {best_test_metrics}")

    

if __name__ == '__main__':
    train2()