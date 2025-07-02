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
    loss_fn = nn.BCEWithLogitsLoss()
    tune_metric = "f1"
    higher_is_better = True

    seed_everything(42)
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

    graph_driver_ids = db_nuovo.table_dict["drivers"].df["driverId"].to_numpy()
    id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}

    # --- INIZIO MODIFICA RIVEDUTA BASATA SULLA TUA CORREZIONE ('qualifying') ---
    
    # Estrai i driverId e le posizioni di 'qualifying' dalla train_table
    train_df_raw = train_table.df
    driver_ids_raw = train_df_raw["driverId"].to_numpy()
    qualifying_positions = train_df_raw["qualifying"].to_numpy()

    # Binarizza le posizioni di 'qualifying': 1 se <= 3, 0 altrimenti (per "top3")
    # binarize_targets richiede un tensore PyTorch e restituisce un tensore.
    # Applica una soglia di 3 per definire "top3".
    binary_top3_labels_raw = binarize_targets(torch.from_numpy(qualifying_positions), threshold=3).numpy().astype(float)


    # Mappa le etichette binarie ai driverId presenti nel grafo completo
    target_vector_official = torch.full((len(graph_driver_ids),), float("nan"))
    for i, driver_id in enumerate(driver_ids_raw):
        if driver_id in id_to_idx:
            target_vector_official[id_to_idx[driver_id]] = binary_top3_labels_raw[i]

    # Assegna il vettore di etichette binarie a data_official['drivers'].y
    data_official['drivers'].y = target_vector_official.float()
    data_official['drivers'].train_mask = ~torch.isnan(target_vector_official)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Crea il grafo per data_full (usato per la ricerca delle metapath)
    data_full, col_stats_dict_full = make_pkey_fkey_graph(
        db_nuovo,
        col_to_stype_dict=col_to_stype_dict_nuovo,
        text_embedder_cfg=None,
        cache_dir=None
    )
    data_full = data_full.to(device)

    # Assicurati che data_full['drivers'].y abbia anch'esso le etichette binarie corrette
    data_full['drivers'].y = target_vector_official.float()
    data_full['drivers'].train_mask = ~torch.isnan(target_vector_official)

    # y_full ora conterrà direttamente le etichette binarie per la ricerca delle metapath
    y_full = data_full['drivers'].y.float()
    train_mask_full = data_full['drivers'].train_mask
    # La riga y_bin_full = binarize_targets(y_full, threshold=10) non è più necessaria qui
    # perché y_full è già binaria e con la soglia corretta.
    # --- FINE MODIFICA RIVEDUTA ---

    hidden_channels = 1024
    out_channels = 512

    metapaths, metapath_counts = greedy_metapath_search_with_bags_learned(
        col_stats_dict = col_stats_dict_full,
        data=data_full,
        y=y_full, # y_full è ora il tensore binario corretto
        train_mask=train_mask_full,
        node_type='drivers',
        L_max=4,
        channels = hidden_channels,
        #beam_width = 4 # Decommenta se vuoi usare beam search
    )

    loader_dict = loader_dict_fn(
        batch_size=1024,
        num_neighbours=512,
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
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        final_out_channels=1, # Corretto: il modello produce un logit per la classificazione binaria
    ).to(device)

    optimizer = torch.optim.Adam(
      model.parameters(),
      lr=0.005,
      weight_decay=0
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=25)

    early_stopping = EarlyStopping(
        patience=30,
        delta=0.0,
        verbose=True,
        path="best_basic_model.pt"
    )

    # Per la valutazione, passiamo le tabelle originali val_table e test_table.
    # evaluate_performance (o task.metrics) dovrebbe essere in grado di estrarre
    # i 'qualifying' da queste tabelle e binarizzarli internamente per l'F1-score.
    
    best_val_metric = -math.inf 
    test_table = task.get_table("test", mask_input_cols=False)
    best_test_metric = -math.inf 
    epochs = 150
    for epoch in range(0, epochs):
      train_loss = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

      train_pred = test(model, loader_dict["train"], device=device, task=task)
      val_pred = test(model, loader_dict["val"], device=device, task=task)
      test_pred = test(model, loader_dict["test"], device=device, task=task)

      # Passa le tabelle originali, non i dataframe delle etichette,
      # dato che evaluate_performance dovrebbe estrarre i target da lì
      train_metrics = evaluate_performance(train_pred, train_table, task.metrics, task=task)
      val_metrics = evaluate_performance(val_pred, val_table, task.metrics, task=task)
      test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)

      scheduler.step(val_metrics[tune_metric])

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
    print(f"best validation results: {best_val_metric}")
    print(f"best test results: {best_test_metric}")


if __name__ == '__main__':
    train2()