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
from utils.mpsgnn_metapath_utils import binarize_targets # Questa funzione non sarà usata per i target principali, ma potrebbe essere ancora rilevante per la logica interna di evaluate_relation_learned se non modificata
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
from utils.EarlyStopping import EarlyStopping
from utils.mpsgnn_metapath_utils import greedy_metapath_search_with_bags_learned, beam_metapath_search_with_bags_learned


def train2():
    dataset = get_dataset("rel-f1", download=True)
    task = get_task("rel-f1", "driver-top3", download=True) 

    # Ottieni le tabelle principali
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    out_channels = 1 # Corretto per la classificazione binaria (output un logit)
    loss_fn = nn.BCEWithLogitsLoss() # Usa la Binary Cross-Entropy Loss per la classificazione
    tune_metric = "f1" # La metrica principale diventa F1-score
    higher_is_better = True # Per F1-score, valori più alti sono migliori

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "./data"

    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)
    db_nuovo, col_to_stype_dict_nuovo = merge_text_columns_to_categorical(db, col_to_stype_dict)

    # Crea il grafo per data_official
    data_official, col_stats_dict_official = make_pkey_fkey_graph(
        db_nuovo,
        col_to_stype_dict=col_to_stype_dict_nuovo,
        text_embedder_cfg = None,
        cache_dir=None
    )

    graph_driver_ids = db_nuovo.table_dict["drivers"].df["driverId"].to_numpy()
    id_to_idx = {driver_id: idx for idx, driver_id in enumerate(graph_driver_ids)}

    # --- INIZIO MODIFICA PER ETICHETTE BINARIE (driver-top3) ---
    # Ottieni le etichette binarie direttamente dal task di Relbench
    # task.get_labels() restituisce un DataFrame con 'row_id' (che è driverId) e 'label' (0 o 1)
    train_labels_df = task.get_labels(train_table)
    val_labels_df = task.get_labels(val_table)
    test_labels_df = task.get_labels(test_table) # Questo DataFrame avrà le etichette reali per il set di test

    # Mappa le etichette binarie corrette ai nodi del grafo in data_official
    target_vector_official = torch.full((len(graph_driver_ids),), float("nan"))
    for _, row in train_labels_df.df.iterrows():
        driver_id = row['row_id'] # 'row_id' è l'ID del driver
        label = row['label']    # 'label' è l'etichetta binaria (0 o 1)
        if driver_id in id_to_idx:
            target_vector_official[id_to_idx[driver_id]] = label

    # Assegna il vettore di etichette binarie a data_official['drivers'].y
    data_official['drivers'].y = target_vector_official.float() # Assicurati che sia float per la loss
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

    # Mappa le etichette binarie corrette ai nodi del grafo in data_full,
    # usando lo stesso vettore di etichette binarie già preparato
    data_full['drivers'].y = target_vector_official.float()
    data_full['drivers'].train_mask = ~torch.isnan(target_vector_official)

    # y_full ora conterrà direttamente le etichette binarie per la ricerca delle metapath
    y_full = data_full['drivers'].y.float()
    train_mask_full = data_full['drivers'].train_mask
    # y_bin_full = binarize_targets(y_full, threshold=10) # Non più strettamente necessaria qui, y_full è già binaria
    # --- FINE MODIFICA PER ETICHETTE BINARIE ---

    hidden_channels = 128
    out_channels = 64

    # Utilizza greedy_metapath_search_with_bags_learned o beam_metapath_search_with_bags_learned
    # y_full è già binaria, quindi la funzione di scoring userà etichette 0/1.
    metapaths, metapath_counts = greedy_metapath_search_with_bags_learned(
        col_stats_dict = col_stats_dict_full,
        data=data_full,
        y=y_full, # y_full è ora il tensore binario
        train_mask=train_mask_full,
        node_type='drivers',
        L_max=2,
        channels = hidden_channels,
        #beam_width = 4 # Decommenta se vuoi usare beam search
    )

    # Configurazione del loader, passa le tabelle corrette
    loader_dict = loader_dict_fn(
        batch_size=256,
        num_neighbours=128,
        data=data_official,
        task=task,
        train_table=train_table, # Tabella non modificata, serve per i loader
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
      lr=0.001,
      weight_decay=0
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=25)

    early_stopping = EarlyStopping(
        patience=30,
        delta=0.0,
        verbose=True,
        path="best_basic_model.pt"
    )

    test_table_eval = task.get_table("test", mask_input_cols=False) # Rinomina per chiarezza

    best_val_metric = -math.inf # F1 è "più alto è meglio", quindi inizia con -inf
    best_test_metric = -math.inf # F1 è "più alto è meglio", quindi inizia con -inf
    epochs = 150
    for epoch in range(0, epochs):
      train_loss = train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)

      # Ottieni le predizioni per train/val/test
      train_pred = test(model, loader_dict["train"], device=device, task=task)
      val_pred = test(model, loader_dict["val"], device=device, task=task)
      test_pred = test(model, loader_dict["test"], device=device, task=task)

      # Calcola le metriche di performance usando i DataFrame delle etichette
      # Assicurati che evaluate_performance sia compatibile con i DataFrame restituiti da task.get_labels
      train_metrics = evaluate_performance(train_pred, train_labels_df, task.metrics, task=task)
      val_metrics = evaluate_performance(val_pred, val_labels_df, task.metrics, task=task)
      test_metrics = evaluate_performance(test_pred, test_labels_df, task.metrics, task=task)

      scheduler.step(val_metrics[tune_metric]) # Scheduler basato sulla metrica di tuning F1

      # Logica di early stopping e salvataggio del modello basata sulla metrica F1
      if (higher_is_better and val_metrics[tune_metric] > best_val_metric):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict()) # Salva il modello con la migliore validazione F1

      if (higher_is_better and test_metrics[tune_metric] > best_test_metric):
          best_test_metric = test_metrics[tune_metric]
          state_dict_test = copy.deepcopy(model.state_dict()) # Salva il modello con il migliore test F1 (opzionale, di solito si usa solo la validazione)

      current_lr = optimizer.param_groups[0]["lr"]
      
      # Stampa le metriche F1 per train, validation e test
      print(f"Epoch: {epoch:02d}, Train {tune_metric}: {train_metrics[tune_metric]:.2f}, Validation {tune_metric}: {val_metrics[tune_metric]:.2f}, Test {tune_metric}: {test_metrics[tune_metric]:.2f}, LR: {current_lr:.6f}")

      early_stopping(val_metrics[tune_metric], model)

      if early_stopping.early_stop:
          print(f"Early stopping triggered at epoch {epoch}")
          break
    print(f"best validation results: {best_val_metric}")
    print(f"best test results: {best_test_metric}")


if __name__ == '__main__':
    train2()