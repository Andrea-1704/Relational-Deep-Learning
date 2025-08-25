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

from model.XMetaPath import MPSGNN
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.XMetapath_utils.XMetapath_metapath_utils import binarize_targets # binarize_targets sarà usata qui
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
from utils.EarlyStopping import EarlyStopping
from utils.XMetapath_utils.XMetapath_metapath_utils import greedy_metapath_search_with_bags_learned, beam_metapath_search_with_bags_learned



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


train_df = train_table.df

assert "driverId" in train_df.columns, "driverId not in train_table"

# Conta quanti driver unici ci sono nel training set
unique_driver_ids = train_df["driverId"].nunique()
print(f"Numero di driver unici nel training set: {unique_driver_ids}")
print("Totale driver nel grafo:", db_nuovo.table_dict["drivers"].df["driverId"].nunique())

drivers_in_graph = set(db_nuovo.table_dict["drivers"].df["driverId"].astype(str))
drivers_in_train = set(train_table.df["driverId"].astype(str))

print(f"Driver del train NON presenti nel grafo: {drivers_in_train - drivers_in_graph}")
print(f"Tutti i driver del train sono nel grafo? {(drivers_in_train - drivers_in_graph) == set()}")

#So:
#the 92 drivers are a subset of the 857 drivers present in the dataset for the 
#task driver top 3.