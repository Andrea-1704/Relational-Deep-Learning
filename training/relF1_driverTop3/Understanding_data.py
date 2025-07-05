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
import torch
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
from relbench.modeling.nn import HeteroEncoder
import sys
import os
sys.path.append(os.path.abspath("."))
from model.MPSGNN_Model import MPSGNN
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.mpsgnn_metapath_utils import binarize_targets # binarize_targets sarà usata qui
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
from utils.EarlyStopping import EarlyStopping
from utils.mpsgnn_metapath_utils import greedy_metapath_search_with_bags_learned, beam_metapath_search_with_bags_learned
#from utils.mapping_utils import get_global_to_local_id_map




dataset = get_dataset("rel-f1", download=True)
task = get_task("rel-f1", "driver-top3", download=True)
train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")
out_channels = 1
tune_metric = "f1"
higher_is_better = True
seed_everything(42) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

#driver_ids_df = db_nuovo.table_dict["drivers"].df["driverId"].to_numpy()

global_to_local = {
    int(global_id): i for i, global_id in enumerate(graph_driver_ids)
}

encoder = HeteroEncoder(
    channels=128,
    node_to_col_names_dict={
        ntype: data_official[ntype].tf.col_names_dict
        for ntype in data_official.node_types
    },
    node_to_col_stats=col_stats_dict_official,
).to(device)
for module in encoder.modules():
    for name, buf in module._buffers.items():
        if buf is not None:
            module._buffers[name] = buf.to(device)
tf_dict = {
    ntype: data_official[ntype].tf.to(device) for ntype in data_official.node_types
} #each node type as a tf.




#Experiments
driver_id_global= 0
print(f"drivers id: {db_nuovo.table_dict['drivers'].df['driverId'].to_numpy()[driver_id_global]}")

local_idx = global_to_local[driver_id_global]
print(f"for node with global {driver_id_global} local is {local_idx}")

surname_first = db_nuovo.table_dict['drivers'].df['merged_text_id'].to_numpy()[local_idx]

print(f"features primo driver: {surname_first}")
#print(f"data official primo driver: {data_official['drivers']}")
embeddings = encoder(tf_dict)['drivers'][local_idx]
print(f"embeddings ricavate nodo 0: {embeddings}")


print(f"for driver id {global_to_local}") #Mapping corresponds for all the tables