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
from model.MPSGNN_Model_old import MPSGNN
from data_management.data import loader_dict_fn, merge_text_columns_to_categorical
from utils.XMetapath_metapath_utils import binarize_targets # binarize_targets sarà usata qui
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train
from utils.EarlyStopping import EarlyStopping
from utils.XMetapath_metapath_utils import greedy_metapath_search_with_bags_learned, beam_metapath_search_with_bags_learned
#from utils.mapping_utils import get_global_to_local_id_map
import os
from typing import Any, Dict, NamedTuple, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_frame import stype
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.utils import sort_edge_index
from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
from relbench.modeling.utils import remove_pkey_fkey, to_unix_time


#NB is this function that creates "edge_index_dict" so 
#if we want to assess which is the mapping between 
#the node index in the edege_index_dict and the one 
#of the embeddings of node embeddings, the answer should
#be found here.
def make_pkey_fkey_graph(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
    
    data = HeteroData()
    col_stats_dict = dict()
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for table_name, table in db.table_dict.items():
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = col_to_stype_dict[table_name]

        # Remove pkey, fkey columns since they will not be used as input
        # feature.
        remove_pkey_fkey(col_to_stype, table)

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            # We need to add edges later, so we need to also keep the fkeys
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

        path = (
            None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt")
        )

        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=path)

        data[table_name].tf = dataset.tensor_frame
        col_stats_dict[table_name] = dataset.col_stats

        # Add time attribute:
        if table.time_col is not None:
            data[table_name].time = torch.from_numpy(
                to_unix_time(table.df[table.time_col])
            )

        # Add edges:
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = df[fkey_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            # Filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            # Ensure no dangling fkeys
            assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            # pkey -> fkey edges.
            # "rev_" is added so that PyG loader recognizes the reverse edges
            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

    data.validate()

    return data, col_stats_dict

"""
data (returned by make_pkey_fkey_graph )[node_type].edge_index_dict
contains the same indeces of the raws in db.table_dict[table_name].df
where db is the parameter in input to edge_index_dict.

SO IN EDGE INDEX DICT WE HAVE THE SAME LOCAL INDEX OF
db.table_dict[table_name].df
"""



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
res_edges = data_official[('drivers', 'rev_f2p_driverId', 'results')].edge_index
#print(sorted(res_edges[1])) #till 20322:: OK!
#print(sorted(res_edges[0])  #till 806:: OK!

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





#driver_id_global= 0
#print(f"drivers id: {db_nuovo.table_dict['drivers'].df['driverId'].to_numpy()}")#til 956

# local_idx = global_to_local[driver_id_global]
# print(f"for node with global {driver_id_global} local is {local_idx}")

# surname_first = db_nuovo.table_dict['drivers'].df['merged_text_id'].to_numpy()[local_idx]

# print(f"features primo driver: {surname_first}")
# #print(f"data official primo driver: {data_official['drivers']}")
# embeddings = encoder(tf_dict)['drivers'][local_idx]
# print(f"embeddings ricavate nodo 0: {embeddings}")


#print(f"results id: {db_nuovo.table_dict['results'].df['resultId'].to_numpy()}") #till 20322


"""
Also encoder returns the results ordered as data_official passed to tf_dict.
And data_official pased to tf_dict is ordered as db.
"""