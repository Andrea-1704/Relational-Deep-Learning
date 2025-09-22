import copy
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
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from torch_geometric.loader import NeighborLoader
import pyg_lib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def edge_dropout(data: HeteroData, drop_prob: float = 0.1) -> HeteroData:
    """
    Rimuove una frazione casuale di archi da ogni tipo di edge.
    Solo per uso durante il training.
    """
    new_data = copy.deepcopy(data)
    for edge_type in data.edge_types:
        edge_index = new_data[edge_type].edge_index
        num_edges = edge_index.size(1)
        if num_edges == 0:
            continue
        mask = torch.rand(num_edges) > drop_prob
        new_data[edge_type].edge_index = edge_index[:, mask]
    return new_data


def train(model, optimizer, loader_dict, device, task, loss_fn) -> float:
    model.train()

    loss_accum = count_accum = 0
    for batch in tqdm(loader_dict["train"]):
        batch = batch.to(device)

        batch = edge_dropout(batch, drop_prob=0.1)  #da provare altri prob

        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        loss = loss_fn(pred.float(), batch[task.entity_table].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

    return loss_accum / count_accum