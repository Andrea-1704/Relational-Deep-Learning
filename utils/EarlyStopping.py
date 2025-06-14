
import torch
import numpy as np
import math
from tqdm import tqdm
import torch_geometric
import torch_frame
from torch_geometric.seed import seed_everything
from relbench.modeling.utils import get_stype_proposal
from collections import defaultdict
import requests
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph
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
from torch.nn import ModuleDict
import torch.nn.functional as F
from torch import nn
import random
from matplotlib import pyplot as plt
from itertools import product
import torch
import numpy as np
import copy
import pandas as pd

class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False, path='checkpoint.pt', higher_is_better=False):
        """
        Args:
            patience (int): Epoche senza miglioramenti da tollerare.
            delta (float): Cambiamento minimo per considerare un miglioramento.
            verbose (bool): Se stampare i messaggi.
            path (str): Dove salvare il modello migliore.
            higher_is_better (bool): True se la metrica va massimizzata (es: AUC), False se minimizzata (es: MAE).
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = None
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.higher_is_better = higher_is_better

    def __call__(self, metric_value, model):
        score = metric_value if self.higher_is_better else -metric_value

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric_value, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(metric_value, model)
            self.counter = 0

    def save_checkpoint(self, metric_value, model):
        if self.verbose:
            print(f"Validation metric migliorata. Salvo modello...")
        torch.save(model.state_dict(), self.path)
        self.best_metric = metric_value
