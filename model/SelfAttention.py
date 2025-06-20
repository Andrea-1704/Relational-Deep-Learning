import torch
import numpy as np
import math
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath("."))


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
from typing import Any, Dict, List, Optional

import torch
import torch_frame
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_geometric.nn import HeteroConv, LayerNorm, PositionalEncoding, SAGEConv
from torch_geometric.typing import EdgeType, NodeType


import torch
from torch import nn
from torch.nn import ModuleDict
from typing import Any, Dict, List
from torch import Tensor
from torch_frame import TensorFrame
import pandas as pd



class FeatureSelfAttentionNet(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        stype_encoder_dict: Dict[torch_frame.stype, torch.nn.Module],
        db,
        node_type,
        n_heads: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        
        self.stype_encoder_dict = stype_encoder_dict
        self.col_names_dict = col_names_dict
        self.col_stats = col_stats
        self.channels = channels
        self.db=db
        self.node_type=node_type
        # Per colonna: costruisco lista di nomi (es. 'age', 'team') -> stype
        self.col_to_stype = {}
        for stype, cols in col_names_dict[node_type].items():
            for col in cols:
                self.col_to_stype[col] = stype

        # Transformer
        self.attn = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=n_heads,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=1,
        )

        self.output_proj = torch.nn.Linear(channels, channels)

    def forward(self, tf: torch_frame.TensorFrame, node_type) -> Tensor:
      embeddings = []
      print(f"prima del for self.col_names_dict è {self.col_names_dict}")
      for stype, cols in self.col_names_dict[node_type].items():
          print(f"per node_type {node_type} abbiiamo stype{stype} e cols{cols}")
          encoder = self.stype_encoder_dict[stype]
          
          for col in cols:
              #x_col = tf.feat_dict[stype][col]
              table = self.db.table_dict[node_type]
              

              x_col = table.df[col]
              #print(f"ecco x_col {x_col} per la colonna {col} della tabella {node_type}")

              # Aggiunta specificazione manuale (evita infer_spec)
              encoder.stype = stype
              encoder.out_channels = self.channels
              print(f"col stats è esattamente {self.col_stats}")
              for col in cols:
                print(f"colonna {self.col_stats[node_type][col]}")
              encoder.stats_list = [
                  self.col_stats[node_type][col] for col in cols
              ]
              print(f"endoer stats list: {encoder.stats_list}")
              # encoder.out_channels = channels
              # encoder.stats_list = [
              #     node_to_col_stats[node_type][col] for col in cols
              # ]
              x_col = table.df[col] 
              encoder.stats_list = [self.col_stats[node_type][col]]
              emb_col = encoder(x_col)
              

              #emb_col = encoder(x_col.unsqueeze(-1) if x_col.ndim == 1 else x_col)
              embeddings.append(emb_col.unsqueeze(1))  # (N, 1, C)


      x = torch.cat(embeddings, dim=1)  # (N, num_features, channels)
      x = self.attn(x)                  # (N, num_features, channels)
      x = x.mean(dim=1)                 # (N, channels)
      return self.output_proj(x)        # (N, channels)


    def reset_parameters(self):
      for encoder in self.stype_encoder_dict.values():
          if hasattr(encoder, "reset_parameters"):
              try:
                  encoder.reset_parameters()
              except Exception as e:
                  print(f"Warning: could not reset {encoder}: {e}")









class MyHeteroEncoder(torch.nn.Module):
    r"""HeteroEncoder based on PyTorch Frame.

    Args:
        channels (int): The output channels for each node type.
        node_to_col_names_dict (Dict[NodeType, Dict[torch_frame.stype, List[str]]]):
            A dictionary mapping from node type to column names dictionary
            compatible to PyTorch Frame.
        torch_frame_model_cls: Model class for PyTorch Frame. The class object
            takes :class:`TensorFrame` object as input and outputs
            :obj:`channels`-dimensional embeddings. Default to
            :class:`torch_frame.nn.ResNet`.
        torch_frame_model_kwargs (Dict[str, Any]): Keyword arguments for
            :class:`torch_frame_model_cls` class. Default keyword argument is
            set specific for :class:`torch_frame.nn.ResNet`. Expect it to
            be changed for different :class:`torch_frame_model_cls`.
        default_stype_encoder_cls_kwargs (Dict[torch_frame.stype, Any]):
            A dictionary mapping from :obj:`torch_frame.stype` object into a
            tuple specifying :class:`torch_frame.nn.StypeEncoder` class and its
            keyword arguments :obj:`kwargs`.
    """

    def __init__(
        self,
        channels: int,
        db,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        torch_frame_model_cls=FeatureSelfAttentionNet,
        torch_frame_model_kwargs: Dict[str, Any] = {
            "channels": 128,
            "num_layers": 4,
        },
        default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                {},
            ),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
        },
    ):
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for node_type in node_to_col_names_dict.keys():
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1]
                )
                for stype in node_to_col_names_dict[node_type].keys()
            }
            torch_frame_model = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=channels,
                col_stats=node_to_col_stats,
                col_names_dict=node_to_col_names_dict,
                stype_encoder_dict=stype_encoder_dict,
                db = db,
                node_type=node_type,
            )
            self.encoders[node_type] = torch_frame_model

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
    ) -> Dict[NodeType, Tensor]:
        x_dict = {
            node_type: self.encoders[node_type](tf, node_type) for node_type, tf in tf_dict.items()
        }
        return x_dict
    



class MyModel(torch.nn.Module):

    def __init__(
        self,
        db,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        shallow_list: List[NodeType] = [],
        id_awareness: bool = False,
        predictor_n_layers : int = 1,
    ):
        super().__init__()
        print(f"dentro encoder col_stata_dict è {col_stats_dict}")
        self.encoder = MyHeteroEncoder(
            channels=channels,
            db = db,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )

        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )

        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=predictor_n_layers,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def encode_node_types(self, batch: HeteroData, node_types: List[str], entity_table) -> Dict[str, Tensor]:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)

        for node_type in node_types:
            if node_type in rel_time_dict:
                x_dict[node_type] = x_dict[node_type] + rel_time_dict[node_type]
            if node_type in self.embedding_dict:
                x_dict[node_type] = x_dict[node_type] + self.embedding_dict[node_type](batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return {ntype: x_dict[ntype] for ntype in node_types if ntype in x_dict}



    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])