import copy
import math
import os
import random
import sys
import requests
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyg_lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    ModuleDict,
    ReLU,
    Sequential,
)
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder import (
    EmbeddingEncoder,
    FeatureEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.models import ResNet

import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import (
    HeteroConv,
    LayerNorm as GNNLayerNorm,
    MLP,
    PositionalEncoding,
    SAGEConv,
)
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.seed import seed_everything
from torch_geometric.typing import EdgeType, NodeType

from relbench.modeling.graph import (
    get_node_train_table_input,
    make_pkey_fkey_graph,
)
from relbench.modeling.nn import (
    HeteroEncoder,
    HeteroGraphSAGE,
    HeteroTemporalEncoder,
)
from relbench.modeling.utils import get_stype_proposal
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder


class FCResidualBlock(Module):
    r"""Fully connected residual block.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        normalization (str, optional): The type of normalization to use.
            :obj:`layer_norm`, :obj:`batch_norm`, or :obj:`None`.
            (default: :obj:`layer_norm`)
        dropout_prob (float): The dropout probability (default: `0.0`, i.e.,
            no dropout).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: str | None = "layer_norm",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.lin1 = Linear(in_channels, out_channels)
        self.lin2 = Linear(out_channels, out_channels)
        self.relu = ReLU()
        self.dropout = Dropout(dropout_prob)

        self.norm1: BatchNorm1d | LayerNorm | None
        self.norm2: BatchNorm1d | LayerNorm | None
        if normalization == "batch_norm":
            self.norm1 = BatchNorm1d(out_channels)
            self.norm2 = BatchNorm1d(out_channels)
        elif normalization == "layer_norm":
            self.norm1 = LayerNorm(out_channels)
            self.norm2 = LayerNorm(out_channels)
        else:
            self.norm1 = self.norm2 = None

        self.shortcut: Linear | None
        if in_channels != out_channels:
            self.shortcut = Linear(in_channels, out_channels)
        else:
            self.shortcut = None

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.shortcut is not None:
            self.shortcut.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        out = self.lin1(x)
        out = self.norm1(out) if self.norm1 else out
        out = self.relu(out)
        out = self.dropout(out)

        out = self.lin2(out)
        out = self.norm2(out) if self.norm2 else out
        out = self.relu(out)
        out = self.dropout(out)

        if self.shortcut is not None:
            x = self.shortcut(x)

        out = out + x

        return out



class FeatureSelfAttentionBlock(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Attention + residual + norm
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feedforward + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class FeatureSelfAttentionNetV2(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1, num_layers: int = 2, pooling: str = 'mean'):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            FeatureSelfAttentionBlock(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.pooling = pooling
        assert pooling in {'mean', 'cls', 'none'}

    def forward(self, x: Tensor) -> Tensor:
        # x: [N, F, C] = [batch, features, channels]
        for layer in self.layers:
            x = layer(x)  # ogni layer output [N, F, C]

        if self.pooling == 'mean':
            return x.mean(dim=1)  # [N, C]
        elif self.pooling == 'cls':
            return x[:, 0, :]     # usa la prima "colonna" come token speciale
        else:  # 'none'
            return x  # [N, F, C]
        

        
class FeatureSelfAttentionNet(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [N, F, C]
        attn_out, _ = self.attn(x, x, x)  # Self-attention tra le feature
        x = self.norm(attn_out + x)       # Residual connection + LayerNorm
        return x.mean(dim=1)              # Aggrega le feature in un'unica embedding per nodo



def extract_column_embeddings(encoder: StypeWiseFeatureEncoder, tf: TensorFrame, out_channels: int) -> Dict[str, Tensor]:
    """
    Function that extracts the embeddings for each column of a node.
    Returns a dictionary {column_name: Tensor[N, C]}.
    """
    x, all_col_names = encoder(tf)  # [N, num_cols * C], List[str]
    N = x.size(0)
    C = out_channels
    num_cols = len(all_col_names)

    x = x.view(N, num_cols, C) #[N, num_cols, C]

    col_emb_dict = {
        col_name: x[:, i, :] for i, col_name in enumerate(all_col_names)
    } #col_name → Tensor[N, C]
    
    return col_emb_dict



class ResNet2(Module):
    """
    To introduce the Self attention mechanism this is the right and only 
    class to change.

    Originally this class was designed to provide the final embeddings
    for the nodes. It used to provides the final embeddings, so we have 
    to break this logic, to get the intermediate embeddings for each 
    column, apply a self attention mechanims in order to weight them 
    and then provide the final embeddings.
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
        | None = None,
        normalization: str | None = "layer_norm",
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        ###new:
        self.col_names = [
            col_name
            for stype, col_list in col_names_dict.items()
            for col_name in col_list
        ]
        ###new:

        #self.feature_attn = FeatureSelfAttentionNet(dim=channels)
        embedding_dim = channels  
        self.feature_attn = FeatureSelfAttentionNet(
            dim=embedding_dim,
            num_heads=4,
            dropout=0.1,
            num_layers=2,
            pooling='mean',  # oppure 'cls' o 'none'
        )

        in_channels = channels 
        self.backbone = Sequential(*[
            FCResidualBlock(
                in_channels if i == 0 else channels,
                channels,
                normalization=normalization,
                dropout_prob=dropout_prob,
            ) for i in range(num_layers)
        ])

        self.decoder = Sequential(
            LayerNorm(channels),
            ReLU(),
            Linear(channels, out_channels),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for block in self.backbone:
            block.reset_parameters()
        self.decoder[0].reset_parameters()
        self.decoder[-1].reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)
        
        #now we extract the embeddings of each of the columns:
        col_emb_dict = extract_column_embeddings(self.encoder, tf, out_channels=128)
        
        col_order = self.col_names  
        x = torch.stack([col_emb_dict[col] for col in col_order], dim=1)  # [N, F, C]
        x = self.feature_attn(x)  #pass the result to self attention module

        x = self.backbone(x)
        out = self.decoder(x)
        return out



class MyHeteroEncoder(torch.nn.Module):
    """
    Is identical to the Relbench version of "HeteroEncoder", but here we are using 
    a custom version of the "ResNet" which is the actual encoder, in order to 
    apply the encoding to each column indipendently and then apply the self
    attention mechanism.

    How does this works?
    follow the description I provided below!
    """

    def __init__(
        self,
        channels: int, #output dimension of the embeddings of the nodes
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        torch_frame_model_cls=ResNet2,
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
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict,
            )
            self.encoders[node_type] = torch_frame_model
        """
        in these rows we are building an encoder for each node type.
        In particular, for each node type, we pass the relevant informations
        of the node to an encoder (in our case the ResNet2) and we store 
        the encoder informations insider this "self.encoders" dict.
        """

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
    ) -> Dict[NodeType, Tensor]:
        """
        Here, in the forward method, for each node type we pass the node
        to the right encoder (using the "self.encoders" dict) and 
        we get as a result the embeddings for each node type.

        So, this is a very key fucntion, because is the function that 
        builds the "x_dict" dictionary.
        """
        x_dict = {
            node_type: self.encoders[node_type](tf) for node_type, tf in tf_dict.items()
        }
        return x_dict   # x_dict = {
                        # "driver": Tensor[N_driver, channels],
                        # "race": Tensor[N_race, channels],
                        # }
            
        
    



class MyModel(torch.nn.Module):
    """
    Is identical to the orioginal version, with the only difference that we are now 
    using a custom version of HeteroEncoder to apply a self attention mechanism 
    between the embeddings of the columns.
    """

    def __init__(
        self,
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

        self.encoder = MyHeteroEncoder(
            channels=channels,
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
    
    def encoder_parameters(self):
        params = list(self.encoder.parameters()) + list(self.temporal_encoder.parameters()) + list(self.gnn.parameters())
        return params
    
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