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



#version 5
class MultiheadAttentionWithBias(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,  # [B, T, C]
        attn_bias: Tensor | None = None  # [T, T] or [B, H, T, T]
    ) -> Tensor:
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim

        qkv = self.qkv_proj(x)  # [B, T, 3 * C]
        q, k, v = qkv.chunk(3, dim=-1)  # ciascuno: [B, T, C]

        # Reshape per head
        q = q.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Attention scores: [B, H, T, T]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)

        # Aggiungi attn_bias (se fornito)
        if attn_bias is not None:
            if attn_bias.dim() == 2:
                attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
            elif attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)  # [B, 1, T, T]
            attn_scores += attn_bias  # broadcast OK

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, v)  # [B, H, T, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        return self.out_proj(attn_output)
    

class FeatureSelfAttentionBlockHighPerfPlus(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadAttentionWithBias(dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 2),
            nn.GLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: Tensor, attn_bias: Tensor = None) -> Tensor:
        # Self-attention
        qkv = self.norm1(x)
        #attn_output, attn_weights = self.attn(qkv, qkv, qkv, need_weights=False)
        attn_output = self.attn(qkv, attn_bias=attn_bias)
        # Applichiamo bias (solo se fornito)
        if attn_bias is not None:
            B, T, C = x.shape
            attn_output += attn_bias[:T, :T].unsqueeze(0)

        x = x + self.dropout1(attn_output)

        # Feed-forward
        x = x + self.ffn(self.norm2(x))
        return x
    






class FeatureSelfAttentionNet(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 6,
        pooling: str = 'cls',
        max_num_columns: int = 64,
        drop_feature_prob: float = 0.1,
    ):
        super().__init__()
        assert pooling in {'mean', 'cls', 'none'}
        self.pooling = pooling
        self.dim = dim
        self.max_num_columns = max_num_columns
        self.drop_feature_prob = drop_feature_prob

        # Column-specific learnable embeddings
        self.column_token_embedding = nn.Parameter(torch.randn(1, max_num_columns, dim))

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_num_columns + 1, dim))

        # Bias colonna-colonna per attention score
        self.col_bias = nn.Parameter(torch.randn(max_num_columns, max_num_columns))

        # CLS token dinamico: MLP sulla media
        self.cls_generator = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Stack di Self-Attention Block
        self.layers = nn.ModuleList([
            FeatureSelfAttentionBlockHighPerfPlus(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [N, F, C] = [batch, num_columns, embedding_dim]
        """
        N, F, C = x.shape
        assert F <= self.max_num_columns, f"F={F} > max={self.max_num_columns}"

        # DropFeature (DropColumn)
        if self.training and self.drop_feature_prob > 0:
            mask = (torch.rand(N, F, device=x.device) > self.drop_feature_prob).float().unsqueeze(-1)
            x = x * mask

        # Add column-specific token embedding
        col_emb = self.column_token_embedding[:, :F, :].expand(N, F, C)
        x = x + col_emb

        # Add positional embedding
        x = x + self.pos_embedding[:, 1:F + 1, :]

        # CLS token dinamico dal contenuto
        if self.pooling == 'cls':
            mean = x.mean(dim=1)  # [N, C]
            cls_token = self.cls_generator(mean).unsqueeze(1)  # [N, 1, C]
            cls_token += self.pos_embedding[:, :1, :]  # Add positional
            x = torch.cat([cls_token, x], dim=1)  # [N, F+1, C]

        # Applica colonna-colonna bias sui pesi attention
        attn_bias = self.col_bias[:F, :F]  # [F, F]

        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias)

        x = self.final_norm(x)

        if self.pooling == 'mean':
            return x.mean(dim=1)
        elif self.pooling == 'cls':
            return x[:, 0, :]
        else:
            return x




#version 4:
# class FeatureSelfAttentionBlockHighPerf(nn.Module):
#     def __init__(self, dim: int, num_heads: int, dropout: float):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
#         self.dropout1 = nn.Dropout(dropout)

#         self.norm2 = nn.LayerNorm(dim)
#         # self.ffn = nn.Sequential(
#         #     nn.Linear(dim, dim * 4),
#         #     nn.GELU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(dim * 4, dim),
#         #     nn.Dropout(dropout),
#         # )

#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.GELU(),
#             nn.Linear(dim * 4, dim * 2),
#             nn.GLU(),  # Gated linear unit
#             nn.Dropout(dropout),
#             nn.Linear(dim, dim),
#         )


#     def forward(self, x: Tensor) -> Tensor:
#         x_attn = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
#         x = x + self.dropout1(x_attn)
#         x = x + self.ffn(self.norm2(x))
#         return x

# class FeatureSelfAttentionNet(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 8,
#         dropout: float = 0.1,
#         num_layers: int = 4,
#         pooling: str = 'cls',
#         max_num_columns: int = 64,
#         drop_feature_prob: float = 0.1,
#     ):
#         super().__init__()
#         assert pooling in {'mean', 'cls', 'none'}
#         self.pooling = pooling
#         self.dim = dim
#         self.max_num_columns = max_num_columns
#         self.drop_feature_prob = drop_feature_prob

#         # Token CLS
#         if pooling == 'cls':
#             self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

#         # Positional embedding per colonna
#         self.pos_embedding = nn.Parameter(torch.randn(1, max_num_columns + 1, dim))

#         # Column-specific learnable embeddings (bias semantico per colonna)
#         self.column_token_embedding = nn.Parameter(torch.randn(1, max_num_columns, dim))

#         # Stack di blocchi Attention + FFN
#         self.layers = nn.ModuleList([
#             FeatureSelfAttentionBlockHighPerf(dim, num_heads, dropout)
#             for _ in range(num_layers)
#         ])

#         self.final_norm = nn.LayerNorm(dim)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         x: [N, F, C] = [batch, num_columns, embedding_dim]
#         """

#         N, F, C = x.shape
#         device = x.device
#         assert F <= self.max_num_columns, f"Received F={F}, but max_num_columns={self.max_num_columns}"

#         # DropFeature (DropColumn)
#         if self.training and self.drop_feature_prob > 0:
#             mask = (torch.rand(N, F, device=device) > self.drop_feature_prob).float().unsqueeze(-1)
#             x = x * mask

#         # Add column token embeddings
#         col_tok_emb = self.column_token_embedding[:, :F, :].expand(N, F, C)
#         x = x + col_tok_emb

#         # Positional embedding
#         pos_emb = self.pos_embedding[:, 1:F + 1, :].expand(N, F, C)
#         x = x + pos_emb

#         # CLS token
#         if self.pooling == 'cls':
#             cls_token = self.cls_token.expand(N, 1, C)
#             cls_pos = self.pos_embedding[:, :1, :].expand(N, 1, C)
#             x = torch.cat([cls_token + cls_pos, x], dim=1)  # [N, F+1, C]

#         # Deep self-attention
#         for layer in self.layers:
#             x = layer(x)

#         x = self.final_norm(x)

#         if self.pooling == 'mean':
#             return x.mean(dim=1)        # [N, C]
#         elif self.pooling == 'cls':
#             return x[:, 0, :]           # [N, C]
#         else:
#             return x                    # [N, F, C]







#version 3:
# class FeatureSelfAttentionNet(nn.Module):
#     """ 
#     Input: [N, F, C], where "N" is the number of nodes;
#     F is the number of features and C the embedding size 
#     for each of them.
    
#     Output: [N, C].
#     """
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 4,
#         dropout: float = 0.1,
#         num_layers: int = 2,
#         pooling: str = 'mean'
#     ):
#         super().__init__()
#         assert pooling in {'mean', 'cls', 'none'}
#         self.pooling = pooling
#         self.dim = dim

#         # Token [CLS] se serve
#         if pooling == 'cls':
#             self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

#         self.layers = nn.ModuleList([
#             FeatureSelfAttentionBlockWithFFN(dim, num_heads, dropout)
#             for _ in range(num_layers)
#         ])
#         """
#         Each layer follows: 
#         1. Self attention between columns
#         2. Feedforward for each column
#         3. Residul connections
#         4. Layer norm
#         """

#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
#         # x: [N, F, C]
#         N = x.size(0)

#         if self.pooling == 'cls':
#             cls_tokens = self.cls_token.expand(N, 1, self.dim)  # [N, 1, C]
#             x = torch.cat([cls_tokens, x], dim=1)  # [N, F+1, C]

#         for layer in self.layers:
#             x = layer(x, mask=mask)  # still [N, F, C] or [N, F+1, C]

#         x = self.norm(x)

#         if self.pooling == 'mean':
#             return x.mean(dim=1)  # [N, C]
#         elif self.pooling == 'cls':
#             return x[:, 0, :]  # [N, C]
#         else:
#             return x  # [N, F, C]




#version2:
# class FeatureSelfAttentionBlock(torch.nn.Module):
#     def __init__(self, dim: int, num_heads: int, dropout: float):
#         super().__init__()
#         self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
#         self.norm1 = torch.nn.LayerNorm(dim)
#         self.norm2 = torch.nn.LayerNorm(dim)

#         self.ffn = torch.nn.Sequential(
#             torch.nn.Linear(dim, dim * 4),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(dropout),
#             torch.nn.Linear(dim * 4, dim),
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         # Attention + residual + norm
#         attn_out, _ = self.attn(x, x, x)
#         x = self.norm1(x + attn_out)

#         # Feedforward + residual + norm
#         ffn_out = self.ffn(x)
#         x = self.norm2(x + ffn_out)

#         return x


# class FeatureSelfAttentionNet(torch.nn.Module):
#     def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1, num_layers: int = 2, pooling: str = 'mean'):
#         super().__init__()
#         self.layers = torch.nn.ModuleList([
#             FeatureSelfAttentionBlock(dim, num_heads, dropout)
#             for _ in range(num_layers)
#         ])
#         self.pooling = pooling
#         assert pooling in {'mean', 'cls', 'none'}

#     def forward(self, x: Tensor) -> Tensor:
#         # x: [N, F, C] = [batch, features, channels]
#         for layer in self.layers:
#             x = layer(x)  # ogni layer output [N, F, C]

#         if self.pooling == 'mean':
#             return x.mean(dim=1)  # [N, C]
#         elif self.pooling == 'cls':
#             return x[:, 0, :]     # usa la prima "colonna" come token speciale
#         else:  # 'none'
#             return x  # [N, F, C]
        

#version 1:    
# class FeatureSelfAttentionNet(torch.nn.Module):
#     def __init__(self, dim: int, num_heads: int = 4):
#         super().__init__()
#         self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
#         self.norm = torch.nn.LayerNorm(dim)

#     def forward(self, x: Tensor) -> Tensor:
#         # x shape: [N, F, C]
#         attn_out, _ = self.attn(x, x, x)  # Self-attention tra le feature
#         x = self.norm(attn_out + x)       # Residual connection + LayerNorm
#         return x.mean(dim=1)              # Aggrega le feature in un'unica embedding per nodo



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
        ]###
        #this "self.col_names" is used to mantain the order
        #between the columns, since col_names_dict is a 
        #dict, so there is no order.


        #self.feature_attn = FeatureSelfAttentionNet(dim=channels)
        embedding_dim = channels  
        self.feature_attn = FeatureSelfAttentionNet(
            dim=embedding_dim,
            num_heads=4,
            dropout=0.1,
            num_layers=2,
            pooling='mean',  # oppure 'cls' o 'none'
        )
        #FeatureSelfAttentionNet will receive embeddings for 
        #each columns of the node and will aggregate the embeddings
        #of the different columns in a single embedding for that node
        #but considering attention weights, which specifies how relevant
        #is each single column for the final embeddings.

        import torch_frame


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