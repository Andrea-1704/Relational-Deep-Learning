import torch
from collections import defaultdict
import copy
from typing import Any, Dict, List
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
import networkx as nx
from torch import nn
from torch_geometric.nn import Linear
from torch_geometric.utils import softmax, degree
from typing import Dict, Tuple


import torch
from torch import nn, Tensor
from torch_geometric.nn import Linear
from torch_geometric.utils import degree
from typing import Dict, Tuple

def _etype_key(etype):  # ('user','follows','user') -> "user__follows__user"
    return "__".join(etype)

class HeteroGraphormerHeteroLite(nn.Module):
    """
    Heterogeneous Graphormer-style layer (lightweight, no SPD):
      - Node-type-specific Q/K/V projections.
      - Per-edge-type head bias + per (src_type,dst_type) head bias.
      - Degree embeddings (log-bucket) added to node inputs (optional).
      - Per-head degree bias on attention logits (optional).
      - Pre-LN Transformer block: MHA(out_proj) + FFN with residuals.
    Complexity: O(E*H) time, O(N + E) memory.
    """
    def __init__(
        self,
        channels: int,
        node_types,
        edge_types,
        device: str = "cuda",
        num_heads: int = 4,
        dropout: float = 0.1,
        # degree encodings:
        use_degree_input: bool = True,
        use_degree_bias: bool = True,
        deg_max_bucket: int = 8,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.C = channels
        self.H = num_heads
        self.Dh = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        # --- Q/K/V projection per node type (heterogeneous) ---
        self.q_lin = nn.ModuleDict({nt: Linear(self.C, self.C) for nt in node_types})
        self.k_lin = nn.ModuleDict({nt: Linear(self.C, self.C) for nt in node_types})
        self.v_lin = nn.ModuleDict({nt: Linear(self.C, self.C) for nt in node_types})
        self.out_lin = Linear(self.C, self.C)

        # --- Transformer FFN ---
        self.ffn = nn.Sequential(
            nn.Linear(self.C, 4 * self.C),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.C, self.C),
            nn.Dropout(dropout),
        )

        # --- Norms (pre-LN) ---
        self.norm1 = nn.LayerNorm(self.C)
        self.norm2 = nn.LayerNorm(self.C)

        self.attn_dropout = nn.Dropout(dropout)

        # --- Per-edge-type head bias (learnable) ---
        self.rel_head_bias = nn.ParameterDict({
            _etype_key(et): nn.Parameter(torch.zeros(self.H)) for et in edge_types
        })

        # --- Per (src_type, dst_type) head bias (learnable) ---
        self.typepair_head_bias = nn.ParameterDict()
        type_pairs = {(s, d) for (s, _, d) in edge_types}
        for (s, d) in type_pairs:
            self.typepair_head_bias[f"{s}__{d}"] = nn.Parameter(torch.zeros(self.H))

        # --- Degree encodings (shared across types; cheap & effective) ---
        self.use_degree_input = use_degree_input
        self.use_degree_bias  = use_degree_bias
        self.deg_max_bucket   = int(deg_max_bucket)
        if self.use_degree_input:
            self.in_deg_emb  = nn.Embedding(self.deg_max_bucket + 1, self.C)
            self.out_deg_emb = nn.Embedding(self.deg_max_bucket + 1, self.C)
            nn.init.normal_(self.in_deg_emb.weight,  std=0.02)
            nn.init.normal_(self.out_deg_emb.weight, std=0.02)

        if self.use_degree_bias:
            # global per-head coefficients for src out-degree / dst in-degree
            self.alpha_head = nn.Parameter(torch.zeros(self.H))
            self.beta_head  = nn.Parameter(torch.zeros(self.H))

        self.to(self.device)

    # ---------- utils ----------
    @staticmethod
    def _bucket_degree(d: Tensor, max_bucket: int) -> Tensor:
        d = d.to(torch.float32).clamp_min_(0)
        b = torch.floor(torch.log2(d + 1.0))
        return torch.clamp(b, 0, max_bucket).to(torch.long)

    @torch.no_grad()
    def _compute_in_out_degree(self, x_dict: Dict[str, Tensor], edge_index_dict):
        in_deg  = {nt: torch.zeros(x_dict[nt].size(0), device=x_dict[nt].device) for nt in x_dict}
        out_deg = {nt: torch.zeros(x_dict[nt].size(0), device=x_dict[nt].device) for nt in x_dict}
        for (src_t, _, dst_t), edge_index in edge_index_dict.items():
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index
            in_deg[dst_t]  += degree(dst, num_nodes=x_dict[dst_t].size(0), dtype=torch.float32)
            out_deg[src_t] += degree(src, num_nodes=x_dict[src_t].size(0), dtype=torch.float32)
        return in_deg, out_deg

    # ---------- forward ----------
    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict) -> Dict[str, Tensor]:
        # ensure device
        for nt in x_dict:
            if x_dict[nt].device != self.device:
                x_dict[nt] = x_dict[nt].to(self.device)

        # (0) degree embeddings in input (optional)
        if self.use_degree_input or self.use_degree_bias:
            in_deg, out_deg = self._compute_in_out_degree(x_dict, edge_index_dict)
        x_in = {}
        for nt, x in x_dict.items():
            if self.use_degree_input:
                in_b  = self._bucket_degree(in_deg[nt],  self.deg_max_bucket)
                out_b = self._bucket_degree(out_deg[nt], self.deg_max_bucket)
                x_in[nt] = x + self.in_deg_emb(in_b) + self.out_deg_emb(out_b)
            else:
                x_in[nt] = x

        # (1) Pre-LN for attention
        x_norm = {nt: self.norm1(x_in[nt]) for nt in x_in}

        # (2) attention per edge type (sparse)
        out_attn = {nt: torch.zeros_like(x_dict[nt], device=self.device) for nt in x_dict}

        for etype, edge_index in edge_index_dict.items():
            if edge_index.numel() == 0:
                continue
            src_t, _, dst_t = etype
            src, dst = edge_index
            src = src.to(torch.long).to(self.device)
            dst = dst.to(torch.long).to(self.device)

            N_src = x_norm[src_t].size(0)
            N_dst = x_norm[dst_t].size(0)
            if src.numel() == 0:
                continue
            if src.max().item() >= N_src or dst.max().item() >= N_dst:
                raise RuntimeError(f"indices out of range in {etype}")

            # type-specific projections
            Q = self.q_lin[dst_t](x_norm[dst_t]).view(N_dst, self.H, self.Dh)  # [N_dst,H,dh]
            K = self.k_lin[src_t](x_norm[src_t]).view(N_src, self.H, self.Dh)  # [N_src,H,dh]
            V = self.v_lin[src_t](x_norm[src_t]).view(N_src, self.H, self.Dh)  # [N_src,H,dh]

            Qe = Q[dst]            # [E,H,dh]
            Ke = K[src]            # [E,H,dh]
            Ve = V[src]            # [E,H,dh]

            attn_scores = (Qe * Ke).sum(dim=-1) / (self.Dh ** 0.5)  # [E,H]

            # per-edge-type head bias
            attn_scores = attn_scores + self.rel_head_bias[_etype_key(etype)]  # broadcast [H] -> [E,H]
            # per (src_type, dst_type) head bias
            attn_scores = attn_scores + self.typepair_head_bias[f"{src_t}__{dst_t}"]

            # optional per-head degree bias
            if self.use_degree_bias:
                src_out = out_deg[src_t][src]  # [E]
                dst_in  = in_deg[dst_t][dst]   # [E]
                log_src = torch.log1p(src_out).unsqueeze(1)  # [E,1]
                log_dst = torch.log1p(dst_in).unsqueeze(1)   # [E,1]
                deg_bias = log_src * self.alpha_head.view(1, self.H) + \
                           log_dst * self.beta_head.view(1, self.H)    # [E,H]
                attn_scores = attn_scores + deg_bias

            # softmax per nodo di destinazione
            attn_weights = torch_geometric.utils.softmax(attn_scores, dst)  # [E,H]
            attn_weights = self.attn_dropout(attn_weights)

            # messaggi
            msg = Ve * attn_weights.unsqueeze(-1)      # [E,H,dh]
            msg = msg.reshape(msg.size(0), self.C)     # concat heads
            msg = self.out_lin(msg)                    # out projection

            out_attn[dst_t].index_add_(0, dst, msg)

        # (3) residuo + FFN + residuo
        y_dict  = {nt: x_dict[nt] + out_attn[nt] for nt in x_dict}
        y_norm  = {nt: self.norm2(y_dict[nt]) for nt in x_dict}
        z_dict  = {nt: self.ffn(y_norm[nt]) for nt in x_dict}
        out_dict = {nt: y_dict[nt] + z_dict[nt] for nt in x_dict}
        return out_dict


class HeteroGraphormer(nn.Module):
    def __init__(self, node_types, edge_types, channels, num_layers=2, device="cuda",
                 num_heads=4, dropout=0.1,
                 use_degree_input=True, use_degree_bias=True, deg_max_bucket=8):
        super().__init__()
        self.layers = nn.ModuleList([
            HeteroGraphormerHeteroLite(
                channels=channels,
                node_types=node_types,
                edge_types=edge_types,
                device=device,
                num_heads=num_heads,
                dropout=dropout,
                use_degree_input=use_degree_input,
                use_degree_bias=use_degree_bias,
                deg_max_bucket=deg_max_bucket,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict, *args, **kwargs):
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        return x_dict

    def reset_parameters(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                try: m.reset_parameters()
                except: pass




class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData, #notice that "data2 is the graph we created with function make_pkey_fkey_graph
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

        self.encoder = HeteroEncoder(
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
        self.gnn = HeteroGraphormer(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            num_layers=num_layers,
            device="cuda",
            num_heads=4,
            dropout=0.1,
            use_degree_input=True,
            use_degree_bias=True,
            deg_max_bucket=8,
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
        #this creates a dictionar for all the nodes: each nodes has its
        #embedding

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )
        #this add the temporal information to the node using the 
        #HeteroTemporalEncoder

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        #add some other shallow embedder

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,#feature of nodes
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )#apply the gnn

        return self.head(x_dict[entity_table][: seed_time.size(0)])#final prediction

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
        # Add ID-awareness to the root node
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

