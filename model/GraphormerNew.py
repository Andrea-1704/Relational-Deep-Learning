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

import torch
from torch import nn, Tensor
from torch_geometric.nn import Linear
from torch_geometric.utils import degree, softmax
from typing import Dict, Tuple

def _etype_key(etype):  # ('user','follows','user') -> "user__follows__user"
    return "__".join(etype)

class HeteroGraphormerLayerComplete(nn.Module):
    """
    Heterogeneous Graphormer-style layer (lightweight):
      - Q/K/V condivisi (come nel tuo codice originale), ma puoi passare a type-specific se vuoi.
      - Per-edge-type head bias: un vettore [H] per relazione (più espressivo del tuo scalare).
      - Time bias per head e per relazione: beta_h^(r) * (-log(1 + dt / tau^(r))).
      - Spatial bias cheap:
          * multirel(s,d): quante altre relazioni collegano la stessa (s,d) nel batch.
          * reciprocity: esiste un arco d->s in QUALSIASI relazione nel batch.
        Entrambi sono per-head (gamma_h^(r), delta_h^(r)).
      - Degree centrality leggera (opzionale): somma scalari ai valori dopo l’aggregazione (come avevi).
    Complessità: O(E*H); nessuna struttura N×N densa; niente NetworkX.
    """
    def __init__(self, channels, edge_types, device, num_heads=4, dropout=0.1,
                 use_degree_bias=True):
        super().__init__()
        self.device = torch.device(device)
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        assert self.channels % num_heads == 0, "channels must be divisible by num_heads"

        # Proiezioni (condivise fra tipi di nodo, come nella tua versione)
        self.q_lin = Linear(channels, channels)
        self.k_lin = Linear(channels, channels)
        self.v_lin = Linear(channels, channels)
        self.out_lin = Linear(channels, channels)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

        # --- Bias per relazione / head ---
        # edge_type bias per-head (=> vector [H])
        self.rel_head_bias = nn.ParameterDict({
            _etype_key(et): nn.Parameter(torch.zeros(num_heads))
            for et in edge_types
        })

        # --- Time bias per head per relazione ---
        # beta^(r)_h e tau^(r) (tau parametrizzato con softplus per positività)
        self.time_beta = nn.ParameterDict({
            _etype_key(et): nn.Parameter(torch.zeros(num_heads))
            for et in edge_types
        })
        self.time_tau_raw = nn.ParameterDict({
            _etype_key(et): nn.Parameter(torch.tensor(1.0))  # softplus -> tau > 0
            for et in edge_types
        })

        # --- Spatial bias cheap per head per relazione ---
        self.gamma_multirel = nn.ParameterDict({
            _etype_key(et): nn.Parameter(torch.zeros(num_heads))
            for et in edge_types
        })
        self.delta_recip = nn.ParameterDict({
            _etype_key(et): nn.Parameter(torch.zeros(num_heads))
            for et in edge_types
        })

        # Degree (come avevi): opzionale
        self.use_degree_bias = bool(use_degree_bias)

        self.to(self.device)

    # -------- utils: total degrees ----------
    def compute_total_degrees(self, x_dict, edge_index_dict):
        in_deg = {nt: torch.zeros(x_dict[nt].size(0), device=self.device) for nt in x_dict}
        out_deg = {nt: torch.zeros(x_dict[nt].size(0), device=self.device) for nt in x_dict}
        for (src_t, _, dst_t), edge_index in edge_index_dict.items():
            if edge_index.numel() == 0: 
                continue
            src, dst = edge_index
            in_deg[dst_t]  += degree(dst, num_nodes=x_dict[dst_t].size(0), dtype=torch.float32)
            out_deg[src_t] += degree(src, num_nodes=x_dict[src_t].size(0), dtype=torch.float32)
        # somma in+out come nel tuo codice originale
        return {nt: in_deg[nt] + out_deg[nt] for nt in x_dict}

    # -------- utils: cheap spatial features built once per batch ----------
    @torch.no_grad()
    def _build_pair_counters(self, edge_index_dict):
        """
        Costruisce:
          - pair_count[(src_t,dst_t)][(s,d)] = numero di edge types che collegano s->d
          - has_reverse[(src_t,dst_t)][(s,d)] = True se esiste qualche relazione con d->s
        Implementato con set/hash per O(E).
        """
        pair_count = {}
        has_reverse = {}
        # map per reverse lookup: for ogni (dst_t, src_t) mantieni set di (d,s) presenti
        reverse_maps = {}

        # 1) costruisci mappe (s,d) per TUTTI gli edge types
        for (src_t, rel, dst_t), edge_index in edge_index_dict.items():
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index
            key_sd = (src_t, dst_t)
            if key_sd not in pair_count:
                pair_count[key_sd] = {}
            if (dst_t, src_t) not in reverse_maps:
                reverse_maps[(dst_t, src_t)] = set()

            # scorri una sola volta gli edge
            s_list = src.tolist()
            d_list = dst.tolist()
            rev_set = reverse_maps[(dst_t, src_t)]
            for s, d in zip(s_list, d_list):
                # count multi-relazioni su stessa coppia (s,d)
                pair_count[key_sd][(s, d)] = pair_count[key_sd].get((s, d), 0) + 1
                # prepara reverse (d,s)
                rev_set.add((d, s))

        # 2) costruisci has_reverse usando le reverse_maps
        for (src_t, rel, dst_t), edge_index in edge_index_dict.items():
            key_sd = (src_t, dst_t)
            if key_sd not in has_reverse:
                has_reverse[key_sd] = set()
            rev_lookup = reverse_maps.get((src_t, dst_t), set())
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index
            for s, d in zip(src.tolist(), dst.tolist()):
                if (s, d) in rev_lookup:
                    has_reverse[key_sd].add((s, d))

        return pair_count, has_reverse

    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict, time_dict: Dict[str, Tensor] = None):
        # x_dict devices
        for nt in x_dict:
            if x_dict[nt].device != self.device:
                x_dict[nt] = x_dict[nt].to(self.device)

        out_dict = {k: torch.zeros_like(v, device=self.device) for k, v in x_dict.items()}

        # Precompute cheap spatial counters
        pair_count, has_reverse = self._build_pair_counters(edge_index_dict)

        # opzionale: degree bias come nel tuo codice
        total_deg = self.compute_total_degrees(x_dict, edge_index_dict) if self.use_degree_bias else None

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if edge_index.numel() == 0:
                continue

            x_src, x_dst = x_dict[src_type], x_dict[dst_type]
            src, dst = edge_index
            src = src.to(torch.long).to(self.device)
            dst = dst.to(torch.long).to(self.device)

            # Proiezioni e reshape heads
            Q = self.q_lin(x_dst).view(-1, self.num_heads, self.head_dim)  # [N_dst,H,d]
            K = self.k_lin(x_src).view(-1, self.num_heads, self.head_dim)  # [N_src,H,d]
            V = self.v_lin(x_src).view(-1, self.num_heads, self.head_dim)  # [N_src,H,d]

            Qe = Q[dst]  # [E,H,d]
            Ke = K[src]  # [E,H,d]
            Ve = V[src]  # [E,H,d]

            attn_scores = (Qe * Ke).sum(dim=-1) / (self.head_dim**0.5)  # [E,H]

            # --- per-edge-type head bias ---
            et_key = _etype_key(edge_type)
            attn_scores = attn_scores + self.rel_head_bias[et_key]  # broadcast [H]

            # --- time bias (se disponibile per i due tipi) ---
            if time_dict is not None and (src_type in time_dict) and (dst_type in time_dict):
                t_src_full = time_dict[src_type].to(self.device)  # [N_src]
                t_dst_full = time_dict[dst_type].to(self.device)  # [N_dst]
                # Assumo tensori numerici (es. unix time o step). Se sono datetime64, converti a float fuori.
                dt = (t_dst_full[dst] - t_src_full[src]).abs().to(torch.float32) + 1e-6  # [E]
                tau = torch.nn.functional.softplus(self.time_tau_raw[et_key]) + 1e-6     # scalar >0
                time_term = -torch.log1p(dt / tau)                                       # [E]
                # per-head beta^(r)
                beta = self.time_beta[et_key].view(1, self.num_heads)                    # [1,H]
                attn_scores = attn_scores + time_term.unsqueeze(1) * beta               # [E,H]

            # --- spatial bias cheap: multirel e reciprocità ---
            key_sd = (src_type, dst_type)
            # counts quante relazioni collegano la coppia (s,d) nel batch
            counts = []
            recips = []
            local_pairs = pair_count.get(key_sd, {})
            local_rec  = has_reverse.get(key_sd, set())
            for s_i, d_i in zip(src.tolist(), dst.tolist()):
                counts.append(local_pairs.get((s_i, d_i), 1) - 1)  # escludi la relazione corrente
                recips.append(1 if (s_i, d_i) in local_rec else 0)
            counts = torch.tensor(counts, dtype=torch.float32, device=self.device)  # [E]
            recips = torch.tensor(recips, dtype=torch.float32, device=self.device)  # [E]

            gamma = self.gamma_multirel[et_key].view(1, self.num_heads)  # [1,H]
            delta = self.delta_recip[et_key].view(1, self.num_heads)     # [1,H]

            if counts.numel() > 0:
                attn_scores = attn_scores + torch.log1p(counts).unsqueeze(1) * gamma
            if recips.numel() > 0:
                attn_scores = attn_scores + recips.unsqueeze(1) * delta

            # softmax per nodo di destinazione
            attn_weights = softmax(attn_scores, dst)  # [E,H]
            attn_weights = self.attn_dropout(attn_weights)

            # messaggi
            out = Ve * attn_weights.unsqueeze(-1)  # [E,H,d]
            out = out.view(-1, self.channels)
            out = self.out_lin(out)

            out_dict[dst_type].index_add_(0, dst, out)

        # Degree "add-on" (come nel tuo) — opzionale e leggero
        if self.use_degree_bias:
            for node_type in out_dict:
                deg_embed = total_deg[node_type].view(-1, 1).expand(-1, self.channels)
                out_dict[node_type] += deg_embed

        # Residual + layer norm
        for node_type in out_dict:
            out_dict[node_type] = self.norm(out_dict[node_type] + x_dict[node_type])

        return out_dict


class HeteroGraphormer(nn.Module):
    def __init__(self, node_types, edge_types, channels, num_layers=2, device="cuda",
                 num_heads=4, dropout=0.1, use_degree_bias=True):
        super().__init__()
        self.layers = nn.ModuleList([
            HeteroGraphormerLayerComplete(
                channels=channels,
                edge_types=edge_types,
                device=device,
                num_heads=num_heads,
                dropout=dropout,
                use_degree_bias=use_degree_bias,
            ) for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict, time_dict=None, *args, **kwargs):
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, time_dict=time_dict)
        return x_dict

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                try:
                    layer.reset_parameters()
                except:
                    pass


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
            num_heads=16,
            dropout=0.1,
            use_degree_bias=True,
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
            x_dict,  # node features
            batch.edge_index_dict,
            time_dict=getattr(batch, "time_dict", None),  # <<<<<< aggiunto
            # batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict non servono qui
        )

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

