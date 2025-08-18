import torch
from collections import defaultdict
import copy
from typing import Any, Dict, List
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
import networkx as nx
from torch import nn
from torch_geometric.nn import Linear
from torch_geometric.utils import softmax, degree
from typing import Dict, Tuple

class HeteroGraphormerLayerComplete(nn.Module):
    def __init__(
        self,
        channels: int,
        edge_types,
        device: str = "cuda",
        num_heads: int = 4,
        dropout: float = 0.1,
        # SPD settings:
        max_spd: int = 3,                # k-hop massimo
        undirected_spd: bool = False,    # se True: SPD su grafo non diretto
        use_reverse_spd: bool = True,    # True: distanza d->s ; False: s->d
        enable_spd: bool = True,         # per debug: disattiva SPD
        # Degree settings:
        deg_max_bucket: int = 8,
        enable_degree: bool = True,      # per debug: disattiva degree
    ):
        super().__init__()
        self.device = torch.device(device)
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        assert self.channels % self.num_heads == 0, "channels must be divisible by num_heads"

        # Proiezioni MHA
        self.q_lin = Linear(self.channels, self.channels)
        self.k_lin = Linear(self.channels, self.channels)
        self.v_lin = Linear(self.channels, self.channels)
        self.out_lin = Linear(self.channels, self.channels)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(self.channels, 4 * self.channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.channels, self.channels),
            nn.Dropout(dropout),
        )

        # Norm (pre-LN)
        self.norm1 = nn.LayerNorm(self.channels)
        self.norm2 = nn.LayerNorm(self.channels)

        self.attn_dropout = nn.Dropout(dropout)

        # Edge-type scalar bias
        self.edge_type_bias = nn.ParameterDict({
            "__".join(et): nn.Parameter(torch.zeros(1))
            for et in edge_types
        })

        # SPD
        self.max_spd = int(max_spd)
        self.undirected_spd = bool(undirected_spd)
        self.use_reverse_spd = bool(use_reverse_spd)
        self.enable_spd = bool(enable_spd)
        # bucket: 1..K, K+1 => ">K/unreachable" ; 0 non usato
        self.spd_emb = nn.Embedding(self.max_spd + 2, self.num_heads)
        nn.init.zeros_(self.spd_emb.weight)

        # Degree embedding (in/out, in input)
        self.deg_max_bucket = int(deg_max_bucket)
        self.enable_degree = bool(enable_degree)
        self.in_deg_emb = nn.Embedding(self.deg_max_bucket + 1, self.channels)
        self.out_deg_emb = nn.Embedding(self.deg_max_bucket + 1, self.channels)
        nn.init.normal_(self.in_deg_emb.weight, std=0.02)
        nn.init.normal_(self.out_deg_emb.weight, std=0.02)

        self.to(self.device)

    # -------------------- utils --------------------
    @staticmethod
    def _bucket_degree(d: Tensor, max_bucket: int) -> Tensor:
        d = d.to(torch.float32).clamp_min_(0)
        b = torch.floor(torch.log2(d + 1.0))
        b = torch.clamp(b, 0, max_bucket).to(torch.long)
        return b

    def _compute_in_out_degree(self, x_dict: Dict[str, Tensor], edge_index_dict):
        in_deg = {nt: torch.zeros(x_dict[nt].size(0), device=x_dict[nt].device) for nt in x_dict}
        out_deg = {nt: torch.zeros(x_dict[nt].size(0), device=x_dict[nt].device) for nt in x_dict}
        for (src_t, _, dst_t), edge_index in edge_index_dict.items():
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index
            in_deg[dst_t]  += degree(dst, num_nodes=x_dict[dst_t].size(0), dtype=torch.float32)
            out_deg[src_t] += degree(src, num_nodes=x_dict[src_t].size(0), dtype=torch.float32)
        return in_deg, out_deg

    # -------------------- SPD globale k-hop --------------------
    @torch.no_grad()
    def _compute_global_spd_buckets(
        self,
        edge_index_dict,
        x_dict: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[Tuple[str,str,str], slice]]:
        # Offsets globali
        node_types = list(x_dict.keys())
        offsets = {}
        n_acc = 0
        for nt in node_types:
            offsets[nt] = n_acc
            n_acc += x_dict[nt].size(0)
        N = n_acc

        # Concat edge globali
        gsrc_all, gdst_all = [], []
        edge_slices = {}
        start = 0
        for etype, edge_index in edge_index_dict.items():
            if edge_index.numel() == 0:
                edge_slices[etype] = slice(start, start)  # vuoto
                continue
            src_t, _, dst_t = etype
            src, dst = edge_index
            # safety: long & device
            src = src.to(torch.long)
            dst = dst.to(torch.long)
            # assert su range
            if src.numel() > 0:
                assert src.max().item() < x_dict[src_t].size(0), f"src index out of range for {etype}"
            if dst.numel() > 0:
                assert dst.max().item() < x_dict[dst_t].size(0), f"dst index out of range for {etype}"

            gsrc = src + offsets[src_t]
            gdst = dst + offsets[dst_t]
            gsrc_all.append(gsrc)
            gdst_all.append(gdst)
            cnt = src.numel()
            edge_slices[etype] = slice(start, start + cnt)
            start += cnt

        if len(gsrc_all) == 0:
            # nessun arco nel batch
            return torch.empty(0, dtype=torch.long, device=self.device), edge_slices

        gsrc_all = torch.cat(gsrc_all, dim=0).to(self.device)
        gdst_all = torch.cat(gdst_all, dim=0).to(self.device)
        E_total = gsrc_all.numel()

        # Direzione per SPD
        if self.use_reverse_spd:
            row = gdst_all
            col = gsrc_all
        else:
            row = gsrc_all
            col = gdst_all

        if self.undirected_spd:
            row = torch.cat([row, col], dim=0)
            col = torch.cat([col, row[:row.numel()//2]], dim=0)  # <- sostituito sotto con versione corretta
            # >>> CORREZIONE: la riga sopra è stata spesso fonte di assert;
            # useremo subito dopo la versione corretta che aggiunge entrambe le direzioni:
        # versione corretta: (assicura stesso num di coppie)
        if self.undirected_spd:
            row = torch.cat([row, col], dim=0)
            col = torch.cat([col, row[:row.numel()//2]], dim=0)  # placeholder per compatibilità
            # ricalcolo davvero simmetrico:
            row = torch.cat([gsrc_all, gdst_all], dim=0) if not self.use_reverse_spd else torch.cat([gdst_all, gsrc_all], dim=0)
            col = torch.cat([gdst_all, gsrc_all], dim=0) if not self.use_reverse_spd else torch.cat([gsrc_all, gdst_all], dim=0)

        # A sparse
        indices = torch.stack([row, col], dim=0)
        values = torch.ones(indices.size(1), device=self.device, dtype=torch.float32)
        A = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

        # Anchor nodes (unici)
        anchor_nodes = gdst_all if self.use_reverse_spd else gsrc_all
        uniq_anchor, inv_anchor = torch.unique(anchor_nodes, return_inverse=True)
        M = uniq_anchor.numel()
        if M == 0:
            return torch.full((E_total,), self.max_spd + 1, dtype=torch.long, device=self.device), edge_slices

        # F0 denso binario (N x M)
        init_inds = torch.stack([uniq_anchor, torch.arange(M, device=self.device)], dim=0)
        init_vals = torch.ones(M, device=self.device, dtype=torch.float32)
        F = torch.sparse_coo_tensor(init_inds, init_vals, (N, M)).coalesce().to_dense()
        F = (F > 0).to(torch.float32)

        # spd predefinito
        spd = torch.full((E_total,), self.max_spd + 1, dtype=torch.long, device=self.device)
        unset = torch.ones((E_total,), dtype=torch.bool, device=self.device)

        # k-hop
        for k in range(1, self.max_spd + 1):
            # F_k = A @ F_{k-1}
            F = torch.sparse.mm(A, F)
            F = (F > 0).to(torch.float32)

            # reachability per-edge:
            probe = gsrc_all if self.use_reverse_spd else gdst_all
            reach = F[probe, inv_anchor] > 0  # [E_total]
            newly = unset & reach
            spd[newly] = k
            unset = unset & (~newly)
            if not unset.any():
                break

        # clamp difensivo (1..K, K+1)
        spd = torch.clamp(spd, 1, self.max_spd + 1)
        return spd, edge_slices

    # -------------------- forward --------------------
    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict) -> Dict[str, Tensor]:
        # allineamento device
        for nt in x_dict:
            if x_dict[nt].device != self.device:
                x_dict[nt] = x_dict[nt].to(self.device)

        # (0) degree embeddings (pre-LN)
        if self.enable_degree:
            in_deg, out_deg = self._compute_in_out_degree(x_dict, edge_index_dict)
            x_in = {}
            for nt, x in x_dict.items():
                in_b = self._bucket_degree(in_deg[nt], self.deg_max_bucket)
                out_b = self._bucket_degree(out_deg[nt], self.deg_max_bucket)
                x_in[nt] = x + self.in_deg_emb(in_b) + self.out_deg_emb(out_b)
        else:
            x_in = x_dict

        # (1) SPD buckets globali (una volta sola)
        if self.enable_spd:
            spd_buckets_all, edge_slices = self._compute_global_spd_buckets(edge_index_dict, x_dict)
        else:
            # costruisci mappe vuote coerenti
            total_edges = sum((edge_index.numel() // 2) for edge_index in edge_index_dict.values())
            spd_buckets_all = torch.full((total_edges,), self.max_spd + 1, dtype=torch.long, device=self.device)
            edge_slices = {}
            start = 0
            for etype, edge_index in edge_index_dict.items():
                E_rel = edge_index.size(1) if edge_index.numel() > 0 else 0
                edge_slices[etype] = slice(start, start + E_rel)
                start += E_rel

        # (2) Pre-LN per attenzione
        x_norm = {nt: self.norm1(x_in[nt]) for nt in x_in}
        out_attn = {nt: torch.zeros_like(x_dict[nt], device=self.device) for nt in x_dict}

        # (3) loop sugli edge type
        for etype, edge_index in edge_index_dict.items():
            src_t, _, dst_t = etype
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index
            src = src.to(torch.long).to(self.device)
            dst = dst.to(torch.long).to(self.device)

            N_src = x_norm[src_t].size(0)
            N_dst = x_norm[dst_t].size(0)
            assert (src.numel() == dst.numel()), "src/dst must have same length"
            if src.numel() == 0:
                continue
            if src.max().item() >= N_src or dst.max().item() >= N_dst:
                raise RuntimeError(f"indices out of range in {etype}")

            # proiezioni
            Q = self.q_lin(x_norm[dst_t]).view(N_dst, self.num_heads, self.head_dim)  # [N_dst,H,d]
            K = self.k_lin(x_norm[src_t]).view(N_src, self.num_heads, self.head_dim)  # [N_src,H,d]
            V = self.v_lin(x_norm[src_t]).view(N_src, self.num_heads, self.head_dim)

            # gather per-edge
            Qe = Q[dst]  # [E_rel,H,d]
            Ke = K[src]  # [E_rel,H,d]
            Ve = V[src]  # [E_rel,H,d]

            # logits
            attn_scores = (Qe * Ke).sum(dim=-1) / (self.head_dim ** 0.5)  # [E_rel,H]

            # SPD bias
            sl = edge_slices[etype]
            spd_slice = spd_buckets_all[sl] if sl.stop > sl.start else torch.empty(0, dtype=torch.long, device=self.device)
            if spd_slice.numel() == 0:
                spd_bias = 0.0
            else:
                # clamp difensivo
                spd_slice = torch.clamp(spd_slice, 1, self.max_spd + 1)
                spd_bias = self.spd_emb(spd_slice)  # [E_rel,H]

            # edge-type scalar bias (broadcast)
            et_bias = self.edge_type_bias["__".join(etype)]  # [1]

            # somma bias
            attn_scores = attn_scores + (spd_bias if isinstance(spd_bias, Tensor) else 0.0) + et_bias

            # softmax per nodo di destinazione
            attn_weights = softmax(attn_scores, dst)  # [E_rel,H]
            attn_weights = self.attn_dropout(attn_weights)

            # messaggi
            msg = Ve * attn_weights.unsqueeze(-1)     # [E_rel,H,d]
            msg = msg.reshape(msg.size(0), self.channels)  # concat heads
            msg = self.out_lin(msg)

            out_attn[dst_t].index_add_(0, dst, msg)

        # (4) residuo + FFN + residuo (pre-LN anche prima di FFN)
        y_dict = {nt: x_dict[nt] + out_attn[nt] for nt in x_dict}
        y_norm = {nt: self.norm2(y_dict[nt]) for nt in x_dict}
        z_dict = {nt: self.ffn(y_norm[nt]) for nt in x_dict}
        out_dict = {nt: y_dict[nt] + z_dict[nt] for nt in x_dict}

        return out_dict

class HeteroGraphormer(torch.nn.Module):
    def __init__(self, node_types, edge_types, channels, num_layers=2, device="cuda"):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            HeteroGraphormerLayerComplete(channels, edge_types, device) for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict, *args, **kwargs):
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        return x_dict

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()





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

