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

class HeteroGraphormerLayerComplete(nn.Module):
    def __init__(self, channels, edge_types, device, num_heads=4, dropout=0.1,
                 max_spd=3, undirected=False, use_reverse=True):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads

        assert self.channels % num_heads == 0, "channels must be divisible by num_heads"

        self.q_lin = Linear(channels, channels)
        self.k_lin = Linear(channels, channels)
        self.v_lin = Linear(channels, channels)
        self.out_lin = Linear(channels, channels)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

        self.edge_type_bias = nn.ParameterDict({
            "__".join(edge_type): nn.Parameter(torch.randn(1))
            for edge_type in edge_types
        })

        self.max_spd = max_spd                 # K: hop massimi (consigliato 2-4)
        self.undirected = undirected           # True = grafo non diretto per SPD
        self.use_reverse = use_reverse         # True = distanza d->s (come nel tuo uso)
        # bucket: 0..max_spd mappano 1..K (useremo 0 per distanza=1), bucket max_spd+1 = ">K/unreachable"
        self.spd_emb = nn.Embedding(self.max_spd + 2, self.num_heads)  # (K+2, H)

    def compute_total_degrees(self, x_dict, edge_index_dict):
        device = self.device
        in_deg = defaultdict(lambda: torch.zeros(0, device=device))
        out_deg = defaultdict(lambda: torch.zeros(0, device=device))
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            src, dst = edge_index

            num_src = x_dict[src_type].size(0)
            num_dst = x_dict[dst_type].size(0)

            if out_deg[src_type].numel() == 0:
                out_deg[src_type] = torch.zeros(num_src, device=device)
            if in_deg[dst_type].numel() == 0:
                in_deg[dst_type] = torch.zeros(num_dst, device=device)

            out_deg[src_type] += degree(src, num_nodes=num_src)
            in_deg[dst_type]  += degree(dst, num_nodes=num_dst)

        return {
            node_type: in_deg[node_type] + out_deg[node_type]
            for node_type in x_dict
        }

    @torch.no_grad()
    def compute_batch_spatial_bias_fast(self, edge_index, num_nodes):
        """
        Calcola un 'reverse' shortest-path (d -> s) fino a K hop in modo vettoriale.
        Ritorna: spd_bucket [E] (long) con valori in {1..K, K+1} dove K+1 = '>K o non raggiungibile'.
        """
        src, dst = edge_index  # [E]
        device = src.device
        K = self.max_spd

        # --- Costruisci adiacenza sparsa del batch ---
        # Verso per SPD: se use_reverse=True, vogliamo d->s, quindi costruiamo A_rev con archi (d->s).
        if self.use_reverse:
            row = dst
            col = src
        else:
            row = src
            col = dst

        if self.undirected:
            row = torch.cat([row, col], dim=0)
            col = torch.cat([col, row[:row.numel()//2]], dim=0)  # aggiungi l'opposto

        # Sparse adjacency (N x N) in COO
        indices = torch.stack([row, col], dim=0)
        values = torch.ones(indices.size(1), device=device, dtype=torch.float32)
        A = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

        # --- Multi-source BFS vettoriale a partire da tutti i nodi 'dst' (se use_reverse) / 'src' altrimenti ---
        # Consideriamo solo i 'destinazioni' unici per ridurre dimensione colonna
        anchor = dst if self.use_reverse else src                   # shape [E]
        uniq_anchor, inv_anchor = torch.unique(anchor, return_inverse=True)  # [M], [E]  (M <= num_nodes)
        M = uniq_anchor.numel()

        # Inizializza fronte: (N x M) denso boole (possiamo usare float e poi >0)
        # F0 ha 1 in (anchor_node, anchor_col)
        init_inds = torch.stack([uniq_anchor, torch.arange(M, device=device)], dim=0)  # [2, M]
        init_vals = torch.ones(M, device=device, dtype=torch.float32)
        F = torch.sparse_coo_tensor(init_inds, init_vals, (num_nodes, M)).coalesce().to_dense()  # (N,M) denso
        F = (F > 0).to(torch.float32)

        # spd per-edge, default K+1 (bucket 'far/unreachable')
        E = src.size(0)
        spd = torch.full((E,), K + 1, dtype=torch.long, device=device)
        unset = torch.ones((E,), dtype=torch.bool, device=device)

        # k-hop espansione: F_k = A @ F_{k-1} (propaga reachability di 1 hop per iterazione)
        # A e F sono su device; torch.sparse.mm(A, F) è denso
        for k in range(1, K + 1):
            F = torch.sparse.mm(A, F)  # (N,N) x (N,M) -> (N,M)
            F = (F > 0).to(torch.float32)  # binarizza

            # Per ogni edge e=(s,d), controlla se s è raggiunto dalla colonna corrispondente a d (inv_anchor[e])
            col_idx = inv_anchor  # [E]
            reach = F[src, col_idx] > 0  # [E] bool
            newly = unset & reach
            spd[newly] = k
            unset = unset & (~newly)
            if not unset.any():
                break

        # Ritorna bucket di distanza: 1..K, K+1 (= non raggiunto entro K)
        return spd


    def forward(self, x_dict, edge_index_dict):
        out_dict = {k: torch.zeros_like(v) for k, v in x_dict.items()}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            x_src, x_dst = x_dict[src_type], x_dict[dst_type]
            src, dst = edge_index

            Q = self.q_lin(x_dst).view(-1, self.num_heads, self.head_dim)
            K = self.k_lin(x_src).view(-1, self.num_heads, self.head_dim)
            V = self.v_lin(x_src).view(-1, self.num_heads, self.head_dim)

            attn_scores = (Q[dst] * K[src]).sum(dim=-1) / self.head_dim**0.5

            # Nuovo spatial bias (batch-local)
            # spatial_bias_tensor = self.compute_batch_spatial_bias(edge_index, x_dst.size(0))
            # attn_scores = attn_scores + spatial_bias_tensor.unsqueeze(-1)

            # bias_name = "__".join(edge_type)
            # attn_scores = attn_scores + self.edge_type_bias[bias_name]
                        # --- SPD bias veloce (k-hop) ---
            spd_bucket = self.compute_batch_spatial_bias_fast(edge_index, x_dst.size(0))   # [E] in {1..K, K+1}
            spd_bias = self.spd_emb(spd_bucket)  # [E, H]  embedding per head

            attn_scores = attn_scores + spd_bias  # broadcast OK: attn_scores [E,H]


            attn_weights = softmax(attn_scores, dst)
            attn_weights = self.dropout(attn_weights)

            out = V[src] * attn_weights.unsqueeze(-1)
            out = out.view(-1, self.channels)

            out_dict[dst_type].index_add_(0, dst, out)

        # Aggiunta degree centrality
        total_deg = self.compute_total_degrees(x_dict, edge_index_dict)
        for node_type in out_dict:
            deg_embed = total_deg[node_type].view(-1, 1).expand(-1, self.channels)
            out_dict[node_type] += deg_embed

        # Residual + layer norm
        for node_type in out_dict:
            out_dict[node_type] = self.norm(out_dict[node_type] + x_dict[node_type])

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

