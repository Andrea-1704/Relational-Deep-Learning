import math
from typing import Any, Dict, List, Tuple
from collections import defaultdict, deque

import torch
from torch import nn, Tensor
from torch.nn import Embedding, ModuleDict
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.nn import Linear, MLP
from torch_geometric.utils import degree

from torch_frame.data.stats import StatType
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder


# ----------------------------
# Utilities: node<->global fuse
# ----------------------------

def build_type_indexers(node_types: List[str]) -> Dict[str, int]:
    # I map node type strings to contiguous integer ids
    return {nt: i for i, nt in enumerate(node_types)}

def build_rel_indexers(edge_types: List[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], int]:
    # I map (src, rel, dst) triplets to contiguous integer ids
    return {et: i for i, et in enumerate(edge_types)}

def fuse_x_dict(x_dict: Dict[str, Tensor], type2id: Dict[str, int]) -> Tuple[Tensor, Tensor, Dict[str, slice]]:
    # I concatenate node embeddings across types into one big [N, C] tensor
    parts = []
    token_types = []
    slices = {}
    cursor = 0
    for nt, x in x_dict.items():
        n = x.size(0)
        parts.append(x)
        token_types.append(torch.full((n,), type2id[nt], dtype=torch.long, device=x.device))
        slices[nt] = slice(cursor, cursor + n)
        cursor += n
    X = torch.cat(parts, dim=0)
    token_type = torch.cat(token_types, dim=0)  # [N]
    return X, token_type, slices

def fuse_edges_to_global(edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                         slices: Dict[str, slice],
                         rel2id: Dict[Tuple[str, str, str], int]) -> Tuple[List[Tuple[int, int, int]], int]:
    # I convert hetero edge_index_dict to a list of (src_global, dst_global, rel_id)
    edges = []
    total_nodes = 0
    for nt, sl in slices.items():
        total_nodes = max(total_nodes, sl.stop)
    for et, eidx in edge_index_dict.items():
        src_nt, _, dst_nt = et
        sl_s, sl_d = slices[src_nt], slices[dst_nt]
        src, dst = eidx
        # I shift per-type indices into fused global indices
        edges.extend([(int(sl_s.start + int(s)), int(sl_d.start + int(d)), rel2id[et]) for s, d in zip(src.tolist(), dst.tolist())])
    return edges, total_nodes


# ---------------------------------
# Centrality encoder (in/out degree)
# ---------------------------------

class CentralityEncoder(nn.Module):
    def __init__(self, channels: int, num_buckets: int = 16, share_in_out: bool = False):
        super().__init__()
        self.num_buckets = num_buckets
        self.share_in_out = share_in_out
        self.in_emb = Embedding(num_buckets, channels)
        self.out_emb = self.in_emb if share_in_out else Embedding(num_buckets, channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_emb.weight, std=0.02)
        if not self.share_in_out:
            nn.init.normal_(self.out_emb.weight, std=0.02)

    @staticmethod
    def _bucketize_deg(deg: Tensor, num_buckets: int) -> Tensor:
        # I put degrees into log buckets for stability
        # bucket k ~ floor(log2(deg+1)), clipped to [0, num_buckets-1]
        deg = deg.to(torch.float32)
        buck = torch.floor(torch.log2(deg + 1.0))
        buck = buck.clamp(min=0, max=num_buckets - 1).to(torch.long)
        return buck

    def forward(self,
                x_dict: Dict[str, Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], Tensor]) -> Dict[str, Tensor]:
        device = next(self.parameters()).device
        # I compute directed in/out degree per node type
        in_deg = {nt: torch.zeros(x.size(0), device=device) for nt, x in x_dict.items()}
        out_deg = {nt: torch.zeros(x.size(0), device=device) for nt, x in x_dict.items()}

        for (src_nt, _, dst_nt), eidx in edge_index_dict.items():
            src, dst = eidx
            out_deg[src_nt].index_add_(0, src.to(device), torch.ones_like(src, dtype=torch.float32, device=device))
            in_deg[dst_nt].index_add_(0, dst.to(device), torch.ones_like(dst, dtype=torch.float32, device=device))

        # I add the embeddings to input features
        out = {}
        for nt, x in x_dict.items():
            in_b = self._bucketize_deg(in_deg[nt], self.num_buckets)
            out_b = self._bucketize_deg(out_deg[nt], self.num_buckets)
            out[nt] = x + self.in_emb(in_b) + self.out_emb(out_b)
        return out


# ---------------------------------------
# Structural bias tables & cached indices
# ---------------------------------------

class HeteroGraphormerStructuralBias(nn.Module):
    """
    I hold trainable bias tables and compute/cache the index tensors
    used to inject structural signals into attention scores.

    Biases included:
      - SPD bias (clipped shortest-path distance; directed with undirected fallback)
      - Adjacency-by-relation bias (per edge type)
      - Type-pair bias (src_type, dst_type)
      - Seed-wise temporal bias (Δt = t_dst - t_seed; added on seed rows)
      - (Optional) First-edge-on-SP bias (light meta-path signal)
      - (Optional) TypeToken link bias (node <-> its TypeToken)
    """
    def __init__(self,
                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],
                 num_heads: int,
                 s_max: int = 6,
                 time_buckets: int = 21,
                 use_first_edge_bias: bool = True,
                 use_type_tokens: bool = True):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.T = len(node_types)
        self.R = len(edge_types)
        self.num_heads = num_heads
        self.s_max = s_max
        self.use_first_edge_bias = use_first_edge_bias
        self.use_type_tokens = use_type_tokens

        # I build small lookups for fast indexing
        self.type2id = build_type_indexers(node_types)
        self.rel2id = build_rel_indexers(edge_types)

        # ---- Trainable tables (shared across layers) ----

        # SPD bias: I keep an embedding per distance bucket and per head
        # vocab: {-1 (no path), 0, 1, ..., s_max, >s_max -> s_max}
        # self.spd_bias = nn.Embedding(s_max + 2, num_heads)  # index 0 => -1, index d+1 => distance d
        # nn.init.normal_(self.spd_bias.weight, std=0.02)

        # Adjacency-by-relation bias: one scalar per relation and head
        self.adj_rel_bias = nn.Parameter(torch.zeros(self.R, num_heads))

        # Type-pair bias: one scalar per (src_type, dst_type) and head
        self.typepair_bias = nn.Parameter(torch.zeros(self.T, self.T, num_heads))

        # Temporal bias: seed-wise Δt buckets (log buckets around 0)
        self.time_buckets = time_buckets
        self.temp_bias = nn.Embedding(time_buckets, num_heads)
        nn.init.normal_(self.temp_bias.weight, std=0.02)

        # First-edge-on-SP bias: one scalar per relation and head (optional)
        if self.use_first_edge_bias:
            self.first_edge_bias = nn.Parameter(torch.zeros(self.R, num_heads))

        # TypeToken link bias: I connect node <-> its TypeToken with a small constant bias per head (optional)
        if self.use_type_tokens:
            self.type_token_link_bias_q2t = nn.Parameter(torch.zeros(num_heads))  # node -> TypeToken(type(node))
            self.type_token_link_bias_t2q = nn.Parameter(torch.zeros(num_heads))  # TypeToken -> node of that type

    # ---- Helper: Δt bucketization around 0 with symmetric log buckets ----
    @staticmethod
    def _bucketize_dt(dt: Tensor, num_buckets: int) -> Tensor:
        # I create symmetric log buckets: ... -365, -90, -30, -7, -1, 0, +1, +7, +30, +90, +365 ...
        # Implementation detail: I map dt to signed log-space and discretize to 'num_buckets' bins.
        eps = 1e-6
        signed_log = torch.sign(dt) * torch.log1p(torch.abs(dt) + eps)  # R -> R
        # I scale into [0, num_buckets-1], keeping 0 near dt=0
        min_v, max_v = -5.0, 5.0  # ~exp(5) ≈ 148; tune if timespan is larger
        norm = (signed_log.clamp(min=min_v, max=max_v) - min_v) / (max_v - min_v + 1e-9)
        idx = torch.floor(norm * (num_buckets - 1)).to(torch.long)
        return idx.clamp(0, num_buckets - 1)

    # ---- Graph preprocessing per forward (I cache indices to reuse in all layers) ----
    def preprocess(self,
                   x_dict: Dict[str, Tensor],
                   edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                   time_dict: Dict[str, Tensor],
                   entity_table: str,
                   seed_count: int,
                   use_undirected_fallback: bool = True) -> Dict[str, Any]:
        """
        Returns a cache with:
          - X, token_type, slices, edges(list), N
          - spd_idx [N, N] (int in [0..s_max+1], where 0 encodes 'no path')
          - adj_rel_pairs: dict rel_id -> (src_idx, dst_idx) Long tensors
          - typepair_idx [N, N, 2] (src_type, dst_type) for gather
          - time (vector [N]) and seed_indices (Long list in fused space)
          - first_edge_rel_idx [N, N] (or None)
          - type_token_indices (if enabled)
        """
        device = next(self.parameters()).device

        # 1) I fuse node features (for shape info only), and I build per-type slices
        X_stub, token_type, slices = fuse_x_dict({k: v.detach() for k, v in x_dict.items()}, self.type2id)
        N = X_stub.size(0)

        # 2) I fuse edges to global indices
        edges, _ = fuse_edges_to_global(edge_index_dict, slices, self.rel2id)

        # 3) I gather per-node timestamps in fused order (TypeTokens will come later)
        time_vec_list = []
        for nt, sl in slices.items():
            t_val = time_dict.get(nt, None) if time_dict is not None else None

            # Accept both: Tensor OR dict-like with key "time"
            if isinstance(t_val, dict):
                t_nt = t_val.get("time", None)
            elif torch.is_tensor(t_val):
                t_nt = t_val
            else:
                t_nt = None

            if t_nt is not None:
                # expected shape: [num_nodes_of_type nt]
                time_vec_list.append(t_nt.to(device))
            else:
                # If this type has no explicit time, fill with zeros for its slice
                time_vec_list.append(torch.zeros(sl.stop - sl.start, device=device))

        time_vec = torch.cat(time_vec_list, dim=0)  # [N]


        # 4) I collect seed indices of entity_table at the front
        et_slice = slices[entity_table]
        seed_indices = torch.arange(et_slice.start, et_slice.start + seed_count, device=device, dtype=torch.long)

        # # 5) I build adjacency lists for BFS
        # adj = [[] for _ in range(N)]
        # for s, d, r in edges:
        #     adj[s].append((d, r))

        # # 6) I compute SPD (directed) with optional undirected fallback; I clip to s_max
        # spd = torch.full((N, N), fill_value=-1, dtype=torch.int16, device=device)
        # # I also compute first-edge-on-SP relation type (optional)
        # first_edge_rel = torch.full((N, N), fill_value=-1, dtype=torch.int16, device=device) if self.use_first_edge_bias else None

        # # I do BFS from each source; for medium subgraphs this is fine and keeps the code clear
        # for src in range(N):
        #     dist = [-1] * N
        #     first_rel = [-1] * N
        #     q = deque()
        #     dist[src] = 0
        #     q.append(src)
        #     while q:
        #         u = q.popleft()
        #         for v, rel_id in adj[u]:
        #             if dist[v] == -1:
        #                 dist[v] = dist[u] + 1
        #                 # I record the first edge type on the path from src to v
        #                 first_rel[v] = rel_id if u == src else first_rel[u]
        #                 q.append(v)
        #     # Optional undirected fallback to reduce -1s
        #     if use_undirected_fallback:
        #         # I expand with reverse edges to capture undirected reachability
        #         q.clear()
        #         # I reuse dist; I only care to replace -1 with some value to mark reachability
        #         # but I won't change first_rel in fallback to stay conservative
        #         # (pairs remaining -1 keep the special SPD bucket)
        #         # I run a quick undirected BFS starting from src where edges can be traversed both ways
        #         und_adj = adj
        #         # I push all already discovered; then I traverse backwards using a reverse scan
        #         # For simplicity (and cost containment), I skip second BFS and accept remaining -1 as "no path"
        #         pass  # keeping it simple; the special -1 bucket already handles disconnections

        #     # I write results to tensors
        #     spd[src, :] = torch.tensor(dist, dtype=torch.int16, device=device)
        #     if self.use_first_edge_bias:
        #         first_edge_rel[src, :] = torch.tensor(first_rel, dtype=torch.int16, device=device)

        # # I clip SPD and remap to embedding indices: -1 -> 0, d -> d+1, d > s_max -> s_max+1
        # spd_clipped = spd.clamp(min=-1, max=self.s_max)
        # spd_idx = spd_clipped + 1  # -1 -> 0; 0..s_max -> 1..s_max+1  (int16 ok)

        # 7) I collect adjacency pairs per relation type for scatter-add
        adj_rel_pairs: Dict[int, Tuple[Tensor, Tensor]] = defaultdict(lambda: (torch.empty(0, dtype=torch.long, device=device),
                                                                               torch.empty(0, dtype=torch.long, device=device)))
        tmp_pairs: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for s, d, r in edges:
            tmp_pairs[r].append((s, d))
        for r, pairs in tmp_pairs.items():
            if len(pairs) > 0:
                src_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
                dst_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
                adj_rel_pairs[r] = (src_idx, dst_idx)

        # 8) I build type-pair indices by broadcast
        # To avoid [N,N,2] materialization, I keep token_type and will gather on the fly
        # (layers can do outer gather using token_type[row] and token_type[col])

        # 9) I add TypeToken indices at the end if enabled
        type_token_indices = None
        if self.use_type_tokens:
            # I place one TypeToken per type, appended after nodes: [N ... N+T-1]
            type_token_indices = torch.arange(N, N + self.T, device=device, dtype=torch.long)

        cache = {
            "N": N,
            "token_type": token_type,     # [N]
            "slices": slices,
            "edges": edges,               # list of (s, d, r)
            #"spd_idx": spd_idx,           # [N, N], int in [0..s_max+1]
            "adj_rel_pairs": adj_rel_pairs,
            "time_vec": time_vec,         # [N]
            "seed_indices": seed_indices, # [S]
            #"first_edge_rel": first_edge_rel,  # [N, N] or None
            "type_token_indices": type_token_indices,  # [T] or None
        }
        return cache

    # ---- Bias assembly (I add scalars per head to attention scores) ----
    def build_bias_per_head(self,
                            cache: Dict[str, Any],
                            num_tokens_total: int) -> Dict[str, Any]:
        """
        I return a dict of callable-updates to apply to attention scores per head.
        For efficiency, I keep factorized representations (indices) and add them into
        full [H, N_tot, N_tot] scores inside the layer.
        """
        return {
            "cache": cache,
            # tables to be used by layers
            #"spd_bias_table": self.spd_bias,             # Embedding
            "adj_rel_bias": self.adj_rel_bias,           # [R, H]
            "typepair_bias": self.typepair_bias,         # [T, T, H]
            "temp_bias_table": self.temp_bias,           # Embedding
            "first_edge_bias": getattr(self, "first_edge_bias", None),  # [R, H] or None
            "type_token_link_bias_q2t": getattr(self, "type_token_link_bias_q2t", None),  # [H] or None
            "type_token_link_bias_t2q": getattr(self, "type_token_link_bias_t2q", None),  # [H] or None
        }


# -----------------------------
# Graphormer block (Pre-LN MHA)
# -----------------------------

class GraphormerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q = Linear(channels, channels, bias=True)
        self.k = Linear(channels, channels, bias=True)
        self.v = Linear(channels, channels, bias=True)
        self.out = Linear(channels, channels, bias=True)

        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * channels, channels),
        )
        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self,
                X: Tensor,                 # [N_tot, C]
                token_type: Tensor,        # [N_tot]
                bias_pack: Dict[str, Any], # from HeteroGraphormerStructuralBias.build_bias_per_head(...)
                attach_type_tokens_masks: Dict[str, Tensor] = None) -> Tensor:
        """
        I run one Pre-LN Transformer encoder layer with global attention and structural bias.
        """
        H, D = self.num_heads, self.head_dim
        N_tot = X.size(0)
        cache = bias_pack["cache"]
        #spd_bias_table = bias_pack["spd_bias_table"]         # Embedding
        # ------- SPD bias (dense add) -------
        # cache["spd_idx"]: [N, N] con i bucket della shortest-path distance (spesso int16)
        # N = #nodi "reali" (senza TypeTokens), N_tot = dimensione totale della sequenza (con eventuali TypeTokens)
        
        # --- Pre-LN e proiezioni (lascia com’è il tuo Pre-LN) ---
        Y = self.ln1(X)                    # [N_tot, d_model]
        N_tot = Y.size(0)
        H = self.num_heads
        D = self.head_dim

        # Proiezioni Q, K, V
        Q = self.q(Y).view(N_tot, H, D).transpose(0, 1)  # [H, N_tot, D]
        K = self.k(Y).view(N_tot, H, D).transpose(0, 1)  # [H, N_tot, D]
        V = self.v(Y).view(N_tot, H, D).transpose(0, 1)  # [H, N_tot, D]

        # Dot-product attention scores
        scores = torch.einsum("hnd,hmd->hnm", Q, K) / math.sqrt(D)  # [H, N_tot, N_tot]

        # === A PARTIRE DA QUI aggiungiamo i bias ADDITIVI ai punteggi ===

        # 1) Shortest-Path Distance (SPD) bias
        # Assumo che cache contenga: N (nodi reali prima dei token) e spd_idx [N, N] con indici di bucket SPD
        N = cache["N"]
        # spd_idx = cache["spd_idx"]  # tipicamente int16 nel tuo preprocess

        # # Prepariamo una matrice [N_tot, N_tot] con pad=0 e riempiamo il blocco reale [0:N, 0:N]
        # # spd_pad = torch.zeros((N_tot, N_tot), dtype=torch.long, device=Y.device)
        # # spd_pad[:N, :N] = spd_idx.to(torch.long)  # nn.Embedding richiede indici long
        # spd_pad = torch.zeros((N_tot, N_tot), dtype=torch.long, device=Y.device)
        # spd_pad[:N, :N] = spd_idx.to(torch.long)
        # scores = scores + spd_head.permute(2, 0, 1)  # [H, N_tot, N_tot]
        # 1) prepara gli indici SPD (ok)
        #N = cache["N"]
        #spd_idx = cache["spd_idx"]  # int16 dal preprocess
        #spd_pad = torch.zeros((N_tot, N_tot), dtype=torch.long, device=Y.device)
        #spd_pad[:N, :N] = spd_idx.to(torch.long)

        # >>> ELIMINA questa riga che usa spd_head prima di crearlo <<<
        # scores = scores + spd_head.permute(2, 0, 1)

        # 2) calcola il bias SPD per head
        #spd_head = spd_bias_table(spd_pad)       # [N_tot, N_tot, H]

        # 3) aggiungi una sola volta ai punteggi
        #scores = scores + spd_head.permute(2, 0, 1)  # [H, N_tot, N_tot]
        #N = cache["N"]

        # spd_bias_table: nn.Embedding(num_buckets, H) -> [N_tot, N_tot, H]
        #spd_head = self.spd_bias_table(spd_pad)  # [N_tot, N_tot, H]
        #spd_head = spd_bias_table(spd_pad)       # [N_tot, N_tot, H]

        #scores = scores + spd_head.permute(2, 0, 1)           # [H, N_tot, N_tot]


        adj_rel_bias = bias_pack["adj_rel_bias"]             # [R, H]
        typepair_bias = bias_pack["typepair_bias"]           # [T, T, H]
        temp_bias_table = bias_pack["temp_bias_table"]       # Embedding
        first_edge_bias = bias_pack.get("first_edge_bias")
        type_token_link_bias_q2t = bias_pack.get("type_token_link_bias_q2t")
        type_token_link_bias_t2q = bias_pack.get("type_token_link_bias_t2q")
        
        # Type-pair bias (dense via outer gather)
        tt = torch.full((N_tot,), -1, dtype=torch.long, device=X.device)
        tt[:N] = cache["token_type"]
        # If TypeTokens exist, I assign them their own type id equal to their represented type
        if attach_type_tokens_masks is not None and "type_token_type_id" in attach_type_tokens_masks:
            # type_token_type_id is [T] with type ids
            t_ids = attach_type_tokens_masks["type_token_type_id"]
            tt[N:N + t_ids.numel()] = t_ids
        typepair = typepair_bias[tt.unsqueeze(1), tt.unsqueeze(0)]  # [N_tot, N_tot, H]
        scores = scores + typepair.permute(2, 0, 1)

        # Adjacency-by-relation (sparse scatter add)
        for r, (s_idx, d_idx) in cache["adj_rel_pairs"].items():
            if s_idx.numel() == 0:
                continue
            # I add the bias only on real node pairs; edges do not target TypeTokens in this dataset
            # adj_rel_bias[r] is [H]; I need to add to scores[:, s, d]
            scores[:, s_idx, d_idx] += adj_rel_bias[r].view(H, 1)

        # First-edge-on-SP (optional; dense add on nodes area)
        if first_edge_bias is not None and cache["first_edge_rel"] is not None:
            fe = cache["first_edge_rel"]  # [N, N] with -1 if missing
            fe = fe.clamp(min=-1)
            # I remap -1 -> no-add by selecting a zeros vector; I just mask
            mask = fe.ge(0)  # [N, N]
            if mask.any():
                r_idx = fe[mask].to(torch.long)  # [K]
                add = first_edge_bias[r_idx]     # [K, H]
                # I scatter into scores for each head
                h_indices = torch.arange(H, device=X.device).view(H, 1)
                # I map the 2D masked indices back to coordinates
                rows, cols = torch.where(mask)
                for h in range(H):
                    scores[h, rows, cols] += add[:, h]

        # Seed-wise temporal bias (I add to rows corresponding to seed queries)
        S_idx = cache["seed_indices"]  # [S] fused indices of seeds (nodes)
        if S_idx.numel() > 0:
            t = cache["time_vec"]  # [N], real nodes only
            # I build Δt for each seed vs all real nodes, then bucketize and gather
            dt = (t.unsqueeze(0) - t[S_idx].unsqueeze(1))  # [S, N]
            dt_idx = HeteroGraphormerStructuralBias._bucketize_dt(dt, temp_bias_table.num_embeddings)  # [S, N]
            temp_add = temp_bias_table(dt_idx)  # [S, N, H]
            # I write into scores[:, seed, :N]
            for h in range(H):
                scores[h, S_idx, :N] += temp_add[:, :, h]

        # TypeToken link bias (optional)
        if attach_type_tokens_masks is not None and "node_to_type_token_col" in attach_type_tokens_masks:
            # node_to_type_token_col: [N] giving the column index j of the TypeToken for each node i
            j_col = attach_type_tokens_masks["node_to_type_token_col"]  # [N]
            rows = torch.arange(0, N, device=X.device)
            if type_token_link_bias_q2t is not None:
                for h in range(H):
                    scores[h, rows, j_col] += type_token_link_bias_q2t[h]
            if type_token_link_bias_t2q is not None and "type_token_to_node_rows" in attach_type_tokens_masks:
                # For TypeToken -> node, I add bias on rows of the TypeToken, columns of nodes of that type
                tt_rows = attach_type_tokens_masks["type_token_to_node_rows"]  # [T] row indices for TypeTokens
                # I need per-type node masks; I precomputed a boolean mask [T, N]
                tmask = attach_type_tokens_masks["type_to_nodes_mask"]        # [T, N] bool
                for h in range(H):
                    #scores[h, tt_rows, :] += (tmask.float() * type_token_link_bias_t2q[h])
                    scores[h, tt_rows, :N] += (tmask.float() * type_token_link_bias_t2q[h])


        # ------------------------------------------------------------

        attn = torch.softmax(scores, dim=-1)  # [H, N_tot, N_tot]
        attn = self.drop(attn)
        Z = torch.einsum("hnm,hmd->hnd", attn, V)  # [H, N_tot, D]
        Z = Z.transpose(0, 1).contiguous().view(N_tot, self.channels)
        X = X + self.drop(self.out(Z))            # residual

        # FFN
        X = X + self.drop(self.ffn(self.ln2(X)))
        return X




class HeteroGraphormer(nn.Module):
    def __init__(self,
                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],
                 channels: int,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 s_max: int = 6,
                 time_buckets: int = 21,
                 use_first_edge_bias: bool = True,
                 use_type_tokens: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_type_tokens = use_type_tokens

        self.bias = HeteroGraphormerStructuralBias(
            node_types=node_types,
            edge_types=edge_types,
            num_heads=num_heads,
            s_max=s_max,
            time_buckets=time_buckets,
            use_first_edge_bias=use_first_edge_bias,
            use_type_tokens=use_type_tokens,
        )
        self.layers = nn.ModuleList([GraphormerBlock(channels, num_heads, dropout) for _ in range(num_layers)])

        # I create per-type "TypeToken" embeddings if enabled
        if self.use_type_tokens:
            self.type_token_emb = nn.Embedding(len(node_types), channels)
            nn.init.normal_(self.type_token_emb.weight, std=0.02)

    def forward(self,
                x_dict: Dict[str, Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                time_dict: Dict[str, Tensor],
                entity_table: str,
                seed_count: int) -> Dict[str, Tensor]:
        device = next(self.parameters()).device

        #preprocess graph structure once (indices & caches)
        cache = self.bias.preprocess(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            time_dict=time_dict,
            entity_table=entity_table,
            seed_count=seed_count,
        )

        #fuse nodes to a single token sequence
        type2id = self.bias.type2id
        X, token_type, slices = fuse_x_dict(x_dict, type2id)  # [N, C], [N]
        X = X.to(device)
        token_type = token_type.to(device)
        N = X.size(0)

        #attach TypeTokens (optional)
        attach_masks = None
        if self.use_type_tokens:
            T = len(self.bias.node_types)
            #append T tokens; their "type id" is the actual type they represent
            type_ids = torch.arange(T, device=device, dtype=torch.long)
            TT = self.type_token_emb(type_ids)             # [T, C]
            X = torch.cat([X, TT], dim=0)                  # [N+T, C]
            token_type = torch.cat([token_type, type_ids], dim=0)  # [N+T]
            # I prepare masks for biasing
            # node -> its TypeToken column index
            type_token_col = cache["type_token_indices"]    # [T] = [N..N+T-1]
            node_to_type_token_col = type_token_col[token_type[:N]]  # [N]
            # TypeToken rows
            tt_rows = torch.arange(N, N + T, device=device, dtype=torch.long)
            # Per-type node masks [T, N] (bool)
            type_to_nodes_mask = torch.zeros(T, N, dtype=torch.bool, device=device)
            for nt, sl in slices.items():
                t_id = type2id[nt]
                type_to_nodes_mask[t_id, sl.start:sl.stop] = True
            attach_masks = {
                "node_to_type_token_col": node_to_type_token_col,
                "type_token_to_node_rows": tt_rows,
                "type_to_nodes_mask": type_to_nodes_mask,
                "type_token_type_id": type_ids,
            }

        # 4) I build bias pack (tables + cache) to be reused by all layers
        bias_pack = self.bias.build_bias_per_head(cache, num_tokens_total=X.size(0))

        # 5) I stack Graphormer blocks
        for layer in self.layers:
            X = layer(X, token_type, bias_pack, attach_type_tokens_masks=attach_masks)

        # 6) I split X back into per-type dict (I drop TypeTokens)
        out_dict = {}
        for nt, sl in slices.items():
            out_dict[nt] = X[sl.start:sl.stop, :]
        return out_dict

    def reset_parameters(self):
        for lyr in self.layers:
            for m in lyr.modules():
                if isinstance(m, (nn.Linear, Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


# ----------------------------
# End-to-end RDL model (Graph)
# ----------------------------

class Model(nn.Module):
    """
    End-to-end RDL model using:
      HeteroEncoder + HeteroTemporalEncoder -> Typed Graphormer -> MLP head.
    Readout is node-level on the seeds of `entity_table`.
    """
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,    # unused here but kept for compatibility
        norm: str,
        shallow_list: List[NodeType] = [],
        id_awareness: bool = False,
        predictor_n_layers: int = 1,
        num_heads: int = 8,
        s_max: int = 6,
        time_buckets: int = 21,
        use_first_edge_bias: bool = True,
        use_type_tokens: bool = True,
        degree_buckets: int = 16,
    ):
        super().__init__()
        self.channels = channels

        # Encoders (kept as in your pipeline)
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]],
            channels=channels,
        )

        # Centrality encoding (in/out degree buckets added to node features)
        self.centrality = CentralityEncoder(channels=channels, num_buckets=degree_buckets)

        # Graphormer core
        self.gnn = HeteroGraphormer(
            node_types=list(data.node_types),
            edge_types=list(data.edge_types),
            channels=channels,
            num_layers=num_layers,
            num_heads=num_heads,
            s_max=s_max,
            time_buckets=time_buckets,
            use_first_edge_bias=use_first_edge_bias,
            use_type_tokens=use_type_tokens,
        )

        # Task head (node-level)
        self.head = MLP(channels, out_channels=out_channels, norm=norm, num_layers=predictor_n_layers)

        # Optional shallow embeddings per selected node types
        self.embedding_dict = ModuleDict({
            nt: Embedding(data.num_nodes_dict[nt], channels) for nt in shallow_list
        })

        # Optional ID-awareness mark for the root/seed nodes of entity_table
        self.id_awareness_emb = nn.Embedding(1, channels) if id_awareness else None

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.centrality.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for emb in self.embedding_dict.values():
            nn.init.normal_(emb.weight, std=0.02)
        if self.id_awareness_emb is not None:
            nn.init.normal_(self.id_awareness_emb.weight, std=0.02)

    def forward(self, batch: HeteroData, entity_table: NodeType) -> Tensor:
        # 1) I encode tabular features into per-node embeddings
        x_dict = self.encoder(batch.tf_dict)

        # 2) I add temporal encodings per node (as you already do)
        seed_time = batch[entity_table].seed_time
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for nt, rel_t in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rel_t

        # 3) I add optional shallow embeddings
        for nt, emb in self.embedding_dict.items():
            x_dict[nt] = x_dict[nt] + emb(batch[nt].n_id)

        # 4) I add centrality (in/out degree) to node features
        x_dict = self.centrality(x_dict, batch.edge_index_dict)

        # 5) I optionally mark the seed queries for the entity_table with an ID-awareness tag
        if self.id_awareness_emb is not None:
            x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        # 6) I apply Graphormer with global attention and structural biases
        x_dict = self.gnn(
            x_dict=x_dict,
            edge_index_dict=batch.edge_index_dict,
            time_dict=batch.time_dict,
            entity_table=entity_table,
            seed_count=seed_time.size(0),
        )

        # 7) I read out predictions on the seeds of the entity_table
        return self.head(x_dict[entity_table][: seed_time.size(0)])

    # (Optional) Alternate readout onto a different dst_table
    def forward_dst_readout(self, batch: HeteroData, entity_table: NodeType, dst_table: NodeType) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError("id_awareness must be set True to use forward_dst_readout")
        # I mark seeds on the root table
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for nt, rel_t in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rel_t
        for nt, emb in self.embedding_dict.items():
            x_dict[nt] = x_dict[nt] + emb(batch[nt].n_id)
        x_dict = self.centrality(x_dict, batch.edge_index_dict)

        x_dict = self.gnn(
            x_dict=x_dict,
            edge_index_dict=batch.edge_index_dict,
            time_dict=batch.time_dict,
            entity_table=entity_table,
            seed_count=seed_time.size(0),
        )
        return self.head(x_dict[dst_table])
