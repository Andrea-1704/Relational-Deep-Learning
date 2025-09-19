from linecache import cache
import math
from typing import Any, Dict, List, Optional, Tuple
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
    #map node type strings to contiguous integer ids
    #basically for each node type we have an index value to represent it.
    return {nt: i for i, nt in enumerate(node_types)}

def build_rel_indexers(edge_types: List[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], int]:
    #map (src, rel, dst) triplets to contiguous integer ids, the same as above but for relations.
    return {et: i for i, et in enumerate(edge_types)}

def fuse_x_dict(x_dict: Dict[str, Tensor], type2id: Dict[str, int]) -> Tuple[Tensor, Tensor, Dict[str, slice]]:
    """
    This function basicly fuses node features from different types into a single tensor. If 
    we have 3 nodes of type drivers with an embedding size of 4 and 2 nodes of type cars with an embedding size of 4,
    the output will be a tensor of size [5, 4]. It also returns a token_type tensor that indicates the type of each node in the fused tensor.
    Additionally, it returns a dictionary of slices that indicate the position of each node type in the fused tensor.
    1. X: [N, C] fused node features
    2. token_type: [N] long tensor with type ids per node
    3. slices: dict of node_type -> slice(start, end) in X

    ex:
    slices = {
        "driver": slice(0, 3),  # driver occupy global indices 0,1,2
        "race": slice(3, 5),    # race occupy global indices 3,4
    }
    """
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
                         rel2id: Dict[Tuple[str, str, str], int]) -> Tuple[Dict[int, Tuple[Tensor, Tensor]], int]:
    """
    Convert a heterogeneous edge_index_dict into a flat list of global edges.

    This function "fuses" per-type edges into a single global index space, so that
    all nodes can be treated as if they were part of one big homogeneous graph.
    For each edge type (src_type, relation, dst_type):
      1. We take its edge_index (two tensors [src, dst] of local indices)
      2. We shift the indices by the slice.start for the corresponding node type
         to obtain global indices in the fused representation
      3. We attach the numeric relation ID (from rel2id) so each edge carries
         information about its type

    Returns:
        edges: list of (global_src, global_dst, rel_id)
        total_nodes: total number of fused nodes (max index + 1)

    Example:
        Suppose we have two node types:
            driver: 3 nodes  -> occupy global positions [0, 1, 2] (slices returned by fuse_x_dict)
            race:   2 nodes  -> occupy global positions [3, 4]

        slices = {
            "driver": slice(0, 3),
            "race": slice(3, 5)
        }

        And one relation ("driver", "participated_in", "race") with local edge_index:

            src = [0, 1, 2]  # local driver indices
            dst = [1, 0, 1]  # local race indices
            rel2id = {("driver", "participated_in", "race"): 0}

        After fusion:
            - src_global = src + slices["driver"].start = [0, 1, 2]
            - dst_global = dst + slices["race"].start   = [4, 3, 4]

        Result:
            edges = [
                (0, 4, 0),  # driver0 -> race1
                (1, 3, 0),  # driver1 -> race0
                (2, 4, 0),  # driver2 -> race1
            ]
            total_nodes = 5 (because we have 3 drivers + 2 races)
    """
    if len(edge_index_dict) == 0:
        total_nodes = max((sl.stop for sl in slices.values()), default=0)
        return {}, total_nodes

    src_all, dst_all, rel_all = [], [], []
    for (src_nt, rel, dst_nt), eidx in edge_index_dict.items():
        src, dst = eidx
        s_off = slices[src_nt].start
        d_off = slices[dst_nt].start
        src_all.append(src + s_off)
        dst_all.append(dst + d_off)
        rel_id = rel2id[(src_nt, rel, dst_nt)]
        rel_all.append(torch.full_like(src, rel_id, dtype=torch.long))

    src_all = torch.cat(src_all, dim=0)
    dst_all = torch.cat(dst_all, dim=0)
    rel_all = torch.cat(rel_all, dim=0)

    adj_rel_pairs: Dict[int, Tuple[Tensor, Tensor]] = {}
    for r in rel_all.unique():
        mask = (rel_all == r)
        adj_rel_pairs[int(r.item())] = (src_all[mask], dst_all[mask])

    total_nodes = max(sl.stop for sl in slices.values())
    return adj_rel_pairs, total_nodes


# ---------------------------------
# Centrality encoder (in/out degree)
# ---------------------------------

class CentralityEncoder(nn.Module):
    """
    Add learnable degree-based encodings to each node representation in a heterogeneous graph.

    Overview
    --------
    For every node, we compute its directed in-degree and out-degree across all relations,
    bucketize those counts with a logarithmic rule, then add two learnable embeddings:
    one for the in-degree bucket and one for the out-degree bucket. This gives the
    transformer a lightweight topological prior without explicitly constructing
    higher-order structural features.

    Arguments
    ---------
    channels : int
        Feature dimensionality of each node embedding.
    num_buckets : int, default 16
        Number of logarithmic buckets used to discretize degree values.
    share_in_out : bool, default False
        If True, reuse the same embedding table for in-degree and out-degree.
        If False, learn two separate tables.

    Inputs
    ------
    x_dict : Dict[str, Tensor]
        A mapping from node type to a tensor of shape [num_nodes_of_type, channels].
        These are the current node features that will be enriched with centrality signals.
    edge_index_dict : Dict[Tuple[str, str, str], LongTensor]
        A mapping from (src_type, relation, dst_type) to a LongTensor of shape [2, num_edges],
        with per-type local indices for src and dst.

    Returns
    -------
    Dict[str, Tensor]
        Same keys as x_dict, each value has shape [num_nodes_of_type, channels].
        For every node feature x, the output is:
            x + in_emb[bucket_in(x)] + out_emb[bucket_out(x)]

    Degree Bucketization
    --------------------
    We map raw degrees to integer buckets with:
        bucket = floor( log2(degree + 1) )
    Then we clamp to [0, num_buckets - 1].
    This compresses large counts while preserving resolution for small counts.

    Numeric Example
    ---------------
    Suppose:
      channels = 2, num_buckets = 4, share_in_out = False

      Node types:
        - "driver" has 3 nodes
        - "race"   has 2 nodes

      Let x_dict be zeros for simplicity:
        x_dict["driver"] = zeros([3, 2])
        x_dict["race"]   = zeros([2, 2])

      Edges (directed from drivers to races):
        edge_index_dict[("driver", "participated_in", "race")] =
            [[0, 0, 1, 2],   # src local indices in "driver"
             [0, 1, 1, 1]]   # dst local indices in "race"

      Degrees computed across all relations:
        driver out-degree: counts by index -> [2, 1, 1]
        driver in-degree:  no incoming edges here -> [0, 0, 0]
        race   in-degree:  counts by index -> [1, 3]
        race   out-degree: none here -> [0, 0]

      Bucketization with floor(log2(deg + 1)):
        deg=0 -> 0,  deg=1 -> 1,  deg=2 -> 1,  deg=3 -> 2

      Therefore:
        driver in-buckets  = [0, 0, 0]
        driver out-buckets = [1, 1, 1]
        race   in-buckets  = [1, 2]
        race   out-buckets = [0, 0]

      For illustration, assume the learnable embeddings currently equal:
        in_emb[0] = [0.00, 0.00]
        in_emb[1] = [0.10, 0.10]
        in_emb[2] = [0.20, 0.20]
        out_emb[0] = [0.00, 0.00]
        out_emb[1] = [1.00, 1.00]
        out_emb[2] = [2.00, 2.00]

      Updated node features (x is zero everywhere in this toy example):
        driver:
          node 0: x + in_emb[0] + out_emb[1] = [1.00, 1.00]
          node 1: x + in_emb[0] + out_emb[1] = [1.00, 1.00]
          node 2: x + in_emb[0] + out_emb[1] = [1.00, 1.00]
        race:
          node 0: x + in_emb[1] + out_emb[0] = [0.10, 0.10]
          node 1: x + in_emb[2] + out_emb[0] = [0.20, 0.20]

    Implementation Notes
    --------------------
    - Degrees are counted with efficient index_add on device.
    - Bucketization is vectorized and numerically stable.
    - Shapes and devices follow the inputs.
    - When share_in_out=True the same table is used for both directions,
      which reduces parameters but ties the two signals.

    """
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
        # put degrees into log buckets for stability
        # bucket k =floor(log2(deg+1)), clipped to [0, num_buckets-1]
        deg = deg.to(torch.float32)
        buck = torch.floor(torch.log2(deg + 1.0))
        buck = buck.clamp(min=0, max=num_buckets - 1).to(torch.long)
        return buck

    def forward(self,
                x_dict: Dict[str, Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], Tensor]) -> Dict[str, Tensor]:
        device = next(self.parameters()).device
        # compute directed in/out degree per node type
        in_deg = {nt: torch.zeros(x.size(0), device=device) for nt, x in x_dict.items()}
        out_deg = {nt: torch.zeros(x.size(0), device=device) for nt, x in x_dict.items()}

        for (src_nt, _, dst_nt), eidx in edge_index_dict.items():
            src, dst = eidx
            out_deg[src_nt].index_add_(0, src.to(device), torch.ones_like(src, dtype=torch.float32, device=device))
            in_deg[dst_nt].index_add_(0, dst.to(device), torch.ones_like(dst, dtype=torch.float32, device=device))

        # Add the embeddings to input features
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
    hold trainable bias tables and compute/cache the index tensors
    used to inject structural signals into attention scores.

    Biases included:
      - SPD bias (clipped shortest-path distance; directed with undirected fallback)
      - Adjacency-by-relation bias (per edge type)
      - Type-pair bias (src_type, dst_type)
      - Seed-wise temporal bias (Δt = t_dst - t_seed; added on seed rows)
    """
    def __init__(self,
                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],
                 num_heads: int,
                 time_buckets: int = 21,):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.T = len(node_types) #num_node types 
        self.R = len(edge_types) #num_edge types
        self.num_heads = num_heads

        self.type2id = build_type_indexers(node_types)
        self.rel2id = build_rel_indexers(edge_types)

        # Per ogni relazione aggiungiamo un bias learnable:
        self.adj_rel_bias = nn.Parameter(torch.zeros(self.R, num_heads))

        # Type-pair bias: one scalar per (src_type, dst_type) and head
        self.typepair_bias = nn.Parameter(torch.zeros(self.T, self.T, num_heads))

        # Temporal bias: seed-wise Δt buckets (log buckets around 0)
        self.time_buckets = time_buckets
        self.temp_bias = nn.Embedding(time_buckets, num_heads)
        nn.init.normal_(self.temp_bias.weight, std=0.02)
        

    @staticmethod
    def _bucketize_dt(dt: Tensor, num_buckets: int) -> Tensor:
        # create symmetric log buckets: ... -365, -90, -30, -7, -1, 0, +1, +7, +30, +90, +365 ...
        # Implementation detail: I map dt to signed log-space and discretize to 'num_buckets' bins.
        eps = 1e-6
        signed_log = torch.sign(dt) * torch.log1p(torch.abs(dt) + eps)  # R -> R
        # I scale into [0, num_buckets-1], keeping 0 near dt=0
        min_v, max_v = -5.0, 5.0 
        norm = (signed_log.clamp(min=min_v, max=max_v) - min_v) / (max_v - min_v + 1e-9)
        idx = torch.floor(norm * (num_buckets - 1)).to(torch.long)
        return idx.clamp(0, num_buckets - 1)


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

        # fuse node features (for shape info only), and I build per-type slices
        X_stub, token_type, slices = fuse_x_dict({k: v.detach() for k, v in x_dict.items()}, self.type2id)
        N = X_stub.size(0)

        # gather per-node timestamps in fused order (TypeTokens will come later)
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


        # collect seed indices of entity_table at the front
        et_slice = slices[entity_table]
        seed_indices = torch.arange(et_slice.start, et_slice.start + seed_count, device=device, dtype=torch.long)

       
        adj_rel_pairs, _ = fuse_edges_to_global(edge_index_dict, slices, self.rel2id)

        # build type-pair indices by broadcast
        # To avoid [N,N,2] materialization, I keep token_type and will gather on the fly
        # (layers can do outer gather using token_type[row] and token_type[col])

        # add TypeToken indices at the end if enabled
        type_token_indices = None
        

        cache = {
            "N": N,
            "token_type": token_type,     # [N]
            "slices": slices,
            "adj_rel_pairs": adj_rel_pairs,
            "time_vec": time_vec,         # [N]
            "seed_indices": seed_indices, # [S]
            "type_token_indices": type_token_indices,  # [T] or None
        }
        return cache


    def build_bias_per_head(self,
                            cache: Dict[str, Any],
                            num_tokens_total: int) -> Dict[str, Any]:
        """
        return a dict of callable-updates to apply to attention scores per head.
        For efficiency, I keep factorized representations (indices) and add them into
        full [H, N_tot, N_tot] scores inside the layer.
        """
        return {
            "cache": cache,
            # tables to be used by layers
            "adj_rel_bias": self.adj_rel_bias,           # [R, H]
            "typepair_bias": self.typepair_bias,         # [T, T, H]
            "temp_bias_table": self.temp_bias,           # Embedding
            "type_token_link_bias_q2t": getattr(self, "type_token_link_bias_q2t", None),  # [H] or None
            "type_token_link_bias_t2q": getattr(self, "type_token_link_bias_t2q", None),  # [H] or None
        }


# -----------------------------
# Graphormer block (Pre-LN MHA)
# -----------------------------
from torch.nn.functional import scaled_dot_product_attention as sdpa

class GraphormerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, dropout: float = 0.005, rel_count: int = 0):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.rel_proj = nn.Linear(channels, channels, bias=True)
        self.rel_gate = nn.Parameter(torch.ones(rel_count, channels))
        self.mp_alpha = nn.Parameter(torch.tensor(-2.0))


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
        # self.ffn = nn.Sequential(
        #     nn.Linear(channels, channels * 4),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(channels * 4, channels * 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(channels *2, channels)
        # )
        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self,
                X: Tensor,                 
                token_type: Tensor,        
                bias_pack: Dict[str, Any],
                attach_type_tokens_masks: Dict[str, Tensor] = None,
                query_idx: Optional[Tensor] = None) -> Tensor:


        
        H, D = self.num_heads, self.head_dim
        N_tot = X.size(0)

        Y = self.ln1(X)
        Q_full = self.q(Y).view(N_tot, H, D).transpose(0, 1)  # [H, N_tot, D]
        K = self.k(Y).view(N_tot, H, D).transpose(0, 1)       # [H, N_tot, D]
        V = self.v(Y).view(N_tot, H, D).transpose(0, 1)       # [H, N_tot, D]

        cache = bias_pack["cache"]
        N = cache["N"]  # numero di nodi reali (senza TypeToken)

        # Se usiamo query seed, limitiamo le righe a query_idx
        if query_idx is not None:
            q_len = int(query_idx.numel())
            Q = Q_full[:, query_idx, :]                       # [H, q_len, D]
            qmap = torch.full((N_tot,), -1, dtype=torch.long, device=X.device)
            qmap[query_idx] = torch.arange(q_len, device=X.device)
        else:
            q_len = N_tot
            Q = Q_full

        # Costruiamo una sola bias map additiva per SDPA
        attn_bias = torch.zeros((H, q_len, N_tot), device=X.device, dtype=Q.dtype)

        # --- Type-pair bias ---
        adj_rel_bias = bias_pack["adj_rel_bias"]             # [R, H]
        typepair_bias = bias_pack["typepair_bias"]           # [T, T, H]
        temp_bias_table = bias_pack["temp_bias_table"]       # nn.Embedding
        first_edge_bias = bias_pack.get("first_edge_bias")
        type_token_link_bias_q2t = bias_pack.get("type_token_link_bias_q2t")
        type_token_link_bias_t2q = bias_pack.get("type_token_link_bias_t2q")

        tt = torch.full((N_tot,), -1, dtype=torch.long, device=X.device)
        tt[:N] = cache["token_type"]
        if attach_type_tokens_masks is not None and "type_token_type_id" in attach_type_tokens_masks:
            t_ids = attach_type_tokens_masks["type_token_type_id"]
            tt[N:N + t_ids.numel()] = t_ids

        if query_idx is None:
            # [N_tot, N_tot, H] -> [H, N_tot, N_tot]
            tp = typepair_bias[tt.unsqueeze(1), tt.unsqueeze(0)].permute(2, 0, 1)
            attn_bias = attn_bias + tp
        else:
            t_q = tt[query_idx]                               # [q_len]
            # [q_len, N_tot, H] -> [H, q_len, N_tot]
            tp_rows = typepair_bias[t_q][:, tt, :].permute(2, 0, 1)
            attn_bias = attn_bias + tp_rows

        # --- Adiacenza per relazione ---
        for r, (s_idx, d_idx) in cache["adj_rel_pairs"].items():
            if s_idx.numel() == 0:
                continue
            if query_idx is None:
                attn_bias[:, s_idx, d_idx] += adj_rel_bias[r].view(H, 1)
            else:
                sel = qmap[s_idx]
                mask = sel.ge(0)
                if mask.any():
                    rows = sel[mask]
                    cols = d_idx[mask]
                    attn_bias[:, rows, cols] += adj_rel_bias[r].view(H, 1)

        # --- First-edge-on-SP (opzionale) ---
        if first_edge_bias is not None and cache["first_edge_rel"] is not None:
            fe = cache["first_edge_rel"].clamp(min=-1)  # [N, N]
            if query_idx is None:
                mask = fe.ge(0)
                if mask.any():
                    rows, cols = torch.where(mask)
                    add = first_edge_bias[fe[rows, cols].to(torch.long)]  # [K, H]
                    for h in range(H):
                        attn_bias[h, rows, cols] += add[:, h]
            else:
                # righe limitate alle query
                fe_q = fe[query_idx, :]                          # [q_len, N]
                mask = fe_q.ge(0)
                if mask.any():
                    qr, qc = torch.where(mask)                    # indici locali [0..q_len)
                    add = first_edge_bias[fe_q[qr, qc].to(torch.long)]
                    for h in range(H):
                        attn_bias[h, qr, qc] += add[:, h]

        # --- Bias temporale seed wise (aggiungo solo sulle righe dei seed) ---
        S_idx = query_idx if query_idx is not None else cache["seed_indices"]
        if S_idx.numel() > 0:
            t = cache["time_vec"]                    # [N]
            dt = (t.unsqueeze(0) - t[S_idx].unsqueeze(1))  # [S, N]
            dt_idx = HeteroGraphormerStructuralBias._bucketize_dt(dt, temp_bias_table.num_embeddings)
            temp_add = temp_bias_table(dt_idx)       # [S, N, H]
            temp_add = temp_add.permute(2, 0, 1)     # [H, S, N]
            if query_idx is None:
                # qui S==|seed| ma righe sono N_tot, sommo solo su quelle S
                # mappo i seed su posizioni locali
                s_map = torch.arange(S_idx.numel(), device=X.device)
                attn_bias[:, S_idx, :N] += temp_add
            else:
                attn_bias[:, torch.arange(S_idx.numel(), device=X.device), :N] += temp_add

        # --- TypeToken link bias opzionale ---
        if attach_type_tokens_masks is not None and "node_to_type_token_col" in attach_type_tokens_masks:
            j_col = attach_type_tokens_masks["node_to_type_token_col"]  # [N]
            if type_token_link_bias_q2t is not None:
                rows = torch.arange(0, N if query_idx is None else q_len, device=X.device)
                for h in range(H):
                    attn_bias[h, rows, j_col] += type_token_link_bias_q2t[h]
            if type_token_link_bias_t2q is not None and "type_token_to_node_rows" in attach_type_tokens_masks:
                tt_rows = attach_type_tokens_masks["type_token_to_node_rows"]  # [T]
                tmask = attach_type_tokens_masks["type_to_nodes_mask"]         # [T, N]
                for h in range(H):
                    attn_bias[h, tt_rows, :N] += (tmask.float() * type_token_link_bias_t2q[h])

        # --- SDPA ---
        attn_out = sdpa(Q, K, V, attn_mask=attn_bias)     # [H, q_len, D]
        Z = attn_out.transpose(0, 1).contiguous().view(q_len, self.channels)  # [q_len, C]

        if query_idx is None:
            X = X + self.drop(self.out(Z))
            # FFN su tutte le righe
            X = X + self.drop(self.ffn(self.ln2(X)))
        else:
            X_new = X
            X_new = X_new.clone()
            X_new[query_idx] = X_new[query_idx] + self.drop(self.out(Z))
            # FFN solo sulle righe query
            X_ln2_q = self.ln2(X_new[query_idx])
            X_new[query_idx] = X_new[query_idx] + self.drop(self.ffn(X_ln2_q))
            X = X_new

        # ----- Residuo 1 hop stile SAGE (con gating e, se serve, limitato alle query) -----
        N = cache["N"]
        Y_norm = self.ln2(X)     # layer norm prima del messaggio

        msg = torch.zeros(N, self.channels, device=X.device)
        deg = torch.zeros(N, 1, device=X.device)
        Yp = self.rel_proj(Y_norm[:N])  # proietto una sola volta

        if query_idx is None:
            for r, (s_idx, d_idx) in cache["adj_rel_pairs"].items():
                if s_idx.numel() == 0: 
                    continue
                proj = Yp[s_idx] * self.rel_gate[r]            # [E_r, C]
                msg.index_add_(0, d_idx, proj)
                deg.index_add_(0, d_idx, torch.ones(d_idx.size(0), 1, device=X.device))
        else:
            qmask = torch.zeros(N, dtype=torch.bool, device=X.device)
            qmask[query_idx] = True
            for r, (s_idx, d_idx) in cache["adj_rel_pairs"].items():
                if s_idx.numel() == 0:
                    continue
                keep = qmask[d_idx]
                if keep.any():
                    s_sel = s_idx[keep]
                    d_sel = d_idx[keep]
                    proj = Yp[s_sel] * self.rel_gate[r]
                    msg.index_add_(0, d_sel, proj)
                    deg.index_add_(0, d_sel, torch.ones(d_sel.size(0), 1, device=X.device))

        deg = deg.clamp_min(1.0)
        mp_out = msg / deg

        delta = torch.zeros_like(X)
        if query_idx is None:
            delta[:N] = self.drop(torch.sigmoid(self.mp_alpha) * mp_out)
        else:
            delta[query_idx] = self.drop(torch.sigmoid(self.mp_alpha) * mp_out[query_idx])
        X = X + delta

        return X





class HeteroGraphormer(nn.Module):
    """
    This is the gnn module that applies multiple Graphormer blocks.
    """
    def __init__(self,
                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],
                 channels: int,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 time_buckets: int = 21,
                 dropout: float = 0.05):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.num_heads = num_heads

        
        self.bias = HeteroGraphormerStructuralBias(
            node_types=node_types,
            edge_types=edge_types,
            num_heads=num_heads,
            time_buckets=time_buckets,
        )
        R = self.bias.R  # numero di edge types

        self.layers = nn.ModuleList([
            GraphormerBlock(channels, num_heads, dropout, rel_count=R)  # <— passalo
            for _ in range(num_layers)
        ])


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

        attach_masks = None
        bias_pack = self.bias.build_bias_per_head(cache, num_tokens_total=X.size(0))

        for li, layer in enumerate(self.layers):
            qidx = cache["seed_indices"] if li == (self.num_layers - 1) else None
            X = layer(X, token_type, bias_pack, attach_type_tokens_masks=attach_masks, query_idx=qidx)


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
        norm: str,
        shallow_list: List[NodeType] = [],
        id_awareness: bool = False,
        predictor_n_layers: int = 1,
        num_heads: int = 8,
        time_buckets: int = 21,
        degree_buckets: int = 16,
    ):
        super().__init__()
        self.channels = channels

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
            time_buckets=time_buckets,
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
        x_dict = self.encoder(batch.tf_dict)

        seed_time = batch[entity_table].seed_time
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for nt, rel_t in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rel_t

        for nt, emb in self.embedding_dict.items():
            x_dict[nt] = x_dict[nt] + emb(batch[nt].n_id)

        x_dict = self.centrality(x_dict, batch.edge_index_dict)

        if self.id_awareness_emb is not None:
            x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        x_dict = self.gnn(
            x_dict=x_dict,
            edge_index_dict=batch.edge_index_dict,
            time_dict=batch.time_dict,
            entity_table=entity_table,
            seed_count=seed_time.size(0),
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(self, batch: HeteroData, entity_table: NodeType, dst_table: NodeType) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError("id_awareness must be set True to use forward_dst_readout")
        
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