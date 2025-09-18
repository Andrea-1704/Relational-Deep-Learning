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
                         rel2id: Dict[Tuple[str, str, str], int]) -> Tuple[List[Tuple[int, int, int]], int]:
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
    edges = []
    total_nodes = 0
    for nt, sl in slices.items():
        total_nodes = max(total_nodes, sl.stop)
    for et, eidx in edge_index_dict.items():
        src_nt, _, dst_nt = et
        sl_s, sl_d = slices[src_nt], slices[dst_nt]
        src, dst = eidx
        edges.extend([(int(sl_s.start + int(s)), int(sl_d.start + int(d)), rel2id[et]) for s, d in zip(src.tolist(), dst.tolist())])
    return edges, total_nodes


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
      - (Optional) First-edge-on-SP bias (light meta-path signal)
      - (Optional) TypeToken link bias (node <-> its TypeToken)
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
        

    # ---- Helper: Δt bucketization around 0 with symmetric log buckets ----
    @staticmethod
    def _bucketize_dt(dt: Tensor, num_buckets: int) -> Tensor:
        # I create symmetric log buckets: ... -365, -90, -30, -7, -1, 0, +1, +7, +30, +90, +365 ...
        # Implementation detail: I map dt to signed log-space and discretize to 'num_buckets' bins.
        eps = 1e-6
        signed_log = torch.sign(dt) * torch.log1p(torch.abs(dt) + eps)  # R -> R
        # I scale into [0, num_buckets-1], keeping 0 near dt=0
        min_v, max_v = -5.0, 5.0 
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

        # fuse node features (for shape info only), and I build per-type slices
        X_stub, token_type, slices = fuse_x_dict({k: v.detach() for k, v in x_dict.items()}, self.type2id)
        N = X_stub.size(0)

        # fuse edges to global indices
        edges, _ = fuse_edges_to_global(edge_index_dict, slices, self.rel2id)

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

       
        #collect adjacency pairs per relation type for scatter-add
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

        # build type-pair indices by broadcast
        # To avoid [N,N,2] materialization, I keep token_type and will gather on the fly
        # (layers can do outer gather using token_type[row] and token_type[col])

        # add TypeToken indices at the end if enabled
        type_token_indices = None
        

        cache = {
            "N": N,
            "token_type": token_type,     # [N]
            "slices": slices,
            "edges": edges,               # list of (s, d, r)
            "adj_rel_pairs": adj_rel_pairs,
            "time_vec": time_vec,         # [N]
            "seed_indices": seed_indices, # [S]
            "type_token_indices": type_token_indices,  # [T] or None
        }
        return cache

    # ---- Bias assembly (I add scalars per head to attention scores) ----
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

class GraphormerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, dropout: float = 0.1, rel_count: int = 0):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.rel_mlps = nn.ModuleDict({str(r): nn.Linear(channels, channels, bias=True)
                                       for r in range(rel_count)})
        self.mp_alpha = nn.Parameter(torch.tensor(0.0))  

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
        
        H, D = self.num_heads, self.head_dim
        N_tot = X.size(0)
        
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
        cache = bias_pack["cache"]       
        N = cache["N"]                   

        
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

        attn = torch.softmax(scores, dim=-1)  # [H, N_tot, N_tot]
        attn = self.drop(attn)
        Z = torch.einsum("hnm,hmd->hnd", attn, V)  # [H, N_tot, D]
        Z = Z.transpose(0, 1).contiguous().view(N_tot, self.channels)
        X = X + self.drop(self.out(Z))            # residual

        # FFN
        X = X + self.drop(self.ffn(self.ln2(X)))

        # ----- Residuo 1-hop stile SAGE 
        N = cache["N"]
        Y_norm = self.ln2(X)  
        if len(self.rel_mlps) == 0:
            R = bias_pack["adj_rel_bias"].size(0)
            for r in range(R):
                self.rel_mlps[str(r)] = nn.Linear(self.channels, self.channels, bias=True)

        msg = torch.zeros(N, self.channels, device=X.device)
        deg = torch.zeros(N, 1, device=X.device)
        for r, (s_idx, d_idx) in cache["adj_rel_pairs"].items():
            if s_idx.numel() == 0: continue
            proj = self.rel_mlps[str(r)](Y_norm[s_idx])        # [E_r, C]
            msg.index_add_(0, d_idx, proj)                     # somma per destinatario
            deg.index_add_(0, d_idx, torch.ones_like(d_idx, dtype=torch.float32).unsqueeze(1))
        deg.clamp_min_(1.0)
        mp_out = msg / deg
        X[:N] = X[:N] + self.drop(torch.sigmoid(self.mp_alpha) * mp_out)
        #


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
                 dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.num_heads = num_heads

        # self.bias = HeteroGraphormerStructuralBias(
        #     node_types=node_types,
        #     edge_types=edge_types,
        #     num_heads=num_heads,
        #     time_buckets=time_buckets,
        # )
        # self.layers = nn.ModuleList([GraphormerBlock(channels, num_heads, dropout) for _ in range(num_layers)])

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
        # if self.use_type_tokens:
        #     T = len(self.bias.node_types)
        #     #append T tokens; their "type id" is the actual type they represent
        #     type_ids = torch.arange(T, device=device, dtype=torch.long)
        #     TT = self.type_token_emb(type_ids)             # [T, C]
        #     X = torch.cat([X, TT], dim=0)                  # [N+T, C]
        #     token_type = torch.cat([token_type, type_ids], dim=0)  # [N+T]
        #     # I prepare masks for biasing
        #     # node -> its TypeToken column index
        #     type_token_col = cache["type_token_indices"]    # [T] = [N..N+T-1]
        #     node_to_type_token_col = type_token_col[token_type[:N]]  # [N]
        #     # TypeToken rows
        #     tt_rows = torch.arange(N, N + T, device=device, dtype=torch.long)
        #     # Per-type node masks [T, N] (bool)
        #     type_to_nodes_mask = torch.zeros(T, N, dtype=torch.bool, device=device)
        #     for nt, sl in slices.items():
        #         t_id = type2id[nt]
        #         type_to_nodes_mask[t_id, sl.start:sl.stop] = True
        #     attach_masks = {
        #         "node_to_type_token_col": node_to_type_token_col,
        #         "type_token_to_node_rows": tt_rows,
        #         "type_to_nodes_mask": type_to_nodes_mask,
        #         "type_token_type_id": type_ids,
        #     }

        bias_pack = self.bias.build_bias_per_head(cache, num_tokens_total=X.size(0))

        for layer in self.layers:
            X = layer(X, token_type, bias_pack, attach_type_tokens_masks=attach_masks)

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