"""
Using paper's metapaths concatenating approach.

This is my latest version so far.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, SAGEConv
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
from torch_geometric.data import HeteroData
from torch_frame.data.stats import StatType
from typing import Any, Dict, List, Tuple
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

class GraphX(nn.Module):
    """
    GraphL: Graphormer-per-hop attentive layer (src -> dst) senza edge features.
    - Attenzione calcolata SOLO sugli archi di edge_index (nessuna matrice NxN).
    - Centrality encoding: somma z_in(deg_in) + z_out(deg_out) ai nodi (src e dst).
    - Bias temporale nei logits: -lambda * Δt  (+ opzionale bucket embedding).
    - Residuo + gate verso x_dst_orig, Pre-LN e FFN.

    Args:
        d_model:       dimensione canali (hidden size).
        num_heads:     teste di attenzione.
        max_deg_bucket: numero massimo di bucket di grado (clip/bucket log).
        time_bias:     'linear' | 'bucket' | 'both' | 'none'
        time_scale:    scala per Δt prima del bias (es. 1.0 se Δt già normalizzato).
        time_max_bucket: numero di bucket log per Δt (se 'bucket' o 'both').
        time_bucket_base: base logaritmica per i bucket di Δt.
        dropout:       dropout su attn e FFN.
    """
    def __init__(self,
                 d_model: int,
                 num_heads: int = 8,
                 max_deg_bucket: int = 512,
                 time_bias: str = 'linear',
                 time_scale: float = 1.0,
                 time_max_bucket: int = 256,
                 time_bucket_base: float = 1.6,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model deve essere multiplo di num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_h = d_model // num_heads
        self.time_bias = time_bias
        self.time_scale = float(time_scale)
        self.time_bucket_base = float(time_bucket_base)

        # Proiezioni Q/K/V e output (condivise sulle teste)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self.mix_attn = nn.Parameter(torch.tensor(0.0))  # scalare learnable in [0,1] via sigmoid

        self.log_tau = nn.Parameter(torch.zeros(self.num_heads))  # τ iniziale = 1.0

        # Centrality encoding (due tabelle: in-degree e out-degree)
        self.deg_in_emb  = nn.Embedding(max_deg_bucket + 1, d_model)
        self.deg_out_emb = nn.Embedding(max_deg_bucket + 1, d_model)
        self.alpha_c = nn.Parameter(torch.tensor(1.0))  # scala learnable

        # Bias temporale
        if self.time_bias in ('linear', 'both'):
            # coefficiente lambda >= 0 per -lambda * Δt
            self.lambda_dt = nn.Parameter(torch.tensor(0.1))
        if self.time_bias in ('bucket', 'both'):
            # embedding di bucket di Δt -> per-head scalar
            self.time_bucket_bias = nn.Embedding(time_max_bucket + 1, num_heads)

        self.alpha_c = nn.Parameter(torch.tensor(0.5))   # invece di 1.0
        self.centrality_drop = nn.Dropout(p=0.1)
        self.mix_attn = nn.Parameter(torch.tensor(0.5))   # 0.5 = metà attn, metà mean


        # LayerNorm e FFN (pre-LN)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

        # Residuo/gate in stile XMetaPathGNNLayer
        self.w_l = nn.Linear(d_model, d_model)
        self.w_0 = nn.Linear(d_model, d_model)
        self.w_1 = nn.Linear(d_model, d_model)
        self.gate = nn.Parameter(torch.tensor(0.5))  # inizializzato a 0.5

        self.attn_drop = nn.Dropout(dropout)

    # ---------- utility ----------
    @staticmethod
    def _bucket_log(x: torch.Tensor, max_bucket: int, base: float) -> torch.Tensor:
        # x: non-negative tensor
        xb = torch.floor(torch.log1p(x) / math.log(base)).long()
        return torch.clamp(xb, 0, max_bucket)

    def _deg_buckets(self, deg: torch.Tensor, max_bucket: int) -> torch.Tensor:
        # deg: LongTensor >=0
        return self._bucket_log(deg.float(), max_bucket=max(self.deg_in_emb.num_embeddings - 1, 1), base=1.6)

    # ---------- forward ----------
    def forward(self,
                h_src: torch.Tensor,          # [Ns, D]
                h_dst: torch.Tensor,          # [Nd, D]
                edge_index: torch.Tensor,     # [2, E] (src_idx, dst_idx) local indices
                x_dst_orig: torch.Tensor,     # [Nd, D] (per il gate)
                t_src: torch.Tensor = None,   # [Ns]   timestamp per src (opzionale se passi edge_dt)
                t_dst: torch.Tensor = None,   # [Nd]   timestamp per dst
                edge_dt: torch.Tensor = None  # [E]    Δt per-edge, opzionale
                ) -> torch.Tensor:
        """
        Output: [Nd, D] — nuove embedding per i dst.
        Nota: si assume (per batch) t_dst[v] >= t_src[u] per ogni arco (u->v).
        """
        device = h_src.device
        Ns, Nd = h_src.size(0), h_dst.size(0)
        D = self.d_model
        u = edge_index[0]  # [E]
        v = edge_index[1]  # [E]
        E = u.numel()

        # --------- Centrality encoding (per relazione) ---------
        # out-degree per src, in-degree per dst (gli altri due sono ~0 in bipartito)
        deg_out_src = torch.bincount(u, minlength=Ns).to(device)
        deg_in_dst  = torch.bincount(v, minlength=Nd).to(device)
        # Se vuoi mettere anche deg_in_src/deg_out_dst (in questo hop spesso 0):
        deg_in_src  = torch.zeros(Ns, device=device, dtype=torch.long)
        deg_out_dst = torch.zeros(Nd, device=device, dtype=torch.long)

        # bucketizzazione (log) e lookup
        b_in_src   = self._bucket_log(deg_in_src.float(),  self.deg_in_emb.num_embeddings - 1,  base=1.6)
        b_out_src  = self._bucket_log(deg_out_src.float(), self.deg_out_emb.num_embeddings - 1, base=1.6)
        b_in_dst   = self._bucket_log(deg_in_dst.float(),  self.deg_in_emb.num_embeddings - 1,  base=1.6)
        b_out_dst  = self._bucket_log(deg_out_dst.float(), self.deg_out_emb.num_embeddings - 1, base=1.6)

        cent_src = self.deg_in_emb(b_in_src) + self.deg_out_emb(b_out_src)
        cent_dst = self.deg_in_emb(b_in_dst) + self.deg_out_emb(b_out_dst)
        h_src = h_src + self.alpha_c * self.centrality_drop(cent_src)
        h_dst = h_dst + self.alpha_c * self.centrality_drop(cent_dst)

        # h_src = h_src + self.alpha_c * (self.deg_in_emb(b_in_src) + self.deg_out_emb(b_out_src))
        # h_dst = h_dst + self.alpha_c * (self.deg_in_emb(b_in_dst) + self.deg_out_emb(b_out_dst))

        # --------- Pre-LN ---------
        Hq = self.ln_q(h_dst)  # queries da dst
        Hk = self.ln_kv(h_src) # keys/values da src

        # --------- Q/K/V e reshape per teste ---------
        Q = self.W_Q(Hq).view(Nd, self.num_heads, self.d_h)     # [Nd, H, d_h]
        K = self.W_K(Hk).view(Ns, self.num_heads, self.d_h)     # [Ns, H, d_h]
        V = self.W_V(Hk).view(Ns, self.num_heads, self.d_h)     # [Ns, H, d_h]

        # --------- Logits solo sugli archi ---------
        # per-edge dot(Q[v], K[u]) / sqrt(d_h)
        logits = (Q[v] * K[u]).sum(dim=-1) / math.sqrt(self.d_h)  # [E, H]

        # --------- Bias temporale sui logits ---------
        if self.time_bias != 'none':
            if edge_dt is None:
                assert (t_src is not None) and (t_dst is not None), \
                    "Passa edge_dt oppure (t_src, t_dst) per costruire Δt."
                dt = (t_dst[v] - t_src[u]).to(h_src.dtype)  # [E]
            else:
                dt = edge_dt.to(h_src.dtype)

            # normalizza/scala Δt se necessario
            if self.time_scale != 1.0:
                dt = dt / self.time_scale
            dt = dt.clamp_min(0)

            # Componente lineare: -lambda * Δt
            if self.time_bias in ('linear', 'both'):
                lam = torch.clamp(self.lambda_dt, min=0.0)
                logits = logits + (-lam * dt).unsqueeze(-1)  # [E, H]

            # Componente a bucket (lookup per-head)
            if self.time_bias in ('bucket', 'both'):
                tb = self._bucket_log(dt, max_bucket=self.time_bucket_bias.num_embeddings - 1,
                                      base=self.time_bucket_base)  # [E]
                logits = logits + self.time_bucket_bias(tb)         # [E, H]

        # ... dopo aver sommato i bias temporali nei logits
        tau = torch.exp(self.log_tau).clamp(min=0.5, max=2.0)   # [H]
        logits = logits / tau                                   # [E, H], broadcast

        # --------- Softmax segmentata su ogni dst (riga v) ---------
        alpha = softmax(logits, v)          # [E, H]
        alpha = self.attn_drop(alpha)

        # --------- Aggregazione pesata (dst raccoglie da src vicini) ---------
        # --------- Aggregazione pesata (dst raccoglie da src vicini) ---------
        H_attn = torch.zeros(Nd, self.num_heads, self.d_h, device=device)
        # somma per ogni v e per head: [Nd, H, d_h]
        H_attn.index_add_(0, v, alpha.unsqueeze(-1) * V[u])

        # Mean aggregator (safety-net robusta)
        deg_v = torch.bincount(v, minlength=Nd).clamp(min=1).to(H_attn.device).view(-1, 1, 1)  # [Nd,1,1]
        H_sum = torch.zeros_like(H_attn)
        H_sum.index_add_(0, v, V[u])                        # somma semplice dei V dei vicini
        H_mean = H_sum / deg_v                              # [Nd, H, d_h]

        # Mix learnable tra attention e mean
        m = torch.sigmoid(self.mix_attn)                    # scalare in (0,1)
        H = m * H_attn + (1.0 - m) * H_mean                 # [Nd, H, d_h]

        # proiezione di uscita
        H = H.reshape(Nd, D)                                # D = self.d_model (già definito sopra)
        H = self.W_O(H)

        # --------- Residuo + gate + FFN (pre-LN) ---------
        g = torch.sigmoid(self.gate)
        out = self.w_l(H) + (1. - g) * self.w_0(h_dst) + g * self.w_1(x_dst_orig)
        out = out + self.ffn(self.ffn_ln(out))
        return out

        
        
        
        
        
        
        # H_attn = torch.zeros(Nd, self.num_heads, self.d_h, device=device)
        # # index_add_: somma per ogni v (per-head)
        # # Mean aggregator (per dst): somma V[u] e dividi per deg(v)
        # deg_v = torch.bincount(v, minlength=Nd).clamp(min=1).unsqueeze(-1).unsqueeze(-1).to(H_attn.device)  # [Nd,1,1]
        # H_sum = torch.zeros(Nd, self.num_heads, self.d_h, device=H_attn.device)
        # H_sum.index_add_(0, v, V[u])           # somma semplice
        # H_mean = H_sum / deg_v                 # [Nd,H,d_h]

        # m = torch.sigmoid(self.mix_attn)       # [1] scalare learnable
        # H_attn = m * H_attn + (1 - m) * H_mean
        # H_attn = H_attn.reshape(Nd, D)
        # H_attn = self.W_O(H_attn)

        # # H_attn.index_add_(0, v, alpha.unsqueeze(-1) * V[u])   # [Nd, H, d_h]
        # # H_attn = H_attn.reshape(Nd, D)
        # # H_attn = self.W_O(H_attn)

        # # --------- Residuo + gate + FFN (pre-LN) ---------
        # g = torch.sigmoid(self.gate)
        # out = self.w_l(H_attn) + (1. - g) * self.w_0(h_dst) + g * self.w_1(x_dst_orig)
        # out = out + self.ffn(self.ffn_ln(out))   # FFN con residuo

        # return out



class MetaPathGNN(nn.Module):
    """
    This is the network that express the GNN operations over a meta path.
    We create a GNN layer for each relation in the metapath. Then, we 
    propagate over the metapath using convolutions.
    Finally we apply a final projection to the initial node embeddings.

    So, we generate embeddings considering the metapath "metapath".
    A metapath is passed, and is a list of tuple (src, rel, dst).

    Here, we use MetaPathGNNLayer as GNN layer, which follows the paper 
    implementation.

    We also fixed a bug that was present in the previous versions: 

    self.convs[conv_idx](
                x=x_dst,
                h=x_dst,
                edge_index=edge_index_remapped
    )

    Here, we were passing to h the current state x, but it is redudant
    since we were passing it already. This was a minor mistake I forgot.


    We also aply a temporal decading weighting for the messages.
    """
    def __init__(self, metapath, hidden_channels, out_channels,
                 dropout_p: float = 0.1,
                 use_time_decay: bool = True,
                 init_lambda: float = 0.1,
                 time_scale: float = 1.0):
        super().__init__()
        self.metapath = metapath
        self.convs = nn.ModuleList()
        for i in range(len(metapath)):
            #self.convs.append(MetaPathGNNLayer(hidden_channels, hidden_channels, relation_index=i))
            #self.convs.append(MetaPathGNNLayer(hidden_channels))
            self.convs.append(
                GraphX(d_model=hidden_channels, num_heads=16,
                    time_bias='linear', time_scale=30.0,
                    time_max_bucket=256, time_bucket_base=1.6,
                    dropout=0.1)
            )


        
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(len(metapath))])
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout_p) for _ in range(len(metapath))])
        
        #UPDATE:
        self.use_time_decay = use_time_decay
        self.time_scale = float(time_scale)
        self.raw_lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(init_lambda))) for _ in range(len(metapath))
        ])

        self.out_proj = nn.Linear(hidden_channels, out_channels)


    def forward(self, x_dict, edge_index_dict,
                node_time_dict: dict = None):
        #edge_type_dict is the list of edge types
        #edge_index_dict contains for each edge_type the edges

        #update, instead of this:
        #h_dict = x_dict.copy()
        #we store x0_dict for x and h_dict that will be updated path by path:
        #x0_dict = {k: v.detach() for k, v in x_dict.items()}   # freezed original features
        #x0_dict = {k: v for k, v in x_dict.items()} 
        x0_dict = {k: v.detach().clone() for k,v in x_dict.items()}
        h_dict  = {k: v.clone()  for k, v in x_dict.items()}   # current state: to update

        def pos_lambda(raw):  # λ > 0
            return F.softplus(raw) + 1e-8

        for i, (src, rel, dst) in enumerate(self.metapath): #already reversed: follow metapath starting from last path!
            conv_idx = i
            edge_index = edge_index_dict[(src, rel, dst)]

            #Store the list of the nodes that are used in the 
            #relation "(src, rel, dst)":
            #this was not mentioned in original paper
            src_nodes = edge_index[0].unique()
            dst_nodes = edge_index[1].unique()
            """
            Example
            src_nodes = [3, 4, 5]
            dst_nodes = [6, 2, 3]
            """

            #To solve the problem mentioend at the beginning of this file, we use a global->
            #to local mapping:
            #src_map = {int(i.item()): i for i, n in enumerate(src_nodes)}
            src_map = {int(n.item()): i for i, n in enumerate(src_nodes)}
            dst_map = {int(n.item()): i for i, n in enumerate(dst_nodes)}
            """
            Example
            if:
            src_nodes = [3, 4, 5]
            dst_nodes = [6, 2, 3]

            then:
            src_map = {3: 0, 4: 1, 5: 2}
            dst_map = {2: 0, 3: 1, 6: 2}
            """

            #Filter: consider only the nodes in the relation
            x_src = h_dict[src][src_nodes]
            x_dst = h_dict[dst][dst_nodes]
            """
            Example
            x_src = [emb(3), emb(4), emb(5)]
            x_dst = [emb(6), emb(2), emb(3)]
            """

            edge_index_remapped = torch.stack([
                torch.tensor([src_map[int(x)] for x in edge_index[0].tolist()], device=edge_index.device, dtype=torch.long),
                torch.tensor([dst_map[int(x)] for x in edge_index[1].tolist()], device=edge_index.device, dtype=torch.long)
            ])
            """
            Example
            if:
            src_map = {3: 0, 4: 1, 5: 2}
            dst_map = {2: 0, 3: 1, 6: 2}

            then:
            edge_index_remapped = tensor([[0, 1, 2],
                                         [0, 1, 2]])
            """

            #UPDATE: take the original x and update h representation:
            x_dst_orig = x0_dict[dst][dst_nodes]   # ORIGINAL
            h_dst_curr = h_dict[dst][dst_nodes]    # CURRENT


            #Δt for edge and weight: exp(-λ Δt)
            edge_weight = None
            # Δt per-edge (senza esponenziale): GraphX lo usa come bias nei logits
            h_src_curr = h_dict[src][src_nodes]
            if self.use_time_decay and (node_time_dict is not None) and (src in node_time_dict) and (dst in node_time_dict):
                t_src_all = node_time_dict[src].float()
                t_dst_all = node_time_dict[dst].float()
                # stessi indici di edge_index (prima del remap)
                t_src_e = t_src_all[edge_index[0]]  # [E_rel]
                t_dst_e = t_dst_all[edge_index[1]]  # [E_rel]
                edge_dt = (t_dst_e - t_src_e).clamp(min=0.0) / float(self.time_scale)
            else:
                # nessun timestamp: bias neutro
                edge_dt = torch.zeros(edge_index.size(1), device=edge_index.device, dtype=torch.float32)

            # hop attenzionale (una sola chiamata!)
            h_dst = self.convs[conv_idx](
                h_src=h_src_curr,
                h_dst=h_dst_curr,
                edge_index=edge_index_remapped,
                x_dst_orig=x_dst_orig,
                edge_dt=edge_dt
            )


            # h_dst = F.relu(h_dst)
            # h_dst = self.norms[conv_idx](h_dst)
            # h_dst = self.dropouts[conv_idx](h_dst)
            h_dst = self.dropouts[conv_idx](h_dst)

            h_dict[dst].index_copy_(0, dst_nodes, h_dst)

           
        target_type = self.metapath[-1][2]      #last dst (== 'drivers')
        return self.out_proj(h_dict[target_type])



class RegressorGating(nn.Module):
    def __init__(self, dim, num_paths, ctx_dim=None, p_drop=0.1):
        super().__init__()
        self.ctx_dim = ctx_dim or dim
        # proiettiamo ogni embedding di metapath a un logit scalare, condizionato dal contesto
        self.ctx_proj = nn.Linear(self.ctx_dim, dim)
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(dim, 1)
        )
        self.out = nn.Linear(dim, 1)

    def forward(self, metapath_embeddings: torch.Tensor, ctx: torch.Tensor):
        # metapath_embeddings: [N, M, D]  ; ctx: [N, C]
        N, M, D = metapath_embeddings.size()
        ctx_e = self.ctx_proj(ctx).unsqueeze(1)            # [N,1,D]
        feats = torch.tanh(metapath_embeddings + ctx_e)    # conditioning
        logits = self.scorer(feats).squeeze(-1)            # [N, M]
        w = torch.softmax(logits, dim=1)                   # [N, M]
        pooled = (w.unsqueeze(-1) * metapath_embeddings).sum(dim=1)  # [N, D]
        return self.out(pooled).squeeze(-1), w



class MetaPathSelfAttention(nn.Module):
    """
    This module applies Transformer-based self-attention between the different metapaths.
    It uses a TransformerEncoder. This module apply self attention between the different
    metapaths. It is mostly used as a source of explainability, in orfer to assess the 
    relevance contribution of every metapath to the final result.
    It was not present in the original paper.
    """
    def __init__(self, dim, num_heads=4, out_dim=1, num_layers=4):
        super().__init__()
        self.out_dim = out_dim
        self.attn_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )

        #UPDATE:
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, out_dim)
        )

    def forward(self, metapath_embeddings: torch.Tensor, return_attention: bool=False):  # [N, M, D]
        """
        metapath_embeddings: [N, M, D]
        return_attention: if True, returns intermediate attention embeddings as well
        """
        assert not torch.isnan(metapath_embeddings).any(), "NaN detected"
        assert not torch.isinf(metapath_embeddings).any(), "Inf detected"

        #UPDATE:
        ctx = self.attn_encoder(metapath_embeddings)
        _, A = self.self_attn(ctx, ctx, ctx, need_weights=True, average_attn_weights=True)
        w = A.mean(dim=1)  
        w = torch.softmax(w, dim=1) 
        gated = ctx * w.unsqueeze(-1)
        pooled = gated.sum(dim=1) 
        out = self.output_proj(pooled).squeeze(-1) 
        if return_attention:
            return out, w
        return out





class XMetaPath2(nn.Module):
    def __init__(self,
                 data: HeteroData,
                 col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
                 metapaths: List[List[int]],  # rel_indices
                 #metapath_counts: Dict[Tuple, int], #statistics of each metapath
                 hidden_channels: int = 64,
                 out_channels: int = 64,
                 num_heads: int = 8,
                 final_out_channels: int = 1,
                 num_layers: int = 4,
                 dropout_p: float = 0.1,
                 time_decay: bool = False,
                 init_lambda: float = 0.1,
                 time_scale: float = 1.0):
        super().__init__()

        self.metapath_models = nn.ModuleList([
            MetaPathGNN(mp, hidden_channels, out_channels,
                        dropout_p=dropout_p,
                        use_time_decay=time_decay,
                        init_lambda=init_lambda,
                        time_scale=time_scale)
            for mp in metapaths
        ]) # we construct a specific MetaPathGNN for each metapath

        #self.regressor = MetaPathSelfAttention(out_channels, num_heads=num_heads, out_dim=final_out_channels, num_layers=num_layers)

        self.regressor = RegressorGating(dim=out_channels, num_paths=len(metapaths), ctx_dim=hidden_channels)

        self.encoder = HeteroEncoder(
            channels=hidden_channels,
             #nt: data.tf_dict[nt].col_names_dict
            node_to_col_names_dict={
                node_type: data.tf_dict[node_type].col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict
        )

        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=hidden_channels,
        )

    def forward(self, batch: HeteroData, entity_table=None):

        seed_time = batch[entity_table].seed_time
        
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time
        
        
        embeddings = [#x_dict, edge_index_dict
            model(
                x_dict, 
                batch.edge_index_dict,
                node_time_dict=batch.time_dict
            )
            for model in self.metapath_models 
        ] #create a list of the embeddings, one for each metapath
        concat = torch.stack(embeddings, dim=1) #concatenate the embeddings 
        #weighted = concat * self.metapath_weights_tensor.view(1, -1, 1) #to consider to add statisitcs
        
        return self.regressor(concat) #finally apply regression; just put weighted instead of concat if statistics
     