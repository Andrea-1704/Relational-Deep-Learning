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
    def __init__(self, channels, edge_types, device, num_heads=4, dropout=0.1):
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

        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Dropout(dropout),
        )

        self.pre_ln_attn = nn.LayerNorm(channels)
        self.pre_ln_ffn  = nn.LayerNorm(channels)

        self.allpairs_K = 3  # raggio SPD (0..K, INF=K+1)
        self.spd_bias = nn.Embedding(self.allpairs_K + 2, self.num_heads)  # per-head bias SPD
        self.dst_chunk_size = 2048  #processa i dst a blocchi


        self.edge_type_bias = nn.ParameterDict({
            "__".join(edge_type): nn.Parameter(torch.randn(1))
            for edge_type in edge_types
        })

    def compute_spd_buckets_allpairs(self, edge_index, n_src, n_dst, K=None):
        """
        Ritorna una matrice [n_dst, n_src] di indici SPD bucketizzati in {0..K, K+1=INF}.
        Usa NetworkX per chiarezza. Per produzione: precompute o scrivere BFS torch-based.
        """
        if K is None:
            K = self.allpairs_K
        src, dst = edge_index
        G = nx.Graph()
        G.add_nodes_from(range(n_src + n_dst))
        # connetti src_i <-> dst_j (non diretto) con offset sui dst
        for s, d in zip(src.tolist(), dst.tolist()):
            G.add_edge(s, n_src + d)

        INF_BUCKET = K + 1
        spd = torch.full((n_dst, n_src), INF_BUCKET, dtype=torch.long, device=self.device)
        # BFS da ciascun dst
        for d in range(n_dst):
            # cutoff=K limita i cammini → costo ridotto
            lengths = nx.single_source_shortest_path_length(G, n_src + d, cutoff=K)
            for node, dist in lengths.items():
                if node < n_src:  # è un nodo src
                    spd[d, node] = min(dist, K)
        return spd  # [nD, nS] long


    # def _attention_block(self, x_dict, edge_index_dict):
    #     """
    #     Multi-Head Attention 'sparse su archi':
    #     - Proietta Q/K/V
    #     - Aggiunge bias (spaziale + tipo di relazione) ai logit
    #     - Softmax per-dst
    #     - Aggrega V pesati e applica W_O (out_lin) + dropout
    #     Ritorna un dict {node_type: [N_type, channels]} con l'output dell'attenzione.
    #     """
    #     H, D = self.num_heads, self.head_dim
    #     out_dict = {nt: torch.zeros_like(x, device=self.device) for nt, x in x_dict.items()}

    #     for edge_type, edge_index in edge_index_dict.items():
    #         src_type, _, dst_type = edge_type
    #         x_src, x_dst = x_dict[src_type], x_dict[dst_type]
    #         src, dst = edge_index  # [E], [E]

    #         # Q, K, V: [N, C] -> [N, H, D]
    #         Q = self.q_lin(x_dst).view(-1, H, D)
    #         K = self.k_lin(x_src).view(-1, H, D)
    #         V = self.v_lin(x_src).view(-1, H, D)

    #         # Logits: [E, H]
    #         attn_scores = (Q[dst] * K[src]).sum(dim=-1) / (D ** 0.5)

    #         # Bias spaziale (batch-local) -> [E] -> [E,1] -> broadcast su H
    #         print(f"computing the batch spatial bias")
    #         spatial_bias_tensor = self.compute_batch_spatial_bias(edge_index, x_dst.size(0))
    #         print(f"The result of the SB è {spatial_bias_tensor}")
    #         attn_scores = attn_scores + spatial_bias_tensor.unsqueeze(-1)

    #         # Bias per tipo di relazione (broadcast su H se scalare)
    #         bias_name = "__".join(edge_type)
    #         attn_scores = attn_scores + self.edge_type_bias[bias_name]

    #         # Softmax per-dst e dropout
    #         attn_weights = softmax(attn_scores, dst)        # [E, H]
    #         attn_weights = self.dropout(attn_weights)

    #         # Aggregazione: [E,H,D] -> flatten heads -> [E, C] e somma su dst
    #         out_e = (V[src] * attn_weights.unsqueeze(-1)).view(-1, self.channels)  # [E, C]
    #         out_dict[dst_type].index_add_(0, dst, out_e)

    #     # Proiezione W_O e dropout su ogni tipo di nodo
    #     for nt in out_dict:
    #         out_dict[nt] = self.dropout(self.out_lin(out_dict[nt]))  # [N_nt, C]

    #     return out_dict


    def forward(self, x_dict, edge_index_dict, batch_dict=None, vnode_idx_dict=None):
        """
        Pre-LN → MHA → Residual → Pre-LN → FFN → Residual
        Mantiene il tuo 'degree centrality' additivo tra MHA e residuo (come avevi),
        ma ora l'attenzione è seguita da W_O + dropout. LN finale (post) non serve più
        perché usiamo schema Pre-LN prima di MHA e prima di FFN.
        """
        # 1) Pre-LN prima del blocco di attenzione (per stabilità stile Graphormer/Pre-LN)
        x_norm = {t: self.pre_ln_attn(x) for t, x in x_dict.items()}

        # 2) MHA con bias strutturali (ritorna già proiettato con W_O)
        attn_out = self._attention_block(x_norm, edge_index_dict)  # {t: [N_t, C]}

        # 2b) (opzionale) Degree centrality additivo come nel tuo codice originale
        total_deg = self.compute_total_degrees(x_dict, edge_index_dict)
        for t in attn_out:
            deg_embed = total_deg[t].view(-1, 1).expand(-1, self.channels)
            attn_out[t] = attn_out[t] + deg_embed

        # 3) Residual dopo MHA
        x_res = {t: x_dict[t] + attn_out[t] for t in x_dict}

        # 4) Pre-LN prima della FFN, poi FFN + residual
        x_ffn_in = {t: self.pre_ln_ffn(x_res[t]) for t in x_res}
        ffn_out  = {t: self.ffn(x_ffn_in[t]) for t in x_ffn_in}
        x_out    = {t: x_res[t] + ffn_out[t] for t in x_res}

        return x_out

    def _attention_block(self, x_dict, edge_index_dict):
        """
        All-pairs attention per blocco (src_type, ->, dst_type):
        - genera TUTTE le coppie (dst, src) (o chunk di dst per memoria)
        - logits = <Q_dst, K_src>/sqrt(d) + bias_rel + bias_SPD
        - softmax per-dst su TUTTE le sorgenti
        - aggrega V_src pesati su ogni dst
        - out_lin + dropout
        """
        H, D, C = self.num_heads, self.head_dim, self.channels
        out_dict = {nt: torch.zeros_like(x, device=self.device) for nt, x in x_dict.items()}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            x_src, x_dst = x_dict[src_type], x_dict[dst_type]
            nS, nD = x_src.size(0), x_dst.size(0)

            # Proiezioni: [N, C] -> [N, H, D]
            K = self.k_lin(x_src).view(nS, H, D)
            V = self.v_lin(x_src).view(nS, H, D)
            Q = self.q_lin(x_dst).view(nD, H, D)

            # Bias relazione (broadcast su H se scalare)
            rel_bias = self.edge_type_bias["__".join(edge_type)]  # shape [1] (scalare)

            # SPD bucket per tutte le coppie (dst, src): [nD, nS]
            # spd_idx = self.compute_spd_buckets_allpairs(edge_index, nS, nD, K=self.allpairs_K)
            # # Embedding SPD per-head: [nD, nS, H]
            # spd_b = self.spd_bias(spd_idx).to(Q.dtype)

            # Processa i dst a blocchi per ridurre picchi di memoria
            chunk = self.dst_chunk_size if self.dst_chunk_size is not None else nD
            for d0 in range(0, nD, chunk):
                d1 = min(d0 + chunk, nD)
                Qc = Q[d0:d1]                         # [d, H, D]
                # logits densi all-pairs: [d, nS, H] = einsum_i,j,h (Q⋅K)
                logits = torch.einsum("i h d, j h d -> i j h", Qc, K) / (D ** 0.5)

                # Aggiungi bias relazione e SPD
                logits = logits + rel_bias.view(1, 1, 1)          # [d, nS, H]
                #logits = logits + spd_b[d0:d1]                    # [d, nS, H]

                # Flatten in lista di coppie (dst_all, src_all)
                d = d1 - d0
                # Ordine: per ogni dst (d0..d1), tutti i src (0..nS-1)
                dst_all = torch.arange(d0, d1, device=self.device).repeat_interleave(nS)  # [d*nS]
                src_all = torch.arange(nS, device=self.device).repeat(d)                  # [d*nS]

                logits_flat = logits.reshape(d * nS, H)     # [d*nS, H]

                # Softmax per-dst su tutte le sorgenti
                attn = softmax(logits_flat, dst_all)        # [d*nS, H]
                attn = self.dropout(attn)

                # Aggregazione: V[src_all]: [d*nS, H, D] → pesa con attn → [d*nS, C]
                V_pairs = V[src_all]                        # [d*nS, H, D]
                out_e = (V_pairs * attn.unsqueeze(-1)).reshape(-1, C)  # [d*nS, C]

                # Somma su ciascun dst (index_add_)
                out_dict[dst_type].index_add_(0, dst_all, out_e)

        # Proiezione W_O + dropout finale per ogni type
        for nt in out_dict:
            out_dict[nt] = self.dropout(self.out_lin(out_dict[nt]))  # [N_nt, C]

        return out_dict



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

    def compute_batch_spatial_bias(self, edge_index, num_nodes):
        # Costruisci grafo da edge_index del batch corrente
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        src, dst = edge_index
        for s, d in zip(src.tolist(), dst.tolist()):
            G.add_edge(s, d)

        spatial_bias = {}
        for node in G.nodes():
            lengths = nx.single_source_dijkstra_path_length(G, node)
            for target, dist in lengths.items():
                spatial_bias[(node, target)] = dist

        # Costruzione tensor di bias dallo spatial_bias
        bias_vals = [spatial_bias.get((d, s), -1.0) for s, d in zip(src.tolist(), dst.tolist())]
        return torch.tensor(bias_vals, dtype=torch.float32, device=self.device)




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

