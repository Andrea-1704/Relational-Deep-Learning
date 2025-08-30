import torch
from torch_geometric.nn import HeteroConv, GATConv
from typing import Any, Dict, List
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
import torch.nn as nn
import torch.nn.functional as F
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder

class HeteroGAT(torch.nn.Module):
    def __init__(self, node_types, edge_types, channels, heads=8, num_layers=2, aggr="sum"):
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.convs = torch.nn.ModuleList()
        self.edge_types = edge_types
        #per ogni layer andiamoa a creare un HeteroConv che contiene 
        #tanti GATConv (che sono già implementati) quanti sono gli 
        #edge type.
        for i in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: GATConv(
                        (-1, -1), channels, heads=heads, concat=False, add_self_loops=False, dropout=0.2
                    )
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)

        self.root_lin = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                nt: nn.Linear(channels, channels, bias=False) for nt in self.node_types
            }) for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.ModuleDict({ nt: nn.LayerNorm(channels) for nt in self.node_types })
            for _ in range(num_layers)
        ])

        self.dropout = 0.1
        self.act = F.elu

    def forward(self, x_dict, edge_index_dict, *args, **kwargs):
        for layer, conv in enumerate(self.convs):
            x_in = x_dict
            x_out = conv(x_dict, edge_index_dict)

            new_dict = {}
            for nt, x in x_out.items():
                x = x + self.root_lin[layer][nt](x_in[nt])   # self-feature
                x = self.norms[layer][nt](x)                # norm
                x = self.act(x)                              # attivazione
                x = F.dropout(x, p=self.dropout, training=self.training)
                # residual leggero se shape combacia
                if x_in[nt].shape == x.shape:
                    x = x + x_in[nt]
                new_dict[nt] = x
            x_dict = new_dict
        return x_dict


    def reset_parameters(self):
        for conv in self.convs:
            for edge_type in self.edge_types:
                conv.convs[edge_type].reset_parameters()
        # === NEW: init dei pesi root ===
        for md in self.root_lin:
            for lin in md.values():
                torch.nn.init.xavier_uniform_(lin.weight)




'''
# It's pretty simple: for each layer of the network we are creating an HeteroConv layer, which can be seen as a wrapper of GNN layers. Then, for each edge_type in that layer we are creating a GAT layer.

# The reason why we need a different GAT layer for each different edge type of a given layer is that each layer requires to be considered separately for it's semantic difference with all the others. 

# So each of the edge type requires its own attention scores.

# NB:

# When we do something like:
# ~~~python
# edge_type: GATConv(
#     (-1, -1), channels, heads=heads, concat=False, add_self_loops=False
# )
# ~~~

# The "(-1, -1)" refers to the dimension of the input channels. In this case we are using 'lazy initialization': PyTorch Geometric will automatically infer the in_channels during the first forward() call by reading the actual dimensions of the source and destination node embeddings.

# This is especially useful in heterogeneous graphs, where dimensions may vary across different types (or be decided dynamically).
'''





class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData, #notice that "data2 is the graph we created with function make_pkey_fkey_graph
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        norm: str,
        aggr: str = "sum",
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
        self.gnn = HeteroGAT(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            heads=8,
            num_layers=num_layers,
            aggr=aggr,
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

