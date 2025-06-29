import torch
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
from relbench.modeling.nn import HeteroEncoder


def binarize_targets(y: torch.Tensor, threshold: float = 10) -> torch.Tensor:
    return (y < threshold).long()

def get_candidate_relations(metadata, current_node_type: str) -> List[Tuple[str, str, str]]:
    return [rel for rel in metadata[1] if rel[0] == current_node_type]

def construct_bags(
    data,
    train_mask: torch.Tensor,
    y: torch.Tensor,
    rel: Tuple[str, str, str],
    node_type: str,
) -> Tuple[List[List[int]], List[float]]:
    """
    Costruisce bags: per ogni nodo target nel training set, crea un bag
    con i nodi raggiunti tramite la relazione 'rel'.
    """
    src, rel_name, dst = rel
    if (src, rel_name, dst) not in data.edge_index_dict:
        return [], []

    edge_index = data.edge_index_dict[(src, rel_name, dst)]
    bags = []
    labels = []

    for i in torch.where(train_mask)[0]:
        node_id = i.item()
        neighbors = edge_index[1][edge_index[0] == node_id]
        if len(neighbors) > 0:
            bags.append(neighbors.tolist())
            labels.append(y[node_id].item())

    return bags, labels


def evaluate_relation_surrogate(
    bags: List[List[int]],
    labels: List[float],
) -> float:
    """
    Surrogate scoring: usa la dimensione del bag come feature predittiva
    e valuta quanto bene correla con le label vere (MAE).
    """
    pred = torch.tensor([len(b) for b in bags], dtype=torch.float)
    true = torch.tensor(labels, dtype=torch.float)
    return F.l1_loss(pred, true).item()



# class ScoringFunctionReg(nn.Module):
#     def __init__(self, in_dim: int):
#         super().__init__()
#         self.theta = nn.Sequential(
#             nn.Linear(in_dim, in_dim),
#             nn.ReLU(),
#             nn.Linear(in_dim, 1)  # da embedding a score scalare
#         )

#     def forward(self, bags: List[torch.Tensor]) -> torch.Tensor:
#         """
#         bags: lista di tensori (uno per ogni bag) con shape [B_i, D] (embedding dei nodi nel bag)
#         Output: predizione scalare per ciascun bag (media pesata)
#         """
#         preds = []
#         for bag in bags:
#             if bag.size(0) == 0:
#                 preds.append(torch.tensor(0.0, device=bag.device))
#                 continue
#             scores = self.theta(bag).squeeze(-1)  # [B_i]
#             weights = torch.softmax(scores, dim=0)  # normalizza i pesi nel bag
#             weighted_avg = torch.sum(weights.unsqueeze(-1) * bag, dim=0)  # [D]
#             pred = weighted_avg.mean()  # riduci a scalare
#             preds.append(pred)
#         return torch.stack(preds)

#     def loss(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         return F.l1_loss(preds, labels)




class ScoringFunctionReg(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.theta = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1)  # da embedding a scalare
        )

    def forward(self, bags: List[torch.Tensor]) -> torch.Tensor:
        """
        Ogni bag è un tensore di shape [B_i, D].
        Ritorna un valore scalare predetto per ciascun bag.
        """
        preds = []
        for bag in bags:
            if bag.size(0) == 0:
                preds.append(torch.tensor(0.0, device=bag.device))
                continue
            scores = self.theta(bag).squeeze(-1)  # [B_i]
            weights = torch.softmax(scores, dim=0)  # [B_i]
            weighted_avg = torch.sum(weights.unsqueeze(-1) * bag, dim=0)  # [D]
            pred = weighted_avg.mean()  # scalare finale
            preds.append(pred)
        return torch.stack(preds)

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(preds, targets)




def evaluate_relation_learned(
    bags: List[List[int]],
    labels: List[float],
    node_embeddings: torch.Tensor,
    epochs: int = 10,
    lr: float = 1e-2,
) -> float:
    """
    Allena ScoringFunctionReg sulle embedding dei nodi nel bag.
    Ritorna la MAE finale.
    """
    device = node_embeddings.device
    in_dim = node_embeddings.size(-1)

    model = ScoringFunctionReg(in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    bag_embeddings = [
        node_embeddings[torch.tensor(bag, device=device)] for bag in bags
    ]
    target_tensor = torch.tensor(labels, device=device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(bag_embeddings)
        loss = model.loss(preds, target_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(bag_embeddings)
        final_mae = model.loss(preds, target_tensor).item()

    return final_mae




def greedy_metapath_search_with_bags_learned(
    data,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    node_type: str,
    col_stats_dict: Dict[str, Dict[str, Dict]],  # per HeteroEncoder
    L_max: int = 3,
    max_rels: int = 10,
) -> List[List[Tuple[str, str, str]]]:
    """
    Costruisce meta-path greedy usando surrogate scoring appreso (MAE).
    """
    device = y.device
    metapaths = []
    current_paths = [[]]

    for level in range(L_max):
        new_paths = []

        for path in current_paths:
            last_ntype = node_type if not path else path[-1][2]

            # Encoder per ottenere tutte le embedding (una volta per step)
            with torch.no_grad():
              encoder = HeteroEncoder(
                  channels=64,
                  node_to_col_names_dict={
                      ntype: data[ntype].tf.col_names_dict
                      for ntype in data.node_types
                  },
                  node_to_col_stats=col_stats_dict,
              ).to(device)

              # forza anche i buffer
              for module in encoder.modules():
                  for name, buf in module._buffers.items():
                      if buf is not None:
                          module._buffers[name] = buf.to(device)

              tf_dict = {
                  ntype: data[ntype].tf.to(device) for ntype in data.node_types if 'tf' in data[ntype]
              }

              node_embeddings_dict = encoder(tf_dict)



            candidate_rels = [
                (src, rel, dst)
                for (src, rel, dst) in data.edge_index_dict.keys()
                if src == last_ntype
            ][:max_rels]

            best_rel = None
            best_score = float("inf")

            for rel in candidate_rels:
                src, _, dst = rel
                node_embeddings = node_embeddings_dict.get(dst)
                if node_embeddings is None:
                    continue

                bags, labels = construct_bags(data, train_mask, y, rel, node_type)
                if len(bags) < 5:
                    continue

                score = evaluate_relation_learned(bags, labels, node_embeddings)
                if score < best_score:
                    best_score = score
                    best_rel = rel

            if best_rel:
                new_paths.append(path + [best_rel])

        current_paths = new_paths
        metapaths.extend(current_paths)

    return metapaths
