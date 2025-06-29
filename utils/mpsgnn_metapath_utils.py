import torch
from torch_geometric.data import HeteroData
from typing import List, Tuple
import torch.nn as nn
import torch.nn.functional as F


def binarize_targets(y: torch.Tensor, threshold: float = 10) -> torch.Tensor:
    return (y < threshold).long()


def get_candidate_relations(metadata, current_node_type: str) -> List[Tuple[str, str, str]]:
    return [rel for rel in metadata[1] if rel[0] == current_node_type]


def surrogate_classification_loss(data: HeteroData,
                                   metapath: List[Tuple[str, str, str]],
                                   y_bin: torch.Tensor,
                                   train_mask: torch.Tensor,
                                   node_type: str) -> float:
    y_vals = y_bin[train_mask]
    pos_mean = y_vals.float().mean()
    neg_mean = 1.0 - pos_mean
    return (1.0 - (pos_mean - neg_mean).abs()).item()


def greedy_metapath_search(data: HeteroData,
                           y_bin: torch.Tensor,
                           train_mask: torch.Tensor,
                           node_type: str,
                           L_max: int = 2,
                           eta: float = 0.4) -> List[List[Tuple[str, str, str]]]:
    metadata = data.metadata()
    current_metapath = []
    all_metapaths = []
    current_node_type = node_type

    for _ in range(L_max):
        candidate_rels = get_candidate_relations(metadata, current_node_type)
        #this one is  working and provides all the paths in which 
        #"node type" appers
        best_rel = None
        best_loss = float('inf')

        for rel in candidate_rels:
            new_path = current_metapath + [rel]
            loss = surrogate_classification_loss(data, new_path, y_bin, train_mask, node_type)
            #print(f"utils la loss è {loss}")
            if loss < best_loss:   
            #i believe there is a mistake in the loss function because it always returns 
            #the same results (is either in the function, or in the parameters, or both)
                best_loss = loss
                best_rel = rel

        if best_rel is None or best_loss > eta:
            break

        current_metapath.append(best_rel)
        current_node_type = best_rel[2]

    if current_metapath:
        all_metapaths.append(current_metapath)

    return all_metapaths


  #new version:

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



def greedy_metapath_search_with_bags(
    data,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    node_type: str,
    L_max: int = 3,
    max_rels: int = 10,
) -> List[List[Tuple[str, str, str]]]:
    """
    Costruisce meta-path greedy selezionando le relazioni che minimizzano MAE
    tra la dimensione dei bags e le label.
    """
    device = y.device
    metapaths = []

    current_paths = [[]]
    for level in range(L_max):
        new_paths = []

        for path in current_paths:
            # Nodo da cui partire (iniziale o ultimo step del path)
            last_ntype = node_type if not path else path[-1][2]

            # Relazioni candidate uscenti da last_ntype
            candidate_rels = [
                (src, rel, dst)
                for (src, rel, dst) in data.edge_index_dict.keys()
                if src == last_ntype
            ][:max_rels]

            best_rel = None
            best_score = float('inf')

            for rel in candidate_rels:
                bags, labels = construct_bags(data, train_mask, y, rel, node_type)
                if len(bags) < 5:
                    continue
                score = evaluate_relation_surrogate(bags, labels)
                if score < best_score:
                    best_score = score
                    best_rel = rel

            if best_rel:
                new_paths.append(path + [best_rel])

        current_paths = new_paths
        metapaths.extend(current_paths)

    return metapaths




class ScoringFunctionReg(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.theta = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1)  # da embedding a score scalare
        )

    def forward(self, bags: List[torch.Tensor]) -> torch.Tensor:
        """
        bags: lista di tensori (uno per ogni bag) con shape [B_i, D] (embedding dei nodi nel bag)
        Output: predizione scalare per ciascun bag (media pesata)
        """
        preds = []
        for bag in bags:
            if bag.size(0) == 0:
                preds.append(torch.tensor(0.0, device=bag.device))
                continue
            scores = self.theta(bag).squeeze(-1)  # [B_i]
            weights = torch.softmax(scores, dim=0)  # normalizza i pesi nel bag
            weighted_avg = torch.sum(weights.unsqueeze(-1) * bag, dim=0)  # [D]
            pred = weighted_avg.mean()  # riduci a scalare
            preds.append(pred)
        return torch.stack(preds)

    def loss(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(preds, labels)

def evaluate_relation_learned(
    bags: List[List[int]],
    labels: List[float],
    node_embeddings: torch.Tensor,
    epochs: int = 10,
    lr: float = 1e-2,
) -> float:
    """
    Addestra una ScoringFunctionReg su bags + labels e ritorna il MAE.
    """
    device = node_embeddings.device
    model = ScoringFunctionReg(node_embeddings.size(-1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepara i dati: converti ogni bag in embedding
    bag_embeddings = [node_embeddings[torch.tensor(bag, device=device)] for bag in bags]
    label_tensor = torch.tensor(labels, device=device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(bag_embeddings)
        loss = model.loss(preds, label_tensor)
        loss.backward()
        optimizer.step()

    # Valutazione finale
    model.eval()
    with torch.no_grad():
        preds = model(bag_embeddings)
        final_mae = model.loss(preds, label_tensor).item()

    return final_mae

