import torch
from torch_geometric.data import HeteroData
from typing import List, Tuple


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
                           eta: float = 0.3) -> List[List[Tuple[str, str, str]]]:
    metadata = data.metadata()
    current_metapath = []
    all_metapaths = []
    current_node_type = node_type

    for _ in range(L_max):
        candidate_rels = get_candidate_relations(metadata, current_node_type)
        best_rel = None
        best_loss = float('inf')

        for rel in candidate_rels:
            new_path = current_metapath + [rel]
            loss = surrogate_classification_loss(data, new_path, y_bin, train_mask, node_type)
            if loss < best_loss:
                best_loss = loss
                best_rel = rel

        if best_rel is None or best_loss > eta:
            break

        current_metapath.append(best_rel)
        current_node_type = best_rel[2]

    if current_metapath:
        all_metapaths.append(current_metapath)

    return all_metapaths
