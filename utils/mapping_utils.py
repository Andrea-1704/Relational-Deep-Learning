from typing import Dict, List
import torch
from torch_geometric.data import HeteroData

def get_global_to_local_id_map(data: HeteroData) -> Dict[str, Dict[int, int]]:
    """
    Maps every type of node from a global ID to a local one.
    """
    id_map = {}
    for ntype in data.node_types:
        if 'n_id' not in data[ntype]:
            raise ValueError(f"data[{ntype}].n_id is missing. Cannot build global→local mapping.")
        id_map[ntype] = {
            global_id.item(): local_idx
            for local_idx, global_id in enumerate(data[ntype].n_id)
        }
    return id_map


def map_bag_global_to_local(bag: List[int], global_to_local: Dict[int, int]) -> List[int]:
    """
    Converts a bag of global node IDs in local node IDs for a given node type.
    Automatically ignores nodes that are not present in mapping.
    """
    return [global_to_local[v] for v in bag if v in global_to_local]
