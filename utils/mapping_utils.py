# mapping_utils.py

from typing import Dict, List
import torch
from torch_geometric.data import HeteroData

def get_global_to_local_id_map(data: HeteroData) -> Dict[str, Dict[int, int]]:
    """
    Costruisce un mapping per ogni tipo di nodo da global ID → local ID.
    Richiede che data[ntype].n_id sia presente (RelBench e NeighborLoader lo forniscono).
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
    Converte una bag di global node IDs in local node IDs per un dato tipo di nodo.
    Ignora automaticamente i nodi non presenti nel mapping.
    """
    return [global_to_local[v] for v in bag if v in global_to_local]
