"""
Refinement of version one with the aim of trying to implement a version 
very close to the original one.
Possibly trying to mimic the result of the authors of the paper.
"""

import torch
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
from relbench.modeling.nn import HeteroEncoder
from collections import defaultdict
#from mapping_utils import get_global_to_local_id_map

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


def train_theta_for_relation(
    bags: List[List[int]],
    labels: List[int],
    node_embeddings: torch.Tensor,
    alpha_prev: Dict[int, float],
    global_to_local_map: Dict[int, int],
    epochs: int = 100,
    lr: float = 0.01,
) -> nn.Linear:
    device = node_embeddings.device
    theta = nn.Linear(node_embeddings.size(-1), 1, bias=False).to(device)
    optimizer = torch.optim.Adam(theta.parameters(), lr=lr)

    bag_embeddings = []
    alpha_values = []
    binary_labels = torch.tensor(labels, device=device)

    for bag in bags:
        local_ids = [global_to_local_map[v] for v in bag if v in global_to_local_map]
        if not local_ids:
            continue
        emb = node_embeddings[torch.tensor(local_ids, device=device)]
        alpha = torch.tensor([alpha_prev.get(v, 1.0) for v in bag if v in global_to_local_map], device=device)
        bag_embeddings.append(emb)
        alpha_values.append(alpha)

    for _ in range(epochs):
        optimizer.zero_grad()
        preds = []
        for emb, alpha in zip(bag_embeddings, alpha_values):
            scores = theta(emb).squeeze(-1)
            weights = torch.softmax(scores * alpha, dim=0)
            weighted_avg = torch.sum(weights.unsqueeze(-1) * emb, dim=0)
            pred = weighted_avg.mean()
            preds.append(pred)
        preds = torch.stack(preds)

        pos = preds[binary_labels == 1]
        neg = preds[binary_labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        loss = torch.sum(torch.sigmoid(neg.unsqueeze(0) - pos.unsqueeze(1)))
        loss.backward()
        optimizer.step()

    return theta



def construct_bags_with_alpha(
    data,
    previous_bags: List[List[int]],
    previous_labels: List[float],
    alpha_prev: Dict[int, float],
    rel: Tuple[str, str, str],
    node_embeddings: torch.Tensor,
    theta: nn.Module,
    src_embeddings: torch.Tensor,
    global_to_local_map: Dict[int, int],  # AGGIUNTO
) -> Tuple[List[List[int]], List[float], Dict[int, float]]:
    edge_index = data.edge_index_dict.get(rel)
    if edge_index is None:
        return [], [], {}

    edge_src, edge_dst = edge_index
    bags = []
    labels = []
    alpha_next = {}

    for bag_v, label in zip(previous_bags, previous_labels):
        bag_u = []

        for v in bag_v:
            if v not in global_to_local_map:
                continue
            local_v = global_to_local_map[v]
            neighbors_u = edge_dst[edge_src == v]
            if len(neighbors_u) == 0:
                continue
            x_v = src_embeddings[local_v]
            theta_xv = theta(x_v).item()
            alpha_v = alpha_prev.get(v, 1.0)

            for u in neighbors_u.tolist():
                alpha_u = theta_xv * alpha_v
                alpha_next[u] = alpha_next.get(u, 0.0) + alpha_u
                bag_u.append(u)

        if len(bag_u) > 0:
            bags.append(bag_u)
            labels.append(label)

    return bags, labels, alpha_next




def evaluate_relation_learned(
    bags: List[List[int]],
    labels: List[float],
    node_embeddings: torch.Tensor,
    alpha_prev: Dict[int, float],
    global_to_local_map: Dict[int, int],
    epochs: int = 100,
    lr: float = 0.01,
) -> Tuple[float, nn.Module]:
    """
    Allena theta su bag di nodi (global ID) usando ranking loss, ritorna MAE finale e theta.
    """
    device = node_embeddings.device
    binary_labels = torch.tensor(labels, device=device)

    # Allena theta con ranking loss
    theta = train_theta_for_relation(
        bags=bags,
        labels=labels,
        node_embeddings=node_embeddings,
        alpha_prev=alpha_prev,
        global_to_local_map=global_to_local_map,
        epochs=epochs,
        lr=lr
    )

    # Valutazione finale
    preds = []
    for bag in bags:
        local_ids = [global_to_local_map[v] for v in bag if v in global_to_local_map]
        if not local_ids:
            preds.append(torch.tensor(0.0, device=device))
            continue
        emb = node_embeddings[torch.tensor(local_ids, device=device)]
        scores = theta(emb).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        weighted_avg = torch.sum(weights.unsqueeze(-1) * emb, dim=0)
        pred = weighted_avg.mean()
        preds.append(pred)

    preds_tensor = torch.stack(preds)
    mae = F.l1_loss(preds_tensor, binary_labels).item()
    return mae, theta




def binarize_targets(y: torch.Tensor, threshold: float = 11) -> torch.Tensor:
    """
    This function trasforms a regression task (like the one of driver position)
    into a binary classification problem. 
    To incorporate the original https://arxiv.org/abs/2412.00521 paper, which 
    was only for binary classification we decided to trnasform our task into a 
    binary classfication task, where the driver position gets converted into a 
    binary label:
    1 if position < threshold;
    o otherwise.
    """
    return (y < threshold).long()




def get_candidate_relations(metadata, current_node_type: str) -> List[Tuple[str, str, str]]:
    """
    This function takes the "metadata" of the grafo (which are basicly all the 
    relevant informations about the graph, such as edge types, node types, etc.)
    and returns all the edges (in tuple format "(src_type, name of relation, 
    dst_type)") that starts from "current_node_type" as "src_type".
    """
    return [rel for rel in metadata[1] if rel[0] == current_node_type]


def construct_bags_with_alpha(
    data,
    previous_bags: List[List[int]],
    previous_labels: List[float],
    alpha_prev: Dict[int, float],     # weights α(v, B) for each v ∈ bag previous
    rel: Tuple[str, str, str],
    node_embeddings: torch.Tensor,
    theta: nn.Module,                 # network to compute Θᵗx_v
    src_embeddings,
) -> Tuple[List[List[int]], List[float], Dict[int, float]]:
    """
    Estend the bags through relation "rel", propagating α following eq. (6) di https://arxiv.org/abs/2412.00521.
    Returns:
    - new bag 
    - labels associated to nodes v
    - new alpha[u] for reached nodes u. This is a dictionary where the key is the node u reached and the value
      is the alfa score for that node.
    """
    edge_index = data.edge_index_dict.get(rel)
    if edge_index is None:
        print(f"this should not have happened, but the relation was not found.")
        return [], [], {}

    edge_src, edge_dst = edge_index #tensor [2, #edges], the first one has the node indexes of the src_type, the second of the dst_type
    bags = [] #the new bags, one for each "v" node.
    labels = [] #for each bag we consider its label, given by the one of the src in relation r.
    alpha_next = {} #the result of the computation of the alfa scores given by equation 6.

    for bag_v, label in zip(previous_bags, previous_labels):
        #the previous bag now becomes a "v" node

        bag_u = [] #new bag for the node (bag) "bag_v"

        for v in bag_v: #for each node in the previous bag 
            neighbors_u = edge_dst[edge_src == v]
            #we consider all the edge indexes of destination type that are linked to the 
            #src type through relation "rel", for which the source was exactly the node "v".
            #  Pratically, here we are going through a 
            #relation rel, for example the "patient->prescription" relation and we are 
            # consideringall the prescription that "father" 
            #node of kind patient had.
            if len(neighbors_u) == 0:
                continue

            #x_v = node_embeddings[v] #take the node embedding of the "father" of the node"
            x_v = src_embeddings[v]

            theta_xv = theta(x_v).item() # Θᵗ x_v scalar
            alpha_v = alpha_prev.get(v, 1.0)

            for u in neighbors_u.tolist():  #consider all the "sons" of node "v" through relation "rel"
                alpha_u = theta_xv * alpha_v #compute the new alfa, according to eq 6
                alpha_next[u] = alpha_next.get(u, 0.0) + alpha_u
                bag_u.append(u)

        if len(bag_u) > 0:
            bags.append(bag_u) #updates the new list of bags
            labels.append(label) #the label of the current bag is the same 
            #as the one that the father bag had.

    return bags, labels, alpha_next




class ScoringFunctionReg(nn.Module):
    """    
    This function is one of possibly infinite different implementation for 
    computing how "significative" is a bag.
    In particular, this approach, which follows https://arxiv.org/abs/2412.00521,
    uses a "mini" neural network taht takes an embedding and produces a score value.
    Each bag is a list of embeddings of the reached nodes at a specific time step
    (each of these nodes share the same node type) and we desire to return a score 
    values to the bag.

    We first apply the theta NN to each of the embeddings of the nodes of the bag, 
    getting its score. 
    Then, we normalize the scores through softmax function in order to obtain the 
    attention weights, these score values corresponds to the "α(v, B)" computed
    by https://arxiv.org/abs/2412.00521 in section 4.1, and formally indicates 
    how much attention we should give to a node of the bag.

    Then, followign the formulation indicated in section 4.1 of the aforementioned
    paper, we simply compute a weighted mean of the embeddings of the nodes in the
    bag.

    Finally, we pass the embeddings of the bag to another NN which computes a
    single prediction score for the bag.

    We are using these two networks to "predict" whether the current bag is 
    able to capture important signals about the predictive label.... DOES THIS MAKE
    SENSE????

    Here we work on a single bag.
    """
    def __init__(self, in_dim: int): #in_dim is the dimension of the embedding of nodes
        super().__init__()
        self.theta = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1)  # from embeddings to scalar
        )
        self.out = nn.Sequential(
          nn.Linear(in_dim, in_dim),
          nn.ReLU(),
          nn.Linear(in_dim, 1)
        ) # final nn on embedding of bag

    def forward(self, bags: List[torch.Tensor]) -> torch.Tensor:
        """
        Each bag is a tensor of shape [B_i, D]
        This function return a scalar value, which represent the 
        prediction of each bag.
        """
        preds = []
        for bag in bags:
            if bag.size(0) == 0:
                print(f"this bag is empty")
                preds.append(torch.tensor(0.0, device=bag.device))
                continue
            scores = self.theta(bag).squeeze(-1)  # [B_i] #alfa scores: one for each v in bag
            weights = torch.softmax(scores, dim=0)  # [B_i] #normalize alfa
            weighted_avg = torch.sum(weights.unsqueeze(-1) * bag, dim=0)  # [D] #mean
            # pred = weighted_avg.mean()  #final scalar -> terrible solution!
            pred = self.out(weighted_avg).squeeze(-1) #apply another nn to indicate the importance of bag
            preds.append(pred)
        return torch.stack(preds)

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the MAE score between the two vectors.
        """
        return F.l1_loss(preds, targets)




def greedy_metapath_search_with_bags_learned(
    data,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    node_type: str,
    col_stats_dict: Dict[str, Dict[str, Dict]],
    L_max: int = 3,
    max_rels: int = 10,
    channels: int = 64,
) -> Tuple[List[List[Tuple[str, str, str]]], Dict[Tuple, int]]:
    device = y.device
    metapaths = []
    metapath_counts = defaultdict(int)
    current_paths = [[]]

    current_bags = [[int(i)] for i in torch.where(train_mask)[0]]
    current_labels = [y[i].item() for i in torch.where(train_mask)[0]]
    alpha = {int(i): 1.0 for i in torch.where(train_mask)[0]}

    for level in range(L_max):
        print(f"level {level}")
        new_paths = []
        new_alpha_all = []
        new_bags_all = []
        new_labels_all = []

        for path in current_paths:
            last_ntype = node_type if not path else path[-1][2]
            print(f"current source node is {last_ntype}")

            with torch.no_grad():
                encoder = HeteroEncoder(
                    channels=channels,
                    node_to_col_names_dict={
                        ntype: data[ntype].tf.col_names_dict for ntype in data.node_types
                    },
                    node_to_col_stats=col_stats_dict,
                ).to(device)
                for module in encoder.modules():
                    for name, buf in module._buffers.items():
                        if buf is not None:
                            module._buffers[name] = buf.to(device)

                tf_dict = {
                    ntype: data[ntype].tf.to(device)
                    for ntype in data.node_types
                    if 'tf' in data[ntype]
                }
                node_embeddings_dict = encoder(tf_dict)

            candidate_rels = [
                (src, rel, dst)
                for (src, rel, dst) in data.edge_index_dict.keys()
                if src == last_ntype
            ][:max_rels]

            best_rel = None
            best_score = float("inf")
            best_theta = None
            best_alpha = None
            best_bags = None
            best_labels = None

            for rel in candidate_rels:
                print(f"considering relation {rel}")
                src, _, dst = rel
                if dst in [step[0] for step in path]:
                    continue
                if dst == node_type:
                    continue

                node_embeddings = node_embeddings_dict.get(dst)
                if node_embeddings is None:
                    continue

                # Allena theta e ottieni score
                score, theta = evaluate_relation_learned(
                    bags=current_bags,
                    labels=current_labels,
                    node_embeddings=node_embeddings_dict[src],
                    alpha_prev=alpha,
                )

                bags, labels, alpha_next = construct_bags_with_alpha(
                    data=data,
                    previous_bags=current_bags,
                    previous_labels=current_labels,
                    alpha_prev=alpha,
                    rel=rel,
                    node_embeddings=node_embeddings,
                    theta=theta,
                    src_embeddings=node_embeddings_dict[src],
                )

                if len(bags) < 5:
                    continue

                print(f"relation {rel} score = {score}")
                if score < best_score:
                    best_score = score
                    best_rel = rel
                    best_theta = theta
                    best_alpha = alpha_next
                    best_bags = bags
                    best_labels = labels

            if best_rel:
                new_path = path + [best_rel]
                new_paths.append(new_path)
                metapath_counts[tuple(new_path)] += 1
                new_alpha_all.append(best_alpha)
                new_bags_all.extend(best_bags)
                new_labels_all.extend(best_labels)

        current_paths = new_paths
        alpha = {k: v for d in new_alpha_all for k, v in d.items()}
        current_bags = new_bags_all
        current_labels = new_labels_all
        metapaths.extend(current_paths)
        print(f"Final metapaths are {metapaths}")

    return metapaths, metapath_counts


def beam_metapath_search_with_bags_learned(
    data,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    node_type: str,
    col_stats_dict: Dict[str, Dict[str, Dict]],
    L_max: int = 3,
    max_rels: int = 10,
    channels: int = 64,
    beam_width: int = 5,
) -> Tuple[List[List[Tuple[str, str, str]]], Dict[Tuple, int]]:
    device = y.device
    global_to_local_id_map = {
        ntype: {
            global_id.item(): local_idx
            for local_idx, global_id in enumerate(data[ntype].n_id)
        }
        for ntype in data.node_types
        if 'n_id' in  data[ntype]
    }

    #print(f"global_to_local_id_map: {enumerate(data['drivers'].n_id)}")
    all_path_info = []  # (score, path, bags, labels, alpha)
    metapath_counts = defaultdict(int)

    # Init: path vuoto → ogni bag è 1 nodo (driver i-esimo)
    current_paths = [[]]
    current_bags = [[int(i)] for i in torch.where(train_mask)[0]]
    current_labels = [y[i].item() for i in torch.where(train_mask)[0]]
    alpha = {int(i): 1.0 for i in torch.where(train_mask)[0]}

    for level in range(L_max):
        print(f"LEVEL {level}")
        next_path_info = []

        for path in current_paths:
            last_ntype = node_type if not path else path[-1][2]
            print(f"Expanding from node type: {last_ntype}")

            with torch.no_grad():
                encoder = HeteroEncoder(
                    channels=channels,
                    node_to_col_names_dict={
                        ntype: data[ntype].tf.col_names_dict for ntype in data.node_types
                    },
                    node_to_col_stats=col_stats_dict,
                ).to(device)
                for module in encoder.modules():
                    for name, buf in module._buffers.items():
                        if buf is not None:
                            module._buffers[name] = buf.to(device)

                tf_dict = {
                    ntype: data[ntype].tf.to(device)
                    for ntype in data.node_types if 'tf' in data[ntype]
                }
                node_embeddings_dict = encoder(tf_dict)

            candidate_rels = [
                (src, rel, dst)
                for (src, rel, dst) in data.edge_index_dict.keys()
                if src == last_ntype
            ][:max_rels]

            for rel in candidate_rels:
                src, _, dst = rel
                if dst in [step[0] for step in path] or dst == node_type:
                    continue

                src_emb = node_embeddings_dict.get(src)
                dst_emb = node_embeddings_dict.get(dst)
                if src_emb is None or dst_emb is None:
                    continue
                    
                score, theta = evaluate_relation_learned(
                    bags=current_bags,
                    labels=current_labels,
                    node_embeddings=node_embeddings_dict[src],
                    alpha_prev=alpha,
                    global_to_local_map=global_to_local_id_map[src]
                )

                bags, labels, alpha_next = construct_bags_with_alpha(
                    data=data,
                    previous_bags=current_bags,
                    previous_labels=current_labels,
                    alpha_prev=alpha,
                    rel=rel,
                    node_embeddings=dst_emb,
                    theta=theta,
                    src_embeddings=src_emb,
                    global_to_local_map=global_to_local_id_map[src]
                )


                if len(bags) < 5:
                    continue

                new_path = path + [rel]
                next_path_info.append((score, new_path, bags, labels, alpha_next))

        # Beam: seleziona top beam_width path con score migliore
        next_path_info.sort(key=lambda x: x[0])
        selected = next_path_info[:beam_width]

        # Prepara per prossimo livello
        current_paths = []
        current_bags = []
        current_labels = []
        alpha = {}

        for score, path, bags, labels, alpha_next in selected:
            current_paths.append(path)
            current_bags.extend(bags)
            current_labels.extend(labels)
            alpha.update(alpha_next)
            all_path_info.append((score, path, bags))

    # Seleziona i migliori beam_width finali tra TUTTI quelli visitati
    all_path_info.sort(key=lambda x: x[0])
    final_selected = all_path_info[:beam_width]
    metapaths = []

    for _, path, bags in final_selected:
        metapaths.append(path)
        metapath_counts[tuple(path)] = len(bags)

    print(f"FINAL METAPATHS (TOP {beam_width}):")
    for path in metapaths:
        print("  ", path)

    return metapaths, metapath_counts
