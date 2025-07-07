import torch
import math
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
from relbench.modeling.nn import HeteroEncoder
from collections import defaultdict
from model.MPSGNN_Model import MPSGNN
from utils.utils import evaluate_performance, evaluate_on_full_train, test, train


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




def evaluate_relation_learned(
    bags: List[List[int]],  #one for each node
    labels: List[float],    #len(labels)==len(bags)
    node_embeddings: torch.Tensor, #embeddings for each node in the graph
    epochs: int = 30,
    lr: float = 1e-2,#should be tuned
) -> float:
    """
    This function follows the algorithm indicated into section 4.4 in 
    https://arxiv.org/abs/2412.00521, by trainign the model on the 
    current bag in order to choose the most "discriminative" relation 
    r_i to be added to the meta path. 
    This function returns a mae value which indicates how predictive 
    is the current bag nodes to make the classification.

    This approach is inspired by the aforementioned paper, but is 
    different in the nature because we need to employ a different
    surrogate function (in the paper the binary ranking loss was used,
    we, instead, use a MAE loss since we need to deal with a regression
    task).
    Notice that in the paper the scoring function "F(B)" was parametrized by
    relations, not for metadapath; We are, instead, building a 
    ScoringFunctionReg from scratch for each relation, reducing the 
    risk of overfitting, but also increasing the complexity of the model.

    We can simply say that the "score" value of a given set of bags, which 
    resemples the "informativeness" of a certain relation in the metapath
    is given by the ability that a network (the same over the different
    relations in order to be comparable) has to assign correctly the label
    of the bags by only locking to the embeddings of nodes "v" inside the
    bag.
    """
    device = node_embeddings.device
    in_dim = node_embeddings.size(-1)

    model = ScoringFunctionReg(in_dim).to(device)#use the class model 
    #defined before
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    bag_embeddings = [
        node_embeddings[torch.tensor(bag, device=device)] for bag in bags
    ] #for each bag take the embeddings of that node, because remember 
    #that a bag is a set of nodes of the same types, but we need to 
    #obtain the embeddings of the nodes. 
    #The result is a list of tensors of shape [B_i, D], where B_i is the 
    #number of nodes in the bag, and D the dimensionality of embeddings.

    target_tensor = torch.tensor(labels, device=device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(bag_embeddings)#forward
        loss = model.loss(preds, target_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(bag_embeddings)
        final_mae = model.loss(preds, target_tensor).item()

    return final_mae




def construct_bags_with_alpha(
    data,
    previous_bags: List[List[int]], 
    previous_labels: List[float],     # list of the "v" nodes
    alpha_prev: Dict[int, float],     # weights α(v, B) for each v ∈ bag previous
    rel: Tuple[str, str, str],
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

    edge_src, edge_dst = edge_index #tensor [2, #edges]
    bags = [] #the new bags, one for each "v" node.
    labels = [] #for each bag we consider its label, given by the one of the src in relation r.
    alpha_next = {} #the result of the computation of the alfa scores given by equation 6.

    for bag_v, label in zip(previous_bags, previous_labels):
        #the previous bag now becomes a "v" node

        bag_u = [] #new bag for the node (bag) "bag_v"

        for v in bag_v: #for each node in the previous bag 

            neighbors_u = edge_dst[edge_src == v] #correct for the first step
            #we consider all the edge indexes of destination type that are linked to the 
            #src type through relation "rel", for which the source was exactly the node "v".
            #  Pratically, here we are going through a 
            #relation rel, for example the "patient->prescription" relation and we are 
            # consideringall the prescription that "father" 
            #node of kind patient had.
            if len(neighbors_u) == 0:
                #could be zero even just because that node simply do not have any of such relations
                #test to understand if we are managing correctly the global and local mapping:
                continue
            
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




def beam_metapath_search_with_bags_learned(
    data: HeteroData, #the result of make_pkey_fkey_graph
    db,   #Object that was passed to make_pkey_fkey_graph to build data
    node_id: str, #ex driverId
    train_mask: torch.Tensor,
    node_type: str, 
    col_stats_dict: Dict[str, Dict[str, Dict]], 
    L_max: int = 3,
    max_rels: int = 10,
    channels : int = 64,
    beam_width: int = 5,  #number of metapaths to look for
) -> Tuple[List[List[Tuple[str, str, str]]], Dict[Tuple, int]]:
    """
    This function provides more than one metapaths by applying a beam search over the 
    metapaths.
    This implementation also do not require to use a hard cutoff value to stop the 
    algorithm from creating sub-optimal and un-usefull long metapaths: 
    it simply considers all the metapaths (also the intermediate ones and their score
    value to be able to consider also intermediate metapaths).
    We also added a statistics count that takes into account the counts of how many 
    times each metapath has been use in the path (for example assuming to have the 
    metapath A->B->C, we count how many A nodes are linked to C nodes throught this
    set of relations).    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metapaths = []
    metapath_counts = {} #for each metapath counts how many bags are presents, so how many istances of that metapath are present
    current_paths = [[]]
    driver_ids_df = db.table_dict[node_type].df[node_id].to_numpy()
    current_bags =  [[int(i)] for i in driver_ids_df if train_mask[i]]
    #current_bags contains the id of the drivers node 
    old_y = data[node_type].y.int().tolist() #ordered as current bags
    print(f"initial y: {old_y}")
    current_labels = []
    for i in range(0, len(old_y)):
        if train_mask[i]:
            current_labels.append(old_y[i])
    # print(len(current_bags))
    print(current_bags)

    # print(len(current_labels))
    # print(current_labels)
    
    assert len(current_bags) == len(current_labels)
    
    alpha = {int(i): 1.0 for i in torch.where(train_mask)[0]}
    all_path_info = [] #memorize all the metapaths with scores, in order
    #to select only the best beam_width at the end

    with torch.no_grad():
        encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                ntype: data[ntype].tf.col_names_dict
                for ntype in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        ).to(device)
        for module in encoder.modules():
            for name, buf in module._buffers.items():
                if buf is not None:
                    module._buffers[name] = buf.to(device)
        tf_dict = {
            ntype: data[ntype].tf.to(device) for ntype in data.node_types
        } #each node type as a tf.
        node_embeddings_dict = encoder(tf_dict)


    for level in range(L_max):
        print(f"we are at level {level}")
        
        next_paths_info = []

        for path in current_paths:
            last_ntype = node_type if not path else path[-1][2]
            print(f"current source node is {last_ntype}")

            candidate_rels = [
                (src, rel, dst)
                for (src, rel, dst) in data.edge_index_dict.keys()
                if src == last_ntype
            ][:max_rels] 

            for rel in candidate_rels: 
                print(f"considering relation {rel}")
                src, _, dst = rel
                if dst in [step[0] for step in path] or dst == node_type:  # avoid loops in met and avoid to return to the source node
                  continue
                
                node_embeddings = node_embeddings_dict.get(dst) #Tensor[num_node_of_kind_dst, embedding_dim]

                if node_embeddings is None:
                    print(f"error: embedding of node {dst} not found")
                    continue

                theta = nn.Linear(node_embeddings.size(-1), 1).to(device) #maybe it should be first learned as in version2
                
                bags, labels, alpha_next = construct_bags_with_alpha(
                    data=data,
                    previous_bags=current_bags,
                    previous_labels=current_labels,
                    alpha_prev=alpha, 
                    rel=rel,
                    #node_embeddings=node_embeddings,
                    theta=theta,
                    src_embeddings = node_embeddings_dict[src]
                )

                if len(bags) < 5:
                    continue
                
                score = evaluate_relation_learned(bags, labels, node_embeddings)
                print(f"relation {rel} allow us to obtain score {score}")

                new_path = path + [rel]

                next_paths_info.append((score, new_path, bags, labels, alpha_next))

        current_paths = []
        current_bags = []
        current_labels = []
        alpha = {}

        for info in next_paths_info:
          _, path, bags, labels, alpha_next = info
          current_paths.append(path)
          current_bags.extend(bags)
          current_labels.extend(labels)
          alpha.update(alpha_next)

        all_path_info.extend(next_paths_info)

    #final selection of the best beamwodth paths:
    all_path_info.sort(key=lambda x:x[0])
    selected = all_path_info[:beam_width]
    for _, path, bags, _, _ in selected:
      metapaths.append(path)
      metapath_counts[tuple(path)] = len(bags)
    print(f"final metapaths are: {metapaths}")
    print(f"metapath counts are: {metapath_counts}")
    return metapaths, metapath_counts


def beam_metapath_search_with_bags_learned_2(
    data: HeteroData, #the result of make_pkey_fkey_graph
    db,   #Object that was passed to make_pkey_fkey_graph to build data
    node_id: str, #ex. driverId
    loader_dict,
    task, 
    loss_fn,
    tune_metric : str,
    higher_is_better: str,
    train_mask: torch.Tensor,
    node_type: str, 
    col_stats_dict: Dict[str, Dict[str, Dict]], 
    L_max: int = 3,
    channels : int = 64,
    number_of_metapaths: int = 5,  #number of metapaths to look for
    out_channels: int = 128,
    hidden_channels: int = 128,
    lr : float = 0.0001,
    wd: float = 0,
    epochs: int = 100,
    max_rel: int = 10
) -> Tuple[List[List[Tuple[str, str, str]]], Dict[Tuple, int]]:
    """
    Avoid score computation, compute results only using mps gnn.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                ntype: data[ntype].tf.col_names_dict
                for ntype in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        ).to(device)
        for module in encoder.modules():
            for name, buf in module._buffers.items():
                if buf is not None:
                    module._buffers[name] = buf.to(device)
        
        tf_dict = {
            ntype: data[ntype].tf.to(device) for ntype in data.node_types if 'tf' in data[ntype]
        }
        node_embeddings_dict = encoder(tf_dict)
    
    metapaths = []
    metapath_counts = {} 
    driver_ids_df = db.table_dict[node_type].df[node_id].to_numpy()
    current_bags =  [[int(i)] for i in driver_ids_df if train_mask[i]]
    old_y = data[node_type].y.int().tolist() #ordered as current bags
    print(f"initial y: {old_y}")
    current_labels = []
    for i in range(0, len(old_y)):
        if train_mask[i]:
            current_labels.append(old_y[i])
    assert len(current_bags) == len(current_labels)
    alpha = {int(i): 1.0 for i in torch.where(train_mask)[0]}
    all_path_info = [] 

    current_paths = [[]]
    for level in range(L_max):
        print(f"we are at level {level}")
        next_paths_info = []

        for path in current_paths:
            last_ntype = node_type if not path else path[-1][2]
            print(f"current source node is {last_ntype}")

            candidate_rels = [
                (src, rel, dst)
                for (src, rel, dst) in data.edge_index_dict.keys()
                if src == last_ntype
            ][:max_rel] 

            for rel in candidate_rels: 
                print(f"considering relation {rel}")
                src, _, dst = rel
                if dst in [step[0] for step in path] or dst == node_type:  # avoid loops in met and avoid to return to the source node
                  continue
                if rel == ('races', 'rev_f2p_raceId', 'standings'): # for some reasons it provokes side assertions
                  continue
                node_embeddings = node_embeddings_dict.get(dst) #Tensor[num_node_of_kind_dst, embedding_dim]
                theta = nn.Linear(node_embeddings.size(-1), 1).to(device) #maybe it should be first learned as in version2
                bags, labels, alpha_next = construct_bags_with_alpha(
                    data=data,
                    previous_bags=current_bags,
                    previous_labels=current_labels,
                    alpha_prev=alpha, 
                    rel=rel,
                    #node_embeddings=node_embeddings,
                    theta=theta,
                    src_embeddings = node_embeddings_dict[src]
                )
                if len(bags) < 5:
                    continue
                #score = evaluate_relation_learned(bags, labels, node_embeddings)
                #print(f"relation {rel} allow us to obtain score {score}")
                new_path = path.copy()
                new_path = new_path.append(rel)
                local_path2 = new_path.copy()
                loc = [local_path2.copy()]
                metapath_counts[tuple(local_path2)] += 1
                model = MPSGNN(
                    data=data,
                    col_stats_dict=col_stats_dict,
                    metadata=data.metadata(),
                    metapath_counts = metapath_counts,
                    metapaths=loc,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    final_out_channels=1,
                ).to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
                #EPOCHS:
                test_table = task.get_table("test", mask_input_cols=False)
                best_test_metrics = -math.inf if higher_is_better else math.inf
                for _ in range(0, epochs):
                    train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)
                    test_pred = test(model, loader_dict["test"], device=device, task=task)
                    test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)
                    if test_metrics[tune_metric] > best_test_metrics and higher_is_better:
                        best_test_metrics = test_metrics[tune_metric]
                    if test_metrics[tune_metric] < best_test_metrics and not higher_is_better:
                        best_test_metrics = test_metrics[tune_metric]
                print(f"For the partial metapath {local_path2.copy()} we obtain F1 test loss equal to {best_test_metrics}")
                next_paths_info.append((best_test_metrics, new_path, bags, labels, alpha_next))

        current_paths = []
        current_bags = []
        current_labels = []
        alpha = {}

        for info in next_paths_info:
          _, path, bags, labels, alpha_next = info
          current_paths.append(path)
          current_bags.extend(bags)
          current_labels.extend(labels)
          alpha.update(alpha_next)

        all_path_info.extend(next_paths_info)

    #final selection of the best beamwodth paths:
    best_score_per_path = {}
    for score, path in all_path_info:
        path_tuple = tuple(path)
        if path_tuple not in best_score_per_path:
            best_score_per_path[path_tuple] = score
    sorted_unique_paths = sorted(best_score_per_path.items(), key=lambda x: x[1], reverse=True)#higher is better
    selected_metapaths = [list(path_tuple) for path_tuple, _ in sorted_unique_paths[:number_of_metapaths]]
    
    return selected_metapaths, metapath_counts



"""
Beam search is a very strong and powerfull version 
and is much more complex than the original one 
used in the paper, which is more like a greedy 
search approch with less coverage.

Despite this we may want to introduce a second 
function with less coverage but more efficient 
for high Lmax:
"""


#Previous version, return only a metaptah, with partial ones
def greedy_metapath_search_with_bags_learned(
    data: HeteroData, #the result of make_pkey_fkey_graph
    db,   #Object that was passed to make_pkey_fkey_graph to build data
    node_id: str, #ex driverId
    loader_dict,
    task, 
    loss_fn,
    tune_metric : str,
    higher_is_better: str,
    train_mask: torch.Tensor,
    node_type: str, 
    col_stats_dict: Dict[str, Dict[str, Dict]], 
    L_max: int = 3,
    channels : int = 64,
    number_of_metapaths: int = 5,  #number of metapaths to look for
    out_channels: int = 128,
    hidden_channels: int = 128,
    lr : float = 0.0001,
    wd: float = 0,
    epochs: int = 100,
    
) -> Tuple[List[List[Tuple[str, str, str]]], Dict[Tuple, int]]:
    
    """
    This is the main component of this set of functions and classes, is the 
    complete algorithm used to implement the meta paths.

    This function searches in a greedy fashion the best meta-paths 
    starting from a node(the TARGET one, for example driver) till "L_max" 
    depth.
    At each step selects the best relation to add to the current path 
    based on a surrogate task score (MAE).

    In the current version of this algorithm we are deliberately avoiding 
    to consider the second stopping criteria indicated in section 4.4 of the 
    reference, in order to avoid to consider a strict threshold for the 
    allowed minimal improvement.
    We also added a statistics count that takes into account the counts of how many 
    times each metapath has been use in the path (for example assuming to have the 
    metapath A->B->C, we count how many A nodes are linked to C nodes throught this
    set of relations).  

    The score that we consider for each of the metapath is not simply the one
    of the compute score, but is the result of a training of the mps gnn!
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                ntype: data[ntype].tf.col_names_dict
                for ntype in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        ).to(device)
        for module in encoder.modules():
            for name, buf in module._buffers.items():
                if buf is not None:
                    module._buffers[name] = buf.to(device)
        
        tf_dict = {
            ntype: data[ntype].tf.to(device) for ntype in data.node_types if 'tf' in data[ntype]
        }
        node_embeddings_dict = encoder(tf_dict)

    
    metapath_counts = defaultdict(int) 
    driver_ids_df = db.table_dict[node_type].df[node_id].to_numpy()
    current_bags =  [[int(i)] for i in driver_ids_df if train_mask[i]]
    old_y = data[node_type].y.int().tolist()
    current_labels = []
    for i in range(0, len(old_y)):
        if train_mask[i]:
            current_labels.append(old_y[i])
    assert len(current_bags) == len(current_labels)
    alpha = {int(i): 1.0 for i in torch.where(train_mask)[0]}
    all_path_info = [] #memorize all the metapaths with scores, in order to select only the best beam_width at the end
    local_path = []
    
    current_paths = [[]] 
    for level in range(L_max):
        print(f"level {level}")
        
        next_paths_info = []
        #current_paths = []
        #current_paths = [(DRIVERS, _, RESULTS)]
        #current_paths = [(RESULTS, _, RACES)]

        for path in current_paths: 
            #path = []
            #path = (DRIVERS, _, RESULTS)
            #path = (RESULTS, _, RACES)
            last_ntype = node_type if not path else path[2]
            #DRIVERS#RESULTS #RACES
            print(f"current source node is {last_ntype}")
           
            candidate_rels = [ #take all the rel that begins from last_ntype
                (src, rel, dst)
                for (src, rel, dst) in data.edge_index_dict.keys()
                if src == last_ntype
            ]

            #choose the best relation beginning from last_ntype:
            best_rel = None
            best_score = float('inf') #score = error, so less is better!
            best_alpha = None
            best_bags = None
            best_labels = None

            for rel in candidate_rels: 
                print(f"considering relation {rel}")
                src, _, dst = rel
                if dst in [step[0] for step in path] or dst == node_type:  # avoid loops in met, avoid to return to the source node
                  continue

                node_embeddings = node_embeddings_dict.get(dst) #access at the value (Tensor[dst, hidden_dim]) for key node type "dst"
                theta = nn.Linear(node_embeddings.size(-1), 1).to(device) #classifier which is used to compute Θᵗx_v
                bags, labels, alpha_next = construct_bags_with_alpha(
                    data=data,
                    previous_bags=current_bags,
                    previous_labels=current_labels,
                    alpha_prev=alpha, #current alfa values for v nodes
                    rel=rel,
                    theta=theta,
                    src_embeddings = node_embeddings_dict[src]
                )
                if len(bags) < 5:
                    continue#this avoid to consider few bags to avoid overfitting
                score = evaluate_relation_learned(bags, labels, node_embeddings) #assign the score value to current split, similar to DECISION TREES
                print(f"relation {rel} allow us to obtain score {score}")
                
                if score < best_score:
                    best_rel = rel
                    best_score = score
                    best_alpha = alpha_next
                    best_bags = bags
                    best_labels = labels
                
                local_path2 = local_path.copy()
                # #even if it is not the best one we memorize it because maybe will
                # #be selected from beam search:
                # local_path2.append(rel)
                # all_path_info.append((score, local_path2.copy()))
                #now is useless since we select through the F1 of the model and we
                #are not going to test it here.
            
            #set best_rel:
            if best_rel:
                print(f"Best relation is {best_rel}")
                local_path.append(best_rel)
                #[(DRIVERS, _, RESULTS)]
                #[(DRIVERS, _, RESULTS), (RESULTS, _, RACES)]
                print(f"Now local path is {local_path}")
                next_paths_info.append((best_score, local_path, best_bags, best_labels, best_alpha))
                #WARNING: SCORE IS COMPUTED ONLY FOR LAST RELATION BUT WE ARE LINKING IT TO THE COMPLETE LOCAL PATH!!!
                metapath_counts[tuple(local_path)] += 1
                

                #FOR THIS ONE WE SHOULD TRAIN A COMPLETE MPS GNN MODEL AND STORE THE SCORE RECEIVED IN TERMS OF 
                #F1, SO HIGHER IS BETTER!
                loc = [local_path.copy()]
                print(f"local path to path is {loc}")
                model = MPSGNN(
                    data=data,
                    col_stats_dict=col_stats_dict,
                    metadata=data.metadata(),
                    metapath_counts = metapath_counts,
                    metapaths=loc,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    final_out_channels=1,
                ).to(device)

                # optimizer = torch.optim.Adam(
                #   model.parameters(),
                #   lr=lr,
                #   weight_decay=wd
                # )
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

                #EPOCHS:
                test_table = task.get_table("test", mask_input_cols=False)
                best_test_metrics = -math.inf 
                for _ in range(0, epochs):
                    train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)
                    test_pred = test(model, loader_dict["test"], device=device, task=task)
                    test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)
                    if test_metrics[tune_metric] > best_test_metrics and higher_is_better:
                        best_test_metrics = test_metrics[tune_metric]
                    if test_metrics[tune_metric] < best_test_metrics and not higher_is_better:
                        best_test_metrics = test_metrics[tune_metric]
                print(f"For the partial metapath {local_path.copy()} we obtain F1 test loss equal to {best_test_metrics}")
                all_path_info.append((best_test_metrics, local_path.copy()))
        
        current_paths = [best_rel] 
        print(f"current path now is equal to {current_paths}\n")
        #current_paths = [(DRIVERS, _, RESULTS)]
        #current_paths = [(RESULTS, _, RACES)]
    
    best_score_per_path = {}
    for score, path in all_path_info:
        path_tuple = tuple(path)
        if path_tuple not in best_score_per_path:
            best_score_per_path[path_tuple] = score
    sorted_unique_paths = sorted(best_score_per_path.items(), key=lambda x: x[1], reverse=True)#higher is better
    selected_metapaths = [list(path_tuple) for path_tuple, _ in sorted_unique_paths[:number_of_metapaths]]
    #print(f"\nfinal metapaths are {selected_metapaths}\n")

    return selected_metapaths, metapath_counts


#SECOND IDEA: DIFFERENT FROM THE ONE OF THE AUTHORS
#WHY DON'T WE TRAIN THE MODEL FOL ALL POSSIBLE COMBINATION, NOT
#ONLY BEST REL AND SELECT THE BEST ONES IN ORDER TO HAVE DIVERSE RELATIONS?
def greedy_metapath_search_with_bags_learned_2(
    data: HeteroData, #the result of make_pkey_fkey_graph
    db,   #Object that was passed to make_pkey_fkey_graph to build data
    node_id: str, #ex. driverId
    loader_dict,
    task, 
    loss_fn,
    tune_metric : str,
    higher_is_better: str,
    train_mask: torch.Tensor,
    node_type: str, 
    col_stats_dict: Dict[str, Dict[str, Dict]], 
    L_max: int = 3,
    channels : int = 64,
    number_of_metapaths: int = 5,  #number of metapaths to look for
    out_channels: int = 128,
    hidden_channels: int = 128,
    lr : float = 0.0001,
    wd: float = 0,
    epochs: int = 100,
    max_rel: int = 10
) -> Tuple[List[List[Tuple[str, str, str]]], Dict[Tuple, int]]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                ntype: data[ntype].tf.col_names_dict
                for ntype in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        ).to(device)
        for module in encoder.modules():
            for name, buf in module._buffers.items():
                if buf is not None:
                    module._buffers[name] = buf.to(device)
        
        tf_dict = {
            ntype: data[ntype].tf.to(device) for ntype in data.node_types if 'tf' in data[ntype]
        }
        node_embeddings_dict = encoder(tf_dict)

    
    metapath_counts = defaultdict(int) 
    driver_ids_df = db.table_dict[node_type].df[node_id].to_numpy()
    current_bags =  [[int(i)] for i in driver_ids_df if train_mask[i]]
    old_y = data[node_type].y.int().tolist()
    current_labels = []
    for i in range(0, len(old_y)):
        if train_mask[i]:
            current_labels.append(old_y[i])
    assert len(current_bags) == len(current_labels)
    alpha = {int(i): 1.0 for i in torch.where(train_mask)[0]}
    all_path_info = [] 
    local_path = []
    
    current_paths = [[]] 
    for level in range(L_max):
        print(f"level {level}")
        next_paths_info = []

        for path in current_paths: 
            last_ntype = node_type if not path else path[2]
            print(f"current source node is {last_ntype}")
           
            candidate_rels = [ 
                (src, rel, dst)
                for (src, rel, dst) in data.edge_index_dict.keys()
                if src == last_ntype
            ][:max_rel]

            best_rel = None
            best_score = float('inf') 
            best_alpha = None
            best_bags = None
            best_labels = None

            for rel in candidate_rels: 
                print(f"considering relation {rel}")
                src, _, dst = rel
                if dst in [step[0] for step in path] or dst == node_type:  
                  continue
                if rel == ('races', 'rev_f2p_raceId', 'standings'): # for some reasons it provokes side assertions
                  continue
                node_embeddings = node_embeddings_dict.get(dst) 
                theta = nn.Linear(node_embeddings.size(-1), 1).to(device) 
                bags, labels, alpha_next = construct_bags_with_alpha(
                    data=data,
                    previous_bags=current_bags,
                    previous_labels=current_labels,
                    alpha_prev=alpha,
                    rel=rel,
                    theta=theta,
                    src_embeddings = node_embeddings_dict[src]
                )
                if len(bags) < 5:
                    continue
                score = evaluate_relation_learned(bags, labels, node_embeddings) 
                print(f"relation {rel} allow us to obtain score {score}")
                if score < best_score:
                    best_rel = rel
                    best_score = score
                    best_alpha = alpha_next
                    best_bags = bags
                    best_labels = labels
                
                local_path2 = local_path.copy()
                local_path2.append(rel)
                loc = [local_path2.copy()]
                metapath_counts[tuple(local_path2)] += 1
                model = MPSGNN(
                    data=data,
                    col_stats_dict=col_stats_dict,
                    metadata=data.metadata(),
                    metapath_counts = metapath_counts,
                    metapaths=loc,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    final_out_channels=1,
                ).to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
                #EPOCHS:
                test_table = task.get_table("test", mask_input_cols=False)
                best_test_metrics = -math.inf if higher_is_better else math.inf
                for _ in range(0, epochs):
                    train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)
                    test_pred = test(model, loader_dict["test"], device=device, task=task)
                    test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)
                    if test_metrics[tune_metric] > best_test_metrics and higher_is_better:
                        best_test_metrics = test_metrics[tune_metric]
                    if test_metrics[tune_metric] < best_test_metrics and not higher_is_better:
                        best_test_metrics = test_metrics[tune_metric]
                print(f"For the partial metapath {local_path2.copy()} we obtain F1 test loss equal to {best_test_metrics}")
                all_path_info.append((best_test_metrics, local_path2.copy()))
            
            #set best_rel:
            if best_rel:
                local_path.append(best_rel)
                print(f"Best relation is {best_rel} and now local path is {local_path}")
                next_paths_info.append((best_score, local_path, best_bags, best_labels, best_alpha))
                metapath_counts[tuple(local_path)] += 1
        
        current_paths = [best_rel] 
        #print(f"current path now is equal to {current_paths}\n")
    
    best_score_per_path = {}
    for score, path in all_path_info:
        path_tuple = tuple(path)
        if path_tuple not in best_score_per_path:
            best_score_per_path[path_tuple] = score
    sorted_unique_paths = sorted(best_score_per_path.items(), key=lambda x: x[1], reverse=True)#higher is better
    selected_metapaths = [list(path_tuple) for path_tuple, _ in sorted_unique_paths[:number_of_metapaths]]
    #print(f"\nfinal metapaths are {selected_metapaths}\n")

    return selected_metapaths, metapath_counts


#Third version: 
#Ignore surrogate task and update local path considering f1 score after training
def greedy_metapath_search_with_bags_learned_3(
    data: HeteroData, #the result of make_pkey_fkey_graph
    db,   #Object that was passed to make_pkey_fkey_graph to build data
    node_id: str, #ex. driverId
    loader_dict,
    task, 
    loss_fn,
    tune_metric : str,
    higher_is_better: str,
    train_mask: torch.Tensor,
    node_type: str, 
    col_stats_dict: Dict[str, Dict[str, Dict]], 
    L_max: int = 3,
    channels : int = 64,
    number_of_metapaths: int = 5,  #number of metapaths to look for
    out_channels: int = 128,
    hidden_channels: int = 128,
    lr : float = 0.0001,
    wd: float = 0,
    epochs: int = 100,
    max_rels: int = 10
) -> Tuple[List[List[Tuple[str, str, str]]], Dict[Tuple, int]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                ntype: data[ntype].tf.col_names_dict
                for ntype in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        ).to(device)
        for module in encoder.modules():
            for name, buf in module._buffers.items():
                if buf is not None:
                    module._buffers[name] = buf.to(device)
        
        tf_dict = {
            ntype: data[ntype].tf.to(device) for ntype in data.node_types if 'tf' in data[ntype]
        }
        node_embeddings_dict = encoder(tf_dict)

    
    metapath_counts = defaultdict(int) 
    driver_ids_df = db.table_dict[node_type].df[node_id].to_numpy()
    current_bags =  [[int(i)] for i in driver_ids_df if train_mask[i]]
    old_y = data[node_type].y.int().tolist()
    current_labels = []
    for i in range(0, len(old_y)):
        if train_mask[i]:
            current_labels.append(old_y[i])
    assert len(current_bags) == len(current_labels)
    alpha = {int(i): 1.0 for i in torch.where(train_mask)[0]}
    all_path_info = [] 
    local_path = []
    
    current_paths = [[]] 
    for level in range(L_max):
        print(f"level {level}")
        
        next_paths_info = []

        for path in current_paths: 
            last_ntype = node_type if not path else path[2]
            print(f"current source node is {last_ntype}")
           
            candidate_rels = [ 
                (src, rel, dst)
                for (src, rel, dst) in data.edge_index_dict.keys()
                if src == last_ntype
            ][:max_rels]

            best_rel = None
            best_score = -math.inf if higher_is_better else math.inf  
            best_alpha = None
            best_bags = None
            best_labels = None

            for rel in candidate_rels: 
                print(f"considering relation {rel}")
                src, _, dst = rel
                if dst in [step[0] for step in path] or dst == node_type:  # avoid loops in met, avoid to return to the source node
                  continue
                if rel == ('races', 'rev_f2p_raceId', 'standings'): # for some reasons it provokes side assertions
                  continue

                node_embeddings = node_embeddings_dict.get(dst) #access at the value (Tensor[dst, hidden_dim]) for key node type "dst"
                theta = nn.Linear(node_embeddings.size(-1), 1).to(device) #classifier which is used to compute Θᵗx_v
                bags, labels, alpha_next = construct_bags_with_alpha(
                    data=data,
                    previous_bags=current_bags,
                    previous_labels=current_labels,
                    alpha_prev=alpha, #current alfa values for v nodes
                    rel=rel,
                    theta=theta,
                    src_embeddings = node_embeddings_dict[src]
                )
                if len(bags) < 5:
                    continue

                local_path2 = local_path.copy()
                #even if it is not the best one we memorize it because maybe will be selected from beam search:

                local_path2.append(rel)
                loc = [local_path2.copy()]
                model = MPSGNN(
                    data=data,
                    col_stats_dict=col_stats_dict,
                    metadata=data.metadata(),
                    metapath_counts = metapath_counts,
                    metapaths=loc,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    final_out_channels=1,
                ).to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
                test_table = task.get_table("test", mask_input_cols=False)
                best_test_metrics = -math.inf if higher_is_better else math.inf
                for _ in range(0, epochs):
                    train(model, optimizer, loader_dict=loader_dict, device=device, task=task, loss_fn=loss_fn)
                    test_pred = test(model, loader_dict["test"], device=device, task=task)
                    test_metrics = evaluate_performance(test_pred, test_table, task.metrics, task=task)
                    if test_metrics[tune_metric] > best_test_metrics and higher_is_better:
                        best_test_metrics = test_metrics[tune_metric]
                    if test_metrics[tune_metric] < best_test_metrics and not higher_is_better:
                        best_test_metrics = test_metrics[tune_metric]
                print(f"For the partial metapath {local_path2.copy()} we obtain F1 test loss equal to {best_test_metrics}")
                all_path_info.append((best_test_metrics, local_path2.copy()))
                score = best_test_metrics #score now is directly the F1 score returneb by training the model on that metapath

                if score > best_score: #higher is better
                    best_rel = rel
                    best_score = score
                    best_alpha = alpha_next
                    best_bags = bags
                    best_labels = labels

            #set best_rel:
            if best_rel:
                local_path.append(best_rel)
                print(f"Best relation is {best_rel} and now local path is {local_path}")
                next_paths_info.append((best_score, local_path, best_bags, best_labels, best_alpha))
                metapath_counts[tuple(local_path)] += 1
                

        
        current_paths = [best_rel] 
        #print(f"current path now is equal to {current_paths}\n")
    
    best_score_per_path = {}
    for score, path in all_path_info:
        path_tuple = tuple(path)
        if path_tuple not in best_score_per_path:
            best_score_per_path[path_tuple] = score
    sorted_unique_paths = sorted(best_score_per_path.items(), key=lambda x: x[1], reverse=True)#higher is better
    selected_metapaths = [list(path_tuple) for path_tuple, _ in sorted_unique_paths[:number_of_metapaths]]
    #print(f"\nfinal metapaths are {selected_metapaths}\n")

    return selected_metapaths, metapath_counts