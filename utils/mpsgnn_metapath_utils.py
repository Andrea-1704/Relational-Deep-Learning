import torch
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
from relbench.modeling.nn import HeteroEncoder
from collections import defaultdict



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




#The previous bag creation function had an important error: it always consider ad node_id
#the first node, the target node. So this function does not provide a solid solution for 
#all the relations in the metapath, after the first one (for which, we have instead 
#a source node that is equal to the target node).
#to solve this major mistake we now provide a different solution that aims to solve the 
#mentioned error, but also to align more closely to section 4.2 of the aforementioned 
#paper by implementing in an integral way the alfa scores values calculation for all
#the node "u" present in the bag, using a recursive function that takes into account
#the "v" nodes of the previous bags.
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




def greedy_metapath_search_with_bags_learned(
    data,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    node_type: str, #target node, "driver" for example
    col_stats_dict: Dict[str, Dict[str, Dict]],  # per HeteroEncoder
    L_max: int = 3,
    max_rels: int = 10,
    channels : int = 64,
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
    """
    device = y.device
    metapaths = [] #returned object
    metapath_counts = defaultdict(int)
    current_paths = [[]] #current partial paths that we are going to expand 

    current_bags = [[int(i)] for i in torch.where(train_mask)[0]] 
    #at the first step, the bags are simply a list of list values, where each list contained inside the 
    #list is the id the driver index node, if that driver is in the train_mask mask.
    current_labels = [y[i].item() for i in torch.where(train_mask)[0]]
    alpha = {int(i): 1.0 for i in torch.where(train_mask)[0]}

    for level in range(L_max): #cycle in the level of metapath
        print(f"level {level}")
        new_paths = []
        new_alpha_all = [] 
        new_bags_all = []
        new_labels_all = []

        for path in current_paths:
            last_ntype = node_type if not path else path[-1][2]
            print(f"current source node is {last_ntype}")
            #if the current next node is empty, start from the target node ("driver")

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
              }#get the features of nodes
              #tf_dict is a dictionary:
              # [node type]:Tensor Frame object containing the columns of such node
              
              
              node_embeddings_dict = encoder(tf_dict)
              #HeteroEncoder takes as input the tf_dict object mentioned before and return 
              #a dictionary in which for each type of node (key of the dictionary) 
              #contains the embeddings for all the nodes of that type: 
              # [node_type]: Tensor[node_type, hidden_dim]

            candidate_rels = [
                (src, rel, dst)
                for (src, rel, dst) in data.edge_index_dict.keys()
                if src == last_ntype
            ][:max_rels] #get "max_rels" relations that start from current
            #node ("last_ntype").

            best_rel = None
            best_score = float("inf")
            best_alpha = None
            best_bags = None
            best_labels = None

            for rel in candidate_rels: 
                print(f"considering relation {rel}")
                src, _, dst = rel
                if dst in [step[0] for step in path]:  # avoid loops in met.
                  continue
                if dst == node_type:
                  continue  # avoid to return to the source node

                node_embeddings = node_embeddings_dict.get(dst) 
                #access at the value (Tensor[dst, hidden_dim]) for key node type "dst"

                if node_embeddings is None:
                    print(f"error: embedding of node {dst} not found")
                    continue

                theta = nn.Linear(node_embeddings.size(-1), 1).to(device) #classifier which is used to compute Θᵗx_v

                bags, labels, alpha_next = construct_bags_with_alpha(
                    data=data,
                    previous_bags=current_bags,
                    previous_labels=current_labels,
                    alpha_prev=alpha, #current alfa values for v nodes
                    rel=rel,
                    node_embeddings=node_embeddings,
                    theta=theta,
                    src_embeddings = node_embeddings_dict[src]
                )

                if len(bags) < 5:
                    continue#this avoid to consider few bags to avoid overfitting

                score = evaluate_relation_learned(bags, labels, node_embeddings) #assign the 
                #score value to current split, similar to DECISION TREES
                print(f"relation {rel} allow us to obtain score {score}")

                if score < best_score:
                    best_score = score
                    best_rel = rel
                    best_alpha = alpha_next
                    #best_nodes = list(set([u for bag in bags for u in bag]))  --> severe error!!!
                    best_bags = bags
                    best_labels = labels


            if best_rel:
                new_paths.append(path + [best_rel]) #add the best_rel to path
                metapath_counts[tuple(path+[best_rel])] += 1
                print(f"The best relation found is {best_rel}")
                new_alpha_all.append(best_alpha) 
                #NB: the best_alpha are the alpha scores returned from the best current relation 
                #"rel" that was found. It is a dictionary that has as keys all the values 
                #of the u nodes and as values the alpha values of those nodes.
                #new_alpha_all is then a list of these dictionaries, where each dictionary
                #contains one key for each of the u nodes, and this is done for each relation
                #in the metapath. In practice, we have a list of elements of the same length as
                #the number of relations in the metapath, and for each of them we have a 
                #dictionary containing for each source node "u" the alpha value.
                #new_nodes_all.append(best_nodes) 

                new_bags_all.extend(best_bags)
                new_labels_all.extend(best_labels)
                #plesase note that these list are inizialized for aeche level indipendently

                 

        current_paths = new_paths
        alpha = {k: v for d in new_alpha_all for k, v in d.items()} #update the alfa values as
        #the last values
        # current_nodes = list(set([u for l in new_nodes_all for u in l])) #update the "u" nodes
        # metapaths.extend(current_paths)
        current_bags = new_bags_all
        current_labels = new_labels_all
        metapaths.extend(current_paths)
        print(f"final metapaths are {metapaths}")

    return metapaths, metapath_counts



def beam_metapath_search_with_bags_learned(
    data,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    node_type: str, 
    col_stats_dict: Dict[str, Dict[str, Dict]], 
    L_max: int = 3,
    max_rels: int = 10,
    channels : int = 64,
    beam_width: int = 5, #number of metapaths to look for
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
    device = y.device
    metapaths = []
    metapath_counts = {} #for each metapath counts how many bags are presents, so how many istances of that metapath are present
    current_paths = [[]]
    current_bags = [[int(i)] for i in torch.where(train_mask)[0]] 
    current_labels = [y[i].item() for i in torch.where(train_mask)[0]]
    alpha = {int(i): 1.0 for i in torch.where(train_mask)[0]}
    all_path_info = [] #memorize all the metapaths with scores, in order
    #to select only the best beam_width at the end

    for level in range(L_max):
        print(f"we are at level {level}")
        #candidate_path_info = []
        next_paths_info = []

        for path in current_paths:
            last_ntype = node_type if not path else path[-1][2]
            print(f"current source node is {last_ntype}")

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

            candidate_rels = [
                (src, rel, dst)
                for (src, rel, dst) in data.edge_index_dict.keys()
                if src == last_ntype
            ][:max_rels] 

            for rel in candidate_rels: 
                print(f"considering relation {rel}")
                src, _, dst = rel
                if dst in [step[0] for step in path]:  # avoid loops in met.
                  continue
                if dst == node_type:
                  continue  # avoid to return to the source node
                node_embeddings = node_embeddings_dict.get(dst) 

                if node_embeddings is None:
                    print(f"error: embedding of node {dst} not found")
                    continue

                theta = nn.Linear(node_embeddings.size(-1), 1).to(device) 
                
                bags, labels, alpha_next = construct_bags_with_alpha(
                    data=data,
                    previous_bags=current_bags,
                    previous_labels=current_labels,
                    alpha_prev=alpha, 
                    rel=rel,
                    node_embeddings=node_embeddings,
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
        


