import torch
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
from relbench.modeling.nn import HeteroEncoder



def binarize_targets(y: torch.Tensor, threshold: float = 10) -> torch.Tensor:
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



def construct_bags(
    data,
    train_mask: torch.Tensor,
    y: torch.Tensor,
    rel: Tuple[str, str, str],
    node_type: str,
) -> Tuple[List[List[int]], List[float]]:
    """
    This function returns the bags for relation "rel" and the correspondings labels. 
    """
    src, rel_name, dst = rel
    if (src, rel_name, dst) not in data.edge_index_dict: #this should be counted as error
        print(f"edge type {rel} not found")
        return [], []

    edge_index = data.edge_index_dict[(src, rel_name, dst)]#taks the edges of that type
    bags = [] #inizialize the bag 
    labels = [] #inizialize the label for each bag (patient -> prescription, we construct
    #bag B for patient p1 which is healthy-> so B is linked to label healthy)

    for i in torch.where(train_mask)[0]:
      #train mask is constructed at the beginning of the train and is simply a boolean vector 
      #of the same size of drivers (target in general) nodes and contains true if that node 
      #is in the train split and we know its label and has some neighbours.
        node_id = i.item()
        neighbors = edge_index[1][edge_index[0] == node_id]
        #we construct the bag considering the rel kind of relation staring from src
        if len(neighbors) > 0:
            bags.append(neighbors.tolist())
            labels.append(y[node_id].item())

    return bags, labels




class ScoringFunctionReg(nn.Module):
    """
    This function is one of possibly infinite different implementation for 
    computing how "significative" is a bag.
    In particular, this approach, which follows https://arxiv.org/abs/2412.00521,
    uses a mini neural network taht takes an embedding and produces a score value.
    Each bag is a list of embeddings of the reached nodes at a specific time step
    (each of these nodes share the same node type) and we desire to return a score 
    values to the bag.

    We first apply the theta NN to each of the embeddings of the nodes of the bag, 
    getting its score. 
    Then, we normalize the scores through softmax function in order to obtain the 
    attention weights, these score values corresponds to the "α(v, B)" computed
    by https://arxiv.org/abs/2412.00521 in section 4.1, and formally indicates 
    how much attention we should "dedicate" to a node of the bag.

    Then, followign the formulation indicated in section 4.1 of the aforementioned
    paper, we simply compute a weighted mean of the embeddings of the nodes in the
    bag.

    Finally, we pass the embeddings of the bag to another NN which computes a
    single prediction score for the bag.

    We are using these two networks to "predict" whether the current bag is 
    able to capture important signals about the predictive label.... DOES THIS MAKE
    SENSE????
    """
    def __init__(self, in_dim: int): #in_dim is the dimension of the embedding of nodes
        super().__init__()
        self.theta = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1)  # from embedding to scalar
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
            scores = self.theta(bag).squeeze(-1)  # [B_i] #alfa scores
            weights = torch.softmax(scores, dim=0)  # [B_i] #normalize alfa
            weighted_avg = torch.sum(weights.unsqueeze(-1) * bag, dim=0)  # [D] #mean
            # pred = weighted_avg.mean()  #final scalar -> terrible solution!
            pred = self.out(weighted_avg).squeeze(-1)
            preds.append(pred)
        return torch.stack(preds)

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the m1 score between the two vectors.
        """
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
