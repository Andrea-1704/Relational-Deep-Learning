"""
This is an innovative implementation that has the aim of 
improving the time required to find the metapaths.

It uses a Reinforcement Learning technique which is pre-
trained (no end-to-end fashion) and then used in order to 
avoid the surrogate task presented in https://arxiv.org/abs/2412.00521
by using instead the agent of the RL technique to choose 
r* at each level.
"""

import torch
import math
from typing import List, Tuple, Dict
from relbench.modeling.nn import HeteroEncoder
from collections import defaultdict
from utils.utils import evaluate_performance, test, train
from model.XMetapath_Model import XMetapath


class RLAgent:
    def __init__(self, tau=1.0, alpha=0.5):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.tau = tau
        self.alpha = alpha
        self.best_score_by_path_global: Dict[Tuple[Tuple[str,str,str], ...], float] = {}  #dict {metapath->score}
        self.statistics_on_mp :  Dict[Tuple[Tuple[str,str,str], ...], int] = {}

    def select_relation(self, state, candidate_rels):
        state_key = tuple(state)
        q = torch.tensor([self.q_table[state_key][r] for r in candidate_rels], dtype=torch.float32)
        tau = max(self.tau, 1e-8)
        logits = q / tau
        logits -= logits.max()  # stabilità
        probs = torch.softmax(logits, dim=0)
        idx = torch.multinomial(probs, 1).item()
        return candidate_rels[idx]
    
    #to return more than one metapath:
    def register_path_score(self, path_list, score: float, higher_is_better: bool):
        key = tuple(path_list)
        prev = self.best_score_by_path_global.get(key) #previous score for same metapath
      
        if prev is None:#if it is the first time we encounter that metapath
            self.best_score_by_path_global[key] = float(score)
            #increment the counter of metapaths:
            self.statistics_on_mp[key] = 1
        else:
            #increment the counter of metapaths:
            self.statistics_on_mp[key] = self.statistics_on_mp[key] + 1
    
            if higher_is_better and score > prev:
                self.best_score_by_path_global[key] = float(score)
            elif (not higher_is_better) and score < prev:
                self.best_score_by_path_global[key] = float(score)

    def update(self, state, rel, reward):
        state_key = tuple(state)
        old_q = self.q_table[state_key][rel]
        self.q_table[state_key][rel] = old_q + self.alpha * (reward - old_q)

    def save(self, path="rl_agent_qtable.pt"):
        torch.save(dict(self.q_table), path)

    def load(self, path="rl_agent_qtable.pt"):
        loaded = torch.load(path, map_location="cpu")
        self.q_table = defaultdict(lambda: defaultdict(float))
        for k, v in loaded.items():
            self.q_table[k] = defaultdict(float, v)



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
    previous_bags: List[List[int]], 
    previous_labels: List[float],     # list of the "v" nodes
    rel: Tuple[str, str, str],
) -> Tuple[List[List[int]], List[float]]:
    """
    Estend the bags through relation "rel"
    Returns:
    - new bag 
    - labels associated to nodes v
    """
    edge_index = data.edge_index_dict.get(rel)
    if edge_index is None:
        print(f"this should not have happened, but the relation was not found.")
        return [], []

    edge_src, edge_dst = edge_index #tensor [2, #edges]
    bags = [] #the new bags, one for each "v" node.
    labels = [] #for each bag we consider its label, given by the one of the src in relation r.

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

            for u in neighbors_u.tolist():  #consider all the "sons" of node "v" through relation "rel"
                bag_u.append(u)

        if len(bag_u) > 0:
            bags.append(bag_u) #updates the new list of bags
            labels.append(label) #the label of the current bag is the same 
            #as the one that the father bag had.

    return bags, labels




def greedy_metapath_search_rl(
    data,
    loader_dict,
    task,
    loss_fn,
    tune_metric,
    higher_is_better,
    train_mask,
    node_type,
    col_stats_dict,
    agent:RLAgent,  # agente RL esterno: is already warmed up!
    L_max=3,
    hidden_channels=128,
    out_channels=128,
    final_out_channels=1,
    epochs=100,
    lr : float = 0.0001,
    wd=0.0,
    epsilon:float = 0.2
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ids = db.table_dict[node_type].df[node_id].to_numpy()
    # current_bags = [[int(i)] for i in ids if train_mask[i]]
    # current_labels = [int(data[node_type].y[i]) for i in range(len(train_mask)) if train_mask[i]]
    # NEW (fix PK→internal indices): build bags/labels from internal indices aligned with masks/edge_index
    idxs = torch.where(train_mask)[0].tolist()             # internal indices 0..num_nodes-1
    current_bags   = [[int(i)] for i in idxs]              # seed one-node bag per training node
    current_labels = [float(data[node_type].y[i]) for i in idxs]  # keep float to support regression
    #NOTA CHE QUESTA VERIFICA E' INUTILE ALMENO PER F1, QUINDI VOLENDO SI POTEVA USARE ANCHE LA VERSIONE COMMENTATA

    metapath_counts = defaultdict(int)
    all_path_info = []
    current_path = []
    current_best_val = -math.inf if higher_is_better else math.inf

    
    val_table = task.get_table("val")

    #Building metapath using reinforcement leanring
    for level in range(L_max):
        print(f"Step {level} - metapath so far: {current_path}")
        last_ntype = node_type if not current_path else current_path[-1][2]
        print(f"Now we are starting from node type {last_ntype}")

        # candidate_rels = [
        #     (src, rel, dst)
        #     for (src, rel, dst) in data.edge_index_dict
        #     if src == last_ntype and dst != node_type and dst not in [r[0] for r in current_path]
        # ]
        # NEW (correct filter): avoid reusing destination types already in the path
        used_dst = {r[2] for r in current_path}
        candidate_rels = [
            (src, rel, dst)
            for (src, rel, dst) in data.edge_index_dict
            if src == last_ntype and dst != node_type and dst not in used_dst
        ]


        if not candidate_rels:
            print("No more candidates.")
            break

        #The agent selects next relation, r* (no surrogate task)
        chosen_rel = agent.select_relation(current_path, candidate_rels)
        print(f"The RL agent has chosen the relation {chosen_rel}")

        #bags expansion for chosen relation
        bags, labels = construct_bags(
            data=data,
            previous_bags=current_bags,
            previous_labels=current_labels,
            rel=chosen_rel,
        )

        if len(bags) < 5:
            agent.update(current_path, chosen_rel, -1.0) #penalty
            print(f"Skipping relation {chosen_rel} (too few valid bags)")
            continue

        #testing on validation after training the MPS GNN 
        mp_candidate = current_path + [chosen_rel]
        print(f"The RL agent chosen r* {chosen_rel} to be added to metapath {current_path}, now we pass to the XMEtapath Model to test it")
        model = XMetapath(
            data=data,
            col_stats_dict=col_stats_dict,
            metapaths=[mp_candidate],
            metapath_counts=agent.statistics_on_mp,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            final_out_channels=final_out_channels,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        #initialize best_val, which keeps track of the performances (best) given by adding the current path to mp:
        best_val = -math.inf if higher_is_better else math.inf

        for _ in range(epochs):
            train(model, optimizer, loader_dict, device, task, loss_fn)
            val_pred = test(model, loader_dict["val"], device=device, task=task)
            val_score = evaluate_performance(val_pred, val_table, task.metrics, task=task)[tune_metric]
            if higher_is_better:
                best_val = max(best_val, val_score)
            else:
                best_val = min(best_val, val_score)
        
        # NEW: log this candidate metapath’s performance into the agent’s global map
        print(f"After testing, the result of metapath {mp_candidate} is {best_val} in the validation set.")
        agent.register_path_score(mp_candidate, best_val, higher_is_better)


        #now we check if current update in path is harmful:
        if higher_is_better:
            if best_val < current_best_val - epsilon * current_best_val:
                #we should stop here the construction of the metapath:
                print("We are stopping here because the last path did not improve")
                agent.update(current_path, chosen_rel, best_val)
                break
            elif best_val > current_best_val:
                #only if this chosen_rel improve performances on gnn we add it to current path
                current_best_val = best_val
                agent.update(current_path, chosen_rel, best_val)
                current_path.append(chosen_rel)
                current_bags, current_labels = bags, labels
                metapath_counts[tuple(current_path)] += 1
                all_path_info.append((best_val, current_path.copy()))
                print(f"For the partial metapath {current_path.copy()} we obtain F1 test loss equal to {best_val}, added {chosen_rel}")
        else:
            # lower-is-better (es. MAE)
            if best_val > current_best_val + epsilon * current_best_val:
                agent.update(current_path, chosen_rel, best_val)
                print(f"Early stop: {chosen_rel} increases loss from {current_best_val:.6f} to {best_val:.6f} (eps={epsilon:.3f}).")
                break
            elif best_val < current_best_val:
                current_best_val = best_val
                agent.update(current_path, chosen_rel, best_val)
                current_path.append(chosen_rel)
                current_bags, current_labels = bags, labels
                metapath_counts[tuple(current_path)] += 1
                all_path_info.append((best_val, current_path.copy()))
                print(f"For the partial metapath {current_path.copy()} we obtain validation loss {best_val:.6f}; added {chosen_rel}")


    return current_path, metapath_counts


#pre training tecnique for the agent of the RL
def warmup_rl_agent(
    agent,
    data,
    loader_dict,
    task,
    loss_fn,
    tune_metric,
    higher_is_better,
    train_mask,
    node_type,
    col_stats_dict,
    num_episodes=5,
    L_max=2,
    epochs=5
):
    print(f"Starting RL agent warm-up for {num_episodes} episodes...")
    for i in range(num_episodes):
        print(f"\n[Warmup Episode {i+1}]")
        _ = greedy_metapath_search_rl(
            agent=agent,
            data=data,
            loader_dict=loader_dict,
            task=task,
            loss_fn=loss_fn,
            tune_metric=tune_metric,
            higher_is_better=higher_is_better,
            train_mask=train_mask,
            node_type=node_type,
            col_stats_dict=col_stats_dict,
            L_max=L_max,
            epochs=epochs,
        )
    print(f"Now the agent is warmed up!!")




"""
Update in order to obtain "number_of_metapaths" metapaths. These metapaths are not
partial version of the "complete" one. I mean that we do not have a complete 
metapath A->B->C->E and then some partial metapaths, but different metapaths 
are returned.
"""

# XMetapath_extension4.py  — sostituisci la tua final_metapath_search_with_rl


def _prefix_overlap(a, b):
    #simple similarity metric: common prefix 
    m = min(len(a), len(b))
    k = 0
    for i in range(m):
        if a[i] == b[i]:
            k += 1
        else:
            break
    return k / m if m > 0 else 0.0

def _is_prefix(a, b):
    # True if a is prefix of b or viceversa.
    if len(a) <= len(b):
        return a == b[:len(a)]
    else:
        return b == a[:len(b)]

def _mmr_select_topk(candidates, scores, k, lam=0.5, sim_fn=_prefix_overlap):
    # candidates: List[Tuple[Triple,...]] ; scores: Dict[path_tuple] -> float
    items = [(tuple(p), scores[tuple(p)]) for p in candidates]
    #sort by score
    items.sort(key=lambda kv: kv[1], reverse=True)

    selected = []
    while items and len(selected) < k:
        best_idx = None
        best_obj = -float("inf")
        for idx, (p, s) in enumerate(items):
            # vincolo "no sottoinsiemi/prefissi"
            if any(_is_prefix(p, q) or _is_prefix(q, p) for q in selected):
                continue
            div = 0.0 if not selected else max(sim_fn(p, q) for q in selected)
            obj = s - lam * div
            if obj > best_obj:
                best_obj, best_idx = obj, idx
        if best_idx is None:
            break
        sel_p, _ = items.pop(best_idx)
        selected.append(sel_p)
    return [list(p) for p in selected]

def final_metapath_search_with_rl(
    agent,
    data,
    loader_dict,
    task,
    loss_fn,
    tune_metric,
    higher_is_better,
    train_mask,
    node_type,
    col_stats_dict,
    L_max=3,
    epochs=100,
    number_of_metapaths=5,
    episodes_for_search=None,     #how many episods to run in the final
    lambda_diversity=0.5,         #trade off: score vs diversity
    require_full_length=True,     #if True we only require metapths of length L_max
    reset_registry=True           #remove candidates from the warm up
):
    print("\n\n Launching final metapath search using trained RL agent \n")

    if reset_registry:
        agent.best_score_by_path_global.clear()
        agent.statistics_on_mp.clear()


    #We need to executes more episodes to find many candidates
    if episodes_for_search is None:
        episodes_for_search = max(3 * number_of_metapaths, 10)

    for ep in range(episodes_for_search):
        print(f"[Final RL — episode {ep+1}/{episodes_for_search}]")
        _path, _counts = greedy_metapath_search_rl(
            data=data,
            loader_dict=loader_dict,
            task=task,
            loss_fn=loss_fn,
            tune_metric=tune_metric,
            higher_is_better=higher_is_better,
            train_mask=train_mask,
            node_type=node_type,
            col_stats_dict=col_stats_dict,
            agent=agent,
            L_max=L_max,
            epochs=epochs,
        )
        # greedy registers candidates and relative scores in agent.best_score_by_path_global

    #candidates from agent's register
    best_map = agent.best_score_by_path_global  # Dict[Tuple[Triple,...]] -> float

    #filter full length
    if require_full_length:
        candidates = [list(p) for p in best_map.keys() if len(p) == L_max]
    else:
        candidates = [list(p) for p in best_map.keys()]

    if not candidates:
        print("WARNING: no full-length metapaths discovered; falling back to any length.")
        candidates = [list(p) for p in best_map.keys()]
        if not candidates:
            return [], defaultdict(int)

    #Top K selection with filter on length and similarity
    selected = _mmr_select_topk(
        candidates=candidates,
        scores=best_map,
        k=number_of_metapaths,
        lam=lambda_diversity,
        sim_fn=_prefix_overlap
    )

    # counters
    metapath_counts = defaultdict(int)
    for p in selected:
        metapath_counts[tuple(p)] += 1

    return selected, metapath_counts
