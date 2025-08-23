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
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict
from relbench.modeling.nn import HeteroEncoder
from collections import defaultdict
from utils.utils import evaluate_performance, test, train
from model.XMetapath_Model import XMetapath
# NEW: -------- Top-K merge utilities and parallel warm-ups --------
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy, random, numpy as np


class RLAgent:
    def __init__(self, tau=1.0, alpha=0.5):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.tau = tau
        self.alpha = alpha
        # NEW: per-run registry of best scores per metapath
        self.best_score_by_path_global: Dict[Tuple[Tuple[str,str,str], ...], float] = {}

    # NEW: reset registry at the start of a run
    def reset_scores(self):
        self.best_score_by_path_global.clear()
    
    # NEW: register/update best score for a metapath
    def register_path_score(self, path_list, score: float, higher_is_better: bool):
        key = tuple(path_list)
        prev = self.best_score_by_path_global.get(key)
        if prev is None:
            self.best_score_by_path_global[key] = float(score)
        else:
            if higher_is_better and score > prev:
                self.best_score_by_path_global[key] = float(score)
            elif (not higher_is_better) and score < prev:
                self.best_score_by_path_global[key] = float(score)

    def select_relation(self, state, candidate_rels):
        state_key = tuple(state)
        q = torch.tensor([self.q_table[state_key][r] for r in candidate_rels], dtype=torch.float32)
        tau = max(self.tau, 1e-8)
        logits = q / tau
        logits -= logits.max()  # stabilità
        probs = torch.softmax(logits, dim=0)
        idx = torch.multinomial(probs, 1).item()
        return candidate_rels[idx]

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
    src_embeddings,
) -> Tuple[List[List[int]], List[float], Dict[int, float]]:
    """
    Estend the bags through relation "rel"
    Returns:
    - new bag 
    - labels associated to nodes v
    """
    edge_index = data.edge_index_dict.get(rel)
    if edge_index is None:
        print(f"this should not have happened, but the relation was not found.")
        return [], [], {}

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
            
            x_v = src_embeddings[v]

            for u in neighbors_u.tolist():  #consider all the "sons" of node "v" through relation "rel"
                bag_u.append(u)

        if len(bag_u) > 0:
            bags.append(bag_u) #updates the new list of bags
            labels.append(label) #the label of the current bag is the same 
            #as the one that the father bag had.

    return bags, labels



"""
Update: we return a certain number of metapaths, the best "number_of_metapaths".
To do so we use a dictionary 'best_score_by_path ' mapping every metapath to the score:
best_score_by_path = {}  # dict: tuple(path) -> float

Just after computing the best score for a path we memorize the performances in the map.
"""
def greedy_metapath_search_rl(
    data,
    db,
    node_id,
    loader_dict,
    task,
    loss_fn,
    tune_metric,
    higher_is_better,
    train_mask,
    node_type,
    col_stats_dict,
    agent,  # agente RL esterno: is already warmed up!
    L_max=3,
    number_of_metapaths=5,
    hidden_channels=128,
    out_channels=128,
    final_out_channels=1,
    epochs=100,
    lr : float = 0.0001,
    wd=0.0,
    epsilon:float = 0.05
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        encoder = HeteroEncoder(
            channels=hidden_channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats_dict,
        ).to(device)
        tf_dict = {nt: data[nt].tf.to(device) for nt in data.node_types if 'tf' in data[nt]}
        node_embeddings_dict = encoder(tf_dict)

    # NEW (sanity checks): sizes aligned to internal node count
    assert data[node_type].y.numel() == data[node_type].num_nodes, f"y size mismatch for {node_type}"
    assert train_mask.numel() == data[node_type].num_nodes, f"train_mask size mismatch for {node_type}"
    # (optional) verify edge index ranges once
    for (src, rel, dst), (edge_src, edge_dst) in data.edge_index_dict.items():
        assert edge_src.min().item() >= 0 and edge_src.max().item() < data[src].num_nodes, f"edge_src out of range for {(src,rel,dst)}"
        assert edge_dst.min().item() >= 0 and edge_dst.max().item() < data[dst].num_nodes, f"edge_dst out of range for {(src,rel,dst)}"
    

    # ids = db.table_dict[node_type].df[node_id].to_numpy()
    # current_bags = [[int(i)] for i in ids if train_mask[i]]
    # current_labels = [int(data[node_type].y[i]) for i in range(len(train_mask)) if train_mask[i]]
    # NEW (fix PK→internal indices): build bags/labels from internal indices aligned with masks/edge_index
    idxs = torch.where(train_mask)[0].tolist()             # internal indices 0..num_nodes-1
    current_bags   = [[int(i)] for i in idxs]              # seed one-node bag per training node
    current_labels = [float(data[node_type].y[i]) for i in idxs]  # keep float to support regression

    metapath_counts = defaultdict(int)
    all_path_info = []
    current_path = []
    current_best_val = -math.inf if higher_is_better else math.inf

    
    val_table = task.get_table("val")

    #Building metapath using reinforcement leanring
    for level in range(L_max):
        print(f"Step {level} - metapath so far: {current_path}")
        last_ntype = node_type if not current_path else current_path[-1][2]

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
        print(f"The current chosen relation by RL is {chosen_rel}")

        #bags expansion
        bags, labels = construct_bags(
            data=data,
            previous_bags=current_bags,
            previous_labels=current_labels,
            rel=chosen_rel,
            src_embeddings=node_embeddings_dict[chosen_rel[0]]
        )

        if len(bags) < 5:
            agent.update(current_path, chosen_rel, -1.0) #penalty
            print(f"Skipping relation {chosen_rel} (too few valid bags)")
            continue

        #testing on validation after training the MPS GNN 
        mp_candidate = current_path + [chosen_rel]
        print(f"Passing to model following metapaths: {mp_candidate}")
        model = XMetapath(
            data=data,
            col_stats_dict=col_stats_dict,
            metapaths=[mp_candidate],
            metapath_counts=metapath_counts,
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
        agent.register_path_score(mp_candidate, best_val, higher_is_better)


        #now we check if current update in path is harmful:
        if higher_is_better:
            if best_val < current_best_val - epsilon * current_best_val:
                #we should stop here the construction of the metapath:
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
    db,
    node_id,
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
            data=data,
            db=db,
            node_id=node_id,
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
            number_of_metapaths=1,  # serve solo a raccogliere reward
        )


#final call with agent already warmed up
def final_metapath_search_with_rl(
    agent,
    data,
    db,
    node_id,
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
    number_of_metapaths=5
):
    print("\n\n Launching final metapath search using trained RL agent \n")
    selected_metapaths, metapath_counts = greedy_metapath_search_rl(
        data=data,
        db=db,
        node_id=node_id,
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
        number_of_metapaths=number_of_metapaths,
    )
    return selected_metapaths, metapath_counts




# NEW: merge multiple {path->score} dicts keeping the best score per path
def merge_best_maps(maps: List[Dict[Tuple[Tuple[str,str,str], ...], float]],
                    higher_is_better: bool) -> Dict[Tuple[Tuple[str,str,str], ...], float]:
    agg: Dict[Tuple[Tuple[str,str,str], ...], float] = {}
    for m in maps:
        for p, s in m.items():
            if (p not in agg) or (higher_is_better and s > agg[p]) or ((not higher_is_better) and s < agg[p]):
                agg[p] = float(s)
    return agg


# NEW: take Top-K paths from a {path->score} dict
def topk_from_best(agg: Dict[Tuple[Tuple[str,str,str], ...], float], K: int, higher_is_better: bool):
    items = sorted(agg.items(), key=lambda kv: kv[1], reverse=higher_is_better)[:K]
    topK_paths  = [list(p) for p, _ in items]
    topK_scores = [s for _, s in items]
    return topK_paths, topK_scores


# NEW: seed helper for reproducibility
def _set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# NEW: one independent warm-up run (thread job)
def _warmup_job(base_args: dict, make_agent_fn: Callable[[], RLAgent], seed: int):
    args = copy.deepcopy(base_args)
    _set_all_seeds(seed)
    agent = make_agent_fn()
    agent.reset_scores()
    args["agent"] = agent
    # run your existing search (unchanged signature/return)
    greedy_metapath_search_rl(**args)
    # collect this run’s registry
    return agent.best_score_by_path_global


# NEW: run multiple independent warm-ups in parallel threads and return global Top-K
def run_warmups_parallel_and_merge(
    base_args: dict,
    make_agent_fn: Callable[[], RLAgent],
    seeds: List[int],
    number_of_runs: int,
    number_of_metapaths: int,
    higher_is_better: bool,
    max_workers: int = None,
):
    """
    Run multiple independent warm-ups (each with its own agent/seed) in parallel threads,
    merge all candidate metapaths, and return global Top-K.
    """
    assert len(seeds) >= number_of_runs, "Provide at least one seed per run."
    maps = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_warmup_job, base_args, make_agent_fn, seeds[i]) for i in range(number_of_runs)]
        for f in as_completed(futs):
            maps.append(f.result())
    global_best = merge_best_maps(maps, higher_is_better)
    topK_paths, topK_scores = topk_from_best(global_best, number_of_metapaths, higher_is_better)
    return topK_paths, topK_scores, global_best