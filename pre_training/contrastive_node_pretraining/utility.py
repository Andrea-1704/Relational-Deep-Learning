import copy
from typing import Dict, Iterable, Optional
from typing import Any, Dict, List
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


class DGIHead(nn.Module):
    """
    Bilinear discriminator per DGI:
      score(z, s) = (W z) · s
    dove s è il summary (graph-level) e z è l'embedding dei nodi.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, z: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        # z: [N, D], summary: [D]
        # out: [N]
        return (self.W(z) * summary).sum(dim=-1)


@torch.no_grad()
def build_shallow_embeddings_if_missing(model: nn.Module,
                                        data,
                                        channels: int,
                                        device: torch.device,
                                        node_types: Optional[Iterable[str]] = None):
    """
    Aggiunge shallow embeddings per tipo di nodo, se non presenti.
    Usa num_nodes globali da `data[node_type].num_nodes`.
    """
    if not hasattr(model, "shallow_embeddings"):
        model.shallow_embeddings = nn.ModuleDict()

    if node_types is None:
        node_types = getattr(data, "node_types", [])

    for ntype in node_types:
        if ntype not in model.shallow_embeddings:
            num_nodes = int(data[ntype].num_nodes)
            emb = nn.Embedding(num_nodes, channels)
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            model.shallow_embeddings[ntype] = emb.to(device)


def compute_summary_from_z_dict(z_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Summary (graph-level) per DGI:
      - media per tipo di nodo
      - poi media tra i tipi
      Assumiamo stesso hidden_dim D per tutti i tipi (come nel tuo backbone).
    """
    per_type = []
    for ntype, z in z_dict.items():
        if z is None or z.numel() == 0:
            continue
        per_type.append(z.mean(dim=0))
    if len(per_type) == 0:
        raise RuntimeError("Empty z_dict in compute_summary_from_z_dict")
    summary = torch.stack(per_type, dim=0).mean(dim=0)  # [D]
    return torch.tanh(summary)

from typing import Dict, Optional, Tuple

def build_time_and_batch_dict(batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """
    Estrae:
      time_dict[ntype]  = batch[ntype].time (se presente)
      batch_dict[ntype] = batch[ntype]      (sub-batch PyG per ntype)
    """
    time_dict: Dict[str, torch.Tensor] = {}
    batch_dict: Dict[str, object] = {}
    for ntype in batch.node_types:
        sub = batch[ntype]
        batch_dict[ntype] = sub
        t = getattr(sub, "time", None)
        print(f"time attribute in build time and batch dict is {t}")
        if t is not None:
            time_dict[ntype] = t
    return time_dict, batch_dict


def _ensure_temp_adapters(model: nn.Module, x_dict: Dict[str, torch.Tensor],
                          rel_time_dict: Dict[str, torch.Tensor]):
    """
    Se l'encoder temporale restituisce una dimensionalità D_t diversa da 'channels',
    aggiunge (una volta sola) un adattatore lineare per ntype: R^{D_t} -> R^{channels}.
    """
    if not hasattr(model, "_temp_adapter"):
        model._temp_adapter = nn.ModuleDict()

    for ntype, x in x_dict.items():
        if ntype not in rel_time_dict or rel_time_dict[ntype] is None:
            continue
        D_x = x.shape[-1]
        D_t = rel_time_dict[ntype].shape[-1]
        if D_x != D_t:
            key = f"{ntype}__{D_t}_to_{D_x}"
            if key not in model._temp_adapter:
                adapter = nn.Linear(D_t, D_x, bias=False)
                nn.init.xavier_uniform_(adapter.weight)
                model._temp_adapter[key] = adapter.to(x.device)
            # applica l'adattatore sul posto
            rel_time_dict[ntype] = model._temp_adapter[key](rel_time_dict[ntype])


def pretrain_forward_embeddings(model: nn.Module,
                                batch,
                                entity_table: str,
                                use_shallow: bool = True,
                                override_n_id: Optional[Dict[str, torch.Tensor]] = None):
    """
    Vista 'encoder -> temporal -> (shallow) -> backbone' SENZA testa supervisionata.
    Ritorna (z_dict, x_dict_before_gnn).
    """
    # 1) Feature encoding (TorchFrame)
    x_dict = model.encoder(batch.tf_dict)  # Dict[ntype, Tensor[N, D]]

    # 2) Temporal encoding: la tua versione richiede time_dict e batch_dict
    if hasattr(model, "temporal_encoder") and model.temporal_encoder is not None:
        time_dict, batch_dict = build_time_and_batch_dict(batch)
        # chiamata nella firma attesa dalla tua HeteroTemporalEncoder
        rel_time_dict = model.temporal_encoder(time_dict, batch_dict)
        print(f"in pretrain rel time dict è {rel_time_dict}")
        if rel_time_dict is not None:
            # Se serve, adatta la dimensionalità (es. D_t != channels)
            _ensure_temp_adapters(model, x_dict, rel_time_dict)
            # somma contributo temporale
            for ntype in x_dict.keys():
                if ntype in rel_time_dict and rel_time_dict[ntype] is not None:
                    x_dict[ntype] = x_dict[ntype] + rel_time_dict[ntype]

    x_dict_before_gnn = {k: v for k, v in x_dict.items()}

    # 3) Shallow embeddings per rendere efficace la corruzione via n_id
    if use_shallow:
        if not hasattr(model, "shallow_embeddings") or len(model.shallow_embeddings) == 0:
            raise RuntimeError(
                "Shallow embeddings not found. "
                "Call build_shallow_embeddings_if_missing(...) before pretraining."
            )
        for ntype, x in x_dict.items():
            if override_n_id is not None and ntype in override_n_id:
                n_id = override_n_id[ntype]
            else:
                n_id = getattr(batch[ntype], "n_id", None)
            if n_id is None:
                raise RuntimeError(f"batch['{ntype}'].n_id not found; cannot use shallow embeddings.")
            shallow = model.shallow_embeddings[ntype](n_id.to(x.device))  # [N, D]
            x_dict[ntype] = x + shallow

    # 4) Backbone eterogeneo
    z_dict = model.gnn(x_dict, batch.edge_index_dict)
    return z_dict, x_dict_before_gnn



@torch.no_grad()
def build_corrupted_n_id(batch) -> Dict[str, torch.Tensor]:
    """
    Crea una vista corrotta: per ogni tipo di nodo, permuta *indipendentemente* gli n_id.
    Non tocchiamo il batch; passiamo questi indici a pretrain_forward_embeddings via 'override_n_id'.
    """
    override = {}
    for ntype in batch.node_types:
        n_id = getattr(batch[ntype], "n_id", None)
        if n_id is None or n_id.numel() == 0:
            continue
        perm = torch.randperm(n_id.size(0), device=n_id.device)
        override[ntype] = n_id[perm]
    return override





def dgi_loss_for_hetero(dgi_head: DGIHead,
                        z_pos: Dict[str, torch.Tensor],
                        z_neg: Dict[str, torch.Tensor],
                        summary: torch.Tensor) -> torch.Tensor:
    """
    Applica DGI per ogni tipo di nodo e media.
    loss = mean_{ntype} [ mean( -log σ(score_pos) - log(1 - σ(score_neg)) ) ]
    """
    losses = []
    for ntype in z_pos.keys():
        zp, zn = z_pos[ntype], z_neg[ntype]
        if zp is None or zn is None or zp.numel() == 0 or zn.numel() == 0:
            continue
        pos = dgi_head(zp, summary)         # [N]
        neg = dgi_head(zn, summary)         # [N]
        loss_nt = F.binary_cross_entropy_with_logits(
            torch.cat([pos, neg], dim=0),
            torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)
        )
        losses.append(loss_nt)
    if len(losses) == 0:
        raise RuntimeError("Empty z_pos/z_neg in dgi_loss_for_hetero")
    return torch.stack(losses, dim=0).mean()







def get_pretrain_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """
    Esclude la testa del task supervisionato dal pretraining.
    Include: encoder, temporal_encoder (se presente), gnn/backbone, shallow embeddings.
    """
    params = []
    if hasattr(model, "encoder") and model.encoder is not None:
        params += list(model.encoder.parameters())
    if hasattr(model, "temporal_encoder") and model.temporal_encoder is not None:
        params += list(model.temporal_encoder.parameters())
    if hasattr(model, "gnn") and model.gnn is not None:
        params += list(model.gnn.parameters())
    if hasattr(model, "shallow_embeddings") and len(model.shallow_embeddings) > 0:
        params += list(model.shallow_embeddings.parameters())
    return params


def pretrain_dgi(model: nn.Module,
                 data,
                 loader,                 # NeighborLoader sullo split TRAIN (eventi ≤ T_train)
                 entity_table: str,
                 hidden_dim: int,
                 device: torch.device,
                 epochs: int = 20,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 use_shallow: bool = True,
                 log_every: int = 50):
    """
    Esegue la fase di pretraining DGI in-domain.
    - Assicurati che `loader` legga SOLO dal training horizon (RelBench).
    - Usa shallow embeddings + permutazione di n_id per la corruzione.
    """
    model.to(device)
    model.train()

    # 1) Shallow embeddings (se richiesto)
    if use_shallow:
        build_shallow_embeddings_if_missing(model, data, channels=hidden_dim, device=device)

    # 2) Discriminatore DGI
    dgi_head = DGIHead(hidden_dim).to(device)

    # 3) Ottimizzatore (solo pretrain params + head DGI)
    pre_params = list(get_pretrain_parameters(model)) + list(dgi_head.parameters())
    optim = torch.optim.Adam(pre_params, lr=lr, weight_decay=weight_decay)

    global_step = 0
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for it, batch in enumerate(loader):
            batch = batch.to(device)

            # POSITIVO
            z_pos, _ = pretrain_forward_embeddings(
                model, batch, entity_table, use_shallow=use_shallow, override_n_id=None
            )
            summary = compute_summary_from_z_dict(z_pos)  # [D]

            # NEGATIVO: permuta n_id per tipo
            override = build_corrupted_n_id(batch)
            z_neg, _ = pretrain_forward_embeddings(
                model, batch, entity_table, use_shallow=use_shallow, override_n_id=override
            )

            # LOSS
            loss = dgi_loss_for_hetero(dgi_head, z_pos, z_neg, summary)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            epoch_loss += float(loss.item())
            global_step += 1

            if log_every and (it + 1) % log_every == 0:
                print(f"[DGI] epoch {ep:03d} iter {it+1:04d} | loss {epoch_loss/(it+1):.4f}")

        print(f"[DGI] epoch {ep:03d} | avg loss {epoch_loss / max(1, len(loader)):.4f}")

    # Al termine: i pesi dell'encoder/backbone sono aggiornati (head del task intatta).
    return model



class LinearProbeHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def extract_entity_embeddings(
    model: nn.Module,
    loader,
    entity_table: str,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    X_list, y_list = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z_dict, _ = pretrain_forward_embeddings(model, batch, entity_table, use_shallow=False)
            z = z_dict[entity_table]                          # [N, D]
            y = batch[entity_table].y.view(-1, 1).to(z.device)  # [N, 1]
            X_list.append(z.detach().cpu())
            y_list.append(y.detach().cpu())
    X = torch.cat(X_list, dim=0)
    Y = torch.cat(y_list, dim=0)
    return X, Y



def run_linear_probe(model: nn.Module,
                     train_loader,
                     val_loader,
                     test_loader,
                     entity_table: str,
                     hidden_dim: int,
                     device: torch.device,
                     lr: float = 1e-3,
                     weight_decay: float = 0.0,
                     epochs: int = 50):
    """
    Congela encoder/backbone e addestra una testa lineare (regressione).
    Ritorna MAE su val/test.
    """
    model.eval()
    for p in get_pretrain_parameters(model):
        p.requires_grad_(False)

    Xtr, Ytr = extract_entity_embeddings(model, train_loader, entity_table, device)
    Xva, Yva = extract_entity_embeddings(model, val_loader, entity_table, device)
    Xte, Yte = extract_entity_embeddings(model, test_loader, entity_table, device)

    device_lp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = LinearProbeHead(hidden_dim, out_dim=1).to(device_lp)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    Xtr, Ytr = Xtr.to(device_lp), Ytr.to(device_lp)
    Xva, Yva = Xva.to(device_lp), Yva.to(device_lp)
    Xte, Yte = Xte.to(device_lp), Yte.to(device_lp)

    best_val = float("inf")
    best_state = None
    for ep in range(1, epochs + 1):
        head.train()
        opt.zero_grad(set_to_none=True)
        pred = head(Xtr)
        loss = loss_fn(pred, Ytr)
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            val_mae = F.l1_loss(head(Xva), Yva).item()
            print(f"[LinearProbe] epoch {ep:03d} | train_loss {loss.item():.4f} | val_MAE {val_mae:.4f}")
            if val_mae < best_val:
                best_val = val_mae
                best_state = copy.deepcopy(head.state_dict())

    if best_state is not None:
        head.load_state_dict(best_state)

    with torch.no_grad():
        test_mae = F.l1_loss(head(Xte), Yte).item()
    print(f"[LinearProbe] best val_MAE {best_val:.4f} | test_MAE {test_mae:.4f}")
    return best_val, test_mae



def fine_tune_supervised(model: nn.Module,
                         train_loader,
                         val_loader,
                         test_loader,
                         entity_table: str,
                         device: torch.device,
                         lr: float = 5e-4,
                         weight_decay: float = 0.0,
                         epochs: int = 100,
                         early_stopping_patience: int = 10):
    """
    Fine-tuning end-to-end (inclusa la head).
    Si assume che il modello abbia attributo `head` che mappa D -> 1 (regressione).
    """
    model.to(device)
    for p in get_pretrain_parameters(model):
        p.requires_grad_(True)
    if hasattr(model, "head") and model.head is not None:
        for p in model.head.parameters():
            p.requires_grad_(True)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    best_state = None
    patience = 0

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            # forward supervisionato "normale"
            # (Assumiamo che il tuo Model.forward(batch, entity_table) restituisca predizioni su entity_table)
            pred = model(batch, entity_table)  # shape [N, 1]
            y = batch[entity_table].y.view(-1, 1).to(pred.device)
            loss = loss_fn(pred, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            train_loss += float(loss.item())

        # Valutazione
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch, entity_table)
                y = batch[entity_table].y.view(-1, 1).to(pred.device)
                val_loss += float(F.l1_loss(pred, y).item())
        val_loss /= max(1, len(val_loader))

        print(f"[FT] epoch {ep:03d} | train_loss {train_loss/max(1,len(train_loader)):.4f} | val_MAE {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print("[FT] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch, entity_table)
            y = batch[entity_table].y.view(-1, 1).to(pred.device)
            test_loss += float(F.l1_loss(pred, y).item())
    test_loss /= max(1, len(test_loader))
    print(f"[FT] best val_MAE {best_val:.4f} | test_MAE {test_loss:.4f}")
    return best_val, test_loss
