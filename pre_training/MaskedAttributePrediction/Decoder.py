import torch
from torch import nn
import torch.nn.functional as Fç

class MAPDecoder(nn.Module):
    def __init__(self, encoder_out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.decoder_dict = nn.ModuleDict()
        self.encoder_out_dim = encoder_out_dim
        self.hidden_dim = hidden_dim

    def add_decoder(self, name: str, out_dim: int, task: str):
        if task == "regression":
            self.decoder_dict[name] = nn.Linear(self.encoder_out_dim, out_dim)
        elif task == "classification":
            self.decoder_dict[name] = nn.Linear(self.encoder_out_dim, out_dim)

    def forward(self, z_dict, batch, mask_info):
        losses = []

        for (node_type, col), info in mask_info.items():
            z = z_dict[node_type]

            # Converto in tensori PyTorch
            indices = torch.tensor(info["indices"], device=z.device)
            true_vals = torch.tensor(info["values"], device=z.device)

            # Estraggo i nodi mascherati
            z_masked = z[indices]
            decoder = self.decoder_dict[f"{node_type}__{col}"]
            pred = decoder(z_masked)

            if pred.numel() == 0:
                continue  # salta se batch vuoto

            #lss diversa per regressione o classificazione
            if true_vals.dtype in [torch.float, torch.float32, torch.float64]:
                loss = F.mse_loss(pred.squeeze(), true_vals.float())
            else:
                loss = F.cross_entropy(pred, true_vals.long())

            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, requires_grad=True, device=z.device)

        return sum(losses)
