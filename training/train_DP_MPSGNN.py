import torch
import torch.nn.functional as F
from model.MPSGNN_Model import MPSGNN
from data_management.data import load_relbench_f1
from mpsgnn_metapath_utils import binarize_targets, greedy_metapath_search


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = load_relbench_f1(split='train')
    data = data.to(device)

    y = data['driver'].y.float()
    train_mask = data['driver'].train_mask

    y_bin = binarize_targets(y, threshold=10)
    metapaths = greedy_metapath_search(data, y_bin, train_mask, node_type='driver', L_max=2)

    model = MPSGNN(
        metadata=data.metadata(),
        metapaths=metapaths,
        hidden_channels=64,
        out_channels=64,
        final_out_channels=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.mse_loss(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        print(f\"Epoch {epoch:03d}, Loss: {loss.item():.4f}\")


if __name__ == '__main__':
    train()
