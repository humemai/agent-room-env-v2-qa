import random
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F


# Define the GNN Model
class GNN(torch.nn.Module):
    """
    class GCNConv(
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool | None = None,
        normalize: bool = True,
        bias: bool = True,
        **kwargs: Any
    )
    """

    def __init__(self, num_features=5, num_outputs=3, **kwargs):
        super(GNN, self).__init__()

        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_outputs)

    # def forward(self, x, edge_index, batch):
    #     x = F.relu(self.conv1(x, edge_index))
    #     x = F.dropout(x, training=self.training)
    #     x = self.conv2(x, edge_index)

    #     # Pool node features within each graph to a single vector
    #     # x = global_mean_pool(x, batch)
    #     # return F.log_softmax(x, dim=1)
    #     return x

    def forward(self, data, policy_type: str) -> torch.Tensor:
        batch_size = data.shape[0]
        if policy_type == "mm":
            return torch.randn(batch_size, len(data[0]["short"]), 3)
        elif policy_type == "explore":
            return torch.randn(batch_size, 1, 5)
        else:
            raise ValueError("Invalid policy type")
