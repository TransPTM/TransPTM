import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import TransformerConv, GCNConv, GATConv


class GNNTrans(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.loss_func = nn.BCELoss()
        self.convs = torch.nn.ModuleList(
            [TransformerConv(in_channels=input_dim, out_channels=hidden_dim, heads=1)] +
            [TransformerConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=1)
                for _ in range(num_layers-1)]
        )

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def get_conv_result(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = F.relu(x, inplace=True)
        return x

    def forward(self, data):
        x, edge_index, batch = data.emb, data.edge_index, data.batch
        idx = (data.ptr + int(len(data.seq[0]) / 2))[:-1]
        #
        x = self.get_conv_result(x, edge_index)

        x = x[idx]
        out = self.mlp(x)

        return out

    def loss(self, pred, label):
        pred, label = pred.reshape(-1), label.reshape(-1)
        return self.loss_func(pred, label)
