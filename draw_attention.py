import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
from model import GNNTrans
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class GNNTransTest(GNNTrans):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__(input_dim, hidden_dim, num_layers)

    def get_conv_result(self, x, edge_index):
        for i in range(self.num_layers):
            x, (_, self.convs[i].alpha) = self.convs[i](x=x, edge_index=edge_index, return_attention_weights=True)
            x = F.relu(x, inplace=True)
        return x

    def forward(self, data):
        x, edge_index, batch = data.emb, data.edge_index, data.batch
        idx = (data.ptr + int(len(data.seq[0]) / 2))[:-1]
        #
        x = self.get_conv_result(x, edge_index)

        x = x[idx]
        out = self.mlp(x)

        return x, out


def get_alpha(model, data):
    model(data)

    # alpha is a "seq_len x seq_len" matrix
    # each row corresponds to a residue and sums to 1
    alpha = model.convs[0].alpha.detach().numpy().reshape(seq_len, seq_len).T  # alpha.sum(axis=1) = [1]
    return alpha


if __name__ == '__main__':
    for seq_len in [21, 25, 31, 35]:
        # seq_len = 15
        train_ls, val_ls, test_ls = torch.load(f'./data/processed/{seq_len}.pt')
        test_dic = {data['unique_id']: data for data in train_ls}

        # load model
        weight = './result/2_layer/11/0_acc0.870_roc0.775_prc0.456.pt'
        num_layers = 2
        model = GNNTransTest(input_dim=1024, hidden_dim=256, num_layers=num_layers)
        model.load_state_dict(torch.load(weight))
        model.eval()

        #
        df = pd.read_csv('./data/dataset.csv')
        df = df.query('set == "train"').reset_index(drop=True)
        # alpha_ls = []
        emb_ls = []
        for unique_id, label in df[['unique_id', 'label']].values:
            unique_id = [unique_id]
            test_data_loader = list(DataLoader([test_dic[uid] for uid in unique_id], batch_size=len(unique_id)))
            # alpha = get_alpha(model, test_data_loader[0])
            # alpha_ls.append(alpha)
            emb_ls.append([model(test_data_loader[0])[0], label])
        torch.save(emb_ls, f'./result/emb_{seq_len}.pt')

        # alpha = np.stack(alpha_ls, 0)
        # np.save(f'./result/attention_{seq_len}.npy', alpha)


