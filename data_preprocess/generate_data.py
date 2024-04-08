import pandas as pd
import numpy as np
import os
from Bio import SeqIO
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


EMB_PATH = './protT5'

res_to_id = {
    "K": 1,
    "A": 2,
    "R": 3,
    "N": 4,
    "D": 5,
    "C": 6,
    "Q": 7,
    "E": 8,
    "G": 9,
    "H": 10,
    "I": 11,
    "L": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
}


class Processor:
    def __init__(self, info_df, length):
        self.info = info_df
        self.length = length

    def run(self):
        train_ls = []
        val_ls = []
        test_ls = []
        for _, record in self.info.iterrows():
            seq = record[f'seq_{self.length}']
            #
            unique_id = record['unique_id'].split(';')
            uniprot = unique_id[0]
            start_idx = int(unique_id[1])
            prot_len = int(unique_id[2])
            label = int(unique_id[3])
            n = int((self.length - 1)/2)
            emb = torch.tensor(
                np.load(os.path.join(EMB_PATH, "{}.npy".format(record['unique_id']))), dtype=torch.float32)
            emb = emb[30-n: 30+n+1, ]

            x = [res_to_id[res] for res in seq]
            x_one_hot = torch.zeros(len(x), 21)
            x_one_hot[range(len(x)), x] = 1
            x = torch.tensor(x, dtype=torch.int32).unsqueeze(1)

            n = len(seq)
            edge_index1 = dense_to_sparse(torch.ones((n, n)))[0]
            a = torch.zeros((n, n))
            a[range(n), np.arange(n)]= 1
            a[range(n-1), np.arange(n-1) + 1] = 1
            a[np.arange(n-1) + 1, np.arange(n-1)] = 1
            idx = int(n / 2)
            a[[idx]*n, range(n)] = 1
            edge_index2 = dense_to_sparse(a)[0]

            data = Data(
                x=x,
                x_one_hot=x_one_hot,
                edge_index1=edge_index1,
                edge_index2=edge_index2,
                emb=emb,
                seq=seq,
                uniprot=uniprot,
                start_idx=start_idx,
                prot_len=prot_len,
                unique_id=';'.join(unique_id),
                y=torch.tensor(label, dtype=torch.float32)
            )
            group = record['set']
            if group == 'train':
                train_ls.append(data)
            elif group == 'val':
                val_ls.append(data)
            elif group == 'test':
                test_ls.append(data)
            else:
                raise Exception('Unknown data group')

        return train_ls, val_ls, test_ls


if __name__ == '__main__':
    df = pd.read_csv('./dataset.csv')

    len_ls = [11, 15, 21, 25, 31, 35, 41, 45, 51, 55, 61]
    for seq_len in len_ls:
        save_path = f'./{seq_len}.pt'
        processor = Processor(df, length=seq_len)
        data_ls = processor.run()
        torch.save(data_ls, save_path)







