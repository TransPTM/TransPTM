from Bio import SeqIO
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def parse_fasta(path, label, lengths):
    result_ls = []
    for record in SeqIO.parse(path, "fasta"):
        uniprot = record.id
        k_idx = int(record.description.split()[1].split('#')[0])
        prot_len = int(record.description.split()[1].split('#')[1])
        seq = str(record.seq)
        seq_ls = []
        for seq_len in lengths:
            n = int((seq_len - 1)/2)
            seq_ls.append(seq[30-n: 30+n+1])

        result_ls.append(['{};{};{};{}'.format(uniprot, k_idx, prot_len, label)] + seq_ls)
    return result_ls


def split_data(data, val_size, test_size):
    pos_data = data.query('label == 1').reset_index(drop=True)
    neg_data = data.query('label == 0').reset_index(drop=True)
    np.random.seed(42)

    train_pos, test_pos = train_test_split(pos_data, test_size=150)
    train_neg, test_neg = train_test_split(neg_data, test_size=test_size)

    train_pos, val_pos = train_test_split(train_pos, test_size=100)
    train_neg, val_neg = train_test_split(train_neg, test_size=val_size)

    train_data = pd.concat([train_pos, train_neg], ignore_index=True)
    train_data['set'] = 'train'
    val_data = pd.concat([val_pos, val_neg], ignore_index=True)
    val_data['set'] = 'val'
    test_data = pd.concat([test_pos, test_neg], ignore_index=True)
    test_data['set'] = 'test'
    out_data = pd.concat([train_data, val_data, test_data], ignore_index=True)
    return out_data


if __name__ == '__main__':
    fasta_ls = [(1, f'../data/fasta/61p.fasta'), (0, f'../data/fasta/61n.fasta')]
    len_ls = [11, 15, 21, 25, 31, 35, 41, 45, 51, 55, 61]
    record_ls = []
    for label, fasta in fasta_ls:
        record_ls += parse_fasta(fasta, label, len_ls)
    columns = ['unique_id'] + [f'seq_{n}' for n in len_ls]
    df = pd.DataFrame(record_ls, columns=columns)
    df['label'] = df['unique_id'].apply(lambda x: int(x.split(';')[-1]))
    df = split_data(df, val_size=0.1, test_size=0.2)
    df.to_csv('../data/dataset.csv', index=False)
