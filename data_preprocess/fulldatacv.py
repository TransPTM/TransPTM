import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
import numpy as np
from model import GNNTrans
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import random

train_ls, val_ls, test_ls = torch.load('25.pt')
full_dataset = train_ls + val_ls + test_ls
random.shuffle(full_dataset)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 100
model = GNNTrans(input_dim=1024, hidden_dim=64, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)


kf = KFold(n_splits=5)
all_acc, all_precision, all_recall, all_f1, all_mcc, all_auc, all_auprc = [], [], [], [], [], [], []
fold = 1
for train_index, val_index in kf.split(full_dataset):
    train_fold = [full_dataset[i] for i in train_index]
    val_fold = [full_dataset[i] for i in val_index]
    train_data_loader = DataLoader(train_fold, batch_size=64, shuffle=True)
    val_data_loader = DataLoader(val_fold, batch_size=64, shuffle=False)
    best_loss = float('inf')
    patience = 25
    counter = 0
    val_loss = 0

    for epoch in range(n_epochs):
        model.train()
        for batch in train_data_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            pred = model(batch)
            loss = model.loss(pred, batch.y)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        for batch in val_data_loader:
            batch = batch.to(device)
            with torch.no_grad():
                pred = model(batch)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        acc = accuracy_score(all_labels, np.round(all_preds))
        auc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)

        all_acc.append(acc)
        all_auc.append(auc)
        all_auprc.append(auprc)

        print(f"Fold: {fold}, Epoch: {epoch + 1}, Accuracy: {acc}, AUC: {auc}, AUPRC: {auprc}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break

    torch.save(model.state_dict(), f"model_fold{fold}.pt")
    fold += 1

