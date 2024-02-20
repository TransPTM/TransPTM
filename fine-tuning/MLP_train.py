import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
import os 
import torch.nn.functional as F
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "prot/prot_t5_xl_uniref50"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5EncoderModel.from_pretrained(model_name).to(device)


class SequenceDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        label = int(row['unique_id'].split(';')[-1])
        sequence = row['seq_25']
        sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        return sequence, label


class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(1024 * 25, 256)  
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Linear(256, 64) 
        self.relu2 = nn.ReLU()  
        self.fc3 = nn.Linear(64, 2)  
        self.relu3 = nn.ReLU()  
        self.fc4 = nn.Linear(2, 1)  

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x) 
        x = self.fc4(x)
        x = torch.sigmoid(x)  
        return x





df = pd.read_csv('datasetft.csv')
train_df = df[df['set'] == 'train']
val_df = df[df['set'] == 'val']

train_dataset = SequenceDataset(train_df)
val_dataset = SequenceDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=6)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=True)


classifier = MLPClassifier().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
criterion = nn.BCELoss()

best_auc = 0.0  
model_save_path = 'mlp_6_1e-4_50.pth'
state_dict = {
    'T5EncoderModel': model.state_dict(),
    'Classifier': classifier.state_dict(),
}

num_epochs = 20
for epoch in range(num_epochs):
    model.train()  
    total_loss, total_accuracy = 0, 0

    for sequences, labels in train_loader:
        labels = labels.to(device).float()
        optimizer.zero_grad()
        
        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=False, padding="longest")
        input_ids = torch.tensor(ids['input_ids'])
        input_ids = input_ids.to(device)
        attention_mask = torch.tensor(ids['attention_mask'])
        attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        predictions = classifier(embeddings).squeeze()

        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        probs = predictions.squeeze() 
        preds = (probs >= 0.5).float() 
        total_accuracy += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        
    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    model.eval()  
    total_val_loss, total_val_accuracy, all_labels, all_predictions = 0, 0, [], []

    with torch.no_grad():
        for sequences, labels in val_loader:
            labels = labels.to(device).float()
            
            ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=False, padding="longest")
            input_ids = torch.tensor(ids['input_ids'])
            input_ids = input_ids.to(device)
            attention_mask = torch.tensor(ids['attention_mask'])
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
            
            predictions = classifier(embeddings).squeeze(-1)
            val_loss = criterion(predictions, labels)
            total_val_loss += val_loss.item()

            probs = predictions
            preds = (probs >= 0.5).float() 
            total_val_accuracy += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(probs.cpu().numpy())
    
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_accuracy = total_val_accuracy / len(val_loader)
    val_auc = roc_auc_score(all_labels, all_predictions)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}, Validation AUC: {val_auc:.4f}")
    
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(state_dict, model_save_path)
        print(f"Saved Best Model with AUC: {best_auc:.4f}")

