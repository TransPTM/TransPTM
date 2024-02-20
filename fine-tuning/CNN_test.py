import torch
from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import re
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 25, 4)  
        self.relu1 = nn.ReLU() 
        self.fc2 = nn.Linear(4, 1) 

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)  
        x = self.fc1(x) 
        x = self.relu1(x)  
        x = self.fc2(x)  
        x = torch.sigmoid(x)  
        return x


model_name = "prot/prot_t5_xl_uniref50"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5EncoderModel.from_pretrained(model_name).to(device)
classifier = Classifier().to(device)

model_save_path = 'cnn_6_1e-6_50.pth'
state_dict = torch.load(model_save_path, map_location=device)
model.load_state_dict(state_dict['T5EncoderModel'])
classifier.load_state_dict(state_dict['Classifier'])

df = pd.read_csv('datasetft.csv')
test_df = df[df['set'] == 'test']
test_dataset = SequenceDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)


classifier.eval()
all_predictions = []
all_labels = []
for sequences, labels in tqdm(test_loader, desc="Testing"):
    labels = labels.to(device)
    
    inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=25)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state
    predictions = classifier(embeddings).squeeze(-1)
    preds = predictions >= 0.5
    all_labels.extend(labels.cpu().numpy())
    all_predictions.extend(preds.cpu().numpy())


all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
    
acc = accuracy_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)
mcc = matthews_corrcoef(all_labels, all_predictions)
auc = roc_auc_score(all_labels, all_predictions)
auprc = average_precision_score(all_labels, all_predictions)

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"AUPRC: {auprc:.4f}")



