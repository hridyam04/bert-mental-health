import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=28).to(device)

def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    labels = df['labels'].apply(eval).tolist()
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    labels_tensor = torch.tensor(labels)
    return TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels_tensor)

train_dataset = load_data('../data/processed/train.csv')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(2):
    print(f"Epoch {epoch+1}")
    for batch in tqdm(train_loader):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels.float())
        loss = outputs.loss
        loss.backward()
        optimizer.step()

model.save_pretrained("../models/bert_model")
tokenizer.save_pretrained("../models/bert_model")
print("Model saved in models/bert_model")
