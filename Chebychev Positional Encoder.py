import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
import copy
import os
import csv



# Set environment variable for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class ListOpsDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # Skip the header
            for row in reader:
                self.samples.append((row[0], int(row[1])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        # Ensure labels are either 0 or 1
        label = label % 2
        tokens = self.tokenizer.encode_plus(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()
        label = torch.tensor(label, dtype=torch.long)
        return input_ids, attention_mask, label

class ChebyshevPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(ChebyshevPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.encoding = self.generate_chebyshev_encodings()

    def generate_chebyshev_encodings(self):
        encodings = np.zeros((self.max_len, self.d_model))
        for pos in range(self.max_len):
            normalized_pos = 2 * pos / (self.max_len - 1) - 1  # Normalize to range [-1, 1]
            for i in range(self.d_model):
                encodings[pos, i] = self.chebyshev_polynomial(normalized_pos, i)
        return torch.tensor(encodings, dtype=torch.float32)

    def chebyshev_polynomial(self, x, n):
        T = [1, x]  # T_0(x) = 1, T_1(x) = x
        for k in range(2, n + 1):
            T.append(2 * x * T[-1] - T[-2])
        return T[n]

    def forward(self, positions):
        batch_size, seq_len = positions.size()
        encodings = self.encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
        return encodings

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = self.bert.config.hidden_size  # Typically 768 for BERT base
        self.chebyshev_pos_enc = ChebyshevPositionalEncoding(d_model=self.hidden_size)
        self.dropout = nn.Dropout(p=0.1)  # Dropout layer with dropout probability of 0.1
        self.classifier = nn.Linear(self.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        pos_enc = self.chebyshev_pos_enc(positions).to(input_ids.device)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0] + pos_enc
        sequence_output = self.dropout(sequence_output)  # Apply dropout
        logits = self.classifier(sequence_output[:, 0, :])
        return logits

class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer and datasets
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = ListOpsDataset(tokenizer, 'output_dir/basic_train.tsv')
val_dataset = ListOpsDataset(tokenizer, 'output_dir/basic_val.tsv')
test_dataset = ListOpsDataset(tokenizer, 'output_dir/basic_test.tsv')

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Initialize model, loss, and optimizer
model = TransformerModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Add L2 regularization with weight_decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) # gradient scheduler
early_stopping = EarlyStopping(patience=3, delta=0.001)

# Training loop with gradient clipping and early stopping
for epoch in range(7):
    print(f"Starting epoch {epoch + 1}")
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    model.train()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        # Debugging: print shapes and values
        print(f"Input shape: {input_ids.shape}, Attention mask shape: {attention_mask.shape}, Labels shape: {labels.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"Labels: {labels}")

        loss = criterion(outputs, labels)
        print(f"Loss: {loss.item()}")

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_predictions += labels.size(0)

        # Print batch information
        print(f"Batch Loss: {loss.item()}, Batch Accuracy: {torch.sum(preds == labels).item() / labels.size(0)}")

    accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_dataloader)}, Accuracy: {accuracy}")
    scheduler.step()

    # Validation step
    model.eval()
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            val_correct_predictions += torch.sum(preds == labels).item()
            val_total_predictions += labels.size(0)

    val_accuracy = val_correct_predictions / val_total_predictions
    val_loss /= len(val_dataloader)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    # Early stopping
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# Load the best model weights
model.load_state_dict(early_stopping.best_model_wts)

# Test step
model.eval()
test_loss = 0.0
test_correct_predictions = 0
test_total_predictions = 0
with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        test_correct_predictions += torch.sum(preds == labels).item()
        test_total_predictions += labels.size(0)

test_accuracy = test_correct_predictions / test_total_predictions
test_loss /= len(test_dataloader)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
