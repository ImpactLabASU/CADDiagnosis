import numpy as np
import pandas as pd
import csv
import os
from joblib import load
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from torch.nn import DataParallel
import csv

writer = SummaryWriter()
current_directory = os.path.dirname(os.path.realpath(__file__))

def calculate_precision(preds, labels):
    _, predicted = torch.max(preds, dim=1)
    true_positives = ((predicted == 1) & (labels == 1)).sum().item()
    false_positives = ((predicted == 1) & (labels == 0)).sum().item()
    precision = true_positives / (true_positives + false_positives + 1e-8)
    return precision
   
def calculate_specificity(preds, labels):
    _, predicted = torch.max(preds, dim=1)
    true_negatives = ((predicted == 0) & (labels == 0)).sum().item()
    false_positives = ((predicted == 1) & (labels == 0)).sum().item()
    specificity = true_negatives / (true_negatives + false_positives + 1e-8)
    return specificity

def calculate_npv(preds, labels):
    _, predicted = torch.max(preds, dim=1)
    true_negatives = ((predicted == 0) & (labels == 0)).sum().item()
    false_negatives = ((predicted == 0) & (labels == 1)).sum().item()
    epsilon = 1e-8
    npv = true_negatives / (true_negatives + false_negatives + 1e-8)
    return npv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#(train_patientNo_list, test_patientNo_list, test_data_list, test_label_list, gender_list) = load('TestDataESE_demographics.joblib')

(train_patientNo_list, test_patientNo_list, test_data_list, test_label_list, gender_list) = load('TestDataESE_unguided_demographics.joblib')


print('test_data_list:', len(test_data_list))
print('test_patientNo_list:', len(test_patientNo_list))
print('test_label:', len(test_label_list))
print('gender:', len(gender_list))

# Define the CSV file name and column names
csv_file = 'TX_TestOutput_unguided_1.csv'
columns = ['PatientNumber', 'Gender', 'ExpectedOutput', 'PredictedOutput']
data_to_write = []

# Zip the test_patientNo_list and test_label_list together to create rows
for patient_number, expected_output, gender in zip(test_patientNo_list, test_label_list, gender_list):
    data_to_write.append([patient_number, gender, expected_output, None])  # Empty PredictedOutput for now


print('Intermediate datya write length:: ', len(data_to_write))

# Write the data to the CSV file
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns)
    writer.writerows(data_to_write)

print(f"CSV file '{csv_file}' has been created with the data.")


test_data_list = np.array(test_data_list)
test_torch_tensor = torch.tensor(test_data_list,dtype=torch.float).to(device)
test_torch_tensor = test_torch_tensor.permute(0, 2, 1).to(device)
# torch_tensor.unsqueeze_(-1)
label = torch.tensor(test_label_list).to(device)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
   
    def __len__(self):
        return len(self.data)
   
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=8, num_layers=3, num_heads=4):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
       
        self.embedding = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

   
    def forward(self, x,mask):
       
        x = self.embedding(x)
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x,src_key_padding_mask=mask)
        x = x.permute(1, 0, 2)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
       

# hyperparameters
input_size = 5
num_classes = 2
hidden_size = 2048
num_layers = 4
num_heads = 4
batch_size = 8
lr = 1e-5
num_epochs = 700

data_test = test_torch_tensor
dataset_test = TimeSeriesDataset(data_test, label)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# prepare model and optimizer
model = TransformerModel(input_size, num_classes, hidden_size, num_layers, num_heads).to(device)
model = DataParallel(model)
model.load_state_dict(torch.load('transformer_model_12leads_allspeed_700_2048.pth_1'))
model.eval()

optimizer = optim.Adam(model.parameters(), lr=lr)


# testing loop

epoch_loss = 0.0
epoch_acc = 0.0
epoch_precision = 0.0
epoch_specificity = 0.0
num_batches = 0
data_to_write = []

for batch_data, batch_labels in dataloader_test:
  optimizer.zero_grad()
  print('batchdata:: ',batch_data.shape)
  mask = (batch_data == -10).all(dim=2).to(device)
  output = model(batch_data.to(device),mask)
  loss = nn.CrossEntropyLoss()(output, batch_labels.type(torch.int64).to(device))
  # calculate accuracy
  preds = torch.argmax(output, dim=1)
  acc = (preds.to(device) == batch_labels.to(device)).sum().item() / batch_labels.size(0)
  batch_precision = calculate_precision(output, batch_labels.to(device))
  batch_specificity = calculate_specificity(output, batch_labels.to(device))
  npv = calculate_npv(output, batch_labels.to(device))
  # update epoch loss and accuracy
  epoch_loss += loss.item()
  epoch_acc += acc
  epoch_precision += batch_precision
  epoch_specificity += batch_specificity
  num_batches += 1
  print(preds)
   
  # Append rows to data_to_write with each batch_label as a separate row in PredictedOutput
  for label in preds:
    data_to_write.append([None, None, None, label.item()])  # Convert tensor to a single item in a list

print('length of pred data:: ', data_to_write)
  # Write the data to the CSV file, appending to the existing file if it already exists
with open(csv_file, 'a', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerows(data_to_write)
   
   
   
epoch_loss /= num_batches
epoch_acc /= num_batches
epoch_precision /= num_batches
epoch_specificity /= num_batches
   
print(f"Testing Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Precision: {epoch_precision:.4f}, NPV: {npv: .2f}, Specificity: {epoch_specificity:.4f}")
