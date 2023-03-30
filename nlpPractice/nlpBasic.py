import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("./Data/diabetes.csv")
X = df[df.columns[:-1]]
y = df['Outcome']

X = X.values
y = torch.tensor(y.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

ms = MinMaxScaler()
ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ms.fit_transform(y_train)
y_test = ms.fit_transform(y_test)

class customDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len
    
train_data = customDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
test_data = customDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = True)

class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        self.layer1 = nn.Linear(8, 64, bias = True)
        self.layer2 = nn.Linear(64, 64, bias = True)
        self.layer3 = nn.Linear(64, 1, bias = True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
    
    def forward(self, inputs):
        x = self.relu(self.layer1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x

epochs = 1001
print_epoch = 100
lr = 1e-2

model = binaryClassification()
model.to(device)
BCE = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

for epoch in range(epochs):
    iteration_loss = 0.
    iteration_accuracy = 0.
    
    model.train()
    for i, data in enumerate(train_loader):
        X, y = data
        X, y = X.to(device), y.to(device)
        y_pred = model(X.float())
        loss = BCE(y_pred, y.reshape(-1, 1).float())
        
        iteration_loss += loss
        iteration_accuracy += accuracy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % print_epoch == 0:
        print(f'Train: epoch: {epoch} - loss: {(iteration_loss/(i+1)):.5f}; acc: {(iteration_accuracy/(i+1)):.3f}')
    
    iteration_loss = 0.
    iteration_accuracy = 0.
    model.eval()
    for i, data in enumerate(test_loader):
        X, y = data
        X, y = X.to(device), y.to(device)
        y_pred = model(X.float())
        loss = BCE(y_pred, y.reshape(-1, 1).float())
        
        iteration_loss += loss
        iteration_accuracy += accuracy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % print_epoch == 0:
        print(f'Test: epoch: {epoch} - loss: {(iteration_loss/(i+1)):.5f}; acc: {(iteration_accuracy/(i+1)):.3f}')
    