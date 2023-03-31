import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv("./Data/training.txt", sep = '\t')
valid_df = pd.read_csv("./Data/validating.txt", sep = '\t')
test_df = pd.read_csv("./Data/testing.txt", sep = '\t')

train_df = train_df.sample(frac = 0.1, random_state = 34)
valid_df = valid_df.sample(frac = 0.1, random_state = 34)
test_df = test_df.sample(frac = 0.1, random_state = 34)

class Datasets(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return self.df.iloc[index, 1], self.df.iloc[index, 2]
    
train_dataset = Datasets(train_df)
train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True, num_workers = 0)

valid_dataset = Datasets(valid_df)
valid_loader = DataLoader(valid_dataset, batch_size = 2, shuffle = True, num_workers = 0)

test_dataset = Datasets(test_df)
test_loader = DataLoader(test_dataset, batch_size = 2, shuffle = True, num_workers = 0)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)

def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(), 'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')
    
def load_checkpoint(load_path, model):
    if load_path == None:
        return
    state_dict = torch.load(load_path, map_location = device)
    print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return
    state_dict = {'train_loss_list': train_loss_list, 'valid_loss_list': valid_loss_list, 'global_steps_list': global_steps_list}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')
    
def load_metrics(load_path):
    if load_path == None:
        return
    state_dict = torch.load(load_path, map_location = device)
    print(f'Model loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

def train(model, optimizer, criterion = nn.BCELoss(), num_epochs = 5, eval_every = len(train_loader) // 2, best_valid_loss = float("Inf")):
    total_correct = 0.0
    total_len = 0.0
    running_loss = 0.0
    valid_running_loss = 0.0
    global_steps = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    
    model.train()
    
    for epoch in range(num_epochs):
        for text, label in train_loader:
            optimizer.zero_grad()
            encoded_list = [tokenizer.encode(t, add_special_tokens = True) for t in text]
            padded_list = [e + [0] * (512 - len(e)) for e in encoded_list]
            sample = torch.tensor(padded_list)
            sample, label = sample.to(device), label.to(device)
            labels = torch.tensor(label)
            outputs = model(sample, labels = labels)
            loss, logits = outputs
            
            pred = torch.argmax(F.softmax(logits), dim = 1)
            correct = pred.eq(labels)
            total_correct += correct.sum().item()
            total_len += len(labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            global_steps += 1
            
            if global_steps % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    for text, label in valid_loader:
                        encoded_list = [tokenizer.encode(t, add_special_tokens = True) for t in text]
                        padded_list = [e + [0] * (512 - len(e)) for e in encoded_list]
                        sample = torch.tensor(padded_list)
                        sample, label = sample.to(device), label.to(device)
                        labels = torch.tensor(label)
                        outputs = model(sample, labels = labels)
                        loss, logits = outputs
                        valid_running_loss += loss.item()
            
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_steps)
                
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()
                
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch + 1, num_epochs, global_steps, num_epochs * len(train_loader), average_train_loss, average_valid_loss))
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint('./Data/model.pt', model, best_valid_loss)
                    save_metrics('./Data/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    save_metrics('./Data/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
    
optimizer = optim.Adam(model.parameters(), lr = 2e-5)
train(model = model, optimizer = optimizer)

train_loss_list, valid_loss_list, global_steps_list = load_metrics('./Data/metrics.pt')

# plt.plot(global_steps_list, train_loss_list, label = 'Training Loss')
# plt.plot(global_steps_list, valid_loss_list, label = 'Validation Loss')
# plt.xlabel('Global Steps')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()                
            
def evaluate(model, test_loader):
    y_pred = []
    y_true = []
    
    model.eval()
    
    with torch.no_grad():
        for text, label in test_loader:
            encoded_list = [tokenizer.encode(t, add_special_tokens = True) for t in text]
            padded_list = [e + [0] * (512 - len(e)) for e in encoded_list]
            sample = torch.tensor(padded_list)
            sample, label - sample.to(device), label.to(device)
            labels = torch.tensor(label)
            output = model(sample, labels = labels)
            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())
            
    print('Classification Result: ')
    print(classification_report(y_true, y_pred, labels = [1, 0], digits = 4))
    
    cm = confusion_matrix(y_true, y_pred, labels = [1, 0])
    ax = plt.subplot()
    sns.heatmap(cm, annot = True, ax = ax, fmt = 'd', cmap = 'Blues')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])
    
    