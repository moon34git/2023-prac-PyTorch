import torch
import torchtext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import string
import random
from torchtext.datasets import IMDB
from torchtext.legacy import datasets

start = time.time()

# Preprocessing Rule
TEXT = torchtext.legacy.data.Field(lower = True, sequential = True, batch_first = True)
LABEL = torchtext.legacy.data.Field(sequential = False, batch_first = True)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(split_ratio = 0.8)

#Make unique tokens
TEXT.build_vocab(train_data, max_size = 10000, min_freq = 10, vectors = None)
LABEL.build_vocab(train_data)

BATCH_SIZE = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

embeding_dim = 100
hidden_size = 300

#BucketIterator is similar to Dataloader
train_iterator, valid_iterator, test_iterator = torchtext.legacy.data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                                            batch_size = BATCH_SIZE, device = device)

vocab_size = len(TEXT.vocab)
n_classes = 2

class BasicRNN(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p = 0.2):
        super(BasicRNN, self).__init__()
        self.n_layers = n_layers
        #Applying word embedding
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.RNN(embed_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True)
        self.out = nn.Linear(self.hidden_dim, n_classes)
                
    def forward(self, x):
        #Transform text into number/vector
        x = self.embed(x)
        
        #Initialize hidden state value as 0
        h_0 = self._init_state(batch_size = x.size(0))
        x, _ = self.rnn(x, h_0)
        
        #Last hidden state value
        h_t = x[:, -1, :]
        
        self.dropout(h_t)
        logit = torch.sigmoid(self.out(h_t))
        return logit
    
    def _init_state(self, batch_size = 1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

model = BasicRNN(n_layers = 1, hidden_dim = 256, n_vocab = vocab_size, embed_dim = 128, n_classes = n_classes, dropout_p = 0.5)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)
        optimizer.zero_grad()
        
        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()
        
        if b % 50 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(e,
                                                                           b * len(x),
                                                                           len(train_iter.dataset),
                                                                           100. * b / len(train_iter),
                                                                           loss.item()))


def evaluate(model, val_iter):
    model.eval()
    corrects, total, total_loss = 0, 0, 0

    for batch in val_iter:
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1) 
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction = "sum")
        total += y.size(0)
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
        
    avg_loss = total_loss / len(val_iter.dataset)
    avg_accuracy = corrects / total
    return avg_loss, avg_accuracy


BATCH_SIZE = 100
LR = 0.001
EPOCHS = 5
for e in range(1, EPOCHS + 1):
    train(model, optimizer, train_iterator)
    val_loss, val_accuracy = evaluate(model, valid_iterator)
    print("[EPOCH: %d], Validation Loss: %5.2f | Validation Accuracy: %5.2f" % (e, val_loss, val_accuracy))


