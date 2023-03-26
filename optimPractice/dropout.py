import torch
import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim

N = 50
noise = 0.3

x_train = torch.unsqueeze(torch.linspace(-1, 1, N), 1)
y_train = x_train + noise * torch.normal(torch.zeros(N, 1), torch.ones(N, 1))

x_test = torch.unsqueeze(torch.linspace(-1, 1, N), 1)
y_test = x_test + noise * torch.normal(torch.zeros(N, 1), torch.ones(N, 1))

# plt.scatter(x_train.data.numpy(), y_train.data.numpy(), c = 'purple', alpha = 0.5, label = 'train')
# plt.scatter(x_test.data.numpy(), y_test.data.numpy(), c = 'yellow', alpha = 0.5, label = 'test')
# plt.legend()
# plt.show()

N_h = 100

model = torch.nn.Sequential(
    nn.Linear(1, N_h),
    nn.ReLU(),
    nn.Linear(N_h, N_h),
    nn.ReLU(),
    nn.Linear(N_h, 1),
)

model_dropout = torch.nn.Sequential(
    nn.Linear(1, N_h),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.Linear(N_h, N_h),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.Linear(N_h, 1),
)

opt = torch.optim.Adam(model.parameters(), lr = 0.01)
opt_dropout = torch.optim.Adam(model_dropout.parameters(), lr = 0.01)
loss_fn = nn.MSELoss()

max_epochs = 1000
for epoch in range(max_epochs):
    pred = model(x_train)
    loss = loss_fn(pred, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    pred_dropout = model_dropout(x_train)
    loss_dropout = loss_fn(pred_dropout, y_train)
    opt_dropout.zero_grad()
    loss_dropout.backward()
    opt_dropout.step()
    
    if epoch % 50 == 0:
        model.eval()
        model_dropout.eval()
        
        test_pred = model(x_test)
        test_loss = loss_fn(test_pred, y_test)
        
        test_pred_dropout = model_dropout(x_test)
        test_loss_dropout = loss_fn(test_pred_dropout, y_test)
        
        plt.scatter(x_train.data.numpy(), y_train.data.numpy(), c = 'purple', alpha = 0.5, label = 'train')
        plt.scatter(x_test.data.numpy(), y_test.data.numpy(), c = 'yellow', alpha = 0.5, label = 'test')
        plt.plot(x_test.data.numpy(), test_pred.data.numpy(), 'b-', lw = 3, label = 'Normal')
        plt.plot(x_test.data.numpy(), test_pred_dropout.data.numpy(), 'g--', lw = 3, label = 'Dropout')
        plt.title(f'Epoch: {epoch}, Loss: {test_loss:>2f}, Loss with Dropout: {test_loss_dropout:>2f}')
        plt.legend()
        model.train()
        model_dropout.train()
        plt.pause(0.05)        
        
    
    