import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = torchvision.datasets.FashionMNIST("./Data", download = True, train = True, transform = transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.FashionMNIST("./Data", download = True, train = False, transform = transforms.Compose([transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 100)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 100)

def show_example(train_dataset):
    labels_map = {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}

    fig = plt.figure(figsize = (8, 8))
    columns = 4
    rows = 5

    for i in range(1, columns * rows + 1):
        img_xy = np.random.randint(len(train_dataset))
        img = train_dataset[img_xy][0][0, :, :]
        fig.add_subplot(rows, columns, i)
        plt.title(labels_map[train_dataset[img_xy][1]])
        plt.axis('off')
        plt.imshow(img, cmap = 'gray')
    plt.show()

class FashionDNN(nn.Module):
    def __init__(self):
        super(FashionDNN, self).__init__()
        self.fc1 = nn.Linear(in_features = 28 * 28, out_features = 256)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features = 256, out_features = 128)
        self.fc3 = nn.Linear(in_features = 128, out_features = 10)
        
    def forward(self, input_data):
        out = input_data.view(-1, 28 * 28)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
learning_rateDNN = 0.001
modelDNN = FashionDNN()
modelDNN.to(device)

criterionDNN = nn.CrossEntropyLoss()
optimizerDNN = torch.optim.Adam(modelDNN.parameters(), lr = learning_rateDNN)

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features = 64 * 6 * 6, out_features = 600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features = 600, out_features = 120)
        self.fc3 = nn.Linear(in_features = 120, out_features = 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

learning_rateCNN = 0.001
modelCNN = FashionCNN()
modelCNN.to(device)

criterionCNN = nn.CrossEntropyLoss()
optimizerCNN = torch.optim.Adam(modelCNN.parameters(), lr = learning_rateCNN)

def implement(train_loader, test_loader, model, criterion, optimizer):
    num_epochs = 5
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []

    prediction_list = []
    labels_list = []

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            train = Variable(images.view(100, 1, 28, 28))
            labels = Variable(labels)
            
            outputs = model(train)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
            
            if not (count % 50):
                total = 0
                correct = 0
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)
                    test = Variable(images.view(100, 1, 28, 28))
                    outputs = model(test)
                    predictions = torch.max(outputs, 1)[1].to(device)
                    prediction_list.append(predictions)
                    correct += (predictions == labels).sum()
                    total += len(labels)
                    
                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            if not (count % 500):
                print(f"Iteration: {count}, Loss: {loss.data:>7f}, Accuracy: {accuracy:>7f}%")
                
implement(train_loader, test_loader, modelDNN, criterionDNN, optimizerDNN)