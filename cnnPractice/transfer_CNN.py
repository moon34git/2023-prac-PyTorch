import os
import time
import copy
import glob
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = models.resnet18(pretrained = True)
    
def set_parameter_requires_grad(model, feature_extracting = True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    
set_parameter_requires_grad(resnet18)

resnet18.fc = nn.Linear(512, 2)

def train():
    torch.multiprocessing.freeze_support()
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.ImageFolder("./Data/catanddog/train", transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, num_workers = 4, shuffle = True)

    def show_example(train_loader):
        samples, labels = next(iter(train_loader))
        classes = {0: 'cat', 1: 'dog'}
        fig = plt.figure(figsize = (16, 24))
        for i in range(24):
            a = fig.add_subplot(4, 6, i + 1)
            a.set_title(classes[labels[i].item()])
            a.axis('off')
            a.imshow(np.transpose(samples[i].numpy(), (1, 2, 0)))
        plt.subplots_adjust(bottom = 0.2, top = 0.6, hspace = 0)

    # show_example(train_loader)  
    
    # resnet18 = models.resnet18(pretrained = True)
    
    # def set_parameter_requires_grad(model, feature_extracting = True):
    #     if feature_extracting:
    #         for param in model.parameters():
    #             param.requires_grad = False
        
    # set_parameter_requires_grad(resnet18)
    
    # resnet18.fc = nn.Linear(512, 2)

    def train_model(model, dataloaders, criterion, optimizer, device, num_epochs = 13, is_train = True):
        since = time.time()
        acc_history = []
        loss_history = []
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders:
                inputs, labels = inputs.to(device), labels.to(device)
                model.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders.dataset)
            epoch_acc = running_corrects.double() / len(dataloaders.dataset)
            
            print(f'Loss: {epoch_loss:>7f} Acc: {epoch_acc:>7f}')
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                
            acc_history.append(epoch_acc.item())
            loss_history.append(epoch_loss)
            torch.save(model.state_dict(), os.path.join('./Data/catanddog/', '{0:0=2d}.pth'.format(epoch)))
            
            print()
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}')
        print(f'Best Acc: {best_acc:>7f}')
        return acc_history, loss_history
    
    params_to_update = []
    for name,param in resnet18.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
                
    optimizer = optim.Adam(params_to_update)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    train_acc_hist, train_loss_hist = train_model(resnet18, train_loader, criterion, optimizer, device)

def test():
    torch.multiprocessing.freeze_support()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    test_dataset = torchvision.datasets.ImageFolder('./Data/catanddog/test', transform = transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, num_workers = 1, shuffle = True)
    
    def eval_model(model, dataloaders, device):
        since = time.time()
        acc_history = []
        best_acc = 0.0
        
        saved_models = glob.glob('./Data/catanddog/' + '*.pth')
        saved_models.sort()
        print('saved_model', saved_models)
        
        for model_path in saved_models:
            print('Loading model', model_path)
            
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model.to(device)
            running_corrects = 0
            
            for inputs, labels in dataloaders:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.no_grad():
                    outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                preds[preds >= 0.5] = 1
                preds[preds < 0.5] = 0
                running_corrects += preds.eq(labels.cpu()).int().sum()
                
            epoch_acc = running_corrects.double() / len(dataloaders.dataset)
            print(f'Acc: {epoch_acc:>3f}')
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
            
            acc_history.append(epoch_acc.item())
            print()
            
        time_elapsed = time.time() - since
        print(f'Validation complete in {time_elapsed // 60}m {time_elapsed % 60}')
        print(f'Best Acc: {best_acc:>4f}')
        return acc_history, model
    
    val_acc_hist, model = eval_model(resnet18, test_loader, device)
    test_data = test_loader
    
    return val_acc_hist, test_data, model
        
def cat_dog(test_loader, model):
    
    def im_convert(tensor):
        image = tensor.clone().detach().numpy()
        image = image.transpose(1, 2, 0)
        image = image * (np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5)))
        image = image.clip(0, 1)
        return image
    
    classes = {0: 'cat', 1: 'dog'}
    images, labels = next(iter(test_loader))
    output = model(images)
    _, preds = torch.max(output, 1)
    
    fig = plt.figure(figsize = (25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 10, idx + 1, xticks = [], yticks = [])
        plt.imshow(im_convert(images[idx]))
        ax.set_title(classes[labels[idx].item()])
        ax.set_title("{}({})".format(str(classes[preds[idx].item()]),str(classes[labels[idx].item()])),color=("green" if preds[idx]==labels[idx] else "red"))  
    plt.show()  
    plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)
    
if __name__ == '__main__':
    # train()
    _, test_loader, model = test()
    cat_dog(test_loader, model)
    