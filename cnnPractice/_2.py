import os
import time
import copy
import glob
import cv2
import shutil
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

def run():
    torch.multiprocessing.freeze_support()
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.ImageFolder("./Data/catanddog/train", transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, num_workers = 8, shuffle = True)

    # def show_example(train_loader):
    samples, labels = next(iter(train_loader))
    classes = {0: 'cat', 1: 'dog'}
    fig = plt.figure(figsize = (16, 24))
    for i in range(24):
        a = fig.add_subplot(4, 6, i + 1)
        a.set_title(classes[labels[i].item()])
        a.axis('off')
        a.imshow(np.transpose(samples[i].numpy(), (1, 2, 0)))
    plt.subplots_adjust(bottom = 0.2, top = 0.6, hspace = 0)
    print("a")
    # show_example(train_loader)


if __name__ == '__main__':
    run()