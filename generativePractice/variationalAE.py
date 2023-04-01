import datetime
import os
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input1 = nn.Linear(input_dim, hidden_dim)
        self.input2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(.2)
        self.training = True
        
    def forward(self, x):
        h_ = self.LeakyReLU(self.input1(x))
        h_ = self.LeakyReLU(self.input2(h_))
        mean = self.mean(h_)
        log_var = self.var(h_)
        return mean, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.hidden1 = nn.Linear(latent_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(.2)
        
    def forward(self, x):
        h = self.LeakyReLU(self.hidden1(x))
        h = self.LeakyReLU(self.hidden2(h))
        x_hat = torch.sigmoid(self.output(h))
        return x_hat
    
class Model(nn.Module):
    def __init__(self, Encoder, Decoder, device):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.device = device
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z
    
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(z)
        return x_hat, mean, log_var
    
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss, KLD

def train(epoch, model, train_loader, optimizer, batch_size, x_dim, device, writer):
    model.train()
    train_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(device)
        
        optimizer.zero_grad()
        
        x_hat, mean, log_var = model(x)
        BCE, KLD = loss_function(x, x_hat, mean, log_var)
        loss = BCE + KLD
        
        writer.add_scalar('Train/Reconstruction Error', BCE.item(), batch_idx + epoch * (len(train_loader.dataset) / batch_size))
        writer.add_scalar('Train/KL-Divergence', KLD.item(), batch_idx + epoch * (len(train_loader.dataset) / batch_size))
        writer.add_scalar('Train/Total Loss', loss.item(), batch_idx + epoch * (len(train_loader.dataset) / batch_size))
        
    train_loss += loss.item()
    loss.backward()
    optimizer.step()
    
    if batch_idx % 100 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(x), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item() / len(x)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    
def test(epoch, model, test_loader, batch_size, x_dim, device, writer):
    
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(device)
            x_hat, mean, log_var = model(x)
            BCE, KLD = loss_function(x, x_hat, mean, log_var)
            loss = BCE + KLD
            
            writer.add_scalar('Test/Reconstruction Error', BCE.item(), batch_idx + epoch * (len(test_loader.dataset) / batch_size))
            writer.add_scalar('Test/KL-Divergence', KLD.item(), batch_idx + epoch * (len(test_loader.dataset) / batch_size))
            writer.add_scalar('Test/Total Loss', loss.item(), batch_idx + epoch * (len(test_loader.dataset) / batch_size))
        
            test_loss += loss.item()
            
            if batch_idx == 0:
                n = min(x.size(0), 8)
                comparison = torch.cat([x[:n], x_hat.view(batch_size, x_dim)[:n]])
                grid = torchvision.utils.make_grid(comparison.cpu())
                writer.add_image('Test image - Above: Real data, below: reconstruction data', grid, epoch)
                

def VAE():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "2"

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root = './Data', train = True, transform = transform, download = True)
    test_dataset = datasets.MNIST(root = './Data', train = False, transform = transform, download = True)

    train_loader = DataLoader(dataset = train_dataset, batch_size = 100, shuffle = True, num_workers = 4, pin_memory = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = 100, shuffle = False, num_workers = 4)
    
    x_dim = 784
    hidden_dim = 400
    latent_dim = 200
    epochs = 30
    batch_size = 100
    
    encoder = Encoder(x_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, x_dim)
    
    model = Model(encoder, decoder, device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    saved_loc = 'scalar/'
    writer = SummaryWriter(saved_loc)
    
    for epoch in tqdm(range(0, epochs)):
        train(epoch, model, train_loader, optimizer, batch_size, x_dim, device, writer)
        test(epoch, model, test_loader, batch_size, x_dim, device, writer)
        print('\n')
    writer.close()
    
if __name__ == '__main__':
    VAE()
    