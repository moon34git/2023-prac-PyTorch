import imageio
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pylab as plt

from torchvision.utils import make_grid, save_image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# matplotlib.style.use('ggplot')

class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.main(x.view(-1, 784))
    
def save_generator_image(image, path):
    save_image(image, path)
    return
    
def train_discriminator(optimizer, real_data, fake_data, device, discriminator, criterion):
    b_size = real_data.size(0)
    real_label = torch.ones(b_size, 1).to(device)
    fake_label = torch.zeros(b_size, 1).to(device)
    optimizer.zero_grad()
    real_output = discriminator(real_data)
    real_loss = criterion(real_output, real_label)
    fake_output = discriminator(fake_data)
    fake_loss = criterion(fake_output, fake_label)
    real_loss.backward()
    fake_loss.backward()
    optimizer.step()
    return real_loss + fake_loss

def train_generator(optimizer, fake_data, device, discriminator, criterion):
    b_size = fake_data.size(0)
    real_label = torch.ones(b_size, 1).to(device)
    optimizer.zero_grad()
    output = discriminator(fake_data)
    loss = criterion(output, real_label)
    loss.backward()
    optimizer.step()
    return loss

def loss_diff(losses_g, losses_d):
    plt.figure()
    losses_g = [f1.item() for f1 in losses_g]
    losses_d = [f2.item() for f2 in losses_d]
    plt.plot(losses_g, label = 'Generator_Loss')
    plt.plot(losses_d, label = 'Discriminator_Loss')
    plt.legend()
    plt.show()
    
def GAN():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "2"

    batch_size = 512
    epochs = 100
    sample_size = 64
    nz = 128
    k = 1

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root = './Data', train = True, transform = transform, download = True)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    
    generator = Generator(nz).to(device)
    discriminator = Discriminator().to(device)
    
    optim_g  = optim.Adam(generator.parameters(), lr = 0.0002)
    optim_d = optim.Adam(discriminator.parameters(), lr = 0.0002)
    
    criterion = nn.BCELoss()
    
    losses_g = []
    losses_d = []
    images = []
    
    generator.train()
    discriminator.train()
    
    for epoch in range(epochs):
        loss_g = 0.0
        loss_d = 0.0
        for idx, data in tqdm(enumerate(train_loader), total = int(len(train_dataset) / train_loader.batch_size)):
            image, _ = data
            image = image.to(device)
            b_size = len(image)
            for _ in range(k):
                fake_data = generator(torch.randn(b_size, nz).to(device)).detach()
                real_data = image
                loss_d += train_discriminator(optim_d, real_data, fake_data, device, discriminator, criterion)
            fake_data = generator(torch.randn(b_size, nz).to(device))
            loss_g += train_generator(optim_g, fake_data, device, discriminator, criterion)
            
        generated_img = generator(torch.randn(b_size, nz).to(device)).cpu().detach()
        generated_img = make_grid(generated_img)
        if epoch % 10 == 0:
            save_generator_image(generated_img, './img/{}.png'.format(epoch))
        images.append(generated_img)
        epoch_loss_g = loss_g / idx
        epoch_loss_d = loss_d / idx
        
        losses_g.append(epoch_loss_g)
        losses_d.append(epoch_loss_d)
        
        print(f'Epoch {epoch} of {epochs}')
        print(f'Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}')
        
        # loss_diff(losses_g, losses_d)
        
    return
        
if __name__ == '__main__':
    GAN()