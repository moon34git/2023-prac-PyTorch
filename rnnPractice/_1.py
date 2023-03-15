import torch
import torchtext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
from torchtext.datasets import IMDB

start = time.time()
train_iter = iter(IMDB(split='train'))
print(next(train_iter)[0])
