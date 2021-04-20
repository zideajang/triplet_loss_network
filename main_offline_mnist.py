import torch

from torch.utils.data import DataLoader,TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE

from src.net.triplet_net import TripletNetwork
from src.mininig import offline
from src.loss_fn import triplet_loss


if __name__ == "__main__":
    device = torch.device('cuda:0')