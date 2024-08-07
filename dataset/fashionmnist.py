import torch
import torchvision
from torchvision.transforms import transforms
import torch.nn as nn


class fashionmnist:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def gettrain(self):
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        return trainloader

    def gettest(self):
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
        return testloader