import torch
import torchvision
from torchvision.transforms import transforms


class mnist:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def gettrain(self):
        # 加载 FashionMNIST 数据集
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=60, shuffle=True)
        return trainloader

    def gettest(self):
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=60, shuffle=False)
        return testloader
