# 定义神经网络模型
import argparse
import os

import torch
from train import train
from dataset.mnist import mnist
from dataset.fashionmnist import fashionmnist
import time
from taylormodel import gettaylormodel
import numpy as np
RANDOM_SEED = 42  # any random number


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def args():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='rm')

    # 添加命令行参数
    parser.add_argument('--dataset', default="mnist", type=str, help='mnist or fashionmnist')
    parser.add_argument('--epochs', type=int, default=100, help='epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--d', type=int,default=2, help='taylor order')
    parser.add_argument('--share', type=bool,default=False, help='taylor order')
    parser.add_argument('--optimizer', type=str,default="adam", help='sgd , adam or adamw')

    return parser.parse_args()

if __name__ == '__main__':
    set_seed(RANDOM_SEED)
    arg = args()
    if arg.dataset == 'mnist':
        dataset = {"trainloader": mnist().gettrain(),
                   "testloader": mnist().gettest()}
    else:
        dataset = {
            "trainloader": fashionmnist().gettrain(),
            "testloader": fashionmnist().gettest()
        }

    parameters = {
                "epochs": arg.epochs,
                  "lr": arg.lr, }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = gettaylormodel(no_share=not arg.share,inputsize=28*28,outputsize=10,d=arg.d)

    s = time.time()
    train(model=model, dataset=dataset, parameters=parameters, timenumber=1,datasetstr=arg.dataset,device=device,optname=arg.optimizer)
    print(time.time() - s)


