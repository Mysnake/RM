import time

import torch
import torch.nn as nn

from torch import optim


def getopimizer(paramters, lr, opimizername):
    if opimizername == "sgd":
        return optim.SGD(paramters, lr=lr)
    if opimizername == "adam":
        return optim.Adam(paramters, lr=lr)
    if opimizername == "adamw":
        return optim.AdamW(paramters, lr=lr)
    if opimizername == "L-BFGS":
        return optim.LBFGS(paramters, lr=lr)


def train(model=None, parameters=None, dataset=None, timenumber = None,device=None,datasetstr=None ,optname="sgd"):
    model.to(device)
    if datasetstr is None:
        raise "dataset is None"
    if timenumber is None:
        raise "timenumber is None"
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = getopimizer(model.parameters(), lr=parameters['lr'], opimizername=optname)
    # optimizer = optim.SGD(model.parameters(), lr=parameters['lr'])
    # 训练模型
    trainlosslist = []
    test_loss_list = []
    traintimelist = []
    acc_list = []
    infertimelist = []
    s = time.time()
    opttimelist = []
    for epoch in range(parameters["epochs"]):  # 迭代5次
        totalopttime = 0.0
        running_loss = 0.0
        optcount = 0
        strain = time.time()
        for i, data in enumerate(dataset["trainloader"], 0):
            optcount += 1
            inputs, labels = data[0].to(device), data[1].to(device)  # 将数据移动到GPU
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            sopttime = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalopttime += time.time() - sopttime
            running_loss += loss.item()
            if i % 100 == 99:  # 每100个小批量数据打印一次损失值
                opttimelist.append(totalopttime / optcount)
                # print("平均优化时间: {}".format(totalopttime / optcount))
                optcount = 0
                totalopttime = 0
                trainlosslist.append(running_loss / 100)
                running_loss = 0.0
        traintimelist.append(time.time() - strain)
        print(('[%d/%d/%d]  {}, {}'.format(optname,datasetstr)) % (epoch + 1, parameters["epochs"],timenumber))
        print('Finished Training Epoch {}'.format(epoch + 1))

        # 在测试集上验证模型性能
        correct = 0
        total = 0
        test_loss = 0.0
        infertime = time.time()
        with torch.no_grad():
            for data in dataset["testloader"]:
                images, labels = data[0].to(device), data[1].to(device)  # 将数据移动到GPU
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        infertimelist.append(time.time() - infertime)
        # print('testloss of the network on the 10000 test images: {}'.format(test_loss / 100))
        test_loss_list.append(test_loss / 100)
        # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        print('Accuracy: {} %'.format(round(100 * correct / total, 2)))
        acc_list.append(round(100 * correct / total, 2))
        # print(total_params)
    e = time.time() - s
    print("第{}次的总时间: {}".format(timenumber,e))
    return {"acc_list": acc_list, #每个epoch计算一次精度
            "train_loss_list": trainlosslist,#每次100次迭代计算一次，一个epoch1000次迭代，也就是一个epoch有10个数据
            "test_loss_list": test_loss_list,#每个epoch计算一次testloss，缩小了100倍
            "infer_time_list": infertimelist,#每个epoch计算一次推理时间
            "total_time": e,                 #一次实验计算一次总时间
            "total_params": total_params,    #参数量
            "train_time_list": traintimelist,#一个epoch一次训练时间，
            "opt_time_list": opttimelist     #100次迭代的优化时间的均值，也就是一个epoch有10个数据
            }
