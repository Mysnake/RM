import torch.nn as nn
import torch
from torch.nn import init
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TaylorLinearNet(nn.Module):

    def __init__(self, input_size, output_size,d):
        super(TaylorLinearNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.d = d
        # self.l0 = nn.Linear(in_features=self.output_size, out_features=1)
        #设计0阶导数


        #这组参数列表是用来构造1阶导数的和，为后续升阶做准备，
        self.d1 = nn.ParameterList(
            [nn.Linear(self.input_size, self.output_size, bias=False).to(device) for i in range(d)])
        #这组参数列表是用来升阶的
        self.dn =nn.ParameterList([
                    nn.ParameterList([
                        nn.ParameterList(
                    [nn.Linear(self.input_size, 1, bias=False).to(device),
                     # nn.Linear(self.output_size, self.input_size, bias=False).to(device)

                     init.kaiming_uniform_(nn.Parameter(torch.Tensor(self.output_size, self.input_size).to(device)), a=math.sqrt(5))
                     ]
                                        )
                    for j in range(i+1)])
                for i in range(d) ])

        # self.d0 = nn.Linear(1,self.output_size, bias=False).to(device)
        self.d0 = nn.Parameter(torch.Tensor(1, self.output_size).to(device))
        init.kaiming_uniform_(self.d0, a=math.sqrt(5))
        self.d1t = nn.Linear(self.input_size, self.output_size,bias=False).to(device)



    def forward(self, x):
        #0阶导数
        res = torch.zeros(x.size(0),self.output_size).to(device)+self.d0
        x = x.view(-1, self.input_size)              #(batch,imagesize)
        #1阶导数
        res += self.d1t(x)
        tx = torch.unsqueeze(x, dim=1)                  #(batch,1,imagesize)
        tx = tx.expand(-1, self.output_size, -1)        #(batch,classnumber,imagesize)
        for idx, (d1v,dnvs) in enumerate(zip(self.d1, self.dn)):
            d1 = d1v(x)
            dt = torch.unsqueeze(d1, dim=2)                   #(batch,classnumber,1)
            tmp_dt = dt
            #直接构造2阶导数
            for jdx,dnv in enumerate(dnvs):
                #tx*dnv[1]可以将复制10份的输入每个维度对应一个参数
                #tmp_dt*(tx*dnv[1])将之前的输入乘以参数的和乘以每个输入维度，也就是升阶
                #dnv[0]()这个线性操作主要目的是求和。
                tmp_dt = dnv[0](tmp_dt*(tx*dnv[1]))#维度为(batch，classnumber,1)
                #tmp_dt = torch.sum (tmp_dt*(tx*dnv[1]),dim=2,keepdim=True)#维度为（batch，class,1）
            # (tmp_dt*(tx*d))
            res += torch.squeeze(tmp_dt, dim=2)
        return res




