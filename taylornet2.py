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
        self.l0 = nn.Linear(in_features=self.output_size, out_features=1)
        #设计0阶导数


        #这组参数列表是用来构造1阶导数的和，为后续升阶做准备，
        # self.d1 = nn.ParameterList(
        #     [nn.Linear(self.input_size, self.output_size, bias=False).to(device) for i in range(d)])
        #这组参数列表是用来升阶的
        self.dn =nn.ParameterList([
                        nn.ParameterList(
                    [nn.Linear(self.input_size, 1, bias=False).to(device),
                     init.kaiming_uniform_(nn.Parameter(torch.Tensor(self.output_size, self.input_size).to(device)), a=math.sqrt(5))])
                for i in range(d)])

        # self.d0 = nn.Linear(1,self.output_size, bias=False).to(device)
        self.d0 = nn.Parameter(torch.Tensor(1, self.output_size).to(device))
        init.kaiming_uniform_(self.d0, a=math.sqrt(5))
        self.d1t = nn.Linear(self.input_size, self.output_size,bias=False).to(device)


    def forward(self, x):
        #0阶导数
        res = torch.zeros(x.size(0),self.output_size).to(device)+self.d0
        x = x.view(-1, self.input_size)                 #(batch,imagesize)
        #1阶导数
        d1t = self.d1t(x)
        res += d1t
        tmp_dt = torch.unsqueeze(d1t, dim=2)
        tx = torch.unsqueeze(x, dim=1)                  #(batch,1,imagesize)
        tx = tx.expand(-1, self.output_size, -1)        #(batch,classnumber,imagesize)
        for idx, dnvs in enumerate(self.dn):
            # d1 = d1v(x)
            # dt = torch.unsqueeze(d1, dim=2)                   #(batch,classnumber,1)
            # tmp_dt = dt
            #直接构造2阶导数
            # for jdx,dnv in enumerate(dnvs):
            # tmp_dt = dnvs[0](tmp_dt*(tx*dnvs[1]))#维度为（batch，class,1）
            tmp_dt = torch.sum(tmp_dt * (tx * dnvs[1]), dim=2, keepdim=True)  # 维度为（batch，class,1）
            res += torch.squeeze(tmp_dt, dim=2)
        return res






class TaylorLinearNet2(nn.Module):

    def __init__(self, input_size, output_size,d):
        super(TaylorLinearNet2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.d = d

        # self.d0 = nn.Linear(1,self.output_size, bias=False).to(device)
        self.d0 = init.kaiming_uniform_(nn.Parameter(torch.Tensor(1, self.output_size).to(device)))
        init.kaiming_uniform_(self.d0, a=math.sqrt(5))
        self.allweight = init.kaiming_uniform_(nn.Parameter(torch.Tensor(self.output_size, self.d,self.input_size).to(device)), a=math.sqrt(5))


    def forward(self, x):
        #0阶导数
        res = torch.zeros(x.size(0),self.output_size).to(device)+self.d0
        x = x.view(-1, self.input_size)                      #(batch,imagesize)
        # x = x.view(-1, self.input_size)-self.point                      #(batch,imagesize)
        x = torch.unsqueeze(x, dim=1)                            #(batch,1,imagesize)
        x = x.expand(-1, self.output_size*self.d, -1)            #(batch,classnumber,imagesize)
        x = x.view(-1, self.output_size, self.d, self.input_size)
        x = x * self.allweight                                   #(batch,classnumber,d,imagesize)
        x = x.sum(3)
        x = x.cumprod(2)
        x = x.sum(2)
        res += x
        return res


# x = torch.arange(0, 6 * 5 * 4).view(6, 4, 5).to(device)
# x = torch.randn((6, 4, 5)).to(device)
# model = TaylorLinearNet(input_size=5*4, output_size=3, d=4).to(device)
# model(x)
