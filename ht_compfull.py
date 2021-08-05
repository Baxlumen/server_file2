import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR

import torchvision
import torchvision.transforms as transforms
from torchvision import models

import tensorly as tl
import tensorly
from itertools import chain
from tensorly import unfold
from tensorly.decomposition import *
from scipy.linalg import svd
from scipy.linalg import norm
import matplotlib.pyplot as plt
import os
import numpy as np
import time

print('==> Loading data..')

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
'''
def sig(x):
	print("activate!")
	return torch.sigmoid(x)
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        #x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.softmax(x)        
        return output

def ht(X,rank):

    U=[0 for x in range(0,2)]
    B=[0 for x in range(0,1)]
    x_mat = unfold(X,0)
    U_,_,_=svd(x_mat)
    U[0]=U_[:,:rank[0]]
        
    x_mat = unfold(X,1)
    U_,_,_=svd(x_mat)
    U[1]=U_[:,:rank[1]]
    U[0]=torch.from_numpy(U[0])
    U[1]=torch.from_numpy(U[1])
    
    B[0] = tl.tenalg.multi_mode_dot(X,(U[0],U[1]),[0,1],transpose=True)

    return U[0],U[1],B[0]

def ht_decomposition_fc_layer(layer, rank):
    l,r,core = ht(layer.weight.data, rank=rank)
    print(core.shape,l.shape,r.shape)
            
    right_layer = torch.nn.Linear(r.shape[0], r.shape[1])
    core_layer = torch.nn.Linear(core.shape[1], core.shape[0])
    left_layer = torch.nn.Linear(l.shape[1], l.shape[0])
    
    left_layer.bias.data = layer.bias.data
    left_layer.weight.data = l
    right_layer.weight.data = r.T
    core_layer.weight.data = core.T  #这句如果不加几乎无影响(暂时)

    new_layers = [right_layer, core_layer, left_layer]
    return nn.Sequential(*new_layers)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

tl.set_backend('pytorch')

ht_net = Net()
linear_layer=ht_net._modules['fc1']
#rank = min(linear_layer.weight.data.numpy().shape) //2
ht_net._modules['fc1']=ht_decomposition_fc_layer(linear_layer, [64,64])

full_net = Net()
print(full_net)
print(ht_net)
print("ht_net have {} paramerters in total".format(sum(x.numel() for x in ht_net.parameters())))
print("full_net have {} paramerters in total".format(sum(x.numel() for x in full_net.parameters())))

#  对于ht_net 的训练
if torch.cuda.is_available():
    #ht_net.cuda()#将所有的模型参数移动到GPU上
    full_net.cuda()

EPOCH = 20
LR = 0.0001
loss_func = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(ht_net.parameters(), LR)
optimizer = torch.optim.Adam(full_net.parameters(), LR)
running_loss=0.0

for epoch in range(EPOCH):
    time_start=time.time()
    for step, (image,label) in enumerate(trainloader):
        optimizer.zero_grad()
        image=image.to(device)
        label = label.to(device)
        output = full_net(image)
        loss = loss_func(output, label) # ouput 是float32 ， label是int64
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if step %100 == 0:
            print('[%d, %5d] loss: %.3f'% (epoch,step,running_loss/100))
            running_loss=0.0
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
print('Finished train')

correct = 0
total = 0
with torch.no_grad():   #关掉计算图，否则W的值会改变
    for data in testloader:  
        images,labels=data
        images = images.to(device)
        labels = labels.to(device)
        output = full_net(images) #每一个output是 4*10 ，每一行为一次预测，10列对应10个数字的预测概率
        _,predicted=torch.max(output.data,1) #按照维度返回所有元素中的最大值  1 是行，返回值[a,b]  a是value，b是index
        total+=labels.size(0) #4 batchsize
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the test images: %f %%' % (100 * correct / total))

#精度 97.15%