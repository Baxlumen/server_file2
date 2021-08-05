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
from tensorly.decomposition import parafac, partial_tucker, tucker

import numpy as np
import time
from torch.autograd import Variable

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=400, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
cnttt = 0

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=5,
                        stride=1,
                        padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.out = nn.Linear(32 * 7 * 7, 10)
        # self.conv1.parameters.

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output




def tucker_decomposition_conv_layer(layer, ranks):
    core, [last, first]  = partial_tucker(layer.weight.data, modes=[0, 1], rank=ranks, init='svd')
    # core, factors = partial_tucker(layer.weight.data, modes=[0, 1, 2, 3], rank=ranks, init='svd')
    #print(core.shape, last.shape, first.shape)

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                out_channels=first.shape[1], kernel_size=1,
                stride=1, padding=0, dilation=layer.dilation, bias=False)

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1],
                out_channels=core.shape[0], kernel_size=layer.kernel_size,
                stride=layer.stride, padding=layer.padding, dilation=layer.dilation, bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
                out_channels=last.shape[0], kernel_size=1, stride=1,
                padding=0, dilation=layer.dilation, bias=True)

    last_layer.bias.data = layer.bias.data

    first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    # for i, l in enumerate(new_layers):
    #    print(i, l.weight.data.shape)
    return nn.Sequential(*new_layers)

def decompose():
    model = torch.load("model").to(device)
    model.eval()
    model.cpu()
    layers = model._modules
    for i, key in enumerate(layers.keys()):
        if i >= len(layers.keys()):
            break
        if isinstance(layers[key], torch.nn.modules.conv.Conv2d):
            conv_layer = layers[key]
            rank = max(conv_layer.weight.data.numpy().shape) // 10
            ranks = [max(int(np.ceil(conv_layer.weight.data.numpy().shape[0] / 3)), 1),
                     max(int(np.ceil(conv_layer.weight.data.numpy().shape[1] / 3)), 1),
                     max(int(np.ceil(conv_layer.weight.data.numpy().shape[2] / 3)), 1),
                     max(int(np.ceil(conv_layer.weight.data.numpy().shape[3] / 3)), 1)]
            layers[key] = tucker_decomposition_conv_layer(conv_layer, ranks)
        torch.save(model, 'model')
    return model

def build(decomp=True):
    print('==> Building model..')
    tl.set_backend('pytorch')
    full_net = CNNNet()
    full_net = full_net.to(device)
    torch.save(full_net, 'model')
    if decomp:
        decompose()
    net = torch.load("model").to(device)
    print('==> Done')
    return net

def gen_noises(model,  layer_ids, std=1, co_matrices=None):
    noises = []
    for i, param in enumerate(model.parameters()):
        if i in layer_ids:
            if co_matrices == None:
                noises.append(torch.randn_like(param) * std) #生成与 param shape一样的随机tensor
            else:
                sz = co_matrices[i].shape[0]
                m = MultivariateNormal(torch.zeros(sz), co_matrices[i])
                noise = m.sample()
                noises.append(noise.reshape(param.shape))
        else:
            noises.append(torch.zeros_like(param))
        noises[-1] = noises[-1].to(device)
    return noises

def es_update(model, epsilons, ls, lr, layer_ids, mode=1, update=True):
    #         模型， 随机出来的tensor，根据随机tensor计算的 loss，学习率,层数【0~9】， mode=2 , updata = True
    device = epsilons[0][0].device
    num_directions = len(epsilons) #40
    elite_rate = 0.2
    elite_num = max(int(elite_rate * num_directions), 1)  #8

    ls = torch.tensor(ls).to(device)
    if mode == 1:
        weight = ls
    else:
        weight = 1 / (ls + 1e-8)
    indices = torch.argsort(weight)[-elite_num:]
    mask = torch.zeros_like(weight)
    mask[indices] = 1

    weight *= mask
    weight = weight / torch.sum(weight)

    grad = []
    for l in epsilons[0]:
        grad.append(torch.zeros_like(l))

    for idx in indices:
        for i, g in enumerate(epsilons[idx]):
            grad[i] += g * weight[idx]
    if update:
        if mode==1:
            i = 0
            for g, param in zip(grad, model.parameters()):
                if i in layer_ids:
                    param.requires_grad = False
                    param -= lr * g
                    param.requires_grad = True
                i += 1
        else:
            i = 0
            # print(len(grad), layer_ids)
            for g, param in zip(grad, model.parameters()):
                if i in layer_ids:
                    # print("update")
                    param.requires_grad = False
                    param += lr * g
                    param.requires_grad = True
                i += 1

    return grad



def add_noises(model, noises, layer_ids):
    i = 0
    for param, noise in zip(model.parameters(), noises):
        if i in layer_ids:
            param.requires_grad = False
            param += noise
            param.requires_grad = True
        i += 1



def remove_noises(model, noises, layer_ids):
    i = 0
    for param, noise in zip(model.parameters(), noises):
        if i in layer_ids:
            param.requires_grad = False
            param -= noise
            param.requires_grad = True
        i += 1

def explore_one_direction(model, inputs, targets, criterion, layer_ids, co_matrices, return_list, if_mirror):
    #               模型   输入[400,1,28,28] 标签：400 ，损失函数，[0~9]   None     用于写入文件  result  True
    ep_rt = []
    ls_rt = []

    epsilon = gen_noises(model, layer_ids, std=0.01, co_matrices=co_matrices) #epsilon len(layer_ids) 随机的tensor
    add_noises(model, epsilon, layer_ids) # 权重矩阵 = 权重矩阵 + 随机矩阵
    outputs = model(inputs) # 前向
    loss = criterion(outputs, targets).item()  #计算损失 （item-> 将tensor 转化为浮点数）
    remove_noises(model, epsilon, layer_ids)  # 权重矩阵 = 权重矩阵 - 随机矩阵

    ep_rt.append(epsilon.copy()) #随机矩阵 list
    ls_rt.append(loss)  # loss 的list

    if if_mirror:        
        for i in range(len(epsilon)): # 每个随机 tensor取数相反数
            epsilon[i] = -epsilon[i]
        add_noises(model, epsilon, layer_ids) #添加 随机数
        outputs = model(inputs)  #前向
        loss = criterion(outputs, targets).item() #计算loss
        remove_noises(model, epsilon, layer_ids) #移除随机数
        ep_rt.append(epsilon)
        ls_rt.append(loss)

    return ep_rt, ls_rt

def test(test_acc, best_acc, model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        print('|', end='')
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 10 == 0:
                print('=', end='')
    acc = 100. * correct / total
    print('|', 'Accuracy:', acc, '% ', correct, '/', total)
    test_acc.append(correct / total)
    return max(acc, best_acc)

full_net = build(decomp=True)

# train之前定义一些参数
model = full_net
num_epoch = 40
lr = 0.5
lr0 = lr
step_size = 3
gamma = 0.5
co_matrices = None
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.001) #优化函数
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.CrossEntropyLoss() #损失函数
    # print(model.parameters())
num_layers = len(model.state_dict()) # 有 num_layers 层  10

train_acc = []
test_acc = []
best_acc = 0
global_mean = []
global_var = []

model.train()
model = model.to(device)
model.share_memory()
early_break = False

es_mode = 2
num_directions = 40
num_directions0 = num_directions

if_alternate = False
fall_cnt = 0
if_es = True
if_bp = False
if_mirror = True

layer_ids = list(range(num_layers))   # 0~9  一共是10层
num_directions = num_directions0   #40
if if_mirror:
    num_directions = num_directions // 2 #  变成  20
lr = lr0  #  0.5

for epoch in range(num_epoch):
    print("\nES layer ", "alternate" if if_alternate else layer_ids, "  Epoch: {}".format(epoch))
    print("|", end="")
    train_loss = 0
    correct = 0
    total = 0            
    epoch_mean = []
    epoch_var = []

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if if_alternate:
            layer_ids = [layer_id]
            layer_id = (layer_id + 1) % num_layers
            print("if_alternate if perform!~~~")
        total += len(inputs)
        ls = []
        epsilons = []
        processes = []
        result = []

        inputs, targets = inputs.to(device), targets.to(device)
        inputs = Variable(inputs)
        inputs.requires_grad = True

        for _ in range(num_directions):
            epsilon, loss = explore_one_direction(model, inputs, targets, criterion, layer_ids, co_matrices, result, if_mirror)
            epsilons.extend(epsilon)  # 每次循环里面有两个 epsilon，一正，一个相反数
            ls.extend(loss)   # 每次也就有两个 loss
            for l in loss:
                train_loss += l

        es_grad = es_update(model, epsilons, ls, lr, layer_ids, es_mode, update=if_es)

        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 10 == 0:
            print('=', end='')
    print('|', 'Accuracy:', 100. * correct / total, '% ', correct, '/', total)
    best_acc = test(test_acc, best_acc, model)

    if epoch % step_size == 0 and epoch:
        lr *= gamma
        lr = max(lr, 0.0125)
        if epoch % (step_size * 2) == 0: 
            num_directions = max(int(num_directions/gamma), num_directions + 1)
        pass
    train_acc.append(correct / total)

    if epoch >= 2:
        if train_acc[-1] - train_acc[-2] < 0.01 and train_acc[-2] - train_acc[-3] < 0.01:
            fall_cnt += 1
        else:
            fall_cnt = 0

    print('Current learning rate: ', lr)
    print('Current num_directions: ', num_directions, "mirror" if if_mirror else "")

    print('Best training accuracy overall: ', best_acc)
