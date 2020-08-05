#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:33:26 2019

@author: yi-chun
"""


# 數學科學院 1801210118 馮逸群 Yi-Chun, Feng


""" 
PARAMETER:

data source: http://yann.lecun.com/exdb/mnist/
   
1 input layer, 3 hidden layers, 1 output layer

input : 28*28 pixel
neuron number in first hidden layer:512
neuron number in second hidden layer:256
neuron number in third hidden layer:128
class in output:10    

activation function:tanh(x)    (including output)
learning rate:0.02
epoch:40

"""




import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


###### neural network #######
class Net(torch.nn.Module):     
    def __init__(self, n_feature=28*28, n_hidden1=512,n_hidden2=256\
                 ,n_hidden3=128,n_output=10):

        super(Net, self).__init__()     
        self.layer1 = torch.nn.Linear(n_feature, n_hidden1) 
        self.layer2 = torch.nn.Linear(n_hidden1, n_hidden2)  
        self.layer3 = torch.nn.Linear(n_hidden2, n_hidden3)         
        self.out = torch.nn.Linear(n_hidden3, n_output)       
        # 3 hidden layers

    def forward(self, x):      
        x = torch.tanh(self.layer1(x))  
        x = torch.tanh(self.layer2(x))  
        x = torch.tanh(self.layer3(x))  
        x = torch.tanh(self.out(x))                 
        return x


##### optimization ######
#net_Adam = Net()
#opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=0.02, betas=(0.9, 0.99))
#optimizer = opt_Adam 


net_RMSprop = Net()
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=0.02, alpha=0.9)
optimizer = opt_RMSprop 

loss_func = torch.nn.CrossEntropyLoss()
BATCH_SIZE = 64


# MNIST DATA

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(\
    root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)










losses=[]
acces=[]
eval_losses = []
eval_acces = []




for epoch in range(40):
    # Training
    train_loss=0
    train_acc=0
    for data in train_loader:
        img,label=data
        img=img.view(img.size(0),-1)
        img=Variable(img)
        label=Variable(label)
        out=net(img)
        loss=loss_func(out,label)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        ACC=num_correct/ img.shape[0]
        train_acc += ACC
        train_loss+=loss.item()
        
    losses.append(train_loss/(len(train_loader)))
    acces.append(train_acc/(len(train_loader)))
    print('Epoch {} Train Loss {} Train  Accuracy {}'.format(
        epoch+1, train_loss / len(train_loader),train_acc / len(train_loader)))





    
    # Testing
    eval_loss=0
    eval_acc=0
    for data in test_loader:
        img,label=data
        img=img.view(img.size(0),-1)
        img=Variable(img)
        label=Variable(label)
        out=net(img)
        loss=loss_func(out,label)
    
        _, pred =out.max(1)
        num_correct = (pred == label).sum().item()
        Acc=num_correct/ img.shape[0]
        eval_acc += Acc
        eval_loss+=loss.item()
    
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))  
    
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
            eval_loss / (len(test_loader)),
            eval_acc / len(test_loader)))



# Result
ax1=plt.subplot(211)
ax1.plot(np.arange(len(acces)),acces,'r',label='train')
ax1.plot(np.arange(len(eval_acces)),eval_acces,'g',label='test')
plt.legend(loc='lower right')
plt.xlabel('epoch')
plt.ylabel('accuracy')


ax2=plt.subplot(212)
ax2.plot(np.arange(len(losses)),losses,'r',label='train')
ax2.plot(np.arange(len(eval_losses)),eval_losses,'g',label='test')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
