#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:33:26 2019

@author: yi-chun
"""


# Yi-Chun, Feng


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
    def __init__(self, n_feature, n_hidden1,n_hidden2,n_hidden3,n_output):#\
                 #n_hidden4, n_output):
        super(Net, self).__init__()     
        self.layer1 = torch.nn.Linear(n_feature, n_hidden1) 
        self.layer2 = torch.nn.Linear(n_hidden1, n_hidden2)  
        self.layer3 = torch.nn.Linear(n_hidden2, n_hidden3)  
        #self.layer4 = torch.nn.Linear(n_hidden3, n_hidden4)  
        self.out = torch.nn.Linear(n_hidden3, n_output)       
        # 4 hidden layers

    def forward(self, x):      
        x = torch.tanh(self.layer1(x))  
        x = torch.tanh(self.layer2(x))  
        x = torch.tanh(self.layer3(x))  
        #x = F.relu(self.layer4(x))  
        x = torch.tanh(self.out(x))                 
        return x

net = Net(n_feature=28*28, n_hidden1=512,\
          n_hidden2=256,n_hidden3=128,n_output=10) #\
          #n_hidden4=20, n_output=10) 



optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  

loss_func = torch.nn.CrossEntropyLoss()

BATCH_SIZE = 1#SGD
BATCH_SIZE2 = 60000#BGD




# MNIST DATA

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(\
    root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_loader2 = DataLoader(train_dataset, batch_size=BATCH_SIZE2, shuffle=True)
test_loader2 = DataLoader(test_dataset, batch_size=BATCH_SIZE2, shuffle=False)








losses=[]
acces=[]
eval_losses = []
eval_acces = []

losses2=[]
acces2=[]
eval_losses2 = []
eval_acces2 = []





for epoch in range(2):
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
        #acc = num_correct / img.shape
        ACC=num_correct/ img.shape[0]
        train_acc += ACC
        train_loss+=loss.item()
        #count+=1
        #print(count)
        #if count%50==0:
            #print('epoch:{},train_loss:{:.6f}'.format(count,\
                  #train_loss/(len(train_dataset))))
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
    
    
    
for epoch in range(2):
    # Training
    train_loss2=0
    train_acc2=0
    for data2 in train_loader2:
        img2,label2=data2
        img2=img2.view(img2.size(0),-1)
        img2=Variable(img2)
        label2=Variable(label2)
        out2=net(img2)
        loss2=loss_func(out2,label2)
    
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()
        _, pred = out2.max(1)
        num_correct2 = (pred == label2).sum().item()
        #acc = num_correct / img.shape
        ACC2=num_correct2/ img2.shape[0]
        train_acc2 += ACC2
        train_loss2+=loss2.item()
        #count+=1
        #print(count)
        #if count%50==0:
            #print('epoch:{},train_loss:{:.6f}'.format(count,\
                  #train_loss/(len(train_dataset))))
    losses2.append(train_loss2/(len(train_loader)))
    acces2.append(train_acc2/(len(train_loader)))
    print('Epoch {} Train Loss2 {} Train  Accuracy2 {}'.format(
        epoch+1, train_loss2 / len(train_loader),train_acc2 / len(train_loader)))





    
    # Testing
    eval_loss2=0
    eval_acc2=0
    for data2 in test_loader2:
        img2,label2=data2
        img2=img2.view(img2.size(0),-1)
        img2=Variable(img2)
        label2=Variable(label2)
        out2=net(img2)
        loss2=loss_func(out2,label2)
    
        _, pred =out2.max(1)
        num_correct2 = (pred == label2).sum().item()
        Acc2=num_correct2/ img2.shape[0]
        eval_acc2 += Acc2
        eval_loss2+=loss2.item()
    
    eval_losses2.append(eval_loss2 / len(test_loader))
    eval_acces2.append(eval_acc2 / len(test_loader))  
    
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
            eval_loss2 / (len(test_loader)),
            eval_acc2 / len(test_loader)))



# Result
plt.figure(num=1, figsize=(8, 8))
#train acc
ax1=plt.subplot(211)
ax1.plot(np.arange(len(acces2)),acces2,'r',label='BGD')
ax1.plot(np.arange(len(acces)),acces,'b',label='SGD')

plt.legend(loc='lower right')
#plt.xlabel('epoch')
plt.ylabel('train accuracy')

# test acc
ax2=plt.subplot(212)
ax2.plot(np.arange(len(eval_acces2)),eval_acces2,'r',label='BGD')
ax2.plot(np.arange(len(eval_acces)),eval_acces,'b',label='SGD')

plt.legend(loc='lower right')
plt.xlabel('epoch')
plt.ylabel('test accuracy')


plt.figure(num=2, figsize=(8, 8))
#train loss
ax3=plt.subplot(211)
ax3.plot(np.arange(len(eval_acces2)),losses2,'r',label='BGD')
ax3.plot(np.arange(len(eval_acces)),losses,'b',label='SGD')

plt.legend(loc='upper right')
#plt.xlabel('epoch')
plt.ylabel('train loss')


# test loss
ax4=plt.subplot(212)
ax4.plot(np.arange(len(eval_losses2)),eval_losses2,'r',label='BGD')
ax4.plot(np.arange(len(eval_losses)),eval_losses,'b',label='SGD')

plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('test loss')


# SGD
plt.figure(num=3, figsize=(8, 8))
ax5=plt.subplot(211)
ax5.plot(np.arange(len(acces)),acces,'r',label='SGD train accuracy')
ax5.plot(np.arange(len(eval_acces)),eval_acces,'b',label='SGD test accuracy')
plt.legend(loc='lower right')
#plt.xlabel('epoch')
plt.ylabel('accuracy')


ax6=plt.subplot(212)
ax6.plot(np.arange(len(losses)),losses,'r',label='SGD train loss')
ax6.plot(np.arange(len(eval_losses)),eval_losses,'b',label='SGD test loss')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')



# BGD
plt.figure(num=4, figsize=(8, 8))
ax7=plt.subplot(211)
ax7.plot(np.arange(len(acces2)),acces2,'r',label='BGD train accuracy')
ax7.plot(np.arange(len(eval_acces2)),eval_acces2,'b',label='BGD test accuracy')
plt.legend(loc='lower right')
#plt.xlabel('epoch')
plt.ylabel('accuracy')


ax8=plt.subplot(212)
ax8.plot(np.arange(len(losses2)),losses2,'r',label='BGD train loss')
ax8.plot(np.arange(len(eval_losses2)),eval_losses2,'b',label='BGD test loss')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')



