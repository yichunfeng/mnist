#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:33:26 2019

@author: yi-chun
"""


# 數學科學院 1801210118 馮逸群 Yi-Chun,Feng


""" 
PARAMETER:

data source: http://yann.lecun.com/exdb/mnist/
   
1 input layer, 4 hidden layers, 1 output layer

input : 28*28 pixel
neuron number in first hidden layer:512
neuron number in second hidden layer:256
neuron number in third hidden layer:128
neuron number in fourth hidden layer:64
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


###### neural network (4 hidden layers)#######
class Net4(torch.nn.Module):     
    def __init__(self, n_feature, n_hidden1,n_hidden2,n_hidden3,n_hidden4,n_output):
        super(Net4, self).__init__()     
        self.layer1 = torch.nn.Linear(n_feature, n_hidden1) 
        self.layer2 = torch.nn.Linear(n_hidden1, n_hidden2)  
        self.layer3 = torch.nn.Linear(n_hidden2, n_hidden3)  
        self.layer4 = torch.nn.Linear(n_hidden3, n_hidden4)  
        self.out = torch.nn.Linear(n_hidden4, n_output)       
        # 4 hidden layers

    def forward(self, x):      
        x = torch.tanh(self.layer1(x))  
        x = torch.tanh(self.layer2(x))  
        x = torch.tanh(self.layer3(x))  
        x = torch.tanh(self.layer4(x))  
        x = torch.tanh(self.out(x))                 
        return x

net4 = Net4(n_feature=28*28, n_hidden1=512,\
          n_hidden2=256,n_hidden3=128,n_hidden4=64,n_output=10) 




###### neural network (3 hidden layers) #######
class Net3(torch.nn.Module):     
    def __init__(self, n_feature, n_hidden1,n_hidden2,n_hidden3,n_output):
        super(Net3, self).__init__()     
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

net3 = Net3(n_feature=28*28, n_hidden1=512,\
          n_hidden2=256,n_hidden3=128,n_output=10) 



###### neural network (2 hidden layers) #######
class Net2(torch.nn.Module):     
    def __init__(self, n_feature, n_hidden1,n_hidden2,n_output):
        super(Net2, self).__init__()     
        self.layer1 = torch.nn.Linear(n_feature, n_hidden1) 
        self.layer2 = torch.nn.Linear(n_hidden1, n_hidden2)  
         
        self.out = torch.nn.Linear(n_hidden2, n_output)       
        # 2 hidden layers

    def forward(self, x):      
        x = torch.tanh(self.layer1(x))  
        x = torch.tanh(self.layer2(x))  
        
        x = torch.tanh(self.out(x))                 
        return x

net2 = Net2(n_feature=28*28, n_hidden1=256,\
          n_hidden2=128,n_output=10) 

optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.02)  
optimizer3 = torch.optim.SGD(net3.parameters(), lr=0.02)  
optimizer4= torch.optim.SGD(net4.parameters(), lr=0.02)  

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










losses2=[]
acces2=[]
eval_losses2 = []
eval_acces2 = []

losses3=[]
acces3=[]
eval_losses3 = []
eval_acces3 = []

losses4=[]
acces4=[]
eval_losses4 = []
eval_acces4 = []

# 2 hidden layers
for epoch in range(40):
    # Training
    train_loss2=0
    train_acc2=0
    for data2 in train_loader:
        img2,label2=data2
        img2=img2.view(img2.size(0),-1)
        img2=Variable(img2)
        label2=Variable(label2)
        out2=net2(img2)
        loss2=loss_func(out2,label2)
    
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
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
    for data2 in test_loader:
        img2,label2=data2
        img2=img2.view(img2.size(0),-1)
        img2=Variable(img2)
        label2=Variable(label2)
        out2=net2(img2)
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


# 3 hidden layers
for epoch in range(40):
    # Training
    train_loss3=0
    train_acc3=0
    for data3 in train_loader:
        img3,label3=data3
        img3=img3.view(img3.size(0),-1)
        img3=Variable(img3)
        label3=Variable(label3)
        out3=net3(img3)
        loss3=loss_func(out3,label3)
    
        optimizer3.zero_grad()
        loss3.backward()
        optimizer3.step()
        _, pred = out3.max(1)
        num_correct3 = (pred == label3).sum().item()
        #acc = num_correct / img.shape
        ACC3=num_correct3/ img3.shape[0]
        train_acc3 += ACC3
        train_loss3+=loss3.item()
        #count+=1
        #print(count)
        #if count%50==0:
            #print('epoch:{},train_loss:{:.6f}'.format(count,\
                  #train_loss/(len(train_dataset))))
    losses3.append(train_loss3/(len(train_loader)))
    acces3.append(train_acc3/(len(train_loader)))
    print('Epoch {} Train Loss3 {} Train  Accuracy3 {}'.format(
        epoch+1, train_loss3 / len(train_loader),train_acc3 / len(train_loader)))





    
    # Testing
    eval_loss3=0
    eval_acc3=0
    for data3 in test_loader:
        img3,label3=data3
        img3=img3.view(img3.size(0),-1)
        img3=Variable(img3)
        label3=Variable(label3)
        out3=net3(img3)
        loss3=loss_func(out3,label3)
    
        _, pred =out3.max(1)
        num_correct3 = (pred == label3).sum().item()
        Acc3=num_correct3/ img3.shape[0]
        eval_acc3 += Acc3
        eval_loss3+=loss3.item()
    
    eval_losses3.append(eval_loss3 / len(test_loader))
    eval_acces3.append(eval_acc3 / len(test_loader))  
    
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
            eval_loss3 / (len(test_loader)),
            eval_acc3 / len(test_loader)))



# 4 hidden layers
for epoch in range(40):
    # Training
    train_loss4=0
    train_acc4=0
    for data4 in train_loader:
        img4,label4=data4
        img4=img4.view(img4.size(0),-1)
        img4=Variable(img4)
        label4=Variable(label4)
        out4=net4(img4)
        loss4=loss_func(out4,label4)
    
        optimizer4.zero_grad()
        loss4.backward()
        optimizer4.step()
        _, pred = out4.max(1)
        num_correct4 = (pred == label4).sum().item()
        #acc = num_correct / img.shape
        ACC4=num_correct4/ img4.shape[0]
        train_acc4 += ACC4
        train_loss4+=loss4.item()
        #count+=1
        #print(count)
        #if count%50==0:
            #print('epoch:{},train_loss:{:.6f}'.format(count,\
                  #train_loss/(len(train_dataset))))
    losses4.append(train_loss4/(len(train_loader)))
    acces4.append(train_acc4/(len(train_loader)))
    print('Epoch {} Train Loss4 {} Train  Accuracy4 {}'.format(
        epoch+1, train_loss4 / len(train_loader),train_acc4 / len(train_loader)))





  
    # Testing
    eval_loss4=0
    eval_acc4=0
    for data4 in test_loader:
        img4,label4=data4
        img4=img4.view(img4.size(0),-1)
        img4=Variable(img4)
        label4=Variable(label4)
        out4=net4(img4)
        loss4=loss_func(out4,label4)
    
        _, pred =out4.max(1)
        num_correct4 = (pred == label4).sum().item()
        Acc4=num_correct4/ img4.shape[0]
        eval_acc4 += Acc4
        eval_loss4+=loss4.item()
    
    eval_losses4.append(eval_loss4 / len(test_loader))
    eval_acces4.append(eval_acc4 / len(test_loader))  
    
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
            eval_loss4 / (len(test_loader)),
            eval_acc4 / len(test_loader)))






# Result
plt.figure(num=1, figsize=(8, 8))
#train acc
ax1=plt.subplot(211)
ax1.plot(np.arange(len(acces2)),acces2,'r',label='train_2 layers')
ax1.plot(np.arange(len(acces3)),acces3,'b',label='train_3 layers')
ax1.plot(np.arange(len(acces4)),acces4,'g',label='train_4 layers')
plt.legend(loc='lower right')
#plt.xlabel('epoch')
plt.ylabel('train accuracy')

# test acc
ax2=plt.subplot(212)
ax2.plot(np.arange(len(eval_acces2)),eval_acces2,'r',label='test_2 layers')
ax2.plot(np.arange(len(eval_acces3)),eval_acces3,'b',label='test_3 layers')
ax2.plot(np.arange(len(eval_acces4)),eval_acces4,'g',label='test_4 layers')
plt.legend(loc='lower right')
plt.xlabel('epoch')
plt.ylabel('test accuracy')


plt.figure(num=2, figsize=(8, 8))
#train loss
ax3=plt.subplot(211)
ax3.plot(np.arange(len(eval_acces2)),losses2,'r',label='train_2 layers')
ax3.plot(np.arange(len(eval_acces3)),losses3,'b',label='train_3 layers')
ax3.plot(np.arange(len(eval_acces4)),losses4,'g',label='train_4 layers')
plt.legend(loc='upper right')
#plt.xlabel('epoch')
plt.ylabel('train loss')


# test loss
ax4=plt.subplot(212)
ax4.plot(np.arange(len(eval_losses2)),eval_losses2,'r',label='test_2 layers')
ax4.plot(np.arange(len(eval_losses3)),eval_losses3,'b',label='test_3 layers')
ax4.plot(np.arange(len(eval_losses4)),eval_losses4,'g',label='test_4 layers')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('test loss')


# 2 hidden layers
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

# 4 hidden layers
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




