# Hand Written Digits Classification by Multilayer Perceptron with PyTorch

This is a quick start for analyzing how the model affects the performance of classification.
The control group is a three-layer perceptron with:
* Batch size: 64
* Activation function: tanh(x)
* Loss function: cross entropy
* Optimizer: stochastic gradient descent
* Learning rate: 0.02
* Training epoch: 40
The treatment groups would change the number of layers, the optimizer, or add regularization, weight initialization. 


## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) 


## Requirements
* Python 3.7.4
* PyTorch v1.5.0
* NumPy
* matplotlib

## Data Preprocess

Transforming images to tensor, and normalizing the images to range [-1, 1]

```
data_tf = transforms.Compose( [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
```

## Performance of Control Group


<img src="https://github.com/yichunfeng/mnist/blob/master/%20mlp_3layer.png" width="500" height="400">

## Treatment Groups 

### Different Numbers of Layers
Experiment in mlp.py demonstrates the performance of 4-layer, 3-layer and 2-layer. 
We kept other conditions remain the same, and only change the number of layers.
The following are the output sizes of each layer.
The 4-layer perceptron:
```
net4 = Net4(n_feature=28*28, n_hidden1=512,\
          n_hidden2=256,n_hidden3=128,n_hidden4=64,n_output=10) 
```
The 3-layer perceptron:
```
net3 = Net3(n_feature=28*28, n_hidden1=512,\
          n_hidden2=256,n_hidden3=128,n_output=10) 
```
The 2-layer perceptron:
```
net2 = Net2(n_feature=28*28, n_hidden1=256,\
          n_hidden2=128,n_output=10) 
```

The comparisons of accuracy and loss are presented below:

<img src="https://github.com/yichunfeng/mnist/blob/master/Accuracy_Layer.png" width="500" height="400">

<img src="https://github.com/yichunfeng/mnist/blob/master/Loss_Layer.png" width="500" height="400">

The following are the illustrations how the relation between the training the testing could be affected when we add or delete one layer:

If we add one layer:

![image](https://github.com/yichunfeng/mnist/blob/master/plus_1_layer.png)

If we delete one layer:

![image](https://github.com/yichunfeng/mnist/blob/master/minus_1_layer.png)


### Different Optimizer
optimization.py and sgd_bgd.py provide different optimizers including Batch Gradient Descent, Root Mean Square Prop and Adaptive Moment Estimation for the multilayer perceptron.
In sgd_bgd.py we simply replace the batch size:
```
# SGD
BATCH_SIZE = 1
# BGD
BATCH_SIZE2 = 60000
```

In optimization.py we try RMSprop and Adam with batch size 64.
```
net_Adam = Net()
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=0.02, betas=(0.9, 0.99))
optimizer = opt_Adam 

net_RMSprop = Net()
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=0.02, alpha=0.9)
optimizer = opt_RMSprop 
```

### Regularization

We added regularizer L1 and L2 in regularization.py.

```
lambda1, lambda2 = 0.5, 0.01

def l1_penalty(var):
    return torch.abs(var).sum()


def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())
```
Therefore, the loss function becomes
```
loss=loss_func(out,label)+lambda1 * l1_penalty(out)
```
and
```
loss=loss_func(out,label)+lambda2 * l2_penalty(out)
```

### Weight Initialization

We applied weight initialization in weight_initialization.py to compare the difference between weights generated from normal distribution and weights by Xavier initialization.

```
def init_weights_xavier(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.01)


def init_weights_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight.data, mean=0, std=1)
        m.bias.data.fill_(0.01)
```
