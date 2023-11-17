"""
TODO: Finish and submit your code for logistic regression and hyperparameter search.

"""
import torch
import torchvision
from torch.autograd import Variable

from posixpath import split
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
np.random.seed(1)


def load_MINST(batch_size_train):
    MNIST_training = torchvision.datasets.MNIST('/MNIST_dataset/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
 
    train_indices = range(0, 48000)
    val_indices = range(48000, 60000)
    MNIST_training_set = Subset(MNIST_training,train_indices)
    MNIST_validation_set = Subset(MNIST_training,val_indices)
    train_loader = torch.utils.data.DataLoader(MNIST_training_set,batch_size=batch_size_train, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(MNIST_validation_set,batch_size=batch_size_train, shuffle=True)
    

    return train_loader,validation_loader
# Multiple Linear regression
class MultipleLinearRegression_MNIST(nn.Module):
    def __init__(self):
        super(MultipleLinearRegression_MNIST, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# Multiple Linear regression
class MultipleLinearRegression_10(nn.Module):
    def __init__(self):
        super(MultipleLinearRegression_10, self).__init__()
        self.fc = nn.Linear(3*32*32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def train_MINSI(epoch,data_loader,model,optimizer,device,log_interval):
  for batch_idx, (data, target) in enumerate(data_loader):
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss_fn = nn.CrossEntropyLoss()
    temploss = loss_fn(output, target)
    #temploss = nn.CrossEntropyLoss(output, one_hot(target,num_classes=10).float())
    l1_lambda = 0.00001
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l2_lambda = 0.000001
    l2_norm = 0
    for i in model.parameters():
        l2_norm += i.pow(2).sum()
    #l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    #loss = 0.5*temploss + l1_lambda * l1_norm + l2_lambda * l2_norm
    loss = temploss +  l2_lambda * l2_norm
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(data_loader.dataset),
        100. * batch_idx / len(data_loader), loss.item()))
  
def eval_MINST(data_loader,model,dataset,device,one_hot):
  loss = 0
  correct = 0
  with torch.no_grad(): # notice the use of no_grad
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        L2_reg = 0
        L2_lam = 0.000001
        for param in model.parameters():
                    L2_reg += torch.norm(param)**2
        mse_loss = F.mse_loss(output, one_hot(target,num_classes=10).float(), reduction='sum')
        total_loss = mse_loss + L2_lam * L2_reg
        loss += total_loss.item()
    loss /= len(data_loader.dataset)
    print(dataset+'set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    return loss, 100. * correct / len(data_loader.dataset)

      
def load_10(batch_size_train):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    CIFAR10_training = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # create a training and a validation set
    CIFAR10_training_set, CIFAR10_validation_set = random_split(CIFAR10_training, [45000, 5000])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(CIFAR10_training_set,
                                            batch_size=batch_size_train,
                                            shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(CIFAR10_validation_set,
                                                    batch_size=batch_size_train,
                                                    shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader,validation_loader,classes

def train_10(epoch,multi_linear_model,train_loader,optimizer,device,log_interval):
    multi_linear_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = multi_linear_model(data)

        loss_fn = nn.CrossEntropyLoss()
        temploss = loss_fn(output, target)
        l1_lambda = 0.0001
        l1_norm = sum(p.abs().sum() for p in multi_linear_model.parameters())
        loss = temploss + l1_lambda * l1_norm

        #loss = F.mse_loss(output, one_hot(target,num_classes=10).float())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
def validation_10(multi_linear_model,validation_loader,device,one_hot):
    multi_linear_model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad(): # notice the use of no_grad
        for data, target in validation_loader:
            data = data.to(device)
            target = target.to(device)
            output = multi_linear_model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            validation_loss += F.mse_loss(output, one_hot(target,num_classes=10).float(), reduction='sum').item()
        validation_loss /= len(validation_loader.dataset)
        print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(validation_loss, correct, len(validation_loader.dataset), 100. * correct / len(validation_loader.dataset)))
    return validation_loss,100. * correct / len(validation_loader.dataset)
        
def logistic_regression(dataset_name, device):
    # TODO: implement logistic regression here
    if dataset_name == "MNIST":
        n_epochs = 9
        batch_size_train = 200
    
        learning_rate = 0.0015555#1e-3
       
        log_interval = 100
        random_seed = 1
        torch.backends.cudnn.enabled = False
        torch.manual_seed(random_seed)
        train_loader,validation_loader= load_MINST(batch_size_train)

        multi_linear_model = MultipleLinearRegression_MNIST().to(device)
        optimizer = optim.Adam(multi_linear_model.parameters(), lr=learning_rate,weight_decay= 0.000011)#,weight_decay=0.0001,#3e-5#,weight_decay=3e-5,0.00008,0.000011,0.00034374957946154284
        one_hot = torch.nn.functional.one_hot
        eval_MINST(validation_loader,multi_linear_model,"Validation",device,one_hot)
        for epoch in range(1, n_epochs + 1):
            train_MINSI(epoch,train_loader,multi_linear_model,optimizer,device,log_interval)
            eval_MINST(validation_loader,multi_linear_model,"Validation",device,one_hot)
        
        submit_model = multi_linear_model
 
    elif dataset_name == "CIFAR10":
        n_epochs = 10
        batch_size_train = 128
        batch_size_test = 1000
        learning_rate = 0.000232#0.000232#0.0001#1e-3
        momentum = 0.5
        log_interval = 100
        random_seed = 1
        torch.backends.cudnn.enabled = False
        torch.manual_seed(random_seed)
        train_loader,validation_loader,classes = load_10(batch_size_train)
        multi_linear_model = MultipleLinearRegression_10().to(device)
        optimizer = optim.Adam(multi_linear_model.parameters(), lr=learning_rate,weight_decay= 0.00153)#0.001 0.00153
        one_hot = one_hot = torch.nn.functional.one_hot

        validation_10(multi_linear_model,validation_loader,device,one_hot)
        for epoch in range(1, n_epochs + 1):
            train_10(epoch,multi_linear_model,train_loader,optimizer,device,log_interval)
            validation_10(multi_linear_model,validation_loader,device,one_hot)
        submit_model = multi_linear_model


        



    results = dict(
        model = submit_model 
    )

    return results

def train_tune(epoch,data_loader,dataname,model,device,log_interval,opt_name,learning_rate,mome,weight_decay):
    
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
    elif opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    if dataname == "MNIST":    
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss_fn = nn.CrossEntropyLoss()
            temploss = loss_fn(output, target)
            #temploss = nn.CrossEntropyLoss(output, one_hot(target,num_classes=10).float())
            #l1_lambda = 0.00001
            #l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_lambda = 0.00001
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            #loss = temploss + l1_lambda * l1_norm + l2_lambda * l2_norm
            loss = temploss + l2_lambda * l2_norm
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
            
    elif dataname == "CIFAR10":
        model.train()
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss_fn = nn.CrossEntropyLoss()
            temploss = loss_fn(output, target)
            l1_lambda = 0.0001
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = temploss + l1_lambda * l1_norm

            #loss = F.mse_loss(output, one_hot(target,num_classes=10).float())
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
    return model


def tune_hyper_parameter(dataset_name, target_metric, device):
    # TODO: implement logistic regression hyper-parameter tuning here

    epochs = 8

   
    log_interval = 100
   

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    #check The database,then apply different optimizer on it.
    if dataset_name == "MNIST":
        batch_size_train = 200
        
        train_loader,validation_loader= load_MINST(batch_size_train)
        adam_learning_rate = np.random.uniform(low=0.001, high=0.01, size=(2,))# 0.0005,.001
        adam_weight_decay = np.random.uniform(low=0.00001, high=0.0003, size=(2,))#0.0000,0.0001
        one_hot = torch.nn.functional.one_hot
        adam_loss = 100000
        adam_acc = 0
        dict_adam_acc = {"adam_lr" : 0,"adam_weight" : 0}
        dict_adam_loss = {"adam_lr" : 0,"adam_weight" : 0}
        #search better learning rate and Weight Decay for Adam
        for adam_lr in adam_learning_rate:
            for adam_weight in adam_weight_decay:
                multi_linear_model = MultipleLinearRegression_MNIST().to(device)

                for epoch in range(1, epochs + 1):
                    if epoch == 1:

                        MNIST_model = train_tune(epoch,train_loader,"MNIST",multi_linear_model,device,log_interval,"Adam",adam_lr,None,adam_weight)
                        loss, acc = eval_MINST(validation_loader,MNIST_model,"Validation",device,one_hot)
                    else:
                        MNIST_model = train_tune(epoch,train_loader,"MNIST",MNIST_model,device,log_interval,"Adam",adam_lr,None,adam_weight)
                        loss, acc = eval_MINST(validation_loader,MNIST_model,"Validation",device,one_hot)
                    if target_metric == 'acc':
                        if adam_acc <  acc:
                            adam_acc = acc
                            dict_adam_acc['adam_lr'] = adam_lr
                            dict_adam_acc['adam_weight'] = adam_weight
                    elif target_metric == 'loss':
                        if adam_loss > loss:
                            adam_loss = loss
                            dict_adam_loss['adam_lr'] = adam_lr
                            dict_adam_loss['adam_weight'] = adam_weight
        #search better learning rate and Weight Decay for SGD
        sgd_learning_rate = np.random.uniform(low=0.00001, high=0.0001, size=(2,))
        sgd_weight_decay = np.random.uniform(low=0.00001, high=0.0001, size=(2,))
        sgd_loss = 1000000
        sgd_acc = 0
        dict_sgd_acc = {"sgd_lr" : 0,"sgd_weight" : 0}
        dict_sgd_loss = {"sgd_lr" : 0,"sgd_weight" : 0}
        for sgd_lr in sgd_learning_rate:
            for sgd_weight in sgd_weight_decay:
                multi_linear_model = MultipleLinearRegression_MNIST().to(device)

                for epoch in range(1, epochs + 1):
                    if epoch == 1:

                        MNIST_model = train_tune(epoch,train_loader,"MNIST",multi_linear_model,device,log_interval,"SGD",sgd_lr,None,sgd_weight)
                        loss, acc = eval_MINST(validation_loader,MNIST_model,"Validation",device,one_hot)
                    else:
                        MNIST_model = train_tune(epoch,train_loader,"MNIST",MNIST_model,device,log_interval,"SGD",sgd_lr,None,sgd_weight)
                        loss, acc = eval_MINST(validation_loader,MNIST_model,"Validation",device,one_hot)
                    if target_metric == 'acc':
                        if sgd_acc <  acc:
                            sgd_acc = acc
                            dict_sgd_acc['sgd_lr'] = sgd_lr
                            dict_sgd_acc['sgd_weight'] = sgd_weight
                    elif target_metric == 'loss':        
                        if sgd_loss > loss:
                            sgd_loss = loss
                            dict_sgd_loss['sgd_lr'] = sgd_lr
                            dict_sgd_loss['sgd_weight'] = sgd_weight

    elif dataset_name == "CIFAR10":
        batch_size_train = 128
        train_loader,validation_loader,classes = load_10(batch_size_train)
        adam_learning_rate = np.random.uniform(low=0.0001, high=0.001, size=(2,))# 0.0005,.001
        adam_weight_decay = np.random.uniform(low=0.001, high=0.002, size=(2,))#0.0000,0.0001
        one_hot = one_hot = torch.nn.functional.one_hot
        adam_loss = 10000000
        adam_acc = 0
        dict_adam_acc = {"adam_lr" : 0,"adam_weight" : 0}
        dict_adam_loss = {"adam_lr" : 0,"adam_weight" : 0}
        #search better learning rate and Weight Decay for Adam
        for adam_lr in adam_learning_rate:
            for adam_weight in adam_weight_decay:
                multi_linear_model = MultipleLinearRegression_10().to(device)

                for epoch in range(1, epochs + 1):
                    if epoch == 1:

                        C_model = train_tune(epoch,train_loader,"CIFAR10",multi_linear_model,device,log_interval,"Adam",adam_lr,None,adam_weight)
                        loss, acc = validation_10(C_model,validation_loader,device,one_hot)
                    else:
                        C_model = train_tune(epoch,train_loader,"CIFAR10",C_model,device,log_interval,"Adam",adam_lr,None,adam_weight)
                        loss, acc = validation_10(C_model,validation_loader,device,one_hot)
                    if target_metric == 'acc':
                        if adam_acc <  acc:
                            adam_acc = acc
                            dict_adam_acc['adam_lr'] = adam_lr
                            dict_adam_acc['adam_weight'] = adam_weight
                    elif target_metric == 'loss':
                        if adam_loss > loss:
                            adam_loss = loss
                            dict_adam_loss['adam_lr'] = adam_lr
                            dict_adam_loss['adam_weight'] = adam_weight
        #search better learning rate and Weight Decay for SGD
        sgd_learning_rate = np.random.uniform(low=0.00001, high=0.0001, size=(2,))
        sgd_weight_decay = np.random.uniform(low=0.00001, high=0.0001, size=(2,))
        sgd_loss = 10000000
        sgd_acc = 0
        dict_sgd_acc = {"sgd_lr" : 0,"sgd_weight" : 0}
        dict_sgd_loss = {"sgd_lr" : 0,"sgd_weight" : 0}
        for sgd_lr in sgd_learning_rate:
            for sgd_weight in sgd_weight_decay:
                multi_linear_model = MultipleLinearRegression_10().to(device)

                for epoch in range(1, epochs + 1):
                    if epoch == 1:

                        C_model = train_tune(epoch,train_loader,"CIFAR10",multi_linear_model,device,log_interval,"SGD",sgd_lr,None,sgd_weight)
                        loss, acc = validation_10(C_model,validation_loader,device,one_hot)
                    else:
                        C_model = train_tune(epoch,train_loader,"CIFAR10",C_model,device,log_interval,"SGD",sgd_lr,None,sgd_weight)
                        loss, acc = validation_10(C_model,validation_loader,device,one_hot)
                    if target_metric == 'acc':
                        if sgd_acc <  acc:
                            sgd_acc = acc
                            dict_sgd_acc['sgd_lr'] = sgd_lr
                            dict_sgd_acc['sgd_weight'] = sgd_weight
                    elif target_metric == 'loss':
                        if sgd_loss > loss:
                            sgd_loss = loss
                            dict_sgd_loss['sgd_lr'] = sgd_lr
                            dict_sgd_loss['sgd_weight'] = sgd_weight


        

    if target_metric == 'acc':   
        if adam_acc < sgd_acc:
            
            best_metric = sgd_acc
            best_params = dict_sgd_acc
        else:
            best_metric = adam_acc
            best_params = dict_adam_acc
    elif target_metric == 'loss':
        if adam_loss < sgd_loss:
            
            best_metric = adam_loss
            best_params = dict_adam_loss
        else:
            best_metric = sgd_loss
            best_params = dict_sgd_loss

   #At first, I found Adam optimizer can have a better Acc to deal with MNIST Dataset with Lr = 0.001 and Weight Decay = 0.00008.
   #Then I called tune_hyper_parameter function, and this function ramdomly selected some numbers and assigned them to Lr and Weight_Decay.
   #And I found   Lr = 0.0015555 and Weight Decay = 0.000011 can get a better Acc on Adam.
   #Similarly,For CIFAR10 Dataset, I found Lr =0.000232 and Weight Decay = 0.00153 can get a better Acc on Adam.
   #By using tune_hyper_parameter function, it can automatically compare which optimizer is better,and I found that Adam is better than SGD for both of two datasets.
  



    return best_params, best_metric
