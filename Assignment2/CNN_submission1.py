
import timeit
from collections import OrderedDict

import torch
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split


#TO DO: Complete this with your CNN architecture. Make sure to complete the architecture requirements.
#The init has in_channels because this changes based on the dataset. 

class Net(nn.Module):
    def __init__(self, in_channels):
        super(Net, self).__init__()
        self.channel = in_channels
        if self.channel == 1:
            self.conv1 = nn.Conv2d(1, 12, kernel_size=3)
            #28-5+1 = 24,12x24x24
            #self.conv2 = nn.Conv2d(12, 16, kernel_size=5)
            #24-5+1 = 20,16x20x20
            #self.conv4 = nn.Conv2d(16, 22, kernel_size=5)
            self.conv2 = nn.Conv2d(12, 32, kernel_size=3)
            self.conv4 = nn.Conv2d(32, 128, kernel_size=3)


            # The linear layers
            self.fc1 = nn.Linear(1152,50)#352
            self.fc3 = nn.Linear(50,10)

            # the upsampling layer
            self.up = nn.Upsample(scale_factor=1.2)
            
            nn.init.kaiming_normal(self.conv1.weight,nonlinearity='relu')
            nn.init.kaiming_normal(self.conv2.weight,nonlinearity='relu')
            nn.init.kaiming_normal(self.conv4.weight,nonlinearity='relu')
            nn.init.kaiming_normal(self.fc1.weight,nonlinearity='relu')
            nn.init.kaiming_normal(self.fc3.weight,nonlinearity='relu')
            

        if self.channel == 3:

            # The convolutional layer
            self.conv1 = nn.Conv2d(3, 12, kernel_size=3)
            self.conv2 = nn.Conv2d(12, 16, kernel_size=3)
            self.conv4 = nn.Conv2d(16, 22, kernel_size=3)

            # The linear layers
            self.fc1 = nn.Linear(1078,40)
            self.fc3 = nn.Linear(40,10)

            # the upsampling layer
            self.up = nn.Upsample(scale_factor=1.2)

            #Since we use relu as activation function, we use He initialization instead of xavier
            # The initialization as required
       
            nn.init.kaiming_normal(self.conv1.weight,nonlinearity='relu')
            nn.init.kaiming_normal(self.conv2.weight,nonlinearity='relu')
            nn.init.kaiming_normal(self.conv4.weight,nonlinearity='relu')
            nn.init.kaiming_normal(self.fc1.weight,nonlinearity='relu')
            nn.init.kaiming_normal(self.fc3.weight,nonlinearity='relu')
            

    def forward(self, x):
        if self.channel == 1:
             # first layer
            x = F.relu(self.conv1(x))#28 -2 = 26, 12 26 26
            x = F.max_pool2d(x, kernel_size=2, stride=2)#12 13 13
            # second layer
            x = F.relu(self.conv2(x))#32 11 11
            x = F.max_pool2d(x, kernel_size=2, stride=2) # 32 5 5
            x = F.dropout(x,0.2)
            #third layer
            x = F.relu(self.conv4(x))#128 3 3

            x = x.view(x.size(0), -1)
            # forth layer,
            x = F.relu(self.fc1(x))
            # fifth layer.
            x = self.fc3(x)
            return x


        
            '''
            # first layer, a convolution layer
            x = F.relu(self.conv1(x))#28-5+1 = 24,12x24x24 12 26 26
            # second layer
            x = F.relu(self.conv2(x))#24-5+1 = 20,16x20x20 #32 20 20  32 24 24
            x = F.max_pool2d(x, kernel_size=2, stride=2)#16x10x10 # 32 10 10 32 12 12 
            x = F.dropout(x,0.2)
            
            # The upsampling layer as required
            x = self.up(x)#16x12x12 # 32 12 12 32 14 14
            #third layer
            x = F.relu(self.conv4(x))#22x8x8 # 64 8 8 256 12 12
            x = F.max_pool2d(x, kernel_size=2, stride=2)#22x4x4 # 64 4 4 128 4 4 256 6 6
            x = F.dropout(x,0.2)


            x = x.view(x.size(0), -1)
            # forth layer, a fully connected layer
            x = F.relu(self.fc1(x))
            # fifth layer. Satisfying the assignment requirment
            x = F.relu(self.fc3(x))
            return x
            '''

        if self.channel == 3:
            # first layer, 
            x = F.relu(self.conv1(x))
            # second layer
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.dropout(x,0.2)
            
            # The upsampling layer 
            x = self.up(x)
            #third layer
            x = F.relu(self.conv4(x))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.dropout(x,0.2)


            x = x.view(x.size(0), -1)
            # forth layer
            x = F.relu(self.fc1(x))
            # fifth layer
            x = self.fc3(x)
            return x



       

#Function to get train and validation datasets. Please do not make any changes to this function.
def load_dataset(
        dataset_name: str,
):
    if dataset_name == "MNIST":
        full_dataset = datasets.MNIST('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]))

        train_dataset, valid_dataset = random_split(full_dataset, [48000, 12000])

    elif dataset_name == "CIFAR10":
        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])

    else:
        raise Exception("Unsupported dataset.")

    return train_dataset, valid_dataset



#TO DO: Complete this function. This should train the model and return the final trained model. 
#Similar to Assignment-1, make sure to print the validation accuracy to see 
#how the model is performing.

def train(
        model,
        train_dataset,
        valid_dataset,
        device,
        dataset_name

):
    batch_size = 128
    #Make sure to fill in the batch size. 
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size, shuffle=False)
    # We are using L2 regularizator here by including the weight decay parameter
    if dataset_name == "CIFAR10":
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.06, weight_decay = 1e-6 , momentum=0.65)
    
        epochs = 28
    if dataset_name == "MNIST":
        #optimizer = torch.optim.SGD(model.parameters(), lr = 0.06, weight_decay = 1e-6 , momentum=0.65)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001 )
    
        epochs = 20
    count = 0
    while count < epochs:
        for batch_index,(data,target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        if count%2==0:
            validation_loss = 0
            
            correct = 0
            with torch.no_grad():
                for data, target in valid_loader:
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
                    validation_loss += F.cross_entropy(output, target).item()
            validation_loss /= len(valid_loader.dataset)
            print('{} set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(dataset_name,validation_loss, correct, len(valid_loader.dataset), 100. * correct / len(valid_loader.dataset)))
        count+=1
    







    results = dict(
        model=model
    )

    return results

def CNN(dataset_name, device):

    #CIFAR-10 has 3 channels whereas MNIST has 1.
    if dataset_name == "CIFAR10":
        in_channels= 3
    elif dataset_name == "MNIST":
        in_channels = 1
    else:
        raise AssertionError(f'invalid dataset: {dataset_name}')

    model = Net(in_channels).to(device)

    train_dataset, valid_dataset = load_dataset(dataset_name)

    results = train(model, train_dataset, valid_dataset, device,dataset_name)

    return results

