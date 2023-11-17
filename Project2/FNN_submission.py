import torch

import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()

        self.loss_type = loss_type
        self.num_classes = num_classes

        """add your code here"""
        self.fc1 = nn.Linear(784, 500)  # First Fully Connected Layer
        self.fc2 = nn.Linear(500, num_classes)  # Second Fully Connected Layer


    def forward(self, x):
        output = None

        """add your code here"""

        # Flatten the input
        x = x.view(x.size(0), -1)

        # Implement the forward pass
        x = F.relu(self.fc1(x))
        output = self.fc2(x)

        return output

    def get_loss(self, output, target):
        if self.loss_type == 'ce':
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, target)
        elif self.loss_type == 'l2':
            # For L2 loss (MSE) we need the target in one-hot encoded form
            target_onehot = torch.zeros(target.size(0), self.num_classes).to(target.device)
            target_onehot.scatter_(1, target.unsqueeze(1), 1)
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, target_onehot)
        else:
            raise ValueError("Invalid loss_type provided. Expected 'ce' or 'l2'.")

        return loss