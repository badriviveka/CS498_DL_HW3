import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 192, 3)
        self.conv3 = nn.Conv2d(192, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Classifier(nn.Module):
    # TODO: implement me
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,128 ,3)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256 ,3)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 3)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256,256, 1)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256*4*4, 150)
        self.fc1_bn = nn.BatchNorm1d(150)
        self.fc2 = nn.Linear(150, 120)
        self.fc2_bn = nn.BatchNorm1d(120)
        self.fc3 = nn.Linear(120, 120)
        self.fc3_bn = nn.BatchNorm1d(120)
        self.fc4 = nn.Linear(120, 150)
        self.fc4_bn = nn.BatchNorm1d(150)
        self.fc5 = nn.Linear(150, 60)
        self.fc5_bn = nn.BatchNorm1d(60)
        self.fc6 = nn.Linear(60, NUM_CLASSES)
        self.dp1 = nn.Dropout(p=0.2)
     

    def forward(self, x):
        # residual=x
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        x = self.pool(F.relu(self.conv5_bn(self.conv5(x))))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = x.view(x.size()[0],256*4*4)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        # x = self.dp1(x)
        residual=x
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.dp1(x)
        x = F.relu(self.fc4_bn(self.fc4(x)))
        # x = self.dp1(x)
        x+=residual
        x = F.relu(self.fc5_bn(self.fc5(x)))
        x = self.dp1(x)
        x = self.fc6(x)
        # x+=residual
        return x




