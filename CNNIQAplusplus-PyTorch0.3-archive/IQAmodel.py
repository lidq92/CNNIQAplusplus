# Implemented by Dingquan Li
# Email: dingquanli@pku.edu.cn
# Date: 2018/4/19
#
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNIQAplusnet(nn.Module):
    def __init__(self, n_classes, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQAplusnet, self).__init__()
        self.conv1  = nn.Conv2d(1, n_kers, ker_size)
        self.fc1    = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2    = nn.Linear(n1_nodes, n2_nodes)
        self.fc3_q  = nn.Linear(n2_nodes, 1)
        self.fc3_d  = nn.Linear(n2_nodes, n_classes)

    def forward(self, x, train=True):

        h = self.conv1(x)

        h1 = F.adaptive_max_pool2d(h, 1)
        h2 = -F.adaptive_max_pool2d(-h, 1)
        h = torch.cat((h1, h2), 1) #
        h = torch.squeeze(torch.squeeze(h, 3), 2)

        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.5, training=train)
        h = F.relu(self.fc2(h))

        q = self.fc3_q(h)
        d = self.fc3_d(h)
        return q, d

class CNNIQAplusplusnet(nn.Module):
    def __init__(self, n_classes, ker_size=3, n1_kers=8, pool_size=2, n2_kers=32, n1_nodes=128, n2_nodes=512):
        super(CNNIQAplusplusnet, self).__init__()
        self.conv1 = nn.Conv2d(1, n1_kers, ker_size)
        self.pool1 = nn.MaxPool2d(pool_size)
        self.conv2 = nn.Conv2d(n1_kers, n2_kers, ker_size)
        self.fc1   = nn.Linear(2 * n2_kers, n1_nodes)
        self.fc2   = nn.Linear(n1_nodes, n2_nodes)
        self.fc3_q = nn.Linear(n2_nodes, 1)
        self.fc3_d = nn.Linear(n2_nodes, n_classes)

    def forward(self, x, train=True):

        h = self.conv1(x)
        h = self.pool1(h)

        h = self.conv2(h)
        h1 = F.adaptive_max_pool2d(h, 1)
        h2 = -F.adaptive_max_pool2d(-h, 1)
        h = torch.cat((h1, h2), 1) #
        h = torch.squeeze(torch.squeeze(h, 3), 2)

        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.5, training=train)
        h = F.relu(self.fc2(h))

        q = self.fc3_q(h)
        d = self.fc3_d(h)
        return q, d