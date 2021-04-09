import torch
import math


class DensityMixture(torch.nn.Module):
    def __init__(self, x_size, theta_size, components_count, hidden_size):
        super(DensityMixture, self).__init__()
        self.components_count = components_count
        self.x_size = x_size
        self.linear_H0 = torch.nn.Linear(theta_size, hidden_size)
        self.linear_H1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_H2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_mean = torch.nn.Linear(hidden_size, components_count*x_size)
        self.linear_sd = torch.nn.Linear(hidden_size, components_count*x_size*x_size)
        self.linear_weight = torch.nn.Linear(hidden_size, components_count)

    def forward(self, x, theta):
        batch_size = x.shape[0]
        x1 = torch.relu(self.linear_H0(theta))
        x2 = torch.relu(self.linear_H1(x1))
        x3 = torch.relu(self.linear_H2(x2))
        mean = self.linear_mean(x3)
        mean = torch.reshape(mean, (batch_size, self.components_count, self.x_size))
        sd = torch.exp(self.linear_sd(x3))
        sd = torch.reshape(sd, (batch_size, self.components_count, self.x_size, self.x_size))
        weight = torch.softmax(self.linear_weight(x3), 1)
        return x, mean, sd, weight
