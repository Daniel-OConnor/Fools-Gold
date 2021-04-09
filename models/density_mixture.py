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
        self.linear_sd = torch.nn.Linear(hidden_size, components_count*(x_size*(x_size-1))//2)
        self.linear_sd_diag = torch.nn.Linear(hidden_size, components_count*x_size)
        self.linear_weight = torch.nn.Linear(hidden_size, components_count)

    def forward(self, x, theta):
        batch_size = x.shape[0]
        x1 = torch.relu(self.linear_H0(theta))
        x2 = torch.relu(self.linear_H1(x1))
        x3 = torch.relu(self.linear_H2(x2))
        mean = self.linear_mean(x3)
        mean = torch.reshape(mean, (batch_size, self.components_count, self.x_size))
        sd_vec = self.linear_sd(x3)
        sd_vec = torch.reshape(sd_vec, (batch_size, self.components_count, (self.x_size*(self.x_size-1))//2))
        sd_vec_diag = torch.exp(self.linear_sd_diag(x3))
        sd_vec_diag = torch.reshape(sd_vec_diag, (batch_size, self.components_count, self.x_size))
        indices = torch.tril_indices(self.x_size, self.x_size, offset=-1)
        sd = torch.zeros((batch_size, self.components_count, self.x_size, self.x_size))
        sd[..., torch.arange(0, self.x_size), torch.arange(0, self.x_size)] = sd_vec_diag
        sd[..., indices[0], indices[1]] = sd_vec
        weight = torch.softmax(self.linear_weight(x3), 1)
        return x, mean, sd, weight
