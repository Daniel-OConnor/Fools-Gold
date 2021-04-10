import torch


class CategoricalDistribution(torch.nn.Module):
    # Here x_size is the number of bins rather than the dimension of x
    def __init__(self, bins, theta_size, hidden_size):
        super(CategoricalDistribution, self).__init__()
        self.linear_H0 = torch.nn.Linear(theta_size, hidden_size)
        self.linear_H1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_H2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_out = torch.nn.Linear(hidden_size, bins)

    def forward(self, x, theta):
        x1 = torch.relu(self.linear_H0(theta))
        x2 = torch.relu(self.linear_H1(x1))
        x3 = torch.relu(self.linear_H2(x2))
        out = torch.softmax(x3, 1)
        return x, out
