import torch

class Ratio(torch.nn.Module):

    def __init__(self, x_size, theta_size, hidden_size):
        super(Ratio, self).__init__()
        self.linear_x = torch.nn.Linear(x_size, hidden_size)
        self.linear_theta0 = torch.nn.Linear(theta_size, hidden_size, bias=False)
        self.linear_theta1 = torch.nn.Linear(theta_size, hidden_size, bias=False)
        self.linear_H1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_H2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_out = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, theta0, theta1):
        x1 = self.linear_x(x) + self.linear_theta0(theta0) + self.linear_theta1(theta1)
        x1 = torch.relu(x1)
        x2 = torch.relu(self.linear_H1(x1))
        x3 = torch.relu(self.linear_H2(x2))
        out = torch.exp(self.linear_out(x3))
        return out
