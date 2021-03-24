import torch

class Ratio(torch.nn.Module):
    def __init__(self, x_size, theta_size, hidden_size):
        super(Ratio, self).__init__()
        self.linear_H0 = torch.nn.Linear(x_size + 2*theta_size, hidden_size)
        self.linear_H1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_H2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_out = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, theta0, theta1):
        batch_sz = x.shape[0]
        x = x.unsqueeze(1)
        theta0 = theta0
        theta1 = theta1
        x0 = torch.cat((x, theta0, theta1), dim=1)
        x1 = torch.tanh(self.linear_H0(x0))
        x2 = torch.tanh(self.linear_H1(x1))
        x3 = torch.tanh(self.linear_H2(x2))
        out = torch.exp(self.linear_out(x3))
        return out.squeeze()
