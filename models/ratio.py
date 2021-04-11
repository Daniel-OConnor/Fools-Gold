import torch


class Ratio(torch.nn.Module):
    def __init__(self, x_size, theta_size, hidden_size):
        super(Ratio, self).__init__()
        self.linear_H0 = torch.nn.Linear(x_size + 2*theta_size, hidden_size)
        self.linear_H1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_H2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_out = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, theta0, theta1):
        # added more intermediate steps for debugging
        x0 = torch.cat((x, theta0, theta1), dim=1)
        y1 = self.linear_H0(x0)
        x1 = torch.tanh(y1)
        y2 = self.linear_H1(x1)
        x2 = torch.tanh(y2)
        y3 = self.linear_H2(x2)
        x3 = torch.tanh(y3)
        y4 = self.linear_out(x3)
        out = torch.exp(y4)
        return out.squeeze(1) # should always be a 1d tensor
