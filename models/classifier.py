import torch


class Classifier(torch.nn.Module):
    def __init__(self, x_size, theta_size, hidden_size):
        super(Classifier, self).__init__()
        self.linear_H0 = torch.nn.Linear(x_size + 2*theta_size, hidden_size)
        self.linear_H1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_H2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_out = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, theta0, theta1):
        x0 = torch.cat((x, theta0, theta1), dim=1)
        x1 = torch.relu(self.linear_H0(x0))
        x2 = torch.relu(self.linear_H1(x1))
        x3 = torch.relu(self.linear_H2(x2))
        out = torch.sigmoid(self.linear_out(x3))
        return out.squeeze()
