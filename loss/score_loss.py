import torch


# Takes the derivative with respect to theta.
# Used in multiple loss functions
def paired_score_loss(input, labels, thetas, target_score):
    theta0, theta1 = thetas
    theta0.retain_grad()
    theta1.retain_grad()
    input.sum().backward(retain_graph=True)
    theta0_grad = theta0.grad
    theta1_grad = theta1.grad
    v0 = (1 - labels) * torch.pow(theta0_grad - target_score, 2).sum()
    v1 = labels * torch.pow(theta1_grad - target_score, 2).sum()
    return torch.mean(v0 + v1)


def score_loss(input, thetas, target_score):
    thetas[0].grad = None
    thetas[0].retain_grad()
    torch.sum(input).backward(retain_graph=True)
    v = torch.pow(thetas[0].grad - target_score, 2)
    return torch.mean(v)
