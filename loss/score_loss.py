import torch


# Takes the derivative with respect to theta.
# Used in multiple loss functions
def paired_score_loss(input, labels, thetas, target_score):
    thetas[0].grad = None
    thetas[1].grad = None
    thetas[0].retain_grad()
    thetas[1].retain_grad()
    torch.sum(input).backward(retain_graph=True)
    labels = torch.unsqueeze(labels, dim=1)
    v0 = (1 - labels) * torch.pow(thetas[0].grad - target_score, 2)
    v1 = labels * torch.pow(thetas[1].grad - target_score, 2)
    return torch.mean(v0+v1)


def score_loss(input, thetas, target_score):
    thetas[0].grad = None
    thetas[0].retain_grad()
    torch.sum(input).backward(retain_graph=True)
    v = torch.pow(thetas[0].grad - target_score, 2)
    return torch.mean(v)