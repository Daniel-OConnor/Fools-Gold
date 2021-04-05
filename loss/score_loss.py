import torch


def score_loss(input, labels, thetas, target_score):
    thetas[0].grad = None
    thetas[1].grad = None
    thetas[0].retain_grad()
    thetas[1].retain_grad()
    log_ratio = torch.log(input)
    torch.sum(log_ratio).backward(retain_graph=True)
    v0 = (1 - labels) * torch.pow(thetas[0].grad - target_score, 2)
    v1 = labels * torch.pow(thetas[1].grad - target_score, 2)
    return torch.mean(v0+v1)