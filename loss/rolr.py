import torch


def rolr(input, labels, _, target):
    v0 = torch.pow(target - input, 2)
    v1 = torch.pow(target - (1/input), 2)
    v2_0 = labels * v0
    v2_1 = (1 - labels) * v1
    return torch.mean(v2_0 + v2_1)


def rascal(input, labels, thetas, target_score, target_ratio, alpha):
    thetas[0].grad = None
    thetas[1].grad = None
    thetas[0].retain_grad()
    thetas[1].retain_grad()
    log_ratio = torch.log(input)
    torch.sum(log_ratio).backward(retain_graph=True)
    v0 = (1-labels) * torch.pow(thetas[0].grad - target_score, 2)
    v1 = labels * torch.pow(thetas[1].grad - target_score, 2)
    return rolr(input, labels, thetas, target_ratio) #+ alpha * torch.mean(v0+v1)
