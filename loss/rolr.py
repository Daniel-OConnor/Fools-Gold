import torch

def rolr(input, target, labels):
    v0 = torch.pow(target - input, 2)
    v1 = torch.pow((1/target)-(1/input), 2)
    v2 = labels*v0 + (1-labels)*v1
    return torch.mean(v2)

def rascal(input, target_ratio, target_score, labels, alpha):
    pass