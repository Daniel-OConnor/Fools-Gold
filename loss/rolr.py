import torch

def rolr(input, labels, target):
    v0 = torch.pow(target - input, 2)
    v1 = torch.pow(target - (1/input), 2)
    v2_0 = labels * v0
    v2_1 = (1 - labels) * v1
    return torch.mean(v2_0 + v2_1)

def rascal(input, target_ratio, target_score, labels, alpha):
    pass