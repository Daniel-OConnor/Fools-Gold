import torch
from loss.score_loss import paired_score_loss


def rolr(input, labels, _, target):
    v0 = torch.pow(target - input, 2)
    v1 = torch.pow(target - (1/input), 2)
    v2_0 = labels * v0
    v2_1 = (1 - labels) * v1
    return torch.mean(v2_0 + v2_1)


def rascal(input, labels, thetas, target_score, target_ratio, alpha):
    log_ratio = torch.log(input)
    return rolr(input, labels, thetas, target_ratio) + alpha * paired_score_loss(log_ratio, labels, thetas, target_score)
