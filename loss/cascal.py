import torch
from loss.score_loss import score_loss


def xe(input, labels):
    v0 = labels * torch.log(input)
    v1 = (1 - labels) * torch.log(1 - input)
    return -torch.mean(v0+v1)


def cascal(input, labels, thetas, target_score, alpha):
    log_ratio = torch.log(1-input) - torch.log(input)
    return xe(input, labels) + alpha * score_loss(log_ratio, labels, thetas, target_score)
