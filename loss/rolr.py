import torch
from loss.score_loss import paired_score_loss


def rolr(input, labels, _, log_target):
    index = labels.unsqueeze(1)
    target = torch.exp(log_target)
    stacked_inputs = torch.stack([input, 1/input], dim=1)
    selected_inputs = torch.gather(stacked_inputs, 1, index).squeeze()
    v = torch.pow(target - selected_inputs, 2)
    return torch.mean(v)


def rascal(input, labels, thetas, target_score, log_target_ratio, alpha):
    log_pred_ratio = torch.log(input)
    rolr_loss = rolr(input, labels, thetas, log_target_ratio)
    score_loss = paired_score_loss(log_pred_ratio, labels, thetas, target_score)
    return rolr_loss + alpha * score_loss
