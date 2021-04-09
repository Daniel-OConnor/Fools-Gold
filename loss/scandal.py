import torch
import math
from loss.score_loss import score_loss


PI = torch.tensor(math.pi)


def gaussian(x, mean, sd): # <-- breaks here
    return 1/(torch.abs(sd)*torch.sqrt(2*PI)) * torch.exp(-0.5*(x-mean)*(x-mean)/(sd*sd))


def prob(x, mean, sd, weight):
    return torch.sum(gaussian(x, mean, sd) * weight, dim=1)


def scandal(input, labels, thetas, target_score, alpha):
    log_prob = torch.log(prob(*input))
    return -torch.mean(log_prob) + alpha * score_loss(log_prob, thetas, target_score)
