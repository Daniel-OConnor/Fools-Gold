import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import math
from loss.score_loss import score_loss


PI = torch.tensor(math.pi)


def gaussian(x, mean, sd):
    return 1/(torch.abs(sd)*torch.sqrt(2*PI)) * torch.exp(-0.5*(x-mean)*(x-mean)/(sd*sd))


def gaussian_mixture_prob(x, mean, sd, weight):
    dist = MultivariateNormal(mean, sd)
    log_prob = torch.logsumexp(dist.log_prob(torch.unsqueeze(x, -1))+torch.log(weight), dim=(1,))
    return log_prob


def scandal(input, labels, thetas, target_score, alpha):
    log_prob = gaussian_mixture_prob(*input)
    return -torch.mean(log_prob) + alpha * score_loss(log_prob, thetas, target_score)
