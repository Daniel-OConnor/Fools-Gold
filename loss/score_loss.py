import torch

# Takes the derivative with respect to theta.
# Used in multiple loss functions
def paired_score_loss(input, labels, thetas, target_score):
    index = labels.repeat(4, 1).t().unsqueeze(2)
    stacked_thetas = torch.stack(thetas, dim=2)
    selected_thetas = torch.gather(stacked_thetas, 2, index).squeeze()
    selected_thetas.grad.zero_()
    selected_thetas.retain_grad()
    input.sum().backward(retain_graph=True)
    theta_grad = selected_thetas.grad
    v = torch.pow(selected_thetas - target_score, 2).sum()
    selected_thetas.grad.zero_()
    return torch.mean(v)

def score_loss(input, thetas, target_score):
    thetas[0].grad = None
    thetas[0].retain_grad()
    torch.sum(input).backward()
    v = torch.pow(thetas[0].grad - target_score, 2)
    return torch.mean(v)
