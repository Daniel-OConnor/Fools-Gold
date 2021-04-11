import torch

# Takes the derivative with respect to theta.
# Used in multiple loss functions
def paired_score_loss(input, labels, thetas, target_score):
    theta_grad = torch.autograd.grad(input.sum(), thetas, retain_graph=True, only_inputs=True)
    theta_size = thetas[0].shape[1]
    index = labels.repeat(theta_size, 1).t().unsqueeze(2)
    stacked_grads = torch.stack(theta_grad, dim=2)
    selected_grads = torch.gather(stacked_grads, 2, index).squeeze()
    v = torch.pow(selected_grads - target_score, 2).sum()
    return torch.mean(v)

def score_loss(input, thetas, target_score):
    thetas[0].grad = None
    thetas[0].retain_grad()
    torch.sum(input).backward()
    v = torch.pow(thetas[0].grad - target_score, 2)
    return torch.mean(v)
