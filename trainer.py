import torch
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train(model, dataset, loss_function, i, optimizer):
    train_iter = tqdm(dataset)
    model.train()
    epoch_loss, num_samples = 0.0, 0
    for batch in train_iter:
        labels, thetas, xs, *targets = batch
        if isinstance(thetas, torch.Tensor):
            thetas = thetas.to(device)
        elif isinstance(thetas, list) or isinstance(thetas, tuple):
            thetas = (thetas[0].to(device), thetas[1].to(device))
        else:
            raise TypeError(str(thetas)+"is neither a tensor or list")
        # labels = labels.to(device)
        if isinstance(xs, torch.Tensor):
            batch_sz = xs.shape[0]
            xs = xs.to(device)
        elif isinstance(xs, list):
            batch_sz = len(xs)
        else:
            raise TypeError(str(xs)+"is neither a tensor or list")
        if len(targets) == 1:
            targets = (targets[0].to(device),)
        else:
            targets = (targets[0].to(device), targets[1].to(device))
        y_hat = model(xs, *thetas)
        loss = loss_function(y_hat, labels, thetas, *targets) * batch_sz
        model.zero_grad()
        loss.backward()
        optimizer.step()
        num_samples += len(labels)
        epoch_loss += loss.item()
        train_iter.set_description("Average loss {:10.4f}".format(epoch_loss/num_samples))
