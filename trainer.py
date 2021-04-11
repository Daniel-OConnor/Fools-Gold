import torch
import numpy as np
from tqdm import tqdm


def train(model, dataset, loss_function, i, optimizer):
    print("Epoch {}...".format(i))
    train_iter = tqdm(dataset)
    model.train()
    epoch_loss, num_samples = 0.0, 0
    for batch in train_iter:
        labels, thetas, xs, *targets = batch
        if isinstance(xs, torch.Tensor):
            batch_sz = xs.shape[0]
        elif isinstance(xs, list):
            batch_sz = len(xs)
        else:
            raise TypeError(str(xs)+"is neither a tensor or list")
        y_hat = model(xs, *thetas)
        loss = loss_function(y_hat, labels, thetas, *targets) * batch_sz
        model.zero_grad()
        loss.backward()
        optimizer.step()
        num_samples += len(labels)
        epoch_loss += loss.item()
        train_iter.set_description("Average loss {:10.4f}".format(epoch_loss/num_samples))
