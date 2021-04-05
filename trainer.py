import torch
import numpy as np
from tqdm import tqdm


def train(model, dataset, loss_function, epochs, optimizer):
    for i in range(epochs):
        print("Epoch {}...".format(i))
        train_iter = tqdm(dataset)
        model.train()
        epoch_loss, num_samples = 0.0, 0
        for batch in train_iter:
            labels, thetas, runs, *targets = batch
            xs = runs[:, -1]
            batch_sz = xs.shape[0]
            y_hat = model(xs, *thetas)
            loss = loss_function(y_hat, labels, thetas, *targets) * batch_sz
            model.zero_grad()
            loss.backward()
            optimizer.step()
            num_samples += len(labels)
            epoch_loss += loss.item()
            train_iter.set_description("Average loss {}".format(epoch_loss/num_samples))