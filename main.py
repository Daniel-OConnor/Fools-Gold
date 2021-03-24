from simulators.dummy_simulator import DummySimulator
from loss.rolr import rolr
from ratio import Ratio
from tqdm import tqdm
import torch
import random
from torch.utils.data import DataLoader
import numpy as np

# training constants
batch_size = 2
epochs = 3
train_fraction = 0.6
num_samples = 1000

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

model = Ratio(1, 1, 50)
model.to(device)

prior = lambda: torch.rand(1).to(device)
sim = DummySimulator()
optimizer = torch.optim.Adam(model.parameters())

# generate pairs of priors
priors = [(prior(), prior()) for _ in range(num_samples)]

# generate data
runs0 = [(sim.simulate(theta0), (theta0, theta1)) for theta0, theta1 in priors]
runs1 = [(sim.simulate(theta1), (theta0, theta1)) for theta0, theta1 in priors]

# generate ratios
ratios0 = [sim.eval_ratio(zs, theta0, theta1) for zs, (theta0, theta1) in runs0]
ratios1 = [sim.eval_ratio(zs, theta1, theta0) for zs, (theta0, theta1) in runs1]

# collate (add labels), shuffle, construct dataloaders
data0 = [(0, zs0, ratio0, ts) for ((zs0, ts), ratio0) in zip(runs0, ratios0)]
data1 = [(1, zs1, ratio1, ts) for ((zs1, ts), ratio1) in zip(runs1, ratios1)]
all_data = data0 + data1
random.shuffle(all_data)
train_data, test_data = all_data[:int(len(all_data)*train_fraction)], all_data[int(len(all_data)*train_fraction):]
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# train
print("Training...")
for i in range(epochs):
    print("Epoch {}...".format(i))
    train_iter = tqdm(train_loader)
    model.train()
    epoch_loss, num_samples = 0.0, 0
    for batch in train_iter:
        model.zero_grad()
        labels, runs, ratios, (theta0, theta1) = batch
        xs = runs[:, -1]
        r_hat = model(xs, theta0, theta1)
        loss = rolr(r_hat, ratios, labels)
        loss.backward()
        optimizer.step()
        num_samples += len(labels)
        epoch_loss += loss.item()
        train_iter.set_description("Average loss {}".format(epoch_loss/num_samples))

# test
print("Testing...")
x, ratio_true, ratio_pred = np.zeros((0)), np.zeros((0)), np.zeros((0))
with torch.no_grad(): # we can only use this when we don't need the score (only the ratio)
    test_perf, num_samples = 0.0, 0
    test_iter = tqdm(test_loader)
    for batch in test_iter:
        model.zero_grad()
        labels, runs, ratios, (theta0, theta1) = batch
        xs = runs[:, -1]
        r_hat = model(xs, theta0, theta1)
        loss = rolr(r_hat, ratios, labels)
        optimizer.step()
        num_samples += len(labels)
        epoch_loss += loss.item()
        test_iter.set_description("Average loss {}".format(epoch_loss/num_samples))
        ratio_true = np.concatenate((ratio_true, ratios))
        ratio_pred = np.concatenate((ratio_pred, r_hat))
        x = np.concatenate((x, xs))

# ugly visualisation code - ignore
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
density_true = gaussian_kde(ratio_true)
density_pred = gaussian_kde(ratio_pred.squeeze())
xs = np.linspace(0,2,200)
density_true.covariance_factor = lambda : .1
density_true._compute_covariance()
density_pred.covariance_factor = lambda : .1
density_pred._compute_covariance()
plt.plot(xs,density_true(xs), "b")
plt.plot(xs,density_pred(xs), "r")
plt.show()

print("Done")