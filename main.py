# %% SETUP
from simulators.dummy_simulator import DummySimulator
from loss.rolr import rolr
from ratio import Ratio
from tqdm import tqdm
import torch
from random import random, shuffle
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

TRAIN = True
# training constants
batch_size = 32
epochs = 2
train_fraction = 0.99
num_priors = 100000
num_sims_per_prior_pair = 1
learning_rate = 0.0002
num_train_samples = int(num_priors * num_sims_per_prior_pair * train_fraction)


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

model = Ratio(1, 1, 200)
model.to(device)

prior = lambda: torch.rand(1).to(device)
sim = DummySimulator()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.load_state_dict(torch.load("model.pt"))
if TRAIN:
    # %% GENERATE DATA
    # generate pairs of priors
    priors = [(prior(), prior()) for _ in range(num_priors)]

    # generate data
    print("Generating data...")
    runs = []
    for ts in tqdm(priors):
        for _ in range(num_sims_per_prior_pair):
            k = 0 if random() < 0.5 else 1
            runs.append((k, sim.simulate(ts[k]), ts))


    # generate ratios
    print("Generating ratios...")
    ratios = [sim.eval_ratio(zs, ts[k], ts[(k + 1) % 2]) for k, zs, ts in tqdm(runs)]

    # collate (add labels), shuffle, construct dataloaders
    data = [(k, zs, ratio, ts) for ((k, zs, ts), ratio) in zip(runs, ratios)]
    shuffle(data)
    train_data, test_data = data[:num_train_samples], data[num_train_samples:]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # %% TRAIN
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
            batch_sz = xs.shape[0]
            r_hat = model(xs, theta0, theta1)
            loss = rolr(r_hat, ratios, labels) * batch_size
            loss.backward()
            optimizer.step()
            num_samples += len(labels)
            epoch_loss += loss.item()
            train_iter.set_description("Average loss {}".format(epoch_loss/num_samples))

    # %% TEST
    print("Testing...")
    epoch_loss, num_samples = 0.0, 0
    x, ratio_true, ratio_pred = np.zeros((0)), np.zeros((0)), np.zeros((0))
    with torch.no_grad(): # we can only use this when we don't need the score (only the ratio)
        test_perf, num_samples = 0.0, 0
        test_iter = tqdm(test_loader)
        for batch in test_iter:
            labels, runs, ratios, (theta0, theta1) = batch
            xs = runs[:, -1]
            batch_sz = xs.shape[0]
            r_hat = model(xs, theta0, theta1)
            loss = rolr(r_hat, ratios, labels)
            num_samples += len(labels)
            epoch_loss += loss.item() * batch_size
            test_iter.set_description("Average loss {}".format(epoch_loss/num_samples))
            ratio_true = np.concatenate((ratio_true, ratios.cpu().detach()))
            ratio_pred = np.concatenate((ratio_pred, r_hat.cpu().detach()))
            x = np.concatenate((x, xs.cpu().detach()))

    torch.save(model.state_dict(), "model.pt")
else:
    model.load_state_dict(torch.load("model.pt"))
# %% ugly visualisation code - ignore
# from scipy.stats import gaussian_kde
# density_true = gaussian_kde(ratio_true)
# density_pred = gaussian_kde(ratio_pred.squeeze())
# xs = np.linspace(0,2,200)
# density_true.covariance_factor = lambda : .01
# density_true._compute_covariance()
# density_pred.covariance_factor = lambda : .01
# density_pred._compute_covariance()


theta0 = 0.2
theta1 = 0.8
# generate data for a single pair of thetas
visual_runs0 = np.array([sim.simulate(theta0)[1].cpu().detach().numpy() for _ in tqdm(range(40000))])
visual_runs1 = np.array([sim.simulate(theta1)[1].cpu().detach().numpy() for _ in tqdm(range(40000))])

xs = np.linspace(-5, 5, 500)
density_true0 = gaussian_kde(visual_runs0)
density_true0.covariance_factor = lambda : .05
density_true0._compute_covariance()
density_true1 = gaussian_kde(visual_runs1)
density_true1.covariance_factor = lambda : .05
density_true1._compute_covariance()
density_pred = [1/model(torch.tensor([x], dtype=torch.float32), torch.tensor([[theta0]], dtype=torch.float32), torch.tensor([[theta1]], dtype=torch.float32)) for x in xs]

plt.plot(xs, density_true0(xs)/density_true1(xs), "r")
#plt.plot(xs, density_true0(xs), "y")
#plt.plot(xs, density_true1(xs), "g")
plt.plot(xs, density_pred, "b")
plt.show()

print("Done")
