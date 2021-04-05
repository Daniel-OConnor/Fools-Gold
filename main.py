# %% SETUP
from simulators.dummy_simulator import DummySimulator
from loss.rolr import rolr
from data_generators import ratio_dataset
from trainer import train
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
train_fraction = 0.9
num_priors = 1000
num_sims_per_prior_pair = 1
learning_rate = 0.0002
num_train_priors = int(num_priors * train_fraction)
num_test_priors = int(num_priors * (1-train_fraction))


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

    train_loader = ratio_dataset(sim, prior, num_train_priors, num_sims_per_prior_pair, batch_size, True)
    test_loader = ratio_dataset(sim, prior, num_train_priors, num_sims_per_prior_pair, batch_size, False)

    # %% TRAIN
    train(model, train_loader, rolr, epochs, optimizer)

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
