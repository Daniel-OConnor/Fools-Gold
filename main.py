from simulators import lotkavolterra
from simulators.lotkavolterra import LotkaVolterra, normalisation_func_brehmer
from loss.rolr import rolr, rascal
from data_generators import ratio_dataset, score_and_ratio_dataset, score_pairs_dataset, score_dataset
from trainer import train
from models.ratio import Ratio
from tqdm import tqdm
import torch
from functools import partial
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

TRAIN = True
# training constants
batch_size = 2 #32
epochs = 5
train_fraction = 1
num_priors = 4 #30000
num_sims_per_prior_pair = 1
learning_rate = 0.00001
num_train_priors = int(num_priors * train_fraction)
num_test_priors = int(num_priors * (1-train_fraction))

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

prior = lambda: lotkavolterra.generate_prior(torch.rand(4), width=0.25).to(device)
sim = LotkaVolterra(normalisation_func=normalisation_func_brehmer)


model = Ratio(sim.x_size, sim.theta_size, 200)
model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if TRAIN:
    # %% GENERATE DATA

    train_loader = score_and_ratio_dataset(sim, prior, num_train_priors, num_sims_per_prior_pair, batch_size, True)
    # %% TRAIN
    train(model, train_loader, partial(rascal, alpha=3), epochs, optimizer)

    torch.save(model.state_dict(), "model.pt")
else:
    model.load_state_dict(torch.load("model.pt"))

"""
theta0 = lotkavolterra.generate_prior(torch.Tensor([0.1, 0.2, 0.3, 0.4]), width=0.25).to(device)
theta1 = lotkavolterra.generate_prior(torch.Tensor([0.9, 0.8, 0.7, 0.6]), width=0.25).to(device)
# generate data for a single pair of thetas
visual_runs0 = np.array([sim.simulate(theta0)[1].cpu().detach().numpy() for _ in tqdm(range(100))])
visual_runs1 = np.array([sim.simulate(theta1)[1].cpu().detach().numpy() for _ in tqdm(range(100))])

xs = np.linspace(-5, 5, 500)
density_true0 = gaussian_kde(visual_runs0)
density_true0.covariance_factor = lambda : .1
density_true0._compute_covariance()
density_true1 = gaussian_kde(visual_runs1)
density_true1.covariance_factor = lambda : .1
density_true1._compute_covariance()

_, mean, sd, weight = model(torch.tensor([[0]], dtype=torch.float32), torch.tensor([[theta0]], dtype=torch.float32))
density_pred = [prob(x, mean, sd, weight) for x in xs]

plt.plot(xs, density_true0(xs), "r")
plt.plot(xs, density_pred, "b")
plt.show()
"""
print("Done")
