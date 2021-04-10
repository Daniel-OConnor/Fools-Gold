# %% SETUP
from simulators.dummy_simulator import DummySimulator
from simulators import lotkavolterra
from simulators.lotkavolterra import LotkaVolterra, normalisation_func_brehmer
from simulators.galton_board import GaltonBoard
from loss.rolr import rolr, rascal
from loss.cascal import cascal
from loss.scandal import scandal, gaussian_mixture_prob, categorical_prob
from data_generators import ratio_dataset, score_and_ratio_dataset, score_pairs_dataset, score_dataset
from trainer import train
from models.ratio import Ratio
from models.classifier import Classifier
from models.density_mixture import DensityMixture
from models.categorical_distribution import CategoricalDistribution
from tqdm import tqdm
import torch
from functools import partial
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

TRAIN = True
# training constants
batch_size = 20 #32
epochs = 300
train_fraction = 1
num_priors = 100 #30000
num_sims_per_prior_pair = 1
learning_rate = 0.00001
num_train_priors = int(num_priors * train_fraction)
num_test_priors = int(num_priors * (1-train_fraction))

prior = lambda: lotkavolterra.generate_prior(torch.rand(4), width=0.25).to(device)
sim = LotkaVolterra(normalisation_func=normalisation_func_brehmer)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

model = DensityMixture(sim.x_size, sim.theta_size, 10, 1000)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if TRAIN:
    # %% GENERATE DATA

    train_loader = score_dataset(sim, prior, num_train_priors, num_sims_per_prior_pair, batch_size, True)
    #test_loader = ratio_dataset(sim, prior, num_test_priors, num_sims_per_prior_pair, batch_size, False)

    # %% TRAIN
    train(model, train_loader, partial(scandal, alpha=3, prob_func=gaussian_mixture_prob), epochs, optimizer)

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

theta0 = torch.Tensor([0.01, 0.5, 1, 0.01])
theta1 = 0.8

x = np.linspace(0, 1, 100)
xs = np.reshape(x, (x.shape[0], 1))
xs = np.concatenate((xs, np.array([[-0.1, 4000, 2000, 0.2, 0.2, 0.02, 0.2, -0.2]]*xs.shape[0])), axis=1)

_, mean, sd, weight = model(torch.tensor([[0]], dtype=torch.float32), torch.unsqueeze(theta0, 0))
with torch.no_grad():
    density_pred = [torch.exp(gaussian_mixture_prob(torch.tensor([x], dtype=torch.float32), mean, sd, weight)) for x in xs]
print(mean, xs[0])
#print(mean, sd)
# generate data for a single pair of thetas
with torch.no_grad():
    visual_runs0 = np.array([sim.simulate(theta0)[-1].cpu().detach().numpy() for _ in tqdm(range(100))])
#visual_runs1 = np.array([sim.simulate(theta1)[1].cpu().detach().numpy() for _ in tqdm(range(1000))])
density_true0 = gaussian_kde(np.swapaxes(visual_runs0, 0, 1))
"""density_true1 = gaussian_kde(visual_runs1)
density_true1.covariance_factor = lambda : .1
density_true1._compute_covariance()"""


#density_pred = [(1-x)/x for x in density_pred]

plt.plot(x, density_true0(np.swapaxes(xs, 0, 1)), "r")
#plt.plot(xs, density_true0(xs), "y")
#plt.plot(xs, density_true1(xs), "g")
plt.plot(x, density_pred, "b")
plt.show()

print("Done")
