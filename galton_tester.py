# %% SETUP
from simulators.dummy_simulator import DummySimulator
from simulators import lotkavolterra
from simulators.lotkavolterra import LotkaVolterra, normalisation_func_brehmer
from simulators.galton_board import GaltonBoard
from loss.rolr import rolr, rascal
from loss.cascal import cascal
from loss.scandal import scandal, gaussian_mixture_prob, categorical_prob
from data_generators import ratio_dataset, score_and_ratio_dataset, score_pairs_dataset, score_dataset
from data_loaders import *
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
import plotter
from test_targets import *

# training constants
TRAIN = True
batch_size = 20 #32
epochs = 5
train_fraction = 1
num_priors = 4000 #30000
num_sims_per_prior_pair = 1
learning_rate = 0.000001
num_train_priors = int(num_priors * train_fraction)
num_test_priors = int(num_priors * (1-train_fraction))

prior = lambda: torch.tensor([-0.7]).to(device) + torch.rand(1).to(device) * torch.tensor([.3])
sim = GaltonBoard(10, 10)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

model = Ratio(sim.x_size, sim.theta_size, 100)
model.to(device)

theta0 = torch.tensor(-0.6)
theta1 = torch.tensor(-0.8)

#model.load_state_dict(torch.load("model.pt"))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# %% GENERATE DATA
if TRAIN:
    train_loader = load_ratio_dataset(20, 200, "galton_data2", "galton_data_")
    baseline = ratio_hist(sim, theta0, theta1, 100, 0, 10)
    for i in range(epochs):
        # %% TRAIN
        train(model, train_loader, rolr, i, optimizer)
        print("test loss: ", test_score_hist_ratio(baseline, model, theta0, theta1, sim, 100))


    torch.save(model.state_dict(), "model.pt")
else:
    model.load_state_dict(torch.load("model.pt"))


with torch.no_grad():
    plotter.PlotClassifierNetwork(model, theta0, theta1, 0, 10, 100).plot()
    plotter.PlotTrueRatioBars(sim, theta0, theta1, 0, 10, 11, 10000).plot()
#plt.plot(xs, density_true0(xs), "y")
#plt.plot(xs, density_true1(xs), "g")
plt.show()
print("Done")
