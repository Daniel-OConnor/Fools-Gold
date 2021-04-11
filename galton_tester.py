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

# training constants
TRAIN = True
batch_size = 20 #32
epochs = 20
train_fraction = 1
num_priors = 4000 #30000
num_sims_per_prior_pair = 1
learning_rate = 0.000001
num_train_priors = int(num_priors * train_fraction)
num_test_priors = int(num_priors * (1-train_fraction))

prior = lambda: ((torch.rand(1).to(device))*0.25)
sim = GaltonBoard(10, 10)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

model = Classifier(sim.x_size, sim.theta_size, 300)
model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# %% GENERATE DATA
if TRAIN:
    train_loader = load_score_pairs_dataset(20, 400, "galton_data", "galton_data_")
    #train_loader = score_dataset(sim, prior, num_train_priors, num_sims_per_prior_pair, batch_size, True)
    # test_loader = ratio_dataset(sim, prior, num_test_priors, num_sims_per_prior_pair, batch_size, False)

    # %% TRAIN
    train(model, train_loader, partial(cascal, alpha=1), epochs, optimizer)

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

theta0 = torch.tensor(0.5)
theta1 = torch.tensor(0)
# generate data for a single pair of thetas
"""visual_runs0 = np.array([sim.simulate(theta0)[-1].cpu().detach().numpy() for _ in tqdm(range(1000))])
#visual_runs1 = np.array([sim.simulate(theta1)[1].cpu().detach().numpy() for _ in tqdm(range(10000))])
xs = list(range(10))



#density_true1 = gaussian_kde(visual_runs1)
#density_true1.covariance_factor = lambda : .1
#density_true1._compute_covariance()
_, p = model(torch.tensor([[0]], dtype=torch.float32), torch.tensor([[theta0]], dtype=torch.float32))
_, p2 = model(torch.tensor([[0]], dtype=torch.float32), torch.tensor([[theta1]], dtype=torch.float32))
print(p)
#density_pred = [(1-x)/x for x in density_pred]
with torch.no_grad():
    density_pred = [torch.exp(categorical_prob(torch.tensor(x, dtype=torch.long), p)) for x in xs]
    density_pred2 = [torch.exp(categorical_prob(torch.tensor(x, dtype=torch.long), p2)) for x in xs]
#print(visual_runs0)"""
#plt.plot(xs, density_pred, "b")
#plt.plot(xs, density_pred2, "b")
#plotter.PlotHist(visual_runs0, [], "r").plot()
#plotter.PlotTrueRatio(sim, theta0, theta1, 0, 10, 1, 1000)
with torch.no_grad():
    plotter.PlotClassifierNetwork(model, theta0, theta1, 0, 10, 100).plot()
    plotter.PlotTrueRatioBars(sim, theta0, theta1, 0, 10, 11, 1000).plot()
#plt.plot(xs, density_true0(xs), "y")
#plt.plot(xs, density_true1(xs), "g")
plt.show()
print("Done")
