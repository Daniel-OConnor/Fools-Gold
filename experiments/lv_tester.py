# %% SETUP
from simulators.dummy_simulator import DummySimulator
from simulators import lotkavolterra
from simulators.lotkavolterra import LotkaVolterra, normalisation_func_brehmer
from simulators.sir_fg import SIR_Sim
from loss.rolr import rolr, rascal
from loss.cascal import xe, cascal
from loss.scandal import scandal, gaussian_mixture_prob, categorical_prob, cross_entropy
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
name = "lv_test_losses.pt"
batch_size = 20 #32
epochs = 2
average = 5
max_samples = 200
train_fraction = 1
num_priors = 4000 #30000
num_sims_per_prior_pair = 1
learning_rate = 0.000001
num_train_priors = int(num_priors * train_fraction)
num_test_priors = int(num_priors * (1-train_fraction))

sim = LotkaVolterra(normalisation_func=normalisation_func_brehmer)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

theta0 = torch.Tensor([0.01*1.0005, 0.5*1.0005, 1*1.0005, 0.01*1.0005])
theta1 = torch.Tensor([0.01, 0.5, 1, 0.01])

runs0 = []

for x in load_dataset_single(1000, "lv_data_single", "lv_data_"):
    runs0.append(x)


runs1 = runs0
for (label, (t0, t1), x, (score0, score1), (logp_0, logp_1)) in load_dataset(1872, "lv_data", "lv_data_"):
    if label == 1:
        runs1.append(x)

print(len(runs0), len(runs1))
baseline = ratio_from_data(sim, runs0, runs1)
runs0 = runs0[:100]
#test_losses = torch.load(name)
#print(test_losses)
test_losses = {}

test_losses["rolr"] = []
for size in range(10, max_samples, 10):
    test = 0
    for _ in range(average):
        model = Ratio(sim.x_size, sim.theta_size, 50)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loader = load_ratio_dataset(20, size, "lv_data", "lv_data_")
        for i in range(epochs):
            # %% TRAIN
            train(model, train_loader, rolr, i, optimizer)
        test += test_score_ratio2(baseline, model, theta0, theta1, runs0, 100)
    print("test loss: ", test)
    test_losses["rolr"].append(test/average)

torch.save(test_losses, name)
test_losses["rascal"] = []
for size in range(10, max_samples, 10):
    test = 0
    for _ in range(average):
        model = Ratio(sim.x_size, sim.theta_size, 50)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loader = load_score_and_ratio_dataset(20, size, "lv_data", "lv_data_")
        for i in range(epochs):
            # %% TRAIN
            train(model, train_loader, partial(rascal, alpha=1), i, optimizer)
        test += test_score_ratio2(baseline, model, theta0, theta1, runs0, 100)
    print("test loss: ", test)
    test_losses["rascal"].append(test/average)

torch.save(test_losses, name)
test_losses["cascal"] = []
for size in range(10, max_samples, 10):
    test = 0
    for _ in range(average):
        model = Classifier(sim.x_size, sim.theta_size, 50)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loader = load_score_pairs_dataset(20, size, "lv_data", "lv_data_")
        for i in range(epochs):
            # %% TRAIN
            train(model, train_loader, partial(cascal, alpha=1), i, optimizer)
        test += test_score_class2(baseline, model, theta0, theta1, runs0, 100)
    print("test loss: ", test)
    test_losses["cascal"].append(test/average)

torch.save(test_losses, name)
test_losses["LRT"] = []
for size in range(10, max_samples, 10):
    test = 0
    for _ in range(average):
        model = Classifier(sim.x_size, sim.theta_size, 50)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loader = load_score_pairs_dataset(20, size, "lv_data", "lv_data_")
        for i in range(epochs):
            # %% TRAIN
            train(model, train_loader, xe, i, optimizer)
        test += test_score_class2(baseline, model, theta0, theta1, runs0, 100)
    print("test loss: ", test)
    test_losses["LRT"].append(test/average)

torch.save(test_losses, name)

test_losses["NDE"] = []
for size in range(10, max_samples, 10):
    test = 0
    for _ in range(average):
        model = DensityMixture(sim.x_size, sim.theta_size, 10, 50)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loader = load_score_dataset(20, size, "lv_data", "lv_data_")
        for i in range(epochs):
            # %% TRAIN
            train(model, train_loader, partial(cross_entropy, prob_func=gaussian_mixture_prob), i, optimizer)
        test += test_score_score2(baseline, model, theta0, theta1, runs0, 100)
    print("test loss: ", test)
    test_losses["NDE"].append(test/average)

torch.save(test_losses, name)
test_losses["scandal"] = []
for size in range(10, max_samples, 10):
    test = 0
    for _ in range(average):
        model = DensityMixture(sim.x_size, sim.theta_size, 10, 50)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loader = load_score_dataset(20, size, "lv_data", "lv_data_")
        for i in range(epochs):
            # %% TRAIN
            train(model, train_loader, partial(scandal, alpha=1, prob_func=gaussian_mixture_prob), i, optimizer)
        test += test_score_score2(baseline, model, theta0, theta1, runs0, 100)
    print("test loss: ", test)
    test_losses["scandal"].append(test/average)

torch.save(test_losses, name)

with torch.no_grad():
    #plotter.PlotClassifierNetwork(model, theta0, theta1, 0, 10, 100).plot()
    #plotter.PlotTrueRatioBars(sim, theta0, theta1, 0, 10, 11, 10000).plot()
    for name, data in test_losses.items():
        if len(data) == 19:
            plt.plot(list(range(1000, 20000, 1000)), data, label=name)
    plt.legend(loc="upper left", fontsize=15)
    plt.ylabel("Mean squared error between true and predicted r", fontsize=15)
    plt.xlabel("Number of training samples", fontsize=15)
    plt.title("Lotka Volterra Simulator Results", fontsize=20)
#plt.plot(xs, density_true1(xs), "g")
plt.show()
print("Done")
