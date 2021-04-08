import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import torch
from tqdm import tqdm
from loss.scandal import prob


# This runs a simulator n times for each of the two parameters theta0 and theta1
# It currently uses tqdm for a progress bar as it takes awhile to complete
# It estimates the probability densities, then divides them to estimate the probability ratio
def plot_true_ratio(sim, theta0, theta1, start, end, steps, n):
    # generate data for a pair of thetas
    visual_runs0 = np.array([sim.simulate(theta0)[1].cpu().detach().numpy() for _ in tqdm(range(n))])
    visual_runs1 = np.array([sim.simulate(theta1)[1].cpu().detach().numpy() for _ in tqdm(range(n))])
    xs = np.linspace(start, end, steps)
    density_true0 = gaussian_kde(visual_runs0)
    density_true0.covariance_factor = lambda: .05
    density_true0._compute_covariance()
    density_true1 = gaussian_kde(visual_runs1)
    density_true1.covariance_factor = lambda: .05
    density_true1._compute_covariance()
    plt.plot(xs, density_true0(xs) / density_true1(xs), "r")
    plt.show()


# This is similar to plot_true_ratio, except it plots the probability density for a single prior theta
def plot_true_likelihood(sim, theta, start, end, steps):
    # generate data for a pair of thetas
    visual_runs = np.array([sim.simulate(theta)[1].cpu().detach().numpy() for _ in tqdm(range(40000))])
    xs = np.linspace(start, end, steps)
    density_true = gaussian_kde(visual_runs)
    density_true.covariance_factor = lambda: .05
    density_true._compute_covariance()
    plt.plot(xs, density_true(xs), "r")
    plt.show()


# This plots the probability density outputted by a density network "model" when given theta as input
def plot_density_network(model, theta, start, end, steps):
    xs = np.linspace(start, end, steps)
    _, mean, sd, weight = model(torch.tensor([[0]], dtype=torch.float32), torch.tensor([[theta]], dtype=torch.float32))
    # density_pred = [(1-x)/x for x in density_pred]
    density_pred = [prob(x, mean, sd, weight) for x in xs]
    plt.plot(xs, density_pred, "b")
    plt.show()


# This plots the probability ratio outputted by a classifier network "model" when given theta0, theta1 as input
def plot_classifier_network(model, theta0, theta1, start, end, steps):
    xs = np.linspace(start, end, steps)

    density_pred = [model(torch.tensor([[x]], dtype=torch.float32), torch.tensor([[theta0]], dtype=torch.float32),
                          torch.tensor([[theta1]], dtype=torch.float32)) for x in xs]
    density_pred = [(1 - x) / x for x in density_pred]

    plt.plot(xs, density_pred, "b")
    plt.show()


# This plots the probability ratio outputted by a ratio network "model" when given theta0, theta1 as input
def plot_ratio_network(model, theta0, theta1, start, end, steps):
    xs = np.linspace(start, end, steps)
    density_pred = [1 / model(torch.tensor([x], dtype=torch.float32), torch.tensor([[theta0]], dtype=torch.float32),
                              torch.tensor([[theta1]], dtype=torch.float32)) for x in xs]

    plt.plot(xs, density_pred, "b")
    plt.show()