from abc import ABCMeta, abstractmethod
import numpy as np
#%matplotlib notebook
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
from tqdm import tqdm
from loss.scandal import gaussian_mixture_prob
from loss.scandal import categorical_prob


def ratio_kde(sim, theta0, theta1, n):
    runs0 = np.array([sim.simulate(theta0)[1].cpu().detach().numpy() for _ in tqdm(range(n))])
    runs1 = np.array([sim.simulate(theta1)[1].cpu().detach().numpy() for _ in tqdm(range(n))])
    density_true0 = gaussian_kde(runs0)
    density_true0.covariance_factor = lambda: .05
    density_true0._compute_covariance()
    density_true1 = gaussian_kde(runs1)
    density_true1.covariance_factor = lambda: .05
    density_true1._compute_covariance()
    return lambda x: density_true0(x)/density_true1(x)


def ratio_from_data(sim, runs0, runs1):
    density_true0 = gaussian_kde([x[0].numpy() for x in runs0])
    density_true1 = gaussian_kde([x[0].numpy() for x in runs1])
    return lambda x: density_true0(x)/density_true1(x)


def ratio_hist(sim, theta0, theta1, n, start, end):
    visual_runs0 = [sim.simulate(theta0)[-1].cpu().detach().numpy()[0] for _ in tqdm(range(n))]
    visual_runs1 = [sim.simulate(theta1)[-1].cpu().detach().numpy()[0] for _ in tqdm(range(n))]
    xs = np.arange(start, end)
    counts0 = [visual_runs0.count(x) + 0.01 for x in xs]
    counts1 = [visual_runs1.count(x) + 0.01 for x in xs]
    ratios = [counts0[i] / counts1[i] for i in range(len(counts0))]
    return ratios


def test_score(base_line, model, theta0, theta1, start, end, steps):
    xs = np.linspace(start, end, steps)
    loss = 0
    with torch.no_grad():
        for x in xs:
            base = base_line(x)
            predicted = model(torch.tensor([[x]]), torch.tensor([[theta0]]), torch.tensor([[theta1]]))
            loss += (base - predicted.detach().numpy()) ** 2
    return loss/len(xs)


def test_score_hist_class(base_line, model, theta0, theta1, sim, n):
    loss = 0
    runs = [sim.simulate(theta0)[-1].cpu().detach().numpy()[0] for _ in range(n)]
    with torch.no_grad():
        for x in runs:
            base = base_line[x]
            x = model(torch.tensor([[x]]), torch.tensor([[theta0]]), torch.tensor([[theta1]]))
            predicted = (1 - x) / x
            loss += (base - predicted.detach().numpy()) ** 2
    return loss/len(runs)


def test_score_hist_ratio(base_line, model, theta0, theta1, sim, n):
    loss = 0
    runs = [sim.simulate(theta0)[-1].cpu().detach().numpy()[0] for _ in range(n)]
    with torch.no_grad():
        for x in runs:
            base = base_line[x]
            predicted = model(torch.tensor([[x]]), torch.tensor([[theta0]]), torch.tensor([[theta1]]))
            loss += (base - predicted.detach().numpy()) ** 2
    return loss/len(runs)


def test_score_hist_score(base_line, model, theta0, theta1, sim, n):
    loss = 0
    runs = [sim.simulate(theta0)[-1].cpu().detach().numpy()[0] for _ in range(n)]
    with torch.no_grad():
        for x in runs:
            base = base_line[x]
            _, p0 = model(torch.tensor([[x]]), torch.tensor([[theta0]]))
            _, p1 = model(torch.tensor([[x]]), torch.tensor([[theta1]]))
            predicted = categorical_prob(torch.tensor([[x]]), p0) / categorical_prob(torch.tensor([[x]]), p1)
            loss += (base - predicted.detach().numpy()[0][0][0]) ** 2
    return loss/len(runs)


def test_score_class(base_line, model, theta0, theta1, sim, n):
    loss = 0
    runs = [sim.simulate(theta0)[-1].cpu().detach().numpy()[0] for _ in range(n)]
    with torch.no_grad():
        for x in runs:
            base = base_line(x)
            x = model(torch.tensor([[x]]), torch.tensor([[theta0]]), torch.tensor([[theta1]]))
            predicted = (1 - x) / x
            loss += (base - predicted.detach().numpy()) ** 2
    return loss/len(runs)


def test_score_ratio(base_line, model, theta0, theta1, sim, n):
    loss = 0
    runs = [sim.simulate(theta0)[-1].cpu().detach().numpy()[0] for _ in range(n)]
    with torch.no_grad():
        for x in runs:
            base = base_line(x)
            predicted = model(torch.tensor([[x]]), torch.tensor([[theta0]]), torch.tensor([[theta1]]))
            loss += (base - predicted.detach().numpy()) ** 2
    return loss/len(runs)


def test_score_score(base_line, model, theta0, theta1, sim, n):
    loss = 0
    runs = [sim.simulate(theta0)[-1].cpu().detach().numpy()[0] for _ in range(n)]
    with torch.no_grad():
        for x in runs:
            base = base_line(x)
            _, mean0, sd0, w0 = model(torch.tensor([[x]]), torch.tensor([[theta0]]))
            _, mean1, sd1, w1 = model(torch.tensor([[x]]), torch.tensor([[theta1]]))
            predicted = gaussian_mixture_prob(torch.tensor([[x]]), mean0, sd0, w0) / gaussian_mixture_prob(torch.tensor([[x]]), mean1, sd1, w1)
            loss += (base - predicted.detach().numpy()) ** 2
    return loss/len(runs)