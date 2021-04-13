# %% SETUP
from simulators.dummy_simulator import DummySimulator
from simulators import lotkavolterra
from simulators.lotkavolterra import LotkaVolterra, normalisation_func_brehmer
from simulators.galton_board import GaltonBoard
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

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)

sim = GaltonBoard(10, 10)

theta0 = torch.tensor(-0.6)
theta1 = torch.tensor(-0.8)

n = 100000

visual_runs0 = [sim.simulate(theta0)[-1].cpu().detach().numpy()[0] for _ in tqdm(range(n))]
visual_runs_less = [sim.simulate(theta0)[-1].cpu().detach().numpy()[0] for _ in tqdm(range(100))]
visual_runs1 = [sim.simulate(theta1)[-1].cpu().detach().numpy()[0] for _ in tqdm(range(n))]

plt.hist(visual_runs0, range(10), density=True, label="-0.6", fill=False, ec="red")
plt.hist(visual_runs_less, range(10), density=True, label="-0.6 (100 simulations)", fill=False, ec="b")
plt.hist(visual_runs1, range(10), density=True, label="-0.8",  fill=False, ec="g")

plt.legend(loc="upper left")

plt.ylabel("Probability")
plt.xlabel("Bin")
plt.title("Generalized Galton Board", fontsize=30)

plt.show()
