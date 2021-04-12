from simulators import lotkavolterra
from simulators.lotkavolterra import LotkaVolterra, normalisation_func_brehmer, default_params
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
from torch.utils.data import DataLoader

TRAIN = True
# training constants
batch_size = 32
epochs = 1
train_fraction = 1
# num_priors = 4 #30000
# num_sims_per_prior_pair = 1
learning_rate = 0.001
# num_train_priors = int(num_priors * train_fraction)
# num_test_priors = int(num_priors * (1-train_fraction))

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


model = Ratio(9, 4, 200)
model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if TRAIN:
    # %% GENERATE DATA
    dataset = []
    print("Loading Data...")
    file_name = "../lv_data/lv_data_0_2929.pt"
    saved_data = torch.load(file_name)
    print("Preparing Data...")
    sum_k = 0
    for (k, ts, x, scores, probs) in tqdm(saved_data):
        sum_k += k
        ts[0].requires_grad_()
        ts[1].requires_grad_()
        x.requires_grad_()
        scores[k].requires_grad_()
        probs[0].requires_grad_()
        probs[1].requires_grad_()
        dataset.append((k, ts, x, scores[k], probs[k] - probs[(k + 1) % 2]))
    train_loader = DataLoader(dataset, batch_size, shuffle=False)# score_and_ratio_dataset(sim, prior, num_train_priors, num_sims_per_prior_pair, batch_size, True)
    # %% TRAIN
    train(model, train_loader, partial(rascal, alpha=0.01), epochs, optimizer)

    torch.save(model.state_dict(), "model.pt")
else:
    model.load_state_dict(torch.load("model.pt"))


sim = LotkaVolterra(normalisation_func=normalisation_func_brehmer)
theta0 = lotkavolterra.generate_prior(torch.Tensor([0.5, -0.5, 0.5, -0.5]), width=0.02).to(device)
theta1 = default_params.to(device)
# generate data for a single 
visual_runs0 = [sim.simulate(theta0)[1].cpu().detach().numpy() for _ in tqdm(range(100))]
visual_runs1 = [sim.simulate(theta1)[1].cpu().detach().numpy() for _ in tqdm(range(100))]

xs = np.linspace(-5, 5, 500)
density_true0 = gaussian_kde(visual_runs0)
density_true0.covariance_factor = lambda : .1
density_true0._compute_covariance()
density_true1 = gaussian_kde(visual_runs1)
density_true1.covariance_factor = lambda : .1
density_true1._compute_covariance()
ratio_true = lambda xs: density_true0(xs)/density_true1(xs)

ratio_pred = [model(x, theta0, theta1) for x in xs]
ratio_true_curve = gaussian_kde()
plt.plot(xs, density_true0(xs), "r")
plt.plot(xs, density_pred, "b")
plt.show()

print("Done")
