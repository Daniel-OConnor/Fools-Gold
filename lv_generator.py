from simulators import lotkavolterra
from simulators.lotkavolterra import LotkaVolterra, normalisation_func_brehmer, default_params
from data_generators import score_and_ratio_dataset
from multiprocessing import Pool
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

num_priors_total = 16
num_workers = 4
priors_per_worker = num_priors_total // num_workers

prior = lambda: lotkavolterra.generate_prior(torch.rand(4), width=0.02)
sim = LotkaVolterra(normalisation_func=normalisation_func_brehmer)

print("Generating data...")

def foo(i):
    θ = prior().detach().requires_grad_()
    with torch.no_grad():
        zs = sim.simulate(θ)
    with torch.no_grad():
        ratio = sim.eval_ratio(zs, θ, default_params).detach()
    score = sim.eval_score(zs, θ).detach()
    return (0, (θ, default_params), zs[-1], score, ratio)

with Pool(num_workers) as p:
    dataset = list(tqdm(p.imap(foo, range(num_priors_total)), total=num_priors_total))

torch.save(dataset, 'lv_data.pt')