import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from random import random


def ratio_dataset(sim, prior, num_priors, num_sims_per_prior_pair, batch_size, shuffle=True):
    print("Generating data...")
    data = []
    for _ in tqdm(range(num_priors)):
        ts = (prior(), prior())
        for _ in range(num_sims_per_prior_pair):
            k = 0 if random() < 0.5 else 1
            with torch.no_grad():
                zs = sim.simulate(ts[k])
                ratio = sim.eval_ratio(zs, ts[k], ts[(k + 1) % 2])
            data.append((k, ts, zs[-1], ratio))
    return DataLoader(data, batch_size, shuffle=shuffle)


def score_dataset(sim, prior, num_priors, num_sims_per_prior_pair, batch_size, shuffle=True):
    print("Generating data...")
    data = []
    for _ in tqdm(range(num_priors)):
        t = (prior(),)
        for _ in range(num_sims_per_prior_pair):
            t = (t[0].detach().requires_grad_(),)
            with torch.no_grad():
                zs = sim.simulate(t[0])
            score = sim.eval_score(zs, t[0])
            data.append((0, t, zs[-1], score))
    return DataLoader(data, batch_size, shuffle=shuffle)


def score_pairs_dataset(sim, prior, num_priors, num_sims_per_prior_pair, batch_size, shuffle=True):
    print("Generating data...")
    data = []
    for _ in tqdm(range(num_priors)):
        ts = (prior(), prior())
        for _ in range(num_sims_per_prior_pair):
            ts = (ts[0].detach().requires_grad_(), ts[1].detach().requires_grad_())
            k = 0 if random() < 0.5 else 1
            with torch.no_grad():
                zs = sim.simulate(ts[k])
            score = sim.eval_score(zs, ts[k])
            data.append((k, ts, zs[-1], score))
    return DataLoader(data, batch_size, shuffle=shuffle)


def score_and_ratio_dataset(sim, prior, num_priors, num_sims_per_prior_pair, batch_size, shuffle=True):
    print("Generating data...")
    data = []
    for _ in tqdm(range(num_priors)):
        ts = (prior(), prior())
        for _ in range(num_sims_per_prior_pair):
            ts = (ts[0].detach().requires_grad_(), ts[1].detach().requires_grad_())
            k = 0 if random() < 0.5 else 1
            with torch.no_grad():
                zs = sim.simulate(ts[k])
            score = sim.eval_score(zs, ts[k])
            ratio = sim.eval_ratio(zs, ts[k], ts[(k + 1) % 2]).detach()
            data.append((k, ts, zs[-1], score, ratio))
    return DataLoader(data, batch_size, shuffle=shuffle)
