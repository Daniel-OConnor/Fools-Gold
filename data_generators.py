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
            zs = sim.simulate(ts[k])
            ratio = sim.eval_ratio(zs, ts[k], ts[(k + 1) % 2])
            data.append((k, ts, zs, ratio))
    return DataLoader(data, batch_size, shuffle=shuffle)


def score_dataset(sim, prior, num_priors, num_sims_per_prior_pair, batch_size, shuffle=True):
    print("Generating data...")
    data = []
    for _ in tqdm(range(num_priors)):
        ts = (prior(), prior())
        for _ in range(num_sims_per_prior_pair):
            k = 0 if random() < 0.5 else 1
            zs = sim.simulate(ts[k])
            score = sim.eval_score(zs, ts[0], ts[1])
            data.append((k, ts, zs, score))
    return DataLoader(data, batch_size, shuffle=shuffle)


def score_and_ratio_dataset(sim, prior, num_priors, num_sims_per_prior_pair, batch_size, shuffle=True):
    print("Generating data...")
    data = []
    for _ in tqdm(range(num_priors)):
        ts = (prior(), prior())
        for _ in range(num_sims_per_prior_pair):
            k = 0 if random() < 0.5 else 1
            zs = sim.simulate(ts[k])
            score = sim.eval_score(zs, ts[0], ts[1])
            ratio = sim.eval_ratio(zs, ts[0], ts[1])
            data.append((k, ts, zs, score, ratio))
    return DataLoader(data, batch_size, shuffle=shuffle)
