import torch
from torch.utils.data import DataLoader
from functools import partial


class GenIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()


def load_dataset(num_files, save_loc, prefix):
    for i in range(num_files):
        saved_data = torch.load("{}/{}{}.pt".format(save_loc, prefix, i))
        for (label, (t0, t1), x, (score0, score1), (logp_0, logp_1)) in saved_data:
            t0.requires_grad_()
            t1.requires_grad_()
            #x.requires_grad_()
            score0.requires_grad_()
            score1.requires_grad_()
            logp_0.requires_grad_()
            logp_1.requires_grad_()
        for row in saved_data:
            yield row


def load_dataset_single(num_files, save_loc, prefix):
    for i in range(num_files):
        saved_data = torch.load("{}/{}{}.pt".format(save_loc, prefix, i))
        for row in saved_data:
            yield row

def load_ratio_dataset_(num_files, save_loc, prefix):
    dataset_loaded = load_dataset(num_files, save_loc, prefix)
    for (label, (t0, t1), x, (score0, score1), (logp_0, logp_1)) in dataset_loaded:
        if label == 0:
            ratio = logp_0-logp_1
        else:
            ratio = logp_1-logp_0
        yield (label, (t0, t1), x, ratio)


def load_ratio_dataset(batch_size, num_files, save_loc, prefix):
    return DataLoader(GenIterableDataset(partial(load_ratio_dataset_, num_files, save_loc, prefix)), batch_size)


def load_score_dataset_(num_files, save_loc, prefix):
    dataset_loaded = load_dataset(num_files, save_loc, prefix)
    for (label, (t0, t1), x, (score0, score1), (logp_0, logp_1)) in dataset_loaded:
        if label == 0:
            yield (label, (t0,), x, score0)
        else:
            yield (label, (t1,), x, score1)


def load_score_dataset(batch_size, num_files, save_loc, prefix):
    return DataLoader(GenIterableDataset(partial(load_score_dataset_, num_files, save_loc, prefix)), batch_size)


def load_score_pairs_dataset_(num_files, save_loc, prefix):
    dataset_loaded = load_dataset(num_files, save_loc, prefix)
    for (label, (t0, t1), x, (score0, score1), (logp_0, logp_1)) in dataset_loaded:
        if label == 0:
            yield (label, (t0, t1), x, score0)
        else:
            yield (label, (t0, t1), x, score1)


def load_score_pairs_dataset(batch_size, num_files, save_loc, prefix):
    return DataLoader(GenIterableDataset(partial(load_score_pairs_dataset_, num_files, save_loc, prefix)), batch_size)


def load_score_and_ratio_dataset_(num_files, save_loc, prefix):
    dataset_loaded = load_dataset(num_files, save_loc, prefix)
    for (label, (t0, t1), x, (score0, score1), (logp_0, logp_1)) in dataset_loaded:
        if label == 0:
            yield (label, (t0, t1), x, score0, logp_0-logp_1)
        else:
            yield (label, (t0, t1), x, score1, logp_1-logp_0)


def load_score_and_ratio_dataset(batch_size, num_files, save_loc, prefix):
    return DataLoader(GenIterableDataset(partial(load_score_and_ratio_dataset_, num_files, save_loc, prefix)), batch_size)
