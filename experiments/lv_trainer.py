# SETUP
from loss.rolr import rolr, rascal
from loss.cascal import xe, cascal
from loss.scandal import scandal, gaussian_mixture_prob, categorical_prob, cross_entropy
from data_loaders import *
from trainer import train
from models.ratio import Ratio
from models.classifier import Classifier
from models.density_mixture import DensityMixture
from tqdm import tqdm
import torch
from functools import partial
from pathlib import Path

# training constants
batch_size = 32 #32
epochs = 2
average = 10
train_fraction = 1
learning_rate = 0.001
experiments = ["rolr", "rascal", "cascal", "scandal", "scandal2", "LRT", "NDE"]

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

training_info = {"base": {"x_size": 9, "theta_size": 4, "optimizer": torch.optim.Adam, "learning_rate": 0.001}

training_info["rolr"].update("model" = lambda: Ratio(9, 4, 50), "hidden_size": 50, "loader": load_ratio_dataset, "loss"= rolr)

training_info["rascal"] = training_info["rolr"].copy()
training_info["rascal"].update("loader" = load_score_and_ratio_dataset, "alpha" = 0.1, "loss"= partial(rascal, alpha=0.1))

training_info["cascal"] = training_info["rascal"].copy()
training_info["cascal"].update("model"= lambda: Classifier(9, 4, 50), "loader" = load_score_pairs_dataset, "alpha" = 0.1, "loss"=partial(cascal, alpha=0.1))

training_info["LRT"] = training_info["cascal"].copy()
training_info["LRT"].update("loss"=xe)

training_info["NDE"] = training_info["base"].copy()
training_info["NDE"].update("model" = lambda: DensityMixture(9, 4, 10, 50), "loader" = load_score_dataset, "loss"=partial(cross_entropy, gaussian_mixture_prob))

training_info["scandal"] = training_info["NDE"].copy()
training_info["scandal"].update("loss": partial(scandal, alpha=0.1, prob_func=gaussian_mixture_prob), "alpha" = 0.1)

training_info["scandal2"] = training_info["scandal"].copy()
training_info["scandal2"].update("loss": partial(scandal, alpha=0.01, prob_func=gaussian_mixture_prob), "alpha" = 0.01)

dataset_sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
pbar = tqdm(total = 490)
for experiments in experiments:
    Path("models/{}".format(experiment)).mkdir(parents=True, exist_ok=True)
    for data_size in dataset_sizes:
        for model_id in range(10):
            pbar.update(1)
            name = "{}_{}_{}".format(experiment, str(data_size), str(model_id))
            model = training_info[experiment]["model"]()
            model.to(device)
            optimizer = training_info[experiment]["optimizer"].to(device)
            train_loader = training_info[experiment]["loader"](32, "../training/data_all_{}.pt".format(data_size))
            for i in range(epochs):
                train(model, train_loader, training_info[experiment]["loss"], i, optimizer)
            torch.save(model.state_dict(), "models/{}/{}.pt".format(experiment, name))


