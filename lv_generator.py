from simulators import lotkavolterra
from simulators.lotkavolterra import LotkaVolterra, normalisation_func_brehmer, default_params
from data_generators import score_and_ratio_dataset
from multiprocessing import Pool
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

num_priors_total = 16 # EDIT
num_workers = 4 # EDIT
num_iterations = 2 # EDIT
prefix = "lv_data_" # EDIT
extension = "pt" # EDIT
save_loc = "lv_data" # EDIT
prior = lambda: lotkavolterra.generate_prior(torch.rand(4), width=0.02) # EDIT
sim = LotkaVolterra(normalisation_func=normalisation_func_brehmer) # EDIT

num_priors_per_iteration = num_priors_total // num_iterations
print("Generating data...")

def foo(i):
    θ = prior().detach().requires_grad_()
    with torch.no_grad():
        zs = sim.simulate(θ)
        logp_0 = sim.log_p(zs, θ).detach()
        logp_1 = sim.eval_ratio(zs, default_params).detach()
    score0 = sim.eval_score(zs, θ).detach()
    score1 = sim.eval_score(zs, default_params).detach()
    return (0, (θ, default_params), zs[-1], (score0, score1), (logp_0, logp_1))

for i in tqdm(range(num_iterations)):
    print("Iteration {} of {}...".format(i+1, num_iterations))
    with Pool(num_workers) as p:
        dataset = list(p.imap(foo, range(num_priors_per_iteration))
    torch.save(dataset, "{}/{}{}.{}".format(save_loc, prefix, i, extension))

# EXAMPLE LOADING CODE

# dataset = []
# for i in range(num_files):
#     saved_data = torch.load("../lv_data/lv_data_{}.pt".format(i))
#     for (_, (t1, t2), x, (score0, score1), (logp_0, logp_1)) in saved_data:
#         t1.requires_grad_()
#         t2.requires_grad_()
#         x.requires_grad_()
#         score0.requires_grad_()
#         score1.requires_grad_()
#         logp_0.requires_grad_()
#         logp_1.requires_grad_()
#     dataset = dataset + saved_data