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
    try:
        θ_0 = prior().detach().requires_grad_()
        θ_1 = default_params.detach().requires_grad_() # prior().detach().requires_grad_()
        with torch.no_grad():
            label = 0 if torch.rand(1) < 0.5 else 1
            if label == 1:
                zs = sim.simulate(θ_1)
            else:
                zs = sim.simulate(θ_0)
            logp_0 = sim.log_p(zs, θ_0).detach()
            logp_1 = sim.log_p(zs, θ_1).detach()
        score0 = sim.eval_score(zs, θ_0).detach()
        score1 = sim.eval_score(zs, θ_1).detach()
        return (label, (θ_0, θ_1), zs[-1], (score0, score1), (logp_0, logp_1))
    except:
        return None

save_iter = tqdm(range(num_iterations))
total_runs = 0
saved_samples = 0
for i in save_iter:
    with Pool(num_workers) as p:
        res = list(p.imap(foo, range(num_priors_per_iteration)))
        dataset = [t for t in res if t is not None]
        total_runs += len(res)
        saved_samples += len(dataset)
        save_iter.set_description("Yield: {}".format(saved_samples/total_runs))
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