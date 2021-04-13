from simulators import lotkavolterra
from simulators.sir_fg import SIR_Sim
from data_generators import score_and_ratio_dataset
from multiprocessing import Pool
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")

num_priors_total = 100000
num_workers = 4
num_iterations = 1000
prefix = "sir_data_"
extension = "pt"
save_loc = "sir_data_single"
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
sim = SIR_Sim()
default_params0 = torch.tensor([0.3])
default_params1 = torch.tensor([0.3003])
num_priors_per_iteration = num_priors_total // num_iterations
print("Generating data...")

def foo(i):
    try:
        θ_1 = default_params1
        with torch.no_grad():
            zs = sim.simulate(θ_1)
            return zs[-1]
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
