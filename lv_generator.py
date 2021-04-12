from simulators import lotkavolterra
from simulators.lotkavolterra import LotkaVolterra, normalisation_func_brehmer, default_params
from data_generators import score_and_ratio_dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import concurrent.futures
from sys import argv
print(argv[1])
torch.multiprocessing.set_sharing_strategy('file_system')
total_iters = 7634
start_num = 7634 - argv[1]
num_workers = 24 # EDIT
num_iterations = 3626 # EDIT
num_samples_total = num_workers * num_iterations # EDIT
prefix = "lv_data_" # EDIT
extension = "pt" # EDIT
save_loc = "lv_data" # EDIT "lv_test_data"
prior0 = lambda: lotkavolterra.generate_prior(torch.rand(4), width=0.02).detach() # EDIT: e.g change this to some fixed value for generating test data
prior1 = lambda: default_params.detach() # EDIT
sim = LotkaVolterra(normalisation_func=normalisation_func_brehmer) # EDIT
num_samples_per_iteration = num_samples_total // num_iterations
print("Generating data...")

def foo(args):
    label, prior0, prior1 = args
    try:
        θ_0 = prior0.requires_grad_()
        θ_1 = prior1.requires_grad_()
        with torch.no_grad():
            if label == 1:
                zs = sim.simulate(θ_1)
            else:
                zs = sim.simulate(θ_0)
            assert(all([z.isfinite().all() for z in zs]))
            logp_0 = sim.log_p(zs, θ_0).detach()
            assert(logp_0.isfinite().all())
            logp_1 = sim.log_p(zs, θ_1).detach()
            assert(logp_1.isfinite().all())
        score0 = sim.eval_score(zs, θ_0).detach()
        assert(score0.isfinite().all())
        score1 = sim.eval_score(zs, θ_1).detach()
        assert(score1.isfinite().all())
        return (label, (θ_0, θ_1), zs[-1], (score0, score1), (logp_0, logp_1))
    except:
        return None

save_iter = tqdm(range(start_num, num_iterations + start_num))
total_runs = 0
saved_samples = 0

random_nums = torch.rand(num_samples_total)
labels = [1 if random_nums[i] > 1.0 else 0 for i in range(num_samples_total)]
args = [(k, prior0(), prior1()) for k in labels]
with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as worker_pool: 
    futures = [worker_pool.submit(foo, arg) for arg in args]
    results_iterator = concurrent.futures.as_completed(futures)
    for i in save_iter:
        res = [results_iterator.__next__().result() for i in range(num_samples_per_iteration)]
        total_runs += len(res)
        dataset = [t for t in res if t is not None]
        saved_samples += len(dataset)
        save_iter.set_description("Yield: {}".format(saved_samples/total_runs))
        torch.save(dataset, "{}/{}{}.{}".format(save_loc, prefix, i, extension))
        del res, dataset

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
