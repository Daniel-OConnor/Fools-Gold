import torch
# from simulator import ProbSimulator
from simulators.simulator import ProbSimulator
from tqdm import tqdm

# TODO:
# * perform pilot run to get normalisation stats

torch_Tensor = torch.cuda.Tensor if torch.cuda.is_available() else torch.Tensor

default_params = torch.Tensor([0.01, 0.5, 1, 0.01])

brehmer_means = torch.Tensor([1.04272841e+02, 7.92735828e+01, 8.56355494e+00, 8.11906932e+00,
            9.75067266e-01, 9.23352650e-01, 9.71107191e-01, 9.11167340e-01,
            4.36308022e-02])
brehmer_stds = torch.Tensor([2.68008281e+01, 2.14120703e+02, 9.00247450e-01, 1.04245882e+00,
        1.13785497e-02, 2.63556410e-02, 1.36672075e-02, 2.76435894e-02,
        1.38785995e-01])

def sample_discrete(distribution: torch.Tensor) -> int:
    """Sample a discrete distribution
    
    Arguments:
        distribution: torch.Tensor, of shape (n), representing an unnormalised mass function
        NOTE: undefined behaviour when there are events with mass == 0
    
    Returns:
        v, the index of the event chosen
    """
    pmf = normalise_mf(distribution)
    cdf = torch.cumsum(pmf, dim=0)
    x = torch.rand(1)
    return torch.searchsorted(cdf, x)

def normalise_mf(distribution: torch.Tensor) -> torch.Tensor:
    # normalise a mass function
    total_mass = distribution.sum()
    return distribution / total_mass

def summary_statistics(zs: torch.Tensor, normalisation_func) -> torch.Tensor:
    """Calculate summary statistics as given in arXiv:1605.06376 appendix F
    
    Arguments:
        zs:                 torch.Tensor                , of shape (n, 2),
                                the time series of the two populations X and Y
        normalisation_func: torch.Tensor -> torch.Tensor, the function for
                                normalising the statistics (optional)
    
    Returns:
        torch.Tensor, of shape (9), the summary statistics
    """
    mean_xy = zs.mean(dim=0)
    var_xy = zs.var(dim=0)
    norm_xy = (zs - mean_xy) / (var_xy.sqrt())
    # "autocorrelation coefficient of each time series at lag 1 and lag 2."
    # see the implementation at:
    # https://github.com/johannbrehmer/goldmine/blob/master/goldmine/simulators/lotka_volterra.py
    # for an explanation
    n = zs.shape[0]
    autocorr_xy_1 = (norm_xy[:-1] * norm_xy[1:]).sum(dim=0) / (n - 1)
    autocorr_xy_2 = (norm_xy[:-2] * norm_xy[2:]).sum(dim=0) / (n - 1)
    # ensure autocorrelations are in the same order as "Mining gold..."
    autocorr = torch.stack([autocorr_xy_1, autocorr_xy_2]).transpose(0, 1).reshape((-1))
    cross_corr = (norm_xy[:, 0] * norm_xy[:, 1]).sum(dim=0).unsqueeze(0) / (n - 1)
    summary = torch.cat([mean_xy, var_xy.log(),
                         autocorr,
                         cross_corr])
    return normalisation_func(summary)

def normalisation_func_brehmer(summary: torch.Tensor) -> torch.Tensor:
    """normalisation func taken from
    https://github.com/johannbrehmer/goldmine/blob/master/goldmine/simulators/lotka_volterra.py

    Arguments:
        summary: torch.Tensor of shape (9), the unnormalised summary statistics
    
    Returns:
        torch.Tensor of shape (9), the normalised summary statistics
    """
    return (summary - brehmer_means) / brehmer_stds

def normalisation_func_FG(summary: torch.Tensor) -> torch.Tensor:
    """normalisation func based on the means and stds from a pilot run of 1000 simulations,
    as described in arXiv:1605.06376 appendix F
    
    Arguments:
        summary: torch.Tensor of shape (9), the unnormalised summary statistics
    
    Returns:
        torch.Tensor of shape (9), the normalised summary statistics
    """
    raise NotImplementedError()

def pilot_run(path: str):
    # perform a pilot run of 1000 runs and save mean and std of summary statistics to path
    lv = LotkaVolterra()
    stats = torch.zeros((1000, 9))
    for i in tqdm(range(1000)):
        _, summary = lv.simulate(default_params)
        stats[i] = summary
    means = stats.mean(dim=0)
    stds = stats.std(dim=0)
    torch.save((means, stds), path)

def generate_prior(t, width=1):
    """Return a set of priors that are uniformly distributed in log-space
    Arguments:
        t:    torch.Tensor of size (n, 4), sampled from a uniform distribution [0, 1)
    """
    modifier = width * (t - 0.5)
    prior = torch.exp(modifier) * default_params
    # prior.requires_grad_()
    return prior

class LotkaVolterra(ProbSimulator):
    x_size = 9
    theta_size = 4
    def __init__(self, init_predators=50, init_prey=100, num_time_units=30, step_size=0.2, normalisation_func=(lambda x: x)):
        self.init_predators = init_predators
        self.init_prey = init_prey
        self.num_time_units = num_time_units
        self.step_size = step_size
        self.num_steps = int(num_time_units / step_size) + 1 # include t=0
        self.normalisation_func = normalisation_func

    def simulate_utopia(self, birth_rate, time, prey_pop):
        total_extra_steps = 0
        X_rand = torch.Tensor()
        Y_all = torch.Tensor()
        DT_CUMSUM = torch.Tensor()
        multiplier = 1000
        curr_time = time
        while curr_time < self.num_time_units - .001:
            est_num_steps = int(((self.num_time_units - curr_time) * multiplier))
            multiplier *= 10
            X_rand_local = torch.rand(est_num_steps) 
            Y_all_local = torch.range(prey_pop + total_extra_steps, prey_pop + est_num_steps + total_extra_steps - 1) #
            Y_rate = birth_rate * Y_all_local
            DT = -(torch.log(1 - X_rand_local)/Y_rate)
            DT_CUMSUM_local = torch.cumsum(DT, 0) + curr_time
            end_idx = torch.searchsorted(DT_CUMSUM_local, self.num_time_units)
            DT_CUMSUM_local = DT_CUMSUM_local[:end_idx]
            X_rand_local = X_rand_local[:end_idx]
            Y_all_local = Y_all_local[:end_idx]
            curr_time = DT_CUMSUM_local[-1]
            DT_CUMSUM = torch.cat([DT_CUMSUM, DT_CUMSUM_local])
            X_rand = torch.cat([X_rand, X_rand_local])
            Y_all = torch.cat([Y_all, Y_all_local])
            total_extra_steps += end_idx
        # X_rand, Y_all, and DT_CUMSUM should be populated
        remaining_pops = torch.stack([DT_CUMSUM, torch.zeros(total_extra_steps), Y_all], dim=1)
        return (remaining_pops, (X_rand, birth_rate, Y_all))

    def simulate(self, θ, epsilon=1e-9):
        """
        Perform a single run of the simulator
        returns a list of torch.Tensor zs, where zs[i] is the i-th latent
        
        For the general structure of simulators in our implementation, probability calculation
        is decoupled from simulation. As such, the method in arXiv:1605.06376 appendix F
        and "Mining gold...", where Gillespie's algorithm is run, but the state is only saved
        at fixed timesteps would not work as we require all latent variables to be able to
        determine the likelihoods in a decoupled manner.
        """
        pops = torch.Tensor([[0., self.init_predators, self.init_prey]])
        time = torch.Tensor([0.])
        reactions = torch.Tensor([[ 1,  0],  # a_0
                                  [-1,  0],  # a_1
                                  [ 0,  1],  # a_2
                                  [ 0, -1]   # a_3
                                 ])

        # Gillespie's algorithm:
        # consider the variant in "Mining gold..." and arXiv:1605.06376 as operating two
        # processes at the same time:
        # * The simulation, where a new reaction occurs after a random amount of time
        # * The observation, where an observation is drawn of the state at each time step
        # We simplify this by "de-interleaving" the two loops into the simulation loop and
        # the sampling/observation loop (sampling is done when calculating summary statistics).
        # This doesn't affect complexity or performance, and IMO, is easier to understand.

        # simulation loop ("Pure Gillespie's algorithm")
        # see https://en.wikipedia.org/wiki/Gillespie_algorithm#Algorithm
        num_iterations = 0
        extra = (None, None)
        while time < self.num_time_units:
            if num_iterations + 1 >= pops.shape[0]:
                pops = torch.cat([pops, torch.zeros(pops.shape)])
            curr_state = pops[num_iterations]
            if curr_state[1] == 0 and curr_state[2] > 0:
                # all predators are dead, need to wrap things up quickly
                extra = self.simulate_utopia(θ[2], time, curr_state[2])
                break
            # Rates of different possible events
            rates = θ * torch.Tensor([curr_state[1] * curr_state[2],
                                      curr_state[1],
                                      curr_state[2],
                                      curr_state[1] * curr_state[2]
                                     ])
            total_rate = rates.sum()
            num_iterations += 1
            if total_rate < epsilon:
                # populations have died out, prevent infinite looping
                pops[num_iterations] = torch.Tensor([self.num_time_units, 0., 0.])
                break
            # sampling exponential distribution with rate (parameter) = total_rate
            delta_t = -(torch.log(1 - torch.rand(1))/total_rate)
            time += delta_t
            next_reaction = sample_discrete(rates)
            new_state = curr_state[1:] + reactions[next_reaction.item()]
            pops[num_iterations] = torch.cat([time, new_state])
        pops = pops[:num_iterations + 1]
        return list(pops) + [extra[1]] + [summary_statistics(self.sample(pops, extra[0]), self.normalisation_func)]

    def sample(self, _pops: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        # sampling loop
        if extra is None:
            pops = _pops
        else:
            pops = torch.cat([_pops, extra])
        num_data = pops.shape[0]
        zs = torch.zeros((self.num_steps, 2))
        zs[0] = pops[0, 1:]
        times = pops[:, 0].contiguous()
        j = 1
        for i in range(1, self.num_steps):
            sample_time = torch.Tensor([self.step_size * i])
            j += torch.searchsorted(, sample_time)
            #while (j < num_data) and (pops[j, 0] <= sample_time):
            #    j += 1
            # pops[j, 1:] is the pop at the earliest time >= sample_time
            zs[i] = pops[j - 1, 1:]
        zs[-1] = pops[-1, 1:] # to get the final population values
        return zs

    def log_p(self, zs, θ):
        """
        Calculate conditional probabilities for a run of the simulator
        
        Arguments:
            zs:        List[torch.Tensor], latent variables
            θ:         torch.Tensor, parameters
        Returns:
            log_p:     torch.Tensor (0 dim), equal to log(ps.prod()),
                   where ps[i] = p(z_i | θ, zs[:i])
        """
        ps = torch.zeros(len(zs) - 1)
        ps[0] = 0 # 1 # initial state
        reaction_lookup = {(0, -1): 3, (1, 0): 0, (-1, 0): 1, (0, 1): 2}
        for i in range(1, len(zs) - 2):
            curr_state = zs[i]
            prev_state = zs[i - 1]
            # calculate probability of event
            delta_pop = torch.round(curr_state[1:] - prev_state[1:])
            reaction = (int(delta_pop[0]), int(delta_pop[1]))
            if reaction == (0, 0):
                # this is the extinction reaction
                # ps[i] = 0
                break
            reaction_idx = reaction_lookup[reaction]
            rates = θ * torch.Tensor([prev_state[1] * prev_state[2],
                                      prev_state[1],
                                      prev_state[2],
                                      prev_state[1] * prev_state[2]
                                     ])
            total_rate = rates.sum()
            prob_event = rates[reaction_idx] / total_rate
            # calculate probability of time
            delta_t = curr_state[0] - prev_state[0]
            # prob_time = total_rate * torch.exp(-delta_t * total_rate)
            ps[i] = prob_event.log() + total_rate.log() - (delta_t * total_rate)
        if zs[-2] is not None:
            X_rand, orig_theta2, Y_all_local = zs[-2]
            ps[-1] = (θ[2] * Y_all_local).log().sum() + (1-X_rand).log().sum() * (θ[2]/orig_theta2)
        return ps.sum()