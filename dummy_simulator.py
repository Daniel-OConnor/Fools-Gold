import torch
from simulator import RatioSimulator, ProbSimulator
from contextlib import nullcontext
import math

PI = torch.tensor(math.pi)

def gaussian(x, mean, sd):
    return 1/(torch.abs(sd)*torch.sqrt(2*PI)) * torch.exp(-0.5*(x-mean)*(x-mean)/(sd*sd))


class DummySimulator(RatioSimulator, ProbSimulator):
    x_size = 1
    theta_size = 1

    def simulate(self, theta):
        r0 = torch.normal(theta, torch.tensor(1.))
        r1 = torch.normal(torch.tensor(0.), r0)
        return torch.tensor([r0, r1])

    def p(self, zs, θ, with_grad=False):
        """
        Calculate conditional probabilities for a run of the simulator
        
        Arguments:
            zs:        torch.Tensor, latent variables
            θ:         torch.Tensor, parameters
            with_grad: bool        , set to false if gradient not needed:
                                        * the computation should be run with torch.no_grad()
        Returns:
            ps: torch.Tensor, where ps[i] = p(z_i|θ, zs[:i])
        """
        context = nullcontext if with_grad else torch.no_grad
        with context():
            p0 = gaussian(zs[0], θ, torch.tensor(1))
            p1 = gaussian(zs[1], 0, zs[0])
        return torch.tensor([p0, p1])

    def ratio(self, zs, θ_0, θ_1, with_grad=False):
        """
        Calculate conditional probability ratios for a run of the simulator
        
        Arguments:
            zs:        torch.Tensor, latent variables
            θ_0:       torch.Tensor, parameters
            θ_1:       torch.Tensor, parameters
            with_grad: bool        , set to false if gradient not needed:
                                        * the computation should be run with torch.no_grad()
        Returns:
            rs: torch.Tensor, where rs[i] = p(z_i | θ_0, zs[:i]) / p(z_i | θ_1, zs[:i])
        """
        context = nullcontext if with_grad else torch.no_grad
        with context():
            r0 = gaussian(zs[0], θ_0, torch.tensor(1)) / gaussian(zs[0], θ_1, torch.tensor(1))
            r1 = gaussian(zs[1], 0, zs[0]) / gaussian(zs[1], 0, zs[0])
        return torch.tensor([r0, r1])

if __name__ == "__main__":
    dummy = DummySimulator()
    for i in range(10):
        print(dummy.sample_R(torch.tensor(1.), torch.tensor(0.5)))
    print()
    for i in range(10):
        print(dummy.sample_T(torch.tensor(1.)))
    print()   
    for i in range(10):
        print(dummy.sample_both(torch.tensor(1.), torch.tensor(0.5)))
