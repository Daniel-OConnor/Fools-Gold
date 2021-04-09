import torch
from .simulator import RatioSimulator, ProbSimulator
from loss.scandal import gaussian

# incorrect interface!!!
class DummySimulator(ProbSimulator):
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
        p0 = gaussian(zs[0], θ, torch.tensor(1))
        p1 = gaussian(zs[1], 0, zs[[0]])
        return torch.cat([p0, p1])
