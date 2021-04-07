import torch
from torch import autograd
from abc import ABCMeta, abstractmethod

class Simulator:
    @abstractmethod
    def simulate(self, θ):
        """
        Perform a single run of the simulator
        returns List[torch.Tensor] zs, where zs[i] is the i-th latent, and
        zs[i] is not dependent on any latent variables zs[j] for j >= i
        """
        pass

    @property
    @abstractmethod
    def x_size(self):
        pass

    @property
    @abstractmethod
    def θ_size(self):
        pass

class RatioSimulator(Simulator):
    @abstractmethod
    def ratio(self, zs, θ_0, θ_1):
        """
        Calculate conditional probability ratios for a run of the simulator
        
        Arguments:
            zs:        torch.Tensor, latent variables
            θ_0:       torch.Tensor, parameters
            θ_1:       torch.Tensor, parameters
        Returns:
            rs: torch.Tensor, where rs[i] = p(z_i | θ_0, zs[:i]) / p(z_i | θ_1, zs[:i])
        """
        pass

    def eval_ratio(self, zs, θ_0, θ_1):
        """returns r(x, zs | θ_0, θ_1)"""
        return torch.prod(self.ratio(zs, θ_0, θ_1))

    def eval_score(self, zs, θ_0, θ_1):
        """
        (adapted from _calculate_T (Daniel))
        Returns:
            * score, t(x, zs | θ_0, θ_1)
            * ratio, r(x, zs | θ_0, θ_1)
        """

        ratio = torch.prod(self.__ratio(zs, θ_0, θ_1, True))
        log_ratio = torch.log(ratio)
        log_ratio.backward()
        return θ_0.grad.detach(), ratio

class ProbSimulator(RatioSimulator):
    @abstractmethod
    def p(self, zs, θ):
        """
        Calculate conditional probabilities for a run of the simulator
        
        Arguments:
            zs:        torch.Tensor, latent variables
            θ:         torch.Tensor, parameters
        Returns:
            ps: torch.Tensor, where ps[i] = p(z_i|θ, zs[:i])
        """
        pass

    def ratio(self, zs, θ_0, θ_1):
        return self.p(zs, θ_0) / self.p(zs, θ_1)

    def eval_score(self, zs, θ):
        """
        (adapted from _calculate_T (Daniel))
        Returns:
            * score, t(x, zs | θ)
            * joint, p(x, zs | θ)
        """
        p = torch.prod(self.p(zs, θ, True))
        log_p = torch.log(p)
        log_p.backward()
        return θ.grad.detach(), p