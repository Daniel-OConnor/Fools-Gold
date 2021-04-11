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
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def x_size(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def θ_size(self):
        raise NotImplementedError()


class RatioSimulator(Simulator):
    @abstractmethod
    def log_ratio(self, zs, θ_0, θ_1):
        """
        Calculate conditional probability ratios for a run of the simulator
        
        Arguments:
            zs:        torch.Tensor, latent variables
            θ_0:       torch.Tensor, parameters
            θ_1:       torch.Tensor, parameters
        Returns:
            log_r:     torch.Tensor (0 dim), equal to log(rs.prod()),
                   where rs[i] = p(z_i | θ_0, zs[:i]) / p(z_i | θ_1, zs[:i])
        """
        raise NotImplementedError()

    def eval_ratio(self, zs, θ_0, θ_1):
        """returns r(x, zs | θ_0, θ_1)"""
        ratio = self.log_ratio(zs, θ_0, θ_1)
        out = ratio.exp()
        return out

    def eval_score(self, zs, θ_0, θ_1):
        """
        (adapted from _calculate_T (Daniel))
        Returns:
            * score, t(x, zs | θ_0, θ_1)
            * ratio, r(x, zs | θ_0, θ_1)
        """
        log_ratio = self.log_ratio(zs, θ_0, θ_1)
        g = torch.autograd.grad(log_ratio, θ_0)[0].detach()
        return g

class ProbSimulator(RatioSimulator):
    @abstractmethod
    def log_p(self, zs, θ):
        """
        Calculate conditional probabilities for a run of the simulator
        
        Arguments:
            zs:        torch.Tensor, latent variables
            θ:         torch.Tensor, parameters
        Returns:
            log_p:     torch.Tensor (0 dim), equal to log(ps.prod()),
                   where ps[i] = p(z_i | θ, zs[:i])
        """
        raise NotImplementedError()

    def log_ratio(self, zs, θ_0, θ_1):
        p1 = self.log_p(zs, θ_0)
        p2 = self.log_p(zs, θ_1)
        return (p1 - p2).sum()

    def eval_score(self, zs, θ):
        """
        (adapted from _calculate_T (Daniel))
        Returns:
            * score, t(x, zs | θ)
            * joint, p(x, zs | θ)
        """
        log_p = self.log_p(zs, θ)
        g = torch.autograd.grad(log_p, θ, allow_unused=True)[0].detach()
        return g
