import torch
from torch import autograd
from abc import ABCMeta, abstractmethod


class Simulator:
    # returns [z_i for each i]
    @abstractmethod
    def simulate(self, theta):
        pass

    # Takes the list of latents, and theta
    # returns [p(z_i|theta, z_j, j<i) for each i]
    @abstractmethod
    def p(self, zs, theta):
        pass

    @property
    @abstractmethod
    def x_size(self):
        pass

    @property
    @abstractmethod
    def theta_size(self):
        pass

    def _calculate_T(self, ps, theta):
        total = torch.tensor(0., requires_grad=True)
        for p in ps:
            total = total + torch.log(p)
        total.backward()
        return theta.grad.detach()

    def sample_T(self, theta):
        theta = theta.detach()
        theta.requires_grad = True
        z = self.simulate(theta)
        t_score = self._calculate_T(self.p(z, theta), theta)
        return z[-1].detach(), t_score

    def sample_R(self, theta0, theta1):
        with torch.no_grad():
            z = self.simulate(theta1)
            ps0 = self.p(z, theta0)
            ps1 = self.p(z, theta1)
            assert(len(ps0) == len(ps1))
            r_score = torch.tensor(1.)
            for i in range(len(ps0)):
                r_score = r_score * (ps0[i]/ps1[i])
            return z[-1], r_score

    def sample_both(self, theta0, theta1):
        theta0 = theta0.detach()
        theta1 = theta1.detach()
        theta0.requires_grad=True
        theta1.requires_grad=True
        
        z = self.simulate(theta1)
        ps0 = self.p(z, theta0)
        ps1 = self.p(z, theta1)

        assert(len(ps0) == len(ps1))
        r_score = torch.tensor(1.)
        for i in range(len(ps0)):
            r_score *= (ps0[i]/ps1[i])

        t_score = self._calculate_T(ps1, theta1)
        return z[-1].detach(), t_score, r_score
