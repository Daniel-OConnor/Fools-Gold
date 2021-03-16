import torch
from torch import autograd
from abc import ABCMeta, abstractmethod

class Simulator:
    # returns (x, [p(z_i|theta, z_j, j<i) for each i])
    @abstractmethod
    def simulate(self, theta):
        pass

    def _calculate_T(self, theta, ps):
        total = torch.tensor(0., requires_grad = True)
        for p in ps:
            total = total + torch.log(p)
        total.backward()
        return theta.grad.detach()

    def sample_T(self, theta):
        theta.requires_grad=True
        x, ps = self.simulate(theta)
        t_score = self._calculate_T(theta, ps)
        return x.detach(), t_score

    def sample_R(self, theta0, theta1):
        with torch.no_grad():
            x0, ps0 = self.simulate(theta0)
            x1, ps1 = self.simulate(theta1)
            assert(len(ps0) == len(ps1))
            r_score = torch.tensor(1.)
            for i in range(len(ps0)):
                r_score = r_score * (ps0[i]/ps1[i])
            return x0, x1, r_score

    def sample_both(self, theta0, theta1):
        theta0.requires_grad=True
        theta1.requires_grad=True
        
        x0, ps0 = self.simulate(theta0)
        x1, ps1 = self.simulate(theta1)
        with torch.no_grad():
            assert(len(ps0) == len(ps1))
            r_score = torch.tensor(1.)
            for i in range(len(ps0)):
                r_score *= (ps0[i]/ps1[i])

        t_score0 = self._calculate_T(theta0, ps0)
        t_score1 = self._calculate_T(theta1, ps1)
        return x0.detach(), x1.detach(), t_score0, t_score1, r_score
