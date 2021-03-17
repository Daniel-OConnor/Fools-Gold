import torch
from simulator import Simulator
import math


def gaussian(x, mean, sd):
    return 1/(abs(sd)*math.sqrt(2*math.pi)) * torch.exp(-0.5*(x-mean)*(x-mean)/(sd*sd))


class DummySimulator(Simulator):
    x_size = 1
    theta_size = 1

    def simulate(self, theta):
        r0 = torch.normal(theta, torch.tensor(1.))
        r1 = torch.normal(torch.tensor(0.), r0)
        return [r0, r1]

    def p(self, z, theta):
        p0 = gaussian(z[0], theta, 1)
        p1 = gaussian(z[1], 0, z[0])
        return [p0, p1]


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
