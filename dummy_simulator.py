import torch
from simulator import Simulator
import math

def guassian(x, mean, sd):
    return 1/(abs(sd)*math.sqrt(2*math.pi)) * torch.exp(-0.5*(x-mean)*(x-mean)/(sd*sd))

class DummySimulator(Simulator):
    def simulate(self, theta):
        r0 = torch.normal(theta, torch.tensor(1.))
        p0 = guassian(r0, theta, 1)
        r1 = torch.normal(torch.tensor(0.), r0)
        p1 = guassian(r1, 0, r0)
        return r1, [p0, p1]

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
