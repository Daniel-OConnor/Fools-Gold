from dummy_simulator import DummySimulator
from rolr import ROLR
from ratio import Ratio
import torch

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

model = Ratio(1, 1, 50)
model.to(device)

def prior():
    return torch.rand(1).to(device)


sim = DummySimulator()
ROLR(sim, model, prior, 500, 500000, 0.001)
