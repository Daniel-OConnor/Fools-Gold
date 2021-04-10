import torch
from .simulator import ProbSimulator
import math
import numpy as np
from random import random


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


class GaltonBoard(ProbSimulator):
    theta_size = 1
    x_size = 1

    def __init__(self, row_count, nail_count):
        self.row_count = row_count
        self.nail_count = nail_count

    # https://github.com/johannbrehmer/goldmine/blob/master/goldmine/simulators/galton.py
    def nail_direction_chance(self, theta, row, nail):
        row_rel = 1. * row / (self.row_count - 1)
        nail_rel = 2. * nail / (self.nail_count - 1) - 1.

        nail_coefficient = ((1. - np.sin(np.pi * row_rel)) * 0.5
                            + np.sin(np.pi * row_rel) * sigmoid(10 * theta * nail_rel))
        return nail_coefficient

    def simulate(self, θ):
        nail = self.nail_count / 2
        zs = []
        for row in range(self.row_count):
            nail_coefficient = self.nail_direction_chance(θ, row, nail)
            max_nail = self.nail_count - 1 if row%2 == 0 else self.nail_count-2
            if (random() < nail_coefficient and nail!=max_nail) or nail == 0:
                zs.extend((1, torch.tensor([nail])))
                if row % 2 != 0:
                    nail += 1
            else:
                zs.extend((0, torch.tensor([nail])))
                if row % 2 == 0:
                    nail -= 0
        # Normalize output
        zs.append(zs[-1]/self.nail_count)
        return zs

    def log_p(self, zs, θ):
        assert len(zs) == 2*self.nail_count + 1
        output = torch.tensor(0., requires_grad=True)
        for row in range(self.row_count):
            nail = zs[2*row+1]
            right = zs[2*row]
            if right == 1:
                output = output + torch.log(self.nail_direction_chance(θ, row, nail))
            else:
                output = output + torch.log(1 - self.nail_direction_chance(θ, row, nail))
        return output

