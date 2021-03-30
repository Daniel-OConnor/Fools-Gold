import torch
from simulator import RatioSimulator, ProbSimulator
from contextlib import nullcontext

class LotkaVolterra(RatioSimulator, ProbSimulator):
    x_size = 1
    theta_size = 1
# 59: A. J. Lotka, Analytical note on certain rhythmic relations in organic systems.
# 60: A. J. Lotka, Undamped oscillations derived from the law of mass action.
# 61: D. T. Gillespie, A general method for numerically simulating the stochastic time evolution of coupled chemical reactions.
# 33: G. Papamakarios, I. Murray, “Fast ε-free inference of simulation models with bayesian conditional density estimation” (appendix F)
# 63:  J. Brehmer, K. Cranmer, G. Louppe, J. Pavez, Code repository for the Lotka–Volterra example in the paper “Mining gold from implicit models to improve likelihoodfree inference.” GitHub. http://github.com/johannbrehmer/goldmine.
