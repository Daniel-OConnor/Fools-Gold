from abc import ABCMeta, abstractmethod
import numpy as np
#%matplotlib notebook
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
from tqdm import tqdm
from loss.scandal import gaussian_mixture_prob
from loss.scandal import categorical_prob

class Plot:
    # can be instantiated with/without any of the optional parameters
    # optional args don't have to be given in order if stated explicitly
    def __init__(self, xs=[], ys=[], colour:str="",
                title:str="", xLabel:str="", yLabel:str="",
                draw:bool=False):
        self.xs = xs; self.ys = ys; self.colour = colour
        self.title = title; self.xLabel = xLabel; self.yLabel = yLabel

        if (draw):
            self.plot()

    # most basic plot function; plots x and y (with optional colour) and shows
    def show(self, testing=False):
        # input checks
        #if (len(self.x)!=len(self.y)): raise ValueError("lengths of x and y don't match")
        if (len(self.xs)==0): raise ValueError("x (and possibly y) are empty!")

        plt.show()
        if (testing): print("graph checked and displayed by show()")
    
    # to-be-implemented by subclass
    @abstractmethod
    def plot(self):
        pass
    
    def assignLabels(self, testing=False):
        if (self.title != ""): plt.title(self.title)
        if (self.xLabel != ""): plt.xlabel(self.xLabel)
        if (self.yLabel != ""): plt.ylabel(self.yLabel)
        if (testing): print("labels attached to graph by assignLabels()")

# chain multiple plots (each a subclass of Plot) together
# plots them all on the same graph
# uses labels (title, xLabel, yLabel) from FIRST plot in list
class MultiPlot():

    def __init__(self, plotlist=[]):
        self.plotlist = plotlist

    # plots all individual plot classes without showing
    # so they appear on same graph
    def plot(self):
        for p in self.plotlist:
            p.plot()

    def show(self):
        plt.show()

class PlotLine(Plot):
    # subclass initialisation includes x/y-ranges for line graphs
    def __init__(self, xs=[], ys=[], colour:str="",
                title:str="", xLabel:str="", yLabel:str="",
                xRange:tuple=None, yRange:tuple=None,
                draw:bool=False):
        self.xRange = xRange; self.yRange = yRange
        super().__init__(xs,ys,colour,title,xLabel,yLabel,draw=draw)        

    # plots line graph given instance info
    # assumes title/labels and x/y-ranges should be used if given (but can be overridden)
    def plot(self, useRanges=True, useLabels=True):
        if (self.xRange != None and self.yRange != None and self.useRanges): plt.axis(list(self.xRange)+list(self.yRange))
        if (self.useLabels): self.assignLabels(True)
        if (self.colour != ""): plt.plot(self.xs,self.ys,color=self.colour)
        else: plt.plot(self.xs,self.ys)
        #self.show(True)

class PlotHist(Plot):

    def __init__(self, xs=[], numBins=10, colour:str="",
                title:str="", xLabel:str="", yLabel:str="",
                freqRange:tuple=None, draw:bool=False):
        self.numBins = numBins; self.freqRange = freqRange
        super().__init__(xs,None,colour,title,xLabel,yLabel,draw=draw)

    # plots histogram given instance info
    # assumes title/labels and x/y-ranges should be used if given (but can be overridden)
    def plot(self, useRanges=True, useLabels=True):
        if (self.freqRange != None and self.useRanges): plt.ylim(freqRange)
        if (self.useLabels): self.assignLabels(True)
        if (self.colour != ""): plt.plot(self.xs,bins=self.numBins,color=self.colour)
        else: plt.hist(self.xs,bins=self.numBins)
        #self.show(True)

# This runs a simulator n times for parameter theta
# It currently uses tqdm for a progress bar as it takes awhile to complete
# It estimates the true probability density (likelihood) function
class PlotTrueLikelihood(PlotLine):

    def __init__(self, sim, theta, start, end, steps, n,
                colour:str="r", title:str="", xLabel:str="", yLabel:str="",
                xRange:tuple=None, yRange:tuple=None, draw:bool=False):
        visual_runs = np.array([sim.simulate(theta)[1].cpu().detach().numpy() for _ in tqdm(range(n))])
        xs = np.linspace(start, end, steps)
        density_true = gaussian_kde(visual_runs)
        density_true.covariance_factor = lambda: .05
        density_true._compute_covariance()
        super().__init__(xs, density_true(xs), colour,title,xLabel,yLabel,xRange,yRange,draw)

# This runs a simulator n times for each of the two parameters theta0 and theta1
# It currently uses tqdm for a progress bar as it takes awhile to complete
# It estimates the probability densities, then divides them to estimate the probability ratio
class PlotTrueRatio(PlotLine):

    def __init__(self, sim, theta0, theta1, start, end, steps, n,
                colour:str="r", title:str="", xLabel:str="", yLabel:str="",
                xRange:tuple=None, yRange:tuple=None, draw:bool=False):
        visual_runs0 = np.array([sim.simulate(theta0)[1].cpu().detach().numpy() for _ in tqdm(range(n))])
        visual_runs1 = np.array([sim.simulate(theta1)[1].cpu().detach().numpy() for _ in tqdm(range(n))])
        xs = np.linspace(start, end, steps)
        density_true0 = gaussian_kde(visual_runs0)
        density_true0.covariance_factor = lambda: .05
        density_true0._compute_covariance()
        density_true1 = gaussian_kde(visual_runs1)
        density_true1.covariance_factor = lambda: .05
        density_true1._compute_covariance()
        super().__init__(xs, density_true0(xs)/density_true1(xs), colour,title,xLabel,yLabel,xRange,yRange,draw)

# This plots the probability density outputted by a density network "model" when given theta as input
class PlotDensityNetwork(PlotLine):

    def __init__(self, model, theta, start, end, steps,
                colour:str="b", title:str="", xLabel:str="", yLabel:str="",
                xRange:tuple=None, yRange:tuple=None, draw:bool=False):
        xs = np.linspace(start, end, steps)
        _, mean, sd, weight = model(torch.tensor([[0]], dtype=torch.float32), torch.tensor([[theta]], dtype=torch.float32))
        density_pred = [gaussian_mixture_prob(x, mean, sd, weight) for x in xs]
        super().__init__(xs, density_pred, colour,title,xLabel,yLabel,xRange,yRange,draw)

# This plots the probability ratio outputted by a classifier network "model" when given theta0, theta1 as input
class PlotClassifierNetwork(PlotLine):

    def __init__(self, model, theta0, theta1, start, end, steps,
                colour:str="b", title:str="", xLabel:str="", yLabel:str="",
                xRange:tuple=None, yRange:tuple=None, draw:bool=False):
        xs = np.linspace(start, end, steps)
        density_pred = [model(torch.tensor([[x]], dtype=torch.float32), torch.tensor([[theta0]], dtype=torch.float32),
                          torch.tensor([[theta1]], dtype=torch.float32)) for x in xs]
        density_pred = [(1 - x) / x for x in density_pred]
        super().__init__(xs, density_pred, colour,title,xLabel,yLabel,xRange,yRange,draw)

# This plots the probability ratio outputted by a ratio network "model" when given theta0, theta1 as input
class PlotClassifierNetwork(PlotLine):

    def __init__(self, model, theta0, theta1, start, end, steps,
                colour:str="b", title:str="", xLabel:str="", yLabel:str="",
                xRange:tuple=None, yRange:tuple=None, draw:bool=False):
        xs = np.linspace(start, end, steps)
        density_pred = [1 / model(torch.tensor([x], dtype=torch.float32), torch.tensor([[theta0]], dtype=torch.float32),
                              torch.tensor([[theta1]], dtype=torch.float32)) for x in xs]
        super().__init__(xs, density_pred, colour,title,xLabel,yLabel,xRange,yRange,draw)


# Categorical equivalent of PlotDensityNetwork
class PlotCategoricalNetwork(PlotHist):

    def __init__(self, model, theta, start, end, steps,
                colour:str="b", title:str="", xLabel:str="", yLabel:str="",
                xRange:tuple=None, yRange:tuple=None, draw:bool=False):
        xs = np.linspace(start, end, steps)
        _, probs = model(torch.tensor([[0]], dtype=torch.float32), torch.tensor([[theta]], dtype=torch.float32))
        density_pred = [categorical_prob(x, probs) for x in xs]
        super().__init__(density_pred, len(list(xs)), colour,title,xLabel,yLabel,xRange,yRange,draw)