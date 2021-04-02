from abc import ABCMeta, abstractmethod
import numpy as n2p
#%matplotlib notebook
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib import gridspec

class Plot:
    # can be instantiated with/without any of the optional parameters
    # optional args don't have to be given in order if stated explicitly
    def __init__(self, x:list=[], y:list=[], colour:str="",
                title:str="", xLabel:str="", yLabel:str="",
                draw:bool=False):
        self.x = x; self.y = y; self.colour = colour
        self.title = title; self.xLabel = xLabel; self.yLabel = yLabel

        if (draw):
            self.plot()

    # most basic plot function; plots x and y (with optional colour) and shows
    def plotShow(self, testing=False):
        # input checks
        #if (len(self.x)!=len(self.y)): raise ValueError("lengths of x and y don't match")
        if (len(self.x)==0): raise ValueError("x (and possibly y) are empty!")

        plt.show()
        if (testing): print("graph checked and displayed by plotShow()")
    
    # to-be-implemented by subclass; full plotting function that calls plotShow at some point
    @abstractmethod
    def plot(self):
        pass
    
    def assignLabels(self, testing=False):
        if (self.title != ""): plt.title(self.title)
        if (self.xLabel != ""): plt.xlabel(self.xLabel)
        if (self.yLabel != ""): plt.ylabel(self.yLabel)
        if (testing): print("labels attached to graph by assignLabels()")



class PlotLine(Plot):
    # subclass initialisation includes x/y-ranges for line graphs
    def __init__(self, x:list=[], y:list=[], colour:str="",
                title:str="", xLabel:str="", yLabel:str="",
                xRange:tuple=None, yRange:tuple=None,
                draw:bool=False):
        self.xRange = xRange; self.yRange = yRange
        super().__init__(x,y,colour,title,xLabel,yLabel,draw=draw)        

    # plots line graph given instance info
    # assumes title/labels and x/y-ranges should be used if given (but can be overridden)
    def plot(self, useRanges=True, useLabels=True):
        if (self.xRange != None and self.yRange != None and self.useRanges): plt.axis(list(self.xRange)+list(self.yRange))
        if (self.useLabels): self.assignLabels(True)
        if (self.colour != ""): plt.plot(self.x,self.y,color=self.colour)
        else: plt.plot(self.x,self.y)
        self.plotShow(True)

class PlotHist(Plot):

    def __init__(self, x:list=[], y:list=[], colour:str="",
                title:str="", xLabel:str="", yLabel:str="",
                numBins:int=10, freqRange:tuple=None,
                draw:bool=False):
        self.numBins = numBins; self.freqRange = freqRange
        super().__init__(x,y,colour,title,xLabel,yLabel,draw=draw)

    # plots histogram given instance info
    # assumes title/labels and x/y-ranges should be used if given (but can be overridden)
    # can also convert x-data to frequencies (YET TO BE IMPLEMENTED)
    def plot(self, useRanges=True, useLabels=True):
        if (self.freqRange != None and self.useRanges): plt.ylim(freqRange)
        if (self.useLabels): self.assignLabels(True)
        if (self.colour != ""): plt.plot(self.x,bins=self.numBins,color=self.colour)
        else: plt.plot(self.x,bins=self.numBins)
        self.plotShow(True)


