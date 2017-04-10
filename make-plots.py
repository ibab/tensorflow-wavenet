"""

for each shapekeys:
    make empty canvas
    for each entry in dict:
        add to plot
    
    save as png

"""

import sys
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt

path = "./"
csvfiles = [f for f in listdir(path) if isfile(join(path, f))]

traildata = {}

interesting_shapekeys = np.arange(77)

for file in csvfiles:
    traildata[file] = np.genfromtxt(file, delimiter=',')
    print traildata[file].shape

#for name,data in traildata.iteritems():
#    plt.imsave(name+"_shapekeys.png", np.kron(data[:,:], np.ones([1,100])))

time = np.arange(len(traildata[csvfiles[0]]))

for key in interesting_shapekeys:
    plt.clf()
    plt.title("shapekey %d" % key)
    plt.xlim([1000,1800])
    #plt.ylim([0,10])
    legend = []
    for name, data in traildata.iteritems():
        plt.plot(time, data[:,key])
        legend.append(name)
    #plt.legend(legend)
    plt.savefig("png/shapekey_%d.png" % key)


