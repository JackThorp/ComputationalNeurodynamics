"""
Computational Neurodynamics

Coursework
Simulates two layers of Izhikevich neurons. Layer 0 consists of 8 modules
of 100 neurons stimulated with a constant base current ? ? ? and layer 1
consists of 200 inhibitory neurons. Connections are described in the code

"""

from BuildModularNetwork import BuildModularNetwork
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
from matplotlib.colors import *
 
T   = 1000  # Simulation time
EXC = 0     # Excitatory layer index
INH = 1     # Inhibitory layer index

Modules = 8
NPerM = 100
NExc = Modules*NPerM
NInh = 200

def main():
   
    net = BuildModularNetwork(Modules, NPerM, NInh, 0.3)

    ## Initialise layers
    for lr in xrange(len(net.layer)):
      net.layer[lr].v = -65 * np.ones(net.layer[lr].N)
      net.layer[lr].u = net.layer[lr].b * net.layer[lr].v
      net.layer[lr].firings = np.array([])

    v1 = np.zeros([T, NExc])
    v2 = np.zeros([T, NInh])

    ## SIMULATE
    for t in xrange(T):
        net.layer[INH].I = np.zeros(NInh)
        net.layer[EXC].I = np.zeros(NExc)
        # Probabilisitcally add 15mA to membrane potential according to poisson distribution
        for n in range(NExc):
            net.layer[EXC].I[n] = 15 if np.random.poisson(0.01) > 0 else 0 
        
        net.Update(t)

        v1[t] = net.layer[EXC].v
        v2[t] = net.layer[INH].v

    ## Retrieve firings for presentation
    firingsExc = net.layer[EXC].firings
    firingsInh = net.layer[INH].firings

    if firingsExc.size != 0:
      v1[firingsExc[:, 0], firingsExc[:, 1]] = 30

    if firingsInh.size != 0:
      v2[firingsInh[:, 0], firingsInh[:, 1]] = 30

    plotFirings(firingsExc)
    plt.show()

def plotNetworkConnections(net):
    """
    Plots connectivity matrices for EXC->EXC and EXC-> INH
    """
    plt.figure(1)
    plt.subplot(11)
    plt.matshow(net.layer[EXC].S[EXC], fignum=1, cmap=plt.get_cmap("cool"))
    plt.subplot(22)
    plt.matshow(net.layer[INH].S[EXC], fignum=2, cmap=plt.get_cmap("cool"))
    #plt.figure(3)
    #plt.matshow(net.layer[0].S[1], fignum=3, cmap=plt.get_cmap("cool"))
    #plt.figure(4)
    #plt.matshow(net.layer[1].S[1], fignum=4, cmap=plt.get_cmap("cool"))

def plotMembranePotentials():
    plt.figure(2)
    plt.subplot(211)
    plt.plot(range(T), v1)
    plt.title('Population 1 membrane potentials')
    plt.ylabel('Voltage (mV)')
    plt.ylim([-90, 40])

    plt.subplot(212)
    plt.plot(range(T), v2)
    plt.title('Population 2 membrane potentials')
    plt.ylabel('Voltage (mV)')
    plt.ylim([-90, 40])
    plt.xlabel('Time (ms)')

def plotInhibitoryFirings(firingsInh):
    if firingsInh.size != 0:
        plt.figure(3)
        plt.scatter(firingsInh[:, 0], firingsInh[:, 1] + 1, marker='.')
        plt.xlim(0, T)
        plt.ylabel('Neuron number')
        plt.ylim(0, NInh+1)
        plt.xlabel('Time (ms)')
        plt.title('Population Inh firings')

def plotFirings(firingsExc):
    """
    Plots a rasta plot of neuron firigns as well as
    average firing rates for individual modules.
    """
    if firingsExc.size != 0:
        plt.figure(7)
        plt.subplot(211)
        plt.scatter(firingsExc[:, 0], firingsExc[:, 1] + 1, marker='.')
        plt.xlim(0, T)
        plt.ylabel('Neuron number')
        plt.ylim(0, NExc+1)
        plt.title('Excitatory neuron firings')

    plt.subplot(212)
    mean_firing = {}
    for point in firingsExc:
        [t, idx] = point
        if t in mean_firing:
            rest = mean_firing[t]
            rest.append(idx)
            mean_firing[t] = rest
        else:
            mean_firing[t] = [idx]
                                              
    out = np.zeros([8, 50])
    for j in range(0, T, 20):
        for k in range(50):
            if j+k in mean_firing:
                idxs = mean_firing[j+k]
                for idx in idxs:
                    module = idx // 100
                    out[module][j//20] += 1 

    out2 = out / 50
    for color in out:
        t = []
        t2 = []
        for i, point in enumerate(color):
            t.append(i * 20)
            t2.append(float(point) / 50)
        plt.plot(t, t2)

if __name__ == '__main__':
    main()
