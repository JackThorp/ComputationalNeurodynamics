"""
Computational Neurodynamics
Exercise 2

(C) Murray Shanahan et al, 2015
"""

from IzNetwork import IzNetwork
import numpy as np
import numpy.random as rn

def BuildModularNetwork(M, NPerM, NInh):
    """
    Constructs two layers of Izhikevich neurons an connects them together.
    Layer 1 contains M modules of NPerMod number of excitatory neurons.
    Layer 2 is an inhibitory pool containing NInh neurons.

    Inhibitory neurons in layer 2 have diffuse connections to all other inhibitory and excitatory neurons
    in the network.

    """

    EXC = 0
    INH = 1

    # Number of excitatory neurons.
    NExc = M*NPerM
    
    # Scaling factors for different neuron connection types. 
    F_EXC_EXC = 17
    F_EXC_INH = 50
    F_INH_EXC = 2 
    F_INH_INH = 1

    # Default conduction delay is 1ms for most connection types.
    # Delay is random between 1ms and 20ms for EXC->EXC connections.
    Dmin = 1    
    Dmax = 20

    net = IzNetwork([NExc, NInh], Dmin)
    
    net.layer[INH].factor[INH] = F_INH_INH  
    net.layer[INH].factor[EXC] = F_EXC_INH
    net.layer[EXC].factor[INH] = F_INH_EXC  
    net.layer[EXC].factor[EXC] = F_EXC_EXC

    # Excitatory layer (regular spiking)    
    r = rn.rand(NExc)
    net.layer[EXC].a = 0.02 * np.ones(NExc)
    net.layer[EXC].b = 0.20 * np.ones(NExc)
    net.layer[EXC].c = -65 + 15*(r**2)
    net.layer[EXC].d = 8 - 6*(r**2)

    # Inhibitory pool (regular spiking)
    r = rn.rand(NInh)
    net.layer[INH].a = 0.02 * np.ones(NInh)
    net.layer[INH].b = 0.20 * np.ones(NInh)
    net.layer[INH].c = -65 + 15*(r**2)
    net.layer[INH].d = 8 - 6*(r**2)

    ## Connectivity matrix (synaptic weights)
    # layer[i].S[j] is the connectivity matrix from layer j to layer i
    # S(i,j) is the strength of the connection from neuron j to neuron i

    # EXC -> EXC connections
    net.layer[EXC].S[EXC] = np.zeros([NExc, NExc])
    for m in range(M):
        c = 0
        while c < 1000:
            i = randomModuleIndex(m, NPerM)
            j = randomModuleIndex(m, NPerM)
            if i != j:
                #print("({}, {})".format(i,j))
                net.layer[EXC].S[EXC][i, j] = 1
                c += 1    

    # EXC -> INH connections
    #net.layer[1].delay[0]  = D * np.ones([N1, N0], dtype=int)

    return net

def randomModuleIndex(M, NPerM):
    """
    Returns random index in module range (M - M+NPerM)
    """
    return M*NPerM + int(rn.rand()*NPerM)
