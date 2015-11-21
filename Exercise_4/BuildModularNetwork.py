"""
Computational Neurodynamics
Exercise 2

(C) Murray Shanahan et al, 2015
"""

from IzNetwork import IzNetwork
import numpy as np
import numpy.random as rn

def BuildModularNetwork(M, NPerM, NInh, p):
    """
    Constructs two layers of Izhikevich neurons an connects them together.
    Layer 1 contains M modules of NPerMod number of excitatory neurons.
    Layer 2 is an inhibitory pool containing NInh neurons.

    Inhibitory neurons in layer 2 have diffuse connections to all other inhibitory and excitatory neurons
    in the network.

    p = rewiring probability

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

    net = IzNetwork([NExc, NInh], Dmax)
    
    net.layer[EXC].S[EXC] = np.zeros([NExc, NExc])
    net.layer[EXC].S[INH] = np.zeros([NExc, NInh])
    net.layer[INH].S[EXC] = np.zeros([NInh, NExc])
    net.layer[INH].S[INH] = np.zeros([NInh, NInh])

    net.layer[INH].factor[INH] = F_INH_INH  
    net.layer[INH].factor[EXC] = F_EXC_INH
    net.layer[EXC].factor[INH] = F_INH_EXC  
    net.layer[EXC].factor[EXC] = F_EXC_EXC

    # Set the conduction delays for connections
    net.layer[EXC].delay[EXC] = rn.randint(Dmin, Dmax, size=(NExc, NExc))
    net.layer[EXC].delay[INH] = Dmin * np.ones([NExc, NInh], dtype=int)
    net.layer[INH].delay[EXC] = Dmin * np.ones([NInh, NExc], dtype=int)
    net.layer[INH].delay[INH] = Dmin * np.ones([NInh, NInh], dtype=int)

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
    for m in range(M):
        c = 0
        while c < 1000:
            from_neuron = randomModuleIndex(m, NPerM)
            
            # Perform probabilistic inter module rewiring
            rewire = rn.random() < p
            if rewire:
                to_module = rn.randint(M)
                to_neuron = randomModuleIndex(to_module, NPerM)
            else:
                to_neuron = randomModuleIndex(m, NPerM)

            # Check not wiring neuron to itself.
            if rewire or from_neuron != to_neuron:
                net.layer[EXC].S[EXC][from_neuron, to_neuron] = 1
                c += 1    

    # EXC -> INH connections (connect 4 EXC neurons to one INH and one only)
    inh_set = range(NInh)
    # assert NPerM is divisible by exc/inh ratio?
    for m in range(M):
        module_set = moduleIndicies(m, NPerM)
        while len(module_set) > 0: 
            [inh_neuron, inh_set] = removeRandomNeuron(inh_set)
            for i in range(4):
                [exc_neuron, module_set] = removeRandomNeuron(module_set)
                net.layer[INH].S[EXC][inh_neuron, exc_neuron] = rn.random()
                

    # INH -> INH & INH -> EXC
    net.layer[INH].S[INH] = np.ones([NInh, NInh]) - np.identity(NInh)
    net.layer[EXC].S[INH] = np.ones([NExc, NInh])
    for i in range(NInh):
        for j in range(NInh):
            net.layer[INH].S[INH][i,j] = rn.random() - 1
        for k in range(NExc):
            net.layer[EXC].S[INH][k,i] = rn.random() - 1

    #print net.layer[EXC].S[INH]

    return net

def removeRandomNeuron(neuron_set):
    n = neuron_set.pop(rn.randint(len(neuron_set)))
    return [n, neuron_set]

def moduleIndicies(m, n_per_m):
    return range(m*n_per_m, (m*n_per_m + n_per_m))

def randomModuleIndex(m, n_per_m):
    """
    Returns random index in module range (M - M+NPerM)
    """
    return rn.choice(moduleIndicies(m, n_per_m), 1)
