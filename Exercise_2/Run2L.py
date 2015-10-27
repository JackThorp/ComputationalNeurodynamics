"""
Computational Neurodynamics
Exercise 2

Simulates two layers of Izhikevich neurons. Layer 0 is stimulated
with a constant base current and layer 1 receives synaptic input
of layer 0.

(C) Murray Shanahan et al, 2015
"""

from Connect2L import Connect2L
import numpy as np
import matplotlib.pyplot as plt

N1 = 4
N2 = 4
T  = 500  # Simulation time
Ib = 5    # Base current

net = Connect2L(N1, N2)

## Initialise layers
for lr in xrange(len(net.layer)):
  net.layer[lr].v = -65 * np.ones(net.layer[lr].N)
  net.layer[lr].u = net.layer[lr].b * net.layer[lr].v
  net.layer[lr].firings = np.array([])

v1 = np.zeros([T, N1])
v2 = np.zeros([T, N2])
u1 = np.zeros([T, N1])
u2 = np.zeros([T, N2])

## SIMULATE
for t in xrange(T):

   # Deliver a constant base current to layer 1
   net.layer[0].I = Ib * np.ones(N1)
   net.layer[1].I = np.zeros(N2)

   net.Update(t)

   v1[t] = net.layer[0].v
   v2[t] = net.layer[1].v
   u1[t] = net.layer[0].u
   u2[t] = net.layer[1].u

## Retrieve firings and add Dirac pulses for presentation
firings1 = net.layer[0].firings
firings2 = net.layer[1].firings

if firings1.size != 0:
  v1[firings1[:, 0], firings1[:, 1]] = 30

if firings2.size != 0:
  v2[firings2[:, 0], firings2[:, 1]] = 30


## Plot membrane potentials
plt.figure(1)
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

## Plot recovery variable
plt.figure(2)
plt.subplot(211)
plt.plot(range(T), u1)
plt.title('Population 1 recovery variables')
plt.ylabel('Voltage (mV)')

plt.subplot(212)
plt.plot(range(T), u2)
plt.title('Population 2 recovery variables')
plt.ylabel('Voltage (mV)')
plt.xlabel('Time (ms)')

## Raster plots of firings
plt.figure(3)
plt.subplot(211)
plt.scatter(firings1[:, 0], firings1[:, 1] + 1, marker='.')
plt.xlim(0, T)
plt.ylabel('Neuron number')
plt.ylim(0, N1+1)
plt.title('Population 1 firings')

plt.subplot(212)
plt.scatter(firings2[:, 0], firings2[:, 1] + 1, marker='.')
plt.xlim(0, T)
plt.ylabel('Neuron number')
plt.ylim(0, N2+1)
plt.xlabel('Time (ms)')
plt.title('Population 2 firings')

plt.show()

