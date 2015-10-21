"""
Computational Neurodynamics
Exercise 1

"""

import numpy as np
import matplotlib.pyplot as plt

m = 1           # mass
c = 0.1         # dampind coefficient
k = 1           # spring constant

dt = 0.5     # Step size for exact solution

# Create time points
Tmin = 0
Tmax = 100 

# arange returns evenly spaced values over interval
T = np.arange(Tmin, Tmax+dt, dt)
y = np.zeros(len(T))
dydt = np.zeros(len(T))
dydt2 = np.zeros(len(T))

# Approximated solution with small integration Step
y[0]        = 1 # Initial value
dydt[0]     = 0 # Initial value 

for t in xrange(1, len(T)):
  dydt2[t]   = -(c*(dydt[t-1]) + k*y[t-1])/m
  dydt[t]    = dydt[t-1] + dt*dydt2[t]
  y[t]       = y[t-1] + dt*dydt[t]

plt.plot(T, y, 'r', label='Euler method $\delta$ t = 0.5')
plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc=0)
plt.show()

