"""
Computational Neurodynamics
Exercise 1

"""

import numpy as np
import matplotlib.pyplot as plt

m = 1           # mass
c = 0.1         # dampind coefficient
k = 1           # spring constant

dt = 0.1     # Step size for exact solution

# Create time points
Tmin = 0
Tmax = 5

# arange returns evenly spaced values over interval
T = np.arange(Tmin, Tmax+dt, dt)
y = np.zeros(len(T))
dydx = np.zeros(len(T))



# Exact solution - calculates e^x for all elements
# y = np.exp(T)

# Approximated solution with small integration Step
y[0]    = 1 # Initial value
dydt[0] = 0 # Initial value

for t in xrange(1, len(T)):
  dydt2[t]   = -(c*(dydt[t-1] + dt*dydt[t-1]) + k*y[t-1])/m
  dydt[t]    = dydt[t-1] + dt*dydt2[t]

# Approximated solution with large integration Step
#y_large[0] = np.exp(Tmin) # Initial value
#for t in xrange(1, len(T_large)):
#  y_large[t] = y_large[t-1] + dt_large*y_large[t-1]   


plt.plot(T, y, 'r', label='Euler method $\delta$ t = 0.5')
plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc=0)
plt.show()

