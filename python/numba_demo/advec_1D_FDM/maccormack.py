#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:36:17 2021

@author: sblair
"""

import numpy as np
import matplotlib.pyplot as plt
import time

Num_ts = 25000;
N = 50000
u = 1

plot_freq = 2500
plot_switch = 1

x_left = -10;
x_right = 10;

x_space = np.linspace(x_left,x_right,num=N,dtype=np.float64);

dx = x_space[1]-x_space[0];
dt = 0.6*dx/u;
nu = u*dx/dt;

f_l = 1.;
f = f_l*np.exp(-(x_space**2));
f[(x_space < -5) & (x_space > -7)] = 1 


# plot the initial condition
if plot_switch == 1:
    plt.figure()
    plt.plot(x_space,f)
    plt.title('Initial Condition');
    plt.grid()

t0 = time.time()
ind = np.arange(N)
x_m = np.roll(ind,1)
x_p = np.roll(ind,-1)

for ts in range(Num_ts):
    if(ts%100 == 0):
        print(f'{"Executing time step %d"}'%ts)
    
    f_tmp1 = f - (u*dt/dx)*(f[x_p]-f);
    f_tmp = 0.5*(f + f_tmp1 - (u*dt/dx)*(f_tmp1 - f_tmp1[x_m]));
    f = f_tmp;
    
    if(plot_switch == 1):
        if(ts%plot_freq == 0):
            plt.figure()
            plt.plot(x_space,f)
            plt.grid()

t1 = time.time()

ex_time = t1 - t0;

print(f'{"Execution time = %g. Average time per DOF update = %g"}'% (ex_time, ex_time/(N*Num_ts)))
