#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 08:53:18 2021

@author: sblair
"""
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt



L_hx = 30; # cm, length of the heat exchanger
nX = 100; # number of points in the x-direction
n_period = 4;
A_lam_ratio = 0.3; # ratio between amplitude and wavelength

def get_B(A):
    return A_lam_ratio*(2*np.pi)/A;

def wave_form_p(x,A):
    return (A/2)*np.sin(get_B(A)*x);

def d_wave_form_p(x,A):
    return get_B(A)*(A/2)*np.cos(get_B(A)*x);

def get_X_max(A):
    return n_period*2.*np.pi/get_B(A);

def chord_length_error(A):
    result = integrate.quad(lambda x: np.sqrt(1.+d_wave_form_p(x,A))**2,
                            0,get_X_max(A));
    chord_length = result[0];
    return chord_length - L_hx;


A = fsolve(chord_length_error,0.1);

print(f'{"Amplitude = %g cm"}'%A);

def wave_form(x):
    return wave_form_p(x,A);


# def d_wave_form(x):
#     return d_wave_form_p(x,A);

# def phi(x):
#     return np.arctan(d_wave_form(x));

# offset = 0.5;

# def offset_x(x):
#     return offset*(-np.sin(phi(x)));

# def offset_y(x):
#     return offset*(np.cos(phi(x)));

print(f'{"A: %12.8f"}'%A);
print(f'{"B: %12.8f"}'%get_B(A));
print(f'{"x_max: %12.8f "}'%get_X_max(A));

xMin = 0;
xMax = get_X_max(A);

X = np.linspace(xMin,xMax,nX);



fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(X,wave_form(X))
plt.grid()
ax.set_aspect('equal',adjustable='box');
plt.show()