# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:50:15 2019

@author: Acer
"""
import matplotlib as mpl
import numpy as np


xi_0   = 0 # initial conditions for RHS
xi_max = 19.46#plasma wavelength = cavity length = 60 micro meters
steps = 400 
xi = np.linspace(xi_0, xi_max, steps) #(START,END,NUMBER OF STEPS BETWEEN LIMITS)
r = np.linspace(0,0.5,steps) #same size as xi

wsigz = 0.25#normX(2.26e-6)
wsigr = 0.25#normX(0.93e-6)
wmu = 16#normX(40e-6)

def DensityProfile(Xi, R, rhobeammax, BeamSigmar, BeamSigmaXi, BeamPosition):
    
    density = rhobeammax * (np.exp(-(np.power((Xi - BeamPosition), 2.))
    /((2 * BeamSigmaXi**2)))) * (np.exp(-(np.power(R, 2.)) 
    / ((2 * BeamSigmar**2))))

    return density

a=DensityProfile(xi,r, 1, wsigr, wsigz, 5)
b=DensityProfile(xi,r, 1, wsigr, wsigz, 10)
c=DensityProfile(xi,r, 1, wsigr, wsigz, 15)
plt.plot(xi,a,xi,b,xi,c)