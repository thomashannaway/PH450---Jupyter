# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:32:15 2019

@author: Acer
"""
import numpy as np
#SCALING PARAMETERS
me = 9.10938356e-31 #kg
qe = 1.60217662e-19 #C
pn = 1e24 #m^-3 plasma density
class Parameter:
    def __init__(self,q,n,m): #charge, density, mass plasma freq
        self.q = q
        self.n = n
        self.m = m
        self.w = (4 * np.pi * q**2 * n / m)**0.5
        self.k = 300000000 / self.w
    
    def frequency(self):
        print (self.w)
        
    def waveno(self):
        print (self.k)
    
plas_freq = Parameter(qe,pn,me)
plas_freq.frequency()

kp = Parameter(qe,pn,me)
kp.waveno()

