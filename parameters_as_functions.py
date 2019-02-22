# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:09:50 2019

@author: Acer
"""

def WP(charge,density,mass):#plasma frequency - rad/s
    return ((4 * np.pi * charge**2 * density) / mass)**0.5

def KP(frequency):#plasma wave no. - rad/m
    return 3e8 / frequency

def norm_E(mass,frequency,charge,):#E field normalisation constant
    return (mass * 3e8 * frequency) / charge

#SCALING PARAMETERS
me = 9.10938356e-31 #kg
qe = 1.60217662e-19 #C
pn = 1e24 #m^-3 plasma density

wp = WP(qe,pn,me)
kp = KP(wp)
E_norm = norm_E(me,wp,qe)