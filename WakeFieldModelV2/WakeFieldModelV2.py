#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:46:29 2019

@author: fhabib
"""

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.integrate import odeint
#from math import sqrt, pi, log
import matplotlib.pyplot as plt

import datetime
now = datetime.datetime.now()
#from matplotlib.ticker import MultipleLocator

# Simple data to display in various forms
mpl.rcParams['font.family'] = 'Times New Roman' #'Arial'
mpl.rcParams['mathtext.fontset'] = 'custom'
#mpl.rcParams['font.sans-serif'] ='Times New Roman' #"Arial"
mpl.rcParams['mathtext.cal'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'#'Arial:italic'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['text.usetex'] = False
hfont = {'fontname':'Times New Roman'}

label_size = 20
fontsize = 20
mpl.rcParams['xtick.labelsize'] = label_size
#mpl.rcParams['labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['lines.linewidth'] = 2.0



def DriverBeam(x=None,r=None, mu=None, sig=None):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * np.exp(-np.power(r, 2.) / (2 * np.power(sig, 2.)))

# Simple Wakefield Model See W.Lu paper
def modelWLu(Rb,xi,mu,sig,nb,r):
    rb,drb =Rb
    #
    Lambda =nb*DriverBeam(xi,r*rb,mu, sig) # A driver with gaussian distribution

    drbdxi = [drb, (4*Lambda * rb**(-3) -  2 * rb * (drb)**2 -  rb**(-1)) ] # System of first order ODEs
    return drbdxi

Rb0    = 0.3 #initial conditions
dRb0   = 0 # 

xi_0   = 0 # initial conditions for RHS
xi_max = 3.60 #max intrgration limits
dxi = 0.01 # size of integration steps
steps = np.int(np.abs(xi_max/dxi)) # number of steps needed to achieve dxi 

initialvalue = [Rb0,dRb0]
#mu=0.6
sig=0.3
mu=0.9
nb=1.0
r=0.000001
solWLu =[[],[]]
xi=np.linspace(xi_0, xi_max, steps) #(START,END,NUMBER OF STEPS BETWEEN LIMITS)
solWLu = odeint(modelWLu, initialvalue,xi,args=(mu,sig,nb,r)); #(func,initial,t)


BlowOutRadius= solWLu[:,0]
#EzField= 0.5*BlowOutRadius*np.gradient(BlowOutRadius)
EzField= 0.5*BlowOutRadius*solWLu[:,1]
NormExField=EzField/EzField.max()

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

fig.set_size_inches(10.0, 10.0, forward=True)
fig.subplots_adjust(top=0.92, bottom=0.0, left=0.10, right=0.95, hspace=0.05,
                        wspace=0.1)
ax2b=ax2.twinx()
plot1ax1=ax1.plot(xi, BlowOutRadius, '--')
plot1ax1=ax1.plot(xi, -BlowOutRadius, '--')
#add_arrow(plot1ax1)
ax1.set_ylabel('r$_{\\rm b} (\\xi)$ (a.u.)',fontsize=fontsize , **hfont)
#ax1.set(title='Non-Linear Wakefield model',font=fontsize , **hfont)

ax2.plot(xi, NormExField, '-',color='red')
#ax2.plot(xi, EzField2, '--')
ax2b.plot(xi, DriverBeam(xi,0,mu, sig),'--',color='blue')

ax2.spines['left'].set_color('red')
ax2.yaxis.label.set_color('red')
ax2.tick_params(axis='y', colors='red')

ax2b.spines['right'].set_color('blue')
ax2b.yaxis.label.set_color('blue')
ax2b.tick_params(axis='y', colors='blue')

#ax2b.plot(xi, EzField2,'--')
ax2.set_xlabel('$\\xi$ (a.u.)',fontsize=fontsize , **hfont)
ax2.set_ylabel('E$_{\\rm z}/E_{\\rm z,max}$',fontsize=fontsize , **hfont)
ax2b.set_ylabel('$n_{\\rm d} (\\xi)/n_{d,max}$',fontsize=fontsize , **hfont)
ax2.set_xlim([0.1,3.5])
TodayDate=str(now.strftime("%d-%m-%Y"))
fig.savefig(TodayDate+'_Wakefield_FHv2.png',format = 'png', dpi=300,bbox_inches='tight')
plt.show()



