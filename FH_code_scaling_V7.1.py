# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:01:37 2019

@author: Acer
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

def WP(density): #plasma frequency from web
    return np.sqrt((density*qe**2)/(eps0*me))

def KP(frequency):#plasma wave no. - rad/m
    return 3e8 / frequency

def max_rho(q_driver, q_electron, r_sig, z_sig):  #max charge density of driver
    return (q_driver)/(np.power(2*np.pi,3/2)*q_electron*np.power(r_sig,2)*z_sig)

def max_E0(mass,frequency,charge):  #max field amplitude/E field normalisation constant
    return (mass * 3e8 * frequency) / charge

#CHARGE DENSITY DISTRIBUTION
def gaussian_rho(x=None,r=None, mu=None, sig_z=None, sig_r=None, rho0=None):
    return rho0 * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig_z, 2.))) * np.exp(-np.power(r, 2.) / (2 * np.power(sig_r, 2.)))

def normX(x):
    return x / kp

def denormX(x):
    return x * kp

def denormField(x):
    return x * E0

def norm_n(x):
    return x / pn

def denorm_n(x):
    return x * pn

#SCALING PARAMETERS
me = 9.10938356e-31 #kg
qe = 1.60217662e-19 #C
pn = 1e23 #plasma density typically between (1-3.3)e23 m^-3
eps0 = 8.854e-12 #C^2 / Nm^2 permittivity of free space
wp = WP(pn)
kp = KP(wp)

#INITIAL CONDITIONS
Rb0    = 0.3 #initial blowout conditions
dRb0   = 0 #initial blowout radius
xi_0   = 0 # initial conditions for RHS
xi_max = 4.1 #plasma wavelength = cavity length = 60 micro meters
dxi = 0.001 # size of integration steps
steps = np.int(np.abs(xi_max/dxi)) # number of steps needed to achieve dxi 
initialvalue = [Rb0,dRb0]
#NORMALIZED DRIVER PARAMETERS 
sig_z = normX(2e-6) #7e-6 
sig_r = normX(5e-6) #5e-6
mu = normX(10e-6) #where the peak is found on comoving axis
r = 0.000001 #radius 
qd = 300e-12 #charge of driver 300 pico C
 

#SCALED DRIVER PARAMETERS
E0 = max_E0(me,wp,qe)
SIG_Z = denormX(sig_z)
SIG_R = denormX(sig_r)
MU = denormX(mu)
MDD = max_rho(qd,qe,SIG_R,SIG_Z)
mdd = norm_n(MDD) #max charge density of driver
#lam=mdd*sig_r**2

#BUNCH PARAMETERS
bsigz=normX(2.26e-6)
bsigr= normX(0.93e-6)
bmu = normX(40e-6)
BSIG_Z = denormX(bsigz)
BSIG_R = denormX(bsigr)
BMU = denormX(bmu)

br = 0.0000001
qb = 2e-12 #charge of bunch
MDB = max_rho(qb,qe,BSIG_R,BSIG_Z) #max charge density of bunch
mdb = norm_n(MDB)
solWLu =[[],[]]
xi=np.linspace(xi_0, xi_max, steps) #(START,END,NUMBER OF STEPS BETWEEN LIMITS)
XI=denormX(xi)

# Simple Wakefield Model See W.Lu paper
def modelWLu(Rb,xi,mu,sig_z,sig_r,mdd,r,br,bmu,bsigz,bsigr,mdb):
    rb,drb =Rb
    Lambda = gaussian_rho(xi,r*rb,mu, sig_z,sig_r,mdd) + gaussian_rho(xi,br*rb,bmu,bsigz,bsigr,mdb) # A driver with gaussian distribution
    drbdxi = [drb, (4*Lambda * rb**(-3) -  2 * rb**(-1) * (drb)**2 -  rb**(-1)) ] # System of first order ODEs
    return drbdxi

solWLu = odeint(modelWLu, initialvalue,xi,args=(mu,sig_z,sig_r,mdd,r,br,bmu,bsigz,bsigr,mdb)); #(func,initial,t)

NormBlowOutRadius = solWLu[:,0] 
BlowOutRadius = denormX(NormBlowOutRadius)
 
#EzField= 0.5*BlowOutRadius*np.gradient(BlowOutRadius)
NormEzField= 0.5*BlowOutRadius*solWLu[:,1]
EzField =denormField(NormEzField)
#NormExField=EzField/EzField.max()

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.set_size_inches(10.0, 10.0, forward=True)
fig.subplots_adjust(top=0.92, bottom=0.0, left=0.10, right=0.95, hspace=0.05,
                        wspace=0.1)
ax2b=ax2.twinx()
plot1ax1=ax1.plot(XI, BlowOutRadius, '--')
plot1ax1=ax1.plot(XI, -BlowOutRadius, '--')
#add_arrow(plot1ax1)
ax1.set_ylabel('r$_{\\rm b} (\\xi)$ ',fontsize=fontsize , **hfont)
#ax1.set(title='Non-Linear Wakefield model',font=fontsize , **hfont)
ax2.plot(XI, EzField, '-',color='red')
#ax2.plot(xi, EzField2, '--')
ax2b.plot(XI, gaussian_rho(XI,r*kp,MU,SIG_Z,SIG_R,MDD),'--',color='blue')
ax2b.plot(XI, gaussian_rho(XI,br*kp,BMU,BSIG_Z,BSIG_R,MDB) ,color='green')
ax2.spines['left'].set_color('red')
ax2.yaxis.label.set_color('red')
ax2.tick_params(axis='y', colors='red')
ax2b.spines['right'].set_color('blue')
ax2b.yaxis.label.set_color('blue')
ax2b.tick_params(axis='y', colors='blue')

#ax2b.plot(xi, EzField2,'--')
ax2.set_xlabel('$\\xi$',fontsize=fontsize , **hfont)
ax2.set_ylabel('E$_{\\rm z}$',fontsize=fontsize , **hfont)
ax2b.set_ylabel('$n_{\\rm d} (\\xi)$',fontsize=fontsize , **hfont)
ax2.set_xlim([0.1*kp,xi_max*kp])
TodayDate=str(now.strftime("%d-%m-%Y"))
fig.savefig(TodayDate+'_Wakefield_FHv2.png',format = 'png', dpi=300,bbox_inches='tight')
plt.show()