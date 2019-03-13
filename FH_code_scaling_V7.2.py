# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:01:37 2019

@author: Acer
"""
import matplotlib as mpl
import numpy as np
from scipy.integrate import odeint
from scipy.constants import m_e,e,c,epsilon_0
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

#FUNCTIONS
def WP(density): #plasma frequency
    return np.sqrt((density*e**2)/(epsilon_0*m_e))

def KP(frequency):#plasma wave no. - rad/m
    return frequency/c

def max_rho(q_driver, r_sig, z_sig): #max charge density of driver
    return (q_driver)/(np.power(2*np.pi,3/2)*e*np.power(r_sig,2)*z_sig)

def max_E0(frequency): #max field amplitude/E field normalisation constant
    return (m_e * c * frequency) / e

def gaussian_rho(x=None,r=None, mu=None, sig_z=None, sig_r=None, rho0=None): #Gaussian charge density distribution
    return rho0 * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig_z, 2.))) * np.exp(-np.power(r, 2.) / (2 * np.power(sig_r, 2.)))

def normX(x): #normalised to kp^-1
    return x * kp

def denormX(x):
    return x / kp

def denormField(x):
    return x * E0

def norm_n(x):
    return x / pn

def denorm_n(x):
    return x * pn

#SCALING PARAMETERS
pn = 5e22 #plasma density typically between (1-3.3)e23 m^-3
wp = WP(pn)
kp = KP(wp)

#INITIAL CONDITIONS
Rb0    = 10 #initial blowout conditions
dRb0   = 0 #initial blowout radius
xi_0   = 0 # initial conditions for RHS
xi_max = 40#plasma wavelength = cavity length = 60 micro meters
dxi = 0.001 # size of integration steps
steps = np.int(np.abs(xi_max/dxi)) # number of steps needed to achieve dxi 
initialvalue = [Rb0,dRb0]
#NORMALIZED DRIVER PARAMETERS (From WLu Paper)
sig_z = normX(30e-6) 
sig_r = normX(10e-6) 
mu = 7#normX(50e-6) #BeamPosition on comoving axis
qd = 2.89e-9 # N = 1.8e10 * e to get charge

#SCALED DRIVER PARAMETERS
E0 = max_E0(wp)
SIG_Z = denormX(sig_z)
SIG_R = denormX(sig_r)
MU = denormX(mu)

solWLu =[[],[]]
xi=np.linspace(xi_0, xi_max, steps) #(START,END,NUMBER OF STEPS BETWEEN LIMITS)
R=np.linspace(0,0.01,steps) #same size as xi
XI=denormX(xi)

#ISSUE 0 - Changing R changes density profile

def rhomax(BeamCharge,BeamSigmar,BeamSigmaXi):
    return (BeamCharge)/(e*(2*np.pi)**(3/2) * BeamSigmar**2 * BeamSigmaXi)

#ISSUE 1 - Scaling not equivalent - separated from ElectronBeamDensity /
#          to investigate. Previous version used scaled args for max and /
#          then normalised for use in solver, For plotting scaled max was used.
RhoMax_ScaledArgs = rhomax(qd,SIG_R,SIG_Z) 
NormRhoMax = norm_n(RhoMax_ScaledArgs) #Max density normalised 
RhoMax_NormArgs = rhomax(qd,sig_r,sig_z) 
denormRhoMax = denorm_n(RhoMax_NormArgs) #Scaled Max Density

def ElectronBeamDensity(Xi=None, R=None, rhobeammax=None,
                        BeamSigmar=None, BeamSigmaXi=None, BeamPosition=None):

    density=rhobeammax*np.exp(-np.power(Xi- BeamPosition, 2.)
    /(2 * np.power(BeamSigmaXi, 2.))) * np.exp(-np.power(R, 2.) 
    / (2 * np.power(BeamSigmar, 2.)))

    return density 


print("This is max rho before electron beam density function: ",RhoMax_ScaledArgs)
check = ElectronBeamDensity(xi,R,RhoMax_ScaledArgs,SIG_R,SIG_Z,MU)
print("This is max rho after electron beam density function: ", max(check))

def Lambda(xi=None, r=None, rhobeammax=None, BeamCharge=None,
           BeamSigmar=None, BeamSigmaXi=None, BeamPosition=None):

    xi, R = np.meshgrid(xi,r)
    ElectronDensity = ElectronBeamDensity(xi,R, rhobeammax,
                      BeamSigmar, BeamSigmaXi, BeamPosition)
                                     
    Lambda=np.trapz(ElectronDensity, axis=0)
    return Lambda

#ISSUE 2 - when lambda is defined outside model, lambda is given to modelWLu/
    #as an arg. When odeint is called, lambda is given in args but odeint /
    # expects a tuple. I put lambda inside model function seems to work.


# Simple Wakefield Model from W.Lu paper
def modelWLu(Rb,xi,R,rhomax, qd,sig_r,sig_z,mu):
    rb,drb =Rb
    lam = Lambda(xi,R,rhomax, qd,sig_r,sig_z,mu)
    drbdxi = [drb, (4*lam*rb**(-3)) - (2*(drb**2)*rb**(-1)) - rb**(-1)]
    # System of first order ODEs
    return drbdxi

solWLu = odeint(modelWLu, initialvalue,xi,args=(R,NormRhoMax,qd,sig_r,sig_z,mu)) #(func,initial,t)

NormBlowOutRadius = solWLu[:,0] 
BlowOutRadius = denormX(NormBlowOutRadius)
 
#EzField= 0.5*BlowOutRadius*np.gradient(BlowOutRadius)
NormEzField= 0.5*BlowOutRadius*solWLu[:,1]
EzField =denormField(NormEzField)
#NormExField=EzField/EzField.max()
XI = denormX(xi)

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
ax2b.plot(XI, ElectronBeamDensity(XI,R/kp,RhoMax_ScaledArgs,SIG_R,SIG_Z,MU),'--',color='blue')
#ax2b.plot(XI, gaussian_rho(XI,br*kp,BMU,BSIG_Z,BSIG_R,MDB) ,color='green')
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
#ax2.set_xlim([0.1*kp,xi_max*kp])
TodayDate=str(now.strftime("%d-%m-%Y"))
fig.savefig(TodayDate+'_Wakefield_FHv2.png',format = 'png', dpi=300,bbox_inches='tight')
plt.show()
