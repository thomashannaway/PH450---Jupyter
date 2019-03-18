# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:27:15 2019

@author: Acer
"""
#IMPORTS
import matplotlib as mpl
import numpy as np
from scipy.integrate import odeint
from scipy.constants import m_e,e,c,epsilon_0
import matplotlib.pyplot as plt
import datetime

#PLOTTING SETTINGS
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
def WP(PlasmaDensity): #plasma frequency rad/s
    return np.sqrt((PlasmaDensity * e **2) / (epsilon_0 * m_e))

def KP(PlasmaFrequency):#plasma wave no. - rad/m
    return PlasmaFrequency / c

def max_E0(frequency): #max field amplitude/E field normalisation constant
    return (m_e * c * frequency) / e

def normX(x): 
    return x * kp

def denormX(x):
    return x / kp

def denormField(x):
    return x * E0

def norm_n(x):
    return x / pn

def denorm_n(x):
    return x * pn


def MaxDensity(BeamCharge, BeamSigmar, BeamSigmaXi):
    
    max_n = (BeamCharge/e) / ((2*np.pi)**(3/2) * 
             BeamSigmar**2 * BeamSigmaXi * pn)
    
    return max_n


def DensityProfile(Xi, R, rhobeammax, BeamSigmar, BeamSigmaXi, BeamPosition):
    
    density = rhobeammax * (np.exp(-(np.power(Xi - BeamPosition, 2.))
    /((2 * BeamSigmaXi**2)))) * (np.exp(-(np.power(R, 2.)) 
    / ((2 * BeamSigmar**2))))

    return density


def Lambda(xi, r, rhobeammax, BeamSigmar, BeamSigmaXi, BeamPosition):
    
    (xi, R) = np.meshgrid(xi,r)

    ElectronDensity = DensityProfile(xi,R, rhobeammax, BeamSigmar,
                        BeamSigmaXi,BeamPosition) #Lambda = int(nb*r dr) so where is r

    Lambda=np.trapz(ElectronDensity, axis=0)
    
    return Lambda



def ODES(Rb, xi, r, rhobeammax, BeamSigmar, BeamSigmaXi, BeamPosition,
         rhowmax, WitSigmar, WitSigmaXi, WitPosition):
    
    rb,drb =Rb
    beamLambda = Lambda(xi, r, rhobeammax, BeamSigmar, BeamSigmaXi, BeamPosition)
    witnessLambda = Lambda(xi, r, rhowmax, WitSigmar, WitSigmaXi, WitPosition)
    lam = beamLambda + witnessLambda
    #drbdxi = [drb, (4*lam*rb**(-3)) - (2*(drb**2)*rb**(-1)) - rb**(-1)]
    #drbdxi = [drb, ((4*lam)/np.power(rb,3)) - ((2*np.power(drb,2))/rb) - (1/rb)]
    drbdxi = [drb, (4*lam*np.power(rb,-3)) - 
              (2*np.power(drb,2)*np.power(rb,-1)) - np.power(rb,-1)]
    

    return drbdxi


#SCALING PARAMETERS
pn = 5e18 #plasma density typically between (1-3.3)e23 m^-3
wp = WP(pn) 
kp = KP(wp)
E0 = max_E0(wp)

#INITIAL CONDITIONS
Rb0    = 0.3 #initial blowout conditions
dRb0   = 0 #initial blowout radius
initialvalue = [Rb0,dRb0]
xi_0   = 0 # initial conditions for RHS
xi_max = 18#plasma wavelength = cavity length = 60 micro meters
steps = 400 
xi = np.linspace(xi_0, xi_max, steps) #(START,END,NUMBER OF STEPS BETWEEN LIMITS)
r = np.linspace(0,0.5,steps) #same size as xi
solWLu =[[],[]]

#NORMALIZED BEAM PARAMETERS (From WLu Paper)
bsigz = 0.5 #np.sqrt(2)
bsigr = 0.5 #1
bmu = 1.95   #BeamPosition on comoving axis
bq = 2.88e-9 # N = 1.8e10 * e to get charge
N = 1.8e10
brhomax = MaxDensity(bq, denormX(bsigr), denormX(bsigz))
#brhomax = MaxDensity(bq, bsigr, bsigz)


#WITNESS PARAMETERS
wsigz = 0.25#normX(2.26e-6)
wsigr = 0.25#normX(0.93e-6)
wmu = 10#normX(40e-6)
wq = 2e-10
wrhomax = MaxDensity(wq,denormX(wsigr),denormX(wsigz))


#solver = odeint(ODES, initialvalue, xi, args = (denormX(r), brhomax, denormX(bsigr), denormX(bsigz), denormX(bmu)))
solver = odeint(ODES, initialvalue, xi, args = (r, brhomax, bsigr, bsigz, bmu, 
                                  wrhomax, wsigr, wsigz, wmu))

BlowOutRadius = solver[:,0]
DerivativeRb = solver[:,1]
EzField= 0.5 * BlowOutRadius * DerivativeRb * E0
XI = denormX(xi)
DenormRb = denormX(BlowOutRadius)
#PLOTTING
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.set_size_inches(10.0, 10.0, forward=True)
fig.subplots_adjust(top=0.92, bottom=0.0, left=0.10, right=0.95, hspace=0.05,
                        wspace=0.1)
ax2b=ax2.twinx()

#PLOT 1
plot1ax1=ax1.plot(XI, DenormRb, '--')
plot1ax1=ax1.plot(XI, -DenormRb, '--')

#add_arrow(plot1ax1)
ax1.set_ylabel('r$_{\\rm b} (\\xi)$ ',fontsize=fontsize , **hfont)
#ax1.set(title='Non-Linear Wakefield model',font=fontsize , **hfont)

#PLOT 2
ax2.plot(XI, EzField, '-',color='red')
#ax2.plot(xi, EzField2, '--')
ax2b.plot(XI, DensityProfile(xi, r, brhomax*pn, bsigr, bsigz, bmu),'--',color='blue')
ax2b.plot(XI, DensityProfile(xi,r,wrhomax*pn,wsigr,wsigz,wmu) ,color='green')
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

#fig.savefig(TodayDate+'_Wakefield_FHv2.png',format = 'png', dpi=300,bbox_inches='tight')
plt.show()
#PARAMETER CHECK
print('Rb0: ', Rb0)
print('Plasma Density: ', pn)
print('brhomax', brhomax)
print('wrhomax', wrhomax)
print('bsigz: ',bsigz)
print('bsigr: ',bsigr)
print('bmu: ',bmu)
print('bq: ',bq)
#print(EzField)
#print('xi', xi)
#print('r',r)