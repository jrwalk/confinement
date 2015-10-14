"""extra plotting routines for parameter covariances
"""
import numpy as np
from SQLpull import *
import buildImodeDBs as b
import readAUGdb as r
import matplotlib.pyplot as plt
import matplotlib as mpl

import scaling.powerlaw.powerlaw as pl

mpl.rc('text',usetex=True)
mpl.rc('text.latex')
mpl.rc('font',size=18)

plt.ion()

# load in datasets
revB,forB = b.buildDB(False)                                                        # old C-Mod data
sql = SQLpull("mode='IMODE'")                                                       # C-Mod pedestal database
aug1 = r.read('/home/jrwalk/ryter/imode_sbset_d_iaea14.csv',mode='I',quiet=True)     # AUG data
aug2 = r.read('/home/jrwalk/ryter/imode_sbset_d_iaea14.csv',mode='IH',quiet=True)   # include IH threshold
#aug = r.concat(aug1,aug2)
aug=aug1
#aug=aug2

# mask out obviously bad values of Prad
mr = np.where(np.logical_and(revB.prad > 0,revB.prad < 2))[0]
revB = revB[mr]
mf = np.where(np.logical_and(forB.prad > 0,forB.prad < 2))[0]
forB = forB[mf]

# mask out low-Bt AUG data?
#mb = np.where(abs(aug.BT) < 2.1)[0]
#aug=aug[mb]

# wrapper functions
def tau_imode(Ip,Bt,nebar,R,epsilon,Ploss):
    """calculates I-mode $\\tau_E$ from input parameters, using C-Mod fit with 
    fixed H-mode-like and L-mode-like size scalings ($\\sim R^2 \\sqrt{\\varepsilon}$
    and $\sim R^{1.5} \\varepsilon^{0.3}$ respectively).

    ARGS:
        Ip: float or float array.
            Plasma current [MA]
        Bt: float or float array.
            Toroidal field [T]
        nebar: float or float array.
            Line-averaged density [10^{20} m^{-3}]
        R: float or float array.
            Major radius [m]
        epsilon: float or float array.
            Inverse aspect ratio
        Ploss: float or float array.
            loss power [MW], Ploss = Pohm + Paux - dW/dt
    """
    tau_1 = 0.056 * Ip**.676 * Bt**.767 * nebar**.006 * R**2 * np.sqrt(epsilon) * Ploss**-.275
    tau_2 = 0.036 * Ip**.679 * Bt**.769 * nebar**.009 * R**1.5 * epsilon**.3 * Ploss**-.278
    return (tau_1,tau_2)

def tau98(Ip,Bt,nebar,R,a,kappa,Ploss):
    """calculates ITER98y2 scaling predicted $\\tau_E$ from input parameters.

    ARGS:
        Ip: float or float array.
            Plasma current [MA]
        Bt: float or float array.
            Toroidal field [T]
        nebar: float or float array.
            Line-averaged density [10^{20} m^{-3}]
        R: float or float array.
            Major radius [m]
        a: float or float array.
            minor radius [m]
        kappa: float or float array.
            elongation []
        Ploss: float or float array.
            loss power [MW], Ploss = Pohm + Paux - dW/dt
    """
    tau = 0.0562 * (nebar*10.)**.41 * 2.**.19 * Ip**.93 * Bt**.15 * R**1.4 * a**.6  * kappa**.78 * Ploss**-0.69
    return tau

def H_imode(Ip,Bt,nebar,R,epsilon,Ploss,tauE):
    """calculates H-factor from I-mode scaling (I-factor?) using C-Mod fit 
    with fixed H-mode-like and L-mode-like size scalings ($\\sim R^2 \\sqrt{\\varepsilon}$
    and $\sim R^{1.5} \\varepsilon^{0.3}$ respectively).

    ARGS:
        Ip: float or float array.
            Plasma current [MA]
        Bt: float or float array.
            Toroidal field [T]
        nebar: float or float array.
            Line-averaged density [10^{20} m^{-3}]
        R: float or float array.
            Major radius [m]
        epsilon: float or float array.
            Inverse aspect ratio
        Ploss: float or float array.
            loss power [MW], Ploss = Pohm + Paux - dW/dt
        tauE: float or float array.
            energy confinement time [s]
    """
    tau_1,tau_2 = tau_imode(Ip,Bt,nebar,R,epsilon,Ploss)
    H1 = tauE/tau_1
    H2 = tauE/tau_2
    return (H1,H2)

def Ploss_sql(obj):
    """generates loss power for SQL data object
    """
    return obj.Picrf + obj.Pohm - obj.dWdt

def Ploss_aug(data):
    """generates loss (total heating) power for AUG data object
    """
    #return (data.POHM + data.PECRH + data.PNBI)/1.e6
    return (data.PLTH)/1.e6

#######################################################################

Ip_cmod = np.concatenate((sql.ip,-1.*revB.ip,forB.ip))
Bt_cmod = np.concatenate((sql.bt,-1.*revB.bt,forB.bt))
nebar_cmod = np.concatenate((sql.nebar/1.e20,revB.nebar,forB.nebar))
R_cmod = np.concatenate((sql.R/100.,revB.rmajor,forB.rmajor))
a_cmod = np.concatenate((sql.a/100.,revB.aminor,forB.aminor))
eps_cmod = a_cmod/R_cmod
kappa_cmod = np.concatenate((sql.kappa,revB.kappa,forB.kappa))
Pow_cmod = np.concatenate((Ploss_sql(sql),revB.ploss,forB.ploss))

tau_imode_sql = tau_imode(sql.ip,sql.bt,sql.nebar/1.e20,sql.R/100.,sql.a/sql.R,Ploss_sql(sql))
tau_imode_revB = tau_imode(-1.*revB.ip,-1.*revB.bt,revB.nebar,revB.rmajor,revB.aminor/revB.rmajor,revB.ploss)
tau_imode_forB = tau_imode(forB.ip,forB.bt,forB.nebar,forB.rmajor,forB.aminor/forB.rmajor,forB.ploss)
tau_imode_cmod = np.concatenate((tau_imode_sql,tau_imode_revB,tau_imode_forB),1)
tauE_cmod = np.concatenate((sql.tauE,revB.taue,forB.taue))

# new H-factor scaling (mods H_imode)

def H_imode_new(Ip,Bt,nebar,R,Ploss,tauE):
    """calculates H-factor from I-mode scaling (I-factor?) using C-Mod fit 
    with fixed H-mode-like and L-mode-like size scalings ($\\sim R^2 \\sqrt{\\varepsilon}$
    and $\sim R^{1.5} \\varepsilon^{0.3}$ respectively).

    ARGS:
        Ip: float or float array.
            Plasma current [MA]
        Bt: float or float array.
            Toroidal field [T]
        nebar: float or float array.
            Line-averaged density [10^{20} m^{-3}]
        R: float or float array.
            Major radius [m]
        Ploss: float or float array.
            loss power [MW], Ploss = Pohm + Paux - dW/dt
        tauE: float or float array.
            energy confinement time [s]
    """
    tau_imode = 0.033 * Ip**.636 * Bt**.689 * nebar**.094 * R**1.87 * Ploss**-.288
    return tauE/tau_imode

H98_cmod = np.concatenate((sql.H,revB.h98,forB.h98))
betaN_cmod = np.concatenate((sql.betan,revB.betan,forB.betan))
HI_cmod = H_imode_new(Ip_cmod,Bt_cmod,nebar_cmod,R_cmod,Pow_cmod,tauE_cmod)

fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.plot(nebar_cmod,H98_cmod,'ob',markersize=7)
ax.plot(np.linspace(0,2.5,10),np.zeros(10)+1,'--k')
ax.set_xlabel('$\\overline{n}_e$ [$10^{20}$ m$^{-3}$]')
ax.set_ylabel('$H_{98}$')
ax.set_xlim([0.5,2.5])
ax.set_ylim([0,1.5])
plt.savefig('/home/jrwalk/graphics/Imode/confinement/covars/nebar_H98.pdf')

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(nebar_cmod,HI_cmod,'ob',markersize=7)
ax.plot(np.linspace(0,2.5,10),np.zeros(10)+1,'--k')
ax.set_xlabel('$\\overline{n}_e$ [$10^{20}$ m$^{-3}$]')
ax.set_ylabel('$H_{I-mode}$')
ax.set_xlim([0.5,2.5])
ax.set_ylim([0,1.5])
plt.savefig('/home/jrwalk/graphics/Imode/confinement/covars/nebar_HI.pdf')

fig3 = plt.figure()
ax = fig3.add_subplot(111)
ax.plot(betaN_cmod,H98_cmod,'ob',markersize=7)
ax.plot(np.linspace(0,2.5,10),np.zeros(10)+1,'--k')
ax.set_xlabel('$\\beta_N$')
ax.set_ylabel('$H_{98}$')
ax.set_xlim([0.4,1.6])
ax.set_ylim([0,1.5])
plt.savefig('/home/jrwalk/graphics/Imode/confinement/covars/betaN_H98.pdf')

fig4 = plt.figure()
ax = fig4.add_subplot(111)
ax.plot(betaN_cmod,HI_cmod,'ob',markersize=7)
ax.plot(np.linspace(0,2.5,10),np.zeros(10)+1,'--k')
ax.set_xlabel('$\\beta_N$')
ax.set_ylabel('$H_{I-mode}$')
ax.set_xlim([0.4,1.6])
ax.set_ylim([0,1.5])
plt.savefig('/home/jrwalk/graphics/Imode/confinement/covars/betaN_HI.pdf')

fig5 = plt.figure()
ax = fig5.add_subplot(111)
ax.plot(Bt_cmod,Pow_cmod,'ob',markersize=7)
ax.set_xlabel('$B_T$ [T]')
ax.set_ylabel('$P_{loss}$ [MW]')
ax.set_xlim([2.0,6.5])
ax.set_ylim([1,6])
plt.savefig('/home/jrwalk/graphics/Imode/confinement/covars/Bt_Ploss.pdf')

fig6 = plt.figure()
ax = fig6.add_subplot(111)
ax.plot(Bt_cmod,Pow_cmod,'ob',markersize=7,label='C-Mod')
ax.plot(abs(aug.BT),Ploss_aug(aug),'sr',markersize=7,label='AUG')
ax.set_xlabel('$B_T$ [T]')
ax.set_ylabel('$P_{loss}$ [MW]')
ax.set_xlim([1.0,6.5])
ax.set_ylim([1,6])
ax.legend(loc='upper left')
plt.savefig('/home/jrwalk/graphics/Imode/confinement/covars/Bt_Ploss_AUG.pdf')

fig7 = plt.figure()
ax = fig7.add_subplot(111)
ax.plot(Ip_cmod,nebar_cmod,'ob',markersize=7,label='C-Mod')
ax.set_xlabel('$I_p$ [MA]')
ax.set_ylabel('$\\overline{n}_e$ [$10^{20}$ m$^{-3}$]')
ax.set_xlim([.5,1.5])
ax.set_ylim([0,2.5])
plt.savefig('/home/jrwalk/graphics/Imode/confinement/covars/Ip_nebar.pdf')

fig8 = plt.figure()
ax = fig8.add_subplot(111)
ax.plot(Ip_cmod,nebar_cmod,'ob',markersize=7,label='C-Mod')
ax.plot(abs(aug1.IP)/1.e6,aug1.NEL/1.e20,'sr',markersize=7,label='AUG')
ax.set_xlabel('$I_p$ [MA]')
ax.set_ylabel('$\\overline{n}_e$ [$10^{20}$ m$^{-3}$]')
ax.set_xlim([.5,1.5])
ax.set_ylim([0,2.5])
ax.legend(loc='lower right')
plt.savefig('/home/jrwalk/graphics/Imode/confinement/covars/Ip_nebar_AUG.pdf')

fig9 = plt.figure()
ax = fig9.add_subplot(111)
ax.plot(nebar_cmod,Pow_cmod,'ob',markersize=7)
ax.set_xlabel('$\\overline{n}_e$')
ax.set_ylabel('$P_{loss}$ [MW]')
ax.set_xlim([0,2.5])
ax.set_ylim([1,6])
plt.savefig('/home/jrwalk/graphics/Imode/confinement/covars/nebar_Ploss.pdf')

fig10 = plt.figure()
ax = fig10.add_subplot(111)
ax.plot(Ip_cmod,Bt_cmod,'ob',markersize=7,label='C-Mod')
ax.set_xlabel('$I_p$ [MA]')
ax.set_ylabel('$B_T$ [T]')
ax.set_xlim([.5,1.5])
ax.set_ylim([1,6.5])
plt.savefig('/home/jrwalk/graphics/Imode/confinement/covars/Ip_Bt.pdf')

fig11 = plt.figure()
ax = fig11.add_subplot(111)
ax.plot(Ip_cmod,Bt_cmod,'ob',markersize=7,label='C-Mod')
ax.plot(abs(aug.IP)/1.e6,abs(aug.BT),'sr',markersize=7,label='AUG')
ax.set_xlabel('$I_p$ [MA]')
ax.set_ylabel('$B_T$ [T]')
ax.set_xlim([.5,1.5])
ax.set_ylim([1,6.5])
ax.legend(loc='upper left')
plt.savefig('/home/jrwalk/graphics/Imode/confinement/covars/Ip_Bt_AUG.pdf')













