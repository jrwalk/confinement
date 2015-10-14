"""least-squares optimizer to model energy confinement time in I-mode using the 
NPL fit based on SQL and IDL-saveset databases.
"""

import numpy as np
from SQLpull import *
import buildImodeDBs as b
import readAUGdb as r
import matplotlib.pyplot as plt
import matplotlib as mpl

import scaling.NPL.Murari as npl
import scaling.NPL.fixedsize as nplf

plt.ion()

def Ploss_sql(obj):
    """generates loss power for SQL data object
    """
    return obj.Picrf + obj.Pohm - obj.dWdt

def Ploss_aug(data):
    """generates loss (total heating) power for AUG data object
    """
    #return (data.POHM + data.PECRH + data.PNBI)/1.e6
    return (data.PLTH)/1.e6

def Pthres(R,a,kappa,nebar,Bt):
    """calculates Martin scaling for H-mode threshold power

    ARGS:
        R: float or float array.
            major radius [m]
        a: float or float array.
            minor radius [m]
        kappa: float or float array.
            elongation []
        nebar: float or float array.
            line-averaged density [10^{20} m^{-3}]
        Bt: float or float array.
            toroidal field [T]
    """
    S = (2.*np.pi)**2 * a * R * np.sqrt(kappa)
    Pth = 0.0488 * nebar**.717 * Bt**.803 * S**.941
    return Pth

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

mpl.rc('text',usetex=True)
mpl.rc('text.latex')
mpl.rc('font',size=18)

plt.ion()

# load in datasets
revB,forB = b.buildDB(False)                                                        # old C-Mod data
sql = SQLpull("mode='IMODE'")                                                       # C-Mod pedestal database
aug1 = r.read('/home/jrwalk/ryter/imode_sbset_d_iaea14.csv',mode='I',quiet=True)     # AUG data
aug2 = r.read('/home/jrwalk/ryter/imode_sbset_d_iaea14.csv',mode='IH',quiet=True)   # include IH threshold
aug = r.concat(aug1,aug2)
#aug=aug1
#aug=aug2

# mask out obviously bad values of Prad
mr = np.where(np.logical_and(revB.prad > 0,revB.prad < 2))[0]
revB = revB[mr]
mf = np.where(np.logical_and(forB.prad > 0,forB.prad < 2))[0]
forB = forB[mf]

# mask out low-Bt AUG data?
#mb = np.where(abs(aug.BT) < 2.1)[0]
#aug=aug[mb]

Ip = np.concatenate((sql.ip,-1.*revB.ip,forB.ip,abs(aug.IP)/1.e6))
Ip_cmod = np.concatenate((sql.ip,-1.*revB.ip,forB.ip))
Bt = np.concatenate((sql.bt,-1.*revB.bt,forB.bt,abs(aug.BT)))
Bt_cmod = np.concatenate((sql.bt,-1.*revB.bt,forB.bt))
nebar = np.concatenate((sql.nebar/1.e19,revB.nebar*10.,forB.nebar*10.,aug.NEL/1.e19))
nebar_cmod = np.concatenate((sql.nebar/1.e19,revB.nebar*10.,forB.nebar*10.))
R = np.concatenate((sql.R/100.,revB.rmajor,forB.rmajor,aug.RGEO))
R_cmod = np.concatenate((sql.R/100.,revB.rmajor,forB.rmajor))
kappa = np.concatenate((sql.kappa,revB.kappa,forB.kappa,aug.KAPPA))
kappa_cmod = np.concatenate((sql.kappa,revB.kappa,forB.kappa))
Pow = np.concatenate((Ploss_sql(sql),revB.ploss,forB.ploss,Ploss_aug(aug)))
Pow_cmod = np.concatenate((Ploss_sql(sql),revB.ploss,forB.ploss))
tauE_cmod = np.concatenate((sql.tauE,revB.taue,forB.taue))
tauE = np.concatenate((tauE_cmod,aug.TAUTH))

##################################################################

NPL_guesses = [.0367,1.006,1.731,1.450,-0.735,0.448,-9.403,-1.365]
NPL_guesses_noR = [.0367,1.006,1.450,-0.735,0.448,-9.403,-1.365]
NPL_guesses_noJET = [0.0647,0.959,1.216,0.280,-0.503,0.279,-13.645,-0.802]

# start with pre-established scaling, incl. JET data
#C-Mod only
tauE_NPL = npl.NPL(NPL_guesses,Ip,R,kappa,Pow,nebar,Bt)
tauE_NPL_Cmod = npl.NPL(NPL_guesses,Ip_cmod,R_cmod,kappa_cmod,Pow_cmod,nebar_cmod,Bt_cmod)
tauE_NPL_AUG = npl.NPL(NPL_guesses,abs(aug.IP)/1.e6,aug.RGEO,aug.KAPPA,Ploss_aug(aug),aug.NEL/1.e19,abs(aug.BT))

fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.plot(tauE_NPL_Cmod,tauE_cmod,'ob',markersize=7)
ax.plot(np.linspace(0,1,10),np.linspace(0,1,10),'--k')
ax.set_xlim([0,.07])
ax.set_ylim([0,.07])
ax.set_xlabel('$\\tau_E$ NPL model')
ax.set_ylabel('$\\tau_E$ measured')
ax.set_title('Murari fit, incl. JET')

# C-Mod + AUG
tauE_NPL = npl.NPL(NPL_guesses,Ip,R,kappa,Pow,nebar,Bt)

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(tauE_NPL_Cmod,tauE_cmod,'ob',markersize=7,label='C-Mod')
ax.plot(tauE_NPL_AUG,aug.TAUTH,'sr',markersize=7,label='AUG')
ax.plot(np.logspace(-2,0,10),np.logspace(-2,0,10),'--k')
ax.set_xlim(1.e-2,.5)
ax.set_ylim(1.e-2,.5)
ax.set_xlabel('$\\tau_E$ NPL model')
ax.set_ylabel('$\\tau_E$ measured')
ax.legend(loc='lower right')
ax.set_title('Murari fit, incl. JET')

# extrapolations
# DIII-D parameters:
R_d3d = 1.67
eps_d3d = .67/1.67
Ip_d3d = 1.5
Bt_d3d = 1.5
ne_d3d = 0.6
P_d3d = 20.0    # check this!!!
kappa_d3d = 1.75
Pth_d3d = Pthres(R_d3d,eps_d3d*R_d3d,kappa_d3d,ne_d3d,Bt_d3d)
tau98_d3d = tau98(Ip_d3d,Bt_d3d,ne_d3d,R_d3d,eps_d3d*R_d3d,kappa_d3d,2.*Pth_d3d)
tau_NPL_d3d = npl.NPL(NPL_guesses,Ip_d3d,R_d3d,kappa_d3d,2.*Pth_d3d,ne_d3d*10.,Bt_d3d)
print("\nDIII-D confinement:")
print("tauE I-mode NPL = %f" % tau_NPL_d3d)
print("tauE ITER98 = %f" % tau98_d3d)
print("H-mode threshold power = %f" % Pth_d3d)

# JET parameters:
R_jet = 3.4
eps_jet = 0.9/3.4
Ip_jet = 3.5
Bt_jet = 3.8
ne_jet = 0.5
P_jet = 30.0
kappa_jet = 1.75
Pth_jet = Pthres(R_jet,eps_jet*R_jet,kappa_jet,ne_jet,Bt_jet)
tau98_jet = tau98(Ip_jet,Bt_jet,ne_jet,R_jet,eps_jet*R_jet,kappa_jet,2.*Pth_jet)
tau_NPL_jet = npl.NPL(NPL_guesses,Ip_jet,R_jet,kappa_jet,2.*Pth_jet,ne_jet*10.,Bt_jet)
print("\nJET confinement")
print("tauE I-mode NPL = %f" % tau_NPL_jet)
print("tauE ITER98 = %f" % tau98_jet)
print("H-mode threshold power = %f" % Pth_jet)

# ITER parameters:
R_iter = 6.2
eps_iter = 2.0/6.2
Ip_iter = 15.0
Bt_iter = 5.3
ne_iter = 1.0
P_iter = 150.0
kappa_iter = 1.75
Pth_iter = Pthres(R_iter,eps_iter*R_iter,kappa_iter,ne_iter,Bt_iter)
tau98_iter = tau98(Ip_iter,Bt_iter,ne_iter,R_iter,eps_iter*R_iter,kappa_iter,P_iter)
tau_NPL_iter = npl.NPL(NPL_guesses,Ip_iter,R_iter,kappa_iter,P_iter,ne_iter*10.,Bt_iter)
print("\nITER confinement")
print("tauE I-mode NPL = %f" % tau_NPL_iter)
print("tauE ITER98 = %f" % tau98_iter)
print("H-mode threshold power = %f" % Pth_iter)

fig3 = plt.figure()
ax = fig3.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(tauE_NPL_Cmod,tauE_cmod,'ob',markersize=7)
ax.plot(tauE_NPL_AUG,aug.TAUTH,'sr',markersize=7)
ax.plot(tau_NPL_d3d,tau_NPL_d3d,'*k',markersize=15)
ax.text(tau_NPL_d3d,tau_NPL_d3d,'DIII-D prediction')
ax.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_d3d,'--r')
ax.plot(tau_NPL_jet,tau_NPL_jet,'*k',markersize=15)
ax.text(tau_NPL_jet,tau_NPL_jet,'JET prediction')
ax.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_jet,'--r')
ax.plot(tau_NPL_iter,tau_NPL_iter,'*k',markersize=15)
ax.text(tau_NPL_iter,tau_NPL_iter,'ITER prediction')
ax.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_iter,'--r')
ax.plot(np.logspace(-2,1,10),np.logspace(-2,1,10),'--k')
ax.set_xlabel("$\\tau_E$ NPL model")
ax.set_ylabel("$\\tau_E$ measured")
ax.set_title('Murari fit, incl. JET')
ax.set_xlim([1.e-2,10])
ax.set_ylim([1.e-2,10])


# start with pre-established scaling, excluding JET data
#C-Mod only
tauE_NPL = npl.NPL(NPL_guesses,Ip,R,kappa,Pow,nebar,Bt)
tauE_NPL_Cmod = npl.NPL(NPL_guesses_noJET,Ip_cmod,R_cmod,kappa_cmod,Pow_cmod,nebar_cmod,Bt_cmod)
tauE_NPL_AUG = npl.NPL(NPL_guesses_noJET,abs(aug.IP)/1.e6,aug.RGEO,aug.KAPPA,Ploss_aug(aug),aug.NEL/1.e19,abs(aug.BT))

fig4 = plt.figure()
ax = fig4.add_subplot(111)
ax.plot(tauE_NPL_Cmod,tauE_cmod,'ob',markersize=7)
ax.plot(np.linspace(0,1,10),np.linspace(0,1,10),'--k')
ax.set_xlim([0,.07])
ax.set_ylim([0,.07])
ax.set_xlabel('$\\tau_E$ NPL model (no JET)')
ax.set_ylabel('$\\tau_E$ measured')
ax.set_title('Murari fit, no JET')

# C-Mod + AUG
tauE_NPL = npl.NPL(NPL_guesses_noJET,Ip,R,kappa,Pow,nebar,Bt)

fig5 = plt.figure()
ax = fig5.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(tauE_NPL_Cmod,tauE_cmod,'ob',markersize=7,label='C-Mod')
ax.plot(tauE_NPL_AUG,aug.TAUTH,'sr',markersize=7,label='AUG')
ax.plot(np.logspace(-2,0,10),np.logspace(-2,0,10),'--k')
ax.set_xlim(1.e-2,.5)
ax.set_ylim(1.e-2,.5)
ax.set_xlabel('$\\tau_E$ NPL model (no JET)')
ax.set_ylabel('$\\tau_E$ measured')
ax.legend(loc='lower right')
ax.set_title('Murari fit, no JET')

# extrapolations
# DIII-D parameters:
R_d3d = 1.67
eps_d3d = .67/1.67
Ip_d3d = 1.5
Bt_d3d = 1.5
ne_d3d = 0.6
P_d3d = 20.0    # check this!!!
kappa_d3d = 1.75
Pth_d3d = Pthres(R_d3d,eps_d3d*R_d3d,kappa_d3d,ne_d3d,Bt_d3d)
tau98_d3d = tau98(Ip_d3d,Bt_d3d,ne_d3d,R_d3d,eps_d3d*R_d3d,kappa_d3d,2.*Pth_d3d)
tau_NPL_d3d = npl.NPL(NPL_guesses_noJET,Ip_d3d,R_d3d,kappa_d3d,2.*Pth_d3d,ne_d3d*10.,Bt_d3d)
print("\nDIII-D confinement:")
print("tauE I-mode NPL = %f" % tau_NPL_d3d)
print("tauE ITER98 = %f" % tau98_d3d)
print("H-mode threshold power = %f" % Pth_d3d)

# JET parameters:
R_jet = 3.4
eps_jet = 0.9/3.4
Ip_jet = 3.5
Bt_jet = 3.8
ne_jet = 0.5
P_jet = 30.0
kappa_jet = 1.75
Pth_jet = Pthres(R_jet,eps_jet*R_jet,kappa_jet,ne_jet,Bt_jet)
tau98_jet = tau98(Ip_jet,Bt_jet,ne_jet,R_jet,eps_jet*R_jet,kappa_jet,2.*Pth_jet)
tau_NPL_jet = npl.NPL(NPL_guesses_noJET,Ip_jet,R_jet,kappa_jet,2.*Pth_jet,ne_jet*10.,Bt_jet)
print("\nJET confinement")
print("tauE I-mode NPL = %f" % tau_NPL_jet)
print("tauE ITER98 = %f" % tau98_jet)
print("H-mode threshold power = %f" % Pth_jet)

# ITER parameters:
R_iter = 6.2
eps_iter = 2.0/6.2
Ip_iter = 15.0
Bt_iter = 5.3
ne_iter = 1.0
P_iter = 150.0
kappa_iter = 1.75
Pth_iter = Pthres(R_iter,eps_iter*R_iter,kappa_iter,ne_iter,Bt_iter)
tau98_iter = tau98(Ip_iter,Bt_iter,ne_iter,R_iter,eps_iter*R_iter,kappa_iter,P_iter)
tau_NPL_iter = npl.NPL(NPL_guesses_noJET,Ip_iter,R_iter,kappa_iter,P_iter,ne_iter*10.,Bt_iter)
print("\nITER confinement")
print("tauE I-mode NPL = %f" % tau_NPL_iter)
print("tauE ITER98 = %f" % tau98_iter)
print("H-mode threshold power = %f" % Pth_iter)

fig6 = plt.figure()
ax = fig6.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(tauE_NPL_Cmod,tauE_cmod,'ob',markersize=7)
ax.plot(tauE_NPL_AUG,aug.TAUTH,'sr',markersize=7)
ax.plot(tau_NPL_d3d,tau_NPL_d3d,'*k',markersize=15)
ax.text(tau_NPL_d3d,tau_NPL_d3d,'DIII-D prediction')
ax.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_d3d,'--r')
ax.plot(tau_NPL_jet,tau_NPL_jet,'*k',markersize=15)
ax.text(tau_NPL_jet,tau_NPL_jet,'JET prediction')
ax.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_jet,'--r')
ax.plot(tau_NPL_iter,tau_NPL_iter,'*k',markersize=15)
ax.text(tau_NPL_iter,tau_NPL_iter,'ITER prediction')
ax.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_iter,'--r')
ax.plot(np.logspace(-2,1,10),np.logspace(-2,1,10),'--k')
ax.set_xlabel("$\\tau_E$ NPL model (no JET)")
ax.set_ylabel("$\\tau_E$ measured")
ax.set_title('Murari fit, no JET')
ax.set_xlim([1.e-2,10])
ax.set_ylim([1.e-2,10])


##################################################################

# now directly fit model
# using only C-Mod data
tau_model,errs,r2,cov = npl.fit_model(NPL_guesses,Ip_cmod,R_cmod,kappa_cmod,Pow_cmod,nebar_cmod,Bt_cmod,tauE_cmod)
tauE_NPL_cmod = npl.NPL(tau_model,Ip_cmod,R_cmod,kappa_cmod,Pow_cmod,nebar_cmod,Bt_cmod)
tauE_NPL_aug = npl.NPL(tau_model,abs(aug.IP)/1.e6,aug.RGEO,aug.KAPPA,Ploss_aug(aug),aug.NEL/1.e19,abs(aug.BT))

pars = tuple(tau_model)
print("\nfitting energy confinement time, C-Mod data")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("R exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("kappa exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("h-param1 exponent = %f +/- %f" % (tau_model[5],errs[5]))
print("h-param2 exponent = %f +/- %f" % (tau_model[6],errs[6]))
print("h-param3 exponent = %f +/- %f" %  (tau_model[7],errs[7]))
print("R^2 = %f" % r2)

fig7 = plt.figure()
ax = fig7.add_subplot(111)
ax.plot(tauE_NPL_cmod,tauE_cmod,'ob',markersize=7)
ax.plot(np.linspace(0,1,10),np.linspace(0,1,10),'--k')
ax.set_xlabel('$\\tau_E$ NPL model')
ax.set_ylabel('$\\tau_E$ measured')
ax.set_title('fitted C-Mod data')
ax.set_xlim([0,.07])
ax.set_ylim([0,.07])

tau_model_cmod = tau_model

# with fixed size
tau_model,errs,r2,cov = nplf.fit_model(NPL_guesses_noR,Ip_cmod,R_cmod,kappa_cmod,Pow_cmod,nebar_cmod,Bt_cmod,tauE_cmod)
tauE_NPL_cmod = nplf.NPL(tau_model,Ip_cmod,R_cmod,kappa_cmod,Pow_cmod,nebar_cmod,Bt_cmod)
tauE_NPL_aug = nplf.NPL(tau_model,abs(aug.IP)/1.e6,aug.RGEO,aug.KAPPA,Ploss_aug(aug),aug.NEL/1.e19,abs(aug.BT))

pars = tuple(tau_model)
print("\nfitting energy confinement time, C-Mod data, fixed R")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("kappa exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("Ploss exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("h-param1 exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("h-param2 exponent = %f +/- %f" % (tau_model[5],errs[5]))
print("h-param3 exponent = %f +/- %f" %  (tau_model[6],errs[6]))
print("R^2 = %f" % r2)

fig8 = plt.figure()
ax = fig8.add_subplot(111)
ax.plot(tauE_NPL_cmod,tauE_cmod,'ob',markersize=7)
ax.plot(np.linspace(0,1,10),np.linspace(0,1,10),'--k')
ax.set_xlabel('$\\tau_E$ NPL model')
ax.set_ylabel('$\\tau_E$ measured')
ax.set_title('fitted C-Mod data, fixed R')
ax.set_xlim([0,.07])
ax.set_ylim([0,.07])

fig9 = plt.figure()
ax = fig9.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(tauE_NPL_cmod,tauE_cmod,'ob',markersize=7,label='C-Mod')
ax.plot(tauE_NPL_aug,aug.TAUTH,'sr',markersize=7,label='AUG')
ax.plot(np.logspace(-2,1,10),np.logspace(-2,1,10),'--k')
ax.set_xlabel('$\\tau_E$ NPL model')
ax.set_ylabel('$\\tau_E$ measured')
ax.set_title('fitted C-Mod data, fixed R')
ax.set_xlim([1.e-2,.5])
ax.set_ylim([1.e-2,.5])

tau_model_fixedsize_cmod = tau_model


# using C-Mod + AUG
tau_model,errs,r2,cov = npl.fit_model(NPL_guesses,Ip,R,kappa,Pow,nebar,Bt,tauE)
tauE_NPL_cmod = npl.NPL(tau_model,Ip_cmod,R_cmod,kappa_cmod,Pow_cmod,nebar_cmod,Bt_cmod)
tauE_NPL_aug = npl.NPL(tau_model,abs(aug.IP)/1.e6,aug.RGEO,aug.KAPPA,Ploss_aug(aug),aug.NEL/1.e19,abs(aug.BT))

pars = tuple(tau_model)
print("\nfitting energy confinement time, C-Mod+AUG data")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("R exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("kappa exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("h-param1 exponent = %f +/- %f" % (tau_model[5],errs[5]))
print("h-param2 exponent = %f +/- %f" % (tau_model[6],errs[6]))
print("h-param3 exponent = %f +/- %f" %  (tau_model[7],errs[7]))
print("R^2 = %f" % r2)

fig10 = plt.figure()
ax = fig10.add_subplot(111)
ax.plot(tauE_NPL_cmod,tauE_cmod,'ob',markersize=7)
ax.plot(np.linspace(0,1,10),np.linspace(0,1,10),'--k')
ax.set_xlabel('$\\tau_E$ NPL model')
ax.set_ylabel('$\\tau_E$ measured')
ax.set_title('fitted C-Mod + AUG')
ax.set_xlim([0,.07])
ax.set_ylim([0,.07])

fig11 = plt.figure()
ax = fig11.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(tauE_NPL_cmod,tauE_cmod,'ob',markersize=7,label='C-Mod')
ax.plot(tauE_NPL_aug,aug.TAUTH,'sr',markersize=7,label='AUG')
ax.plot(np.logspace(-2,1,10),np.logspace(-2,1,10),'--k')
ax.set_xlabel('$\\tau_E$ NPL model')
ax.set_ylabel('$\\tau_E$ measured')
ax.set_title('fitted C-Mod + AUG')
ax.set_xlim([1.e-2,.5])
ax.set_ylim([1.e-2,.5])

tau_NPL_d3d = npl.NPL(tau_model,Ip_d3d,R_d3d,kappa_d3d,2.*Pth_d3d,ne_d3d*10.,Bt_d3d)
print("\nDIII-D confinement:")
print("tauE I-mode NPL = %f" % tau_NPL_d3d)
print("tauE ITER98 = %f" % tau98_d3d)
print("H-mode threshold power = %f" % Pth_d3d)

tau_NPL_jet = npl.NPL(tau_model,Ip_jet,R_jet,kappa_jet,2.*Pth_jet,ne_jet*10.,Bt_jet)
print("\nJET confinement")
print("tauE I-mode NPL = %f" % tau_NPL_jet)
print("tauE ITER98 = %f" % tau98_jet)
print("H-mode threshold power = %f" % Pth_jet)

tau_NPL_iter = npl.NPL(tau_model,Ip_iter,R_iter,kappa_iter,P_iter,ne_iter*10.,Bt_iter)
print("\nITER confinement")
print("tauE I-mode NPL = %f" % tau_NPL_iter)
print("tauE ITER98 = %f" % tau98_iter)
print("H-mode threshold power = %f" % Pth_iter)

fig12 = plt.figure()
ax = fig12.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(tauE_NPL_cmod,tauE_cmod,'ob',markersize=7)
ax.plot(tauE_NPL_aug,aug.TAUTH,'sr',markersize=7)
ax.plot(tau_NPL_d3d,tau_NPL_d3d,'*k',markersize=15)
ax.text(tau_NPL_d3d,tau_NPL_d3d,'DIII-D prediction')
ax.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_d3d,'--r')
ax.plot(tau_NPL_jet,tau_NPL_jet,'*k',markersize=15)
ax.text(tau_NPL_jet,tau_NPL_jet,'JET prediction')
ax.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_jet,'--r')
ax.plot(tau_NPL_iter,tau_NPL_iter,'*k',markersize=15)
ax.text(tau_NPL_iter,tau_NPL_iter,'ITER prediction')
ax.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_iter,'--r')
ax.plot(np.logspace(-2,1,10),np.logspace(-2,1,10),'--k')
ax.set_xlabel("$\\tau_E$ NPL model (no JET)")
ax.set_ylabel("$\\tau_E$ measured")
ax.set_title('Fitted C-Mod + AUG')
ax.set_xlim([1.e-2,10])
ax.set_ylim([1.e-2,10])

tau_model_cmod_aug = tau_model


###################################################3
# saturation factors

NPL_guesses = [.0367,1.006,1.731,1.450,-0.735,0.448,-9.403,-1.365]
NPL_guesses_noJET = [0.0647,0.959,1.216,0.280,-0.503,0.279,-13.645,-0.802]

# saturation curves
x = np.linspace(0,30,1000)
h_NPL = npl.hfactor(NPL_guesses[5:],x,5.4)
h_NPL_noJET = npl.hfactor(NPL_guesses_noJET[5:],x,5.4)
h_cmod = npl.hfactor(tau_model_cmod[5:],x,5.4)
h_cmod_fixedsize = nplf.hfactor(tau_model_fixedsize_cmod[4:],x,5.4)
h_cmod_aug = npl.hfactor(tau_model_cmod_aug[5:],x,5.4)

# points
points_cmod = npl.hfactor(tau_model_cmod[5:],nebar_cmod,Bt_cmod)
points_cmod_fixedsize = nplf.hfactor(tau_model_fixedsize_cmod[4:],nebar_cmod,Bt_cmod)
points_cmod_aug = npl.hfactor(tau_model_cmod_aug[5:],nebar_cmod,Bt_cmod)
points_aug = npl.hfactor(tau_model_cmod[5:],aug.NEL/1.e19,aug.BT)
points_aug_fixedsize = nplf.hfactor(tau_model_fixedsize_cmod[4:],aug.NEL/1.e19,aug.BT)
points_aug_cmod = npl.hfactor(tau_model_cmod_aug[5:],aug.NEL/1.e19,aug.BT)

fig13 = plt.figure()
ax = fig13.add_subplot(111)
ax.plot(x,h_NPL,'g',label='original fit')
#ax.plot(x,h_NPL_noJET,'c',label='w/o JET')
ax.plot(x,h_cmod,'b',label='C-Mod fit')
ax.plot(x,h_cmod_fixedsize,'m',label='C-Mod fixed size')
ax.plot(x,h_cmod_aug,'r',label='C-Mod+AUG fit')
ax.set_xlabel('$\\overline{n}_e$ [$10^{19}$ m$^{-3}$]')
ax.set_ylabel('h-factor')
ax.set_xlim([0,25])

# tauE density response
n_tau_NPL = npl.NPL(NPL_guesses,1.0,0.6,1.7,4.0,x,5.4)
n_tau_cmod = npl.NPL(tau_model_cmod,1.0,0.6,1.7,4.0,x,5.4)
n_tau_cmod_fixedsize = nplf.NPL(tau_model_fixedsize_cmod,1.0,0.6,1.7,4.0,x,5.4)
n_tau_cmod_aug = npl.NPL(tau_model_cmod_aug,1.0,0.6,1.7,4.0,x,5.4)

fig14 = plt.figure()
ax = fig14.add_subplot(111)
ax.plot(x,n_tau_NPL,'g',label='original fit')
ax.plot(x,n_tau_cmod,'b',label='C-Mod fit')
ax.plot(x,n_tau_cmod_fixedsize,'m',label='C-Mod fixed size')
ax.plot(x,n_tau_cmod_aug,'r',label='C-Mod+AUG fit')
ax.set_xlabel('$\\overline{n}_e$ [$10^{19}$ m$^{-3}$]')
ax.set_ylabel('$\\tau_E$ [s]')
ax.set_xlim([0,25])






