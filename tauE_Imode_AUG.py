"""least-squares regression optimizer to model energy confinement time in I-mode, using C-Mod and AUG data
"""
import numpy as np
from SQLpull import *
import buildImodeDBs as b
import readAUGdb as r
import matplotlib.pyplot as plt
import matplotlib as mpl
import powerlaw as pl

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
#aug=aug1
aug=aug2

# mask out obviously bad values of Prad
mr = np.where(np.logical_and(revB.prad > 0,revB.prad < 2))[0]
revB = revB[mr]
mf = np.where(np.logical_and(forB.prad > 0,forB.prad < 2))[0]
forB = forB[mf]

# mask out low-Bt AUG data?
mb = np.where(abs(aug.BT) < 2.1)[0]
aug=aug[mb]

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

def H89(Ip,Bt,nebar,R,a,kappa,Ptot,tauE):
    """wrapper to get H89 for AUG data (not written in dataset).
    
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
            elongation [dimenionless]
        Ptot: float or float array.
            total power [MW], Ptot = Pohm + Paux
        tauE: float or float array.
            energy confinement time [s]
    """
    tau89 = .048 * nebar**.1 * np.sqrt(2) * Ip**.85 * Bt**.2 * R**1.2 * a**.3 * kappa**.5 * Ptot**-0.5
    return tauE/tau89

def Ploss_sql(obj):
    """generates loss power for SQL data object
    """
    return obj.Picrf + obj.Pohm - obj.dWdt

def Ploss_aug(data):
    """generates loss (total heating) power for AUG data object
    """
    #return (data.POHM + data.PECRH + data.PNBI)/1.e6
    return (data.PLTH)/1.e6

##################################################################

# generate datasets

tau_imode_sql = tau_imode(sql.ip,sql.bt,sql.nebar/1.e20,sql.R/100.,sql.a/sql.R,Ploss_sql(sql))
tau_imode_revB = tau_imode(-1.*revB.ip,-1.*revB.bt,revB.nebar,revB.rmajor,revB.aminor/revB.rmajor,revB.ploss)
tau_imode_forB = tau_imode(forB.ip,forB.bt,forB.nebar,forB.rmajor,forB.aminor/forB.rmajor,forB.ploss)
tau_imode_cmod = np.concatenate((tau_imode_sql,tau_imode_revB,tau_imode_forB),1)
tauE_cmod = np.concatenate((sql.tauE,revB.taue,forB.taue))

tau_imode_aug = tau_imode(abs(aug.IP)/1.e6,abs(aug.BT),aug.NEL/1.e20,aug.RGEO,aug.AMIN/aug.RGEO,Ploss_aug(aug))
# generate power-type masks
m1 = np.where(np.logical_and(aug.PECRH != 0.0,aug.PNBI != 0.0))[0]  # both powers
m2 = np.where(aug.PECRH == 0.0)[0]                                  # NBI only
m3 = np.where(aug.PNBI == 0.0)[0]                                   # ECRH only
m4 = np.where(aug.IP < 0.0)[0]                                      # LSN
m5 = np.where(aug.IP > 0.0)[0]                                      # USN

tau_imode_all = np.concatenate((tau_imode_cmod,tau_imode_aug),1)
tauE_all = np.concatenate((tauE_cmod,aug.TAUTH))

# start with pre-established scaling

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_xscale('log')
ax1.set_yscale('log')
#ax1.plot(tau_imode_sql[0],sql.tauE,'ob',markersize=7)
#ax1.plot(tau_imode_revB[0],revB.taue,'sr',markersize=7)
#ax1.plot(tau_imode_forB[0],forB.taue,'^g',markersize=7)
ax1.plot(tau_imode_cmod[0],tauE_cmod,'ob',markersize=7,label='C-Mod')
ax1.plot(tau_imode_aug[0],aug.TAUTH,'sr',markersize=7,label='AUG')
ax1.plot(np.logspace(-2,0,10),np.logspace(-2,0,10),'--k')
ax1.set_xlabel("$0.056 * R^{2} * \\varepsilon^{0.5} * I_p^{0.68} * B_t^{0.77} * \\overline{n}_e^{0.01} * P_{loss}^{-0.28}$")
ax1.set_ylabel("$\\tau_E$ [s]")
ax1.set_xlim(1.e-2,.5)
ax1.set_ylim(1.e-2,.5)
ax1.legend(loc='lower right')
plt.savefig('/home/jrwalk/graphics/Imode/AUG/tauE_Imode_fixed-H-like.pdf')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_xscale('log')
ax2.set_yscale('log')
#ax2.plot(tau_imode_sql[1],sql.tauE,'ob',markersize=7)
#ax2.plot(tau_imode_revB[1],revB.taue,'sr',markersize=7)
#ax2.plot(tau_imode_forB[1],forB.taue,'^g',markersize=7)
ax2.plot(tau_imode_cmod[1],tauE_cmod,'ob',markersize=7,label='C-Mod')
ax2.plot(tau_imode_aug[1],aug.TAUTH,'sr',markersize=7,label='AUG')
ax2.plot(np.logspace(-2,0,10),np.logspace(-2,0,10),'--k')
ax2.set_xlabel("$0.036 * R^{1.5} * \\varepsilon^{0.3} * I_p^{0.68} * B_t^{0.77} * \\overline{n}_e^{0.01} * P_{loss}^{-0.28}$")
ax2.set_ylabel("$\\tau_E$ [s]")
ax2.set_xlim(1.e-2,.5)
ax2.set_ylim(1.e-2,.5)
ax2.legend(loc='lower right')
plt.savefig('/home/jrwalk/graphics/Imode/AUG/tauE_Imode_fixed-L-like.pdf')

##################################################################

# fit out full dataset

Ip = np.concatenate((sql.ip,-1.*revB.ip,forB.ip,abs(aug.IP)/1.e6))
Ip_cmod = np.concatenate((sql.ip,-1.*revB.ip,forB.ip))
Bt = np.concatenate((sql.bt,-1.*revB.bt,forB.bt,abs(aug.BT)))
Bt_cmod = np.concatenate((sql.bt,-1.*revB.bt,forB.bt))
nebar = np.concatenate((sql.nebar/1.e20,revB.nebar,forB.nebar,aug.NEL/1.e20))
nebar_cmod = np.concatenate((sql.nebar/1.e20,revB.nebar,forB.nebar))
R = np.concatenate((sql.R/100.,revB.rmajor,forB.rmajor,aug.RGEO))
R_cmod = np.concatenate((sql.R/100.,revB.rmajor,forB.rmajor))
a = np.concatenate((sql.a/100.,revB.aminor,forB.aminor,aug.AMIN))
a_cmod = np.concatenate((sql.a/100.,revB.aminor,forB.aminor))
eps = a/R
eps_cmod = a_cmod/R_cmod
kappa = np.concatenate((sql.kappa,revB.kappa,forB.kappa,aug.KAPPA))
kappa_cmod = np.concatenate((sql.kappa,revB.kappa,forB.kappa))
Pow = np.concatenate((Ploss_sql(sql),revB.ploss,forB.ploss,Ploss_aug(aug)))
Pow_cmod = np.concatenate((Ploss_sql(sql),revB.ploss,forB.ploss))

# full fit

tau_model,errs,r2,cov = pl.fit_model(tauE_all,[.056,.68,.77,.01,2.,.5,-.28],Ip,Bt,nebar,R,eps,Pow)
tauE_cmod_mod = pl.linmodel(tau_model,Ip_cmod,Bt_cmod,nebar_cmod,R_cmod,eps_cmod,Pow_cmod)
tauE_aug_mod = pl.linmodel(tau_model,abs(aug.IP)/1.e6,abs(aug.BT),aug.NEL/1.e20,aug.RGEO,aug.AMIN/aug.RGEO,Ploss_aug(aug))

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("C-Mod plus AUG databases")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("R exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("epsilon exponent = %f +/- %f" % (tau_model[5],errs[5]))
print("Ploss exponent = %f +/- %f" % (tau_model[6],errs[6]))
print("R^2 = %f" % r2)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.set_xscale('log')
ax3.set_yscale('log')
#ax3.plot(tauE_mod,tauE_all,'ob',markersize=7)
ax3.plot(tauE_cmod_mod,tauE_cmod,'ob',markersize=7,label='C-Mod')
ax3.plot(tauE_aug_mod,aug.TAUTH,'sr',markersize=7,label='AUG')
ax3.plot(np.logspace(-2,0,10),np.logspace(-2,0,10),'--k')
ax3.set_xlim([1.e-2,.5])
ax3.set_ylim([1.e-2,.5])
ax3.set_xlabel("$%.3f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * R^{%.2f} * \\varepsilon^{%.2f} * P_{loss}^{%.2f}$" % pars)
ax3.set_ylabel("$\\tau_E$ [s]")
ax3.legend(loc='lower right')
plt.savefig('/home/jrwalk/graphics/Imode/AUG/tauE_Imode_free_R_eps.pdf')

# drop epsilon scaling

tau_model,errs,r2,cov = pl.fit_model(tauE_all,[.056,.68,.77,.01,2.,-.28],Ip,Bt,nebar,R,Pow)
tauE_cmod_mod = pl.linmodel(tau_model,Ip_cmod,Bt_cmod,nebar_cmod,R_cmod,Pow_cmod)
tauE_aug_mod = pl.linmodel(tau_model,abs(aug.IP)/1.e6,abs(aug.BT),aug.NEL/1.e20,aug.RGEO,Ploss_aug(aug))

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("C-Mod plus AUG databases, drop epsilon scaling")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("R exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("Ploss exponent = %f +/- %f" % (tau_model[5],errs[5]))
print("R^2 = %f" % r2)

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.set_xscale('log')
ax4.set_yscale('log')
#ax4.plot(tauE_mod,tauE_all,'ob',markersize=7)
ax4.plot(tauE_cmod_mod,tauE_cmod,'ob',markersize=7,label='C-Mod')
ax4.plot(tauE_aug_mod,aug.TAUTH,'sr',markersize=7,label='AUG')
ax4.plot(np.logspace(-2,0,10),np.logspace(-2,0,10),'--k')
ax4.set_xlim([1.e-2,.5])
ax4.set_ylim([1.e-2,.5])
ax4.set_xlabel("$%.3f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * R^{%.2f} * P_{loss}^{%.2f}$" % pars)
ax4.set_ylabel("$\\tau_E$ [s]")
ax4.legend(loc='lower right')
plt.savefig('/home/jrwalk/graphics/Imode/AUG/tauE_Imode_free_R.pdf')

# mask off different AUG power profiles
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.set_xscale('log')
ax5.set_yscale('log')
#ax5.plot(tauE_mod,tauE_all,'ob',markersize=7)
ax5.plot(tauE_cmod_mod,tauE_cmod,'ob',markersize=7,label='C-Mod')
ax5.plot(tauE_aug_mod[m1],aug.TAUTH[m1],'sr',markersize=7,label='AUG, ECRH+NBI')
ax5.plot(tauE_aug_mod[m2],aug.TAUTH[m2],'sc',markersize=7,label='AUG, NBI only')
ax5.plot(tauE_aug_mod[m3],aug.TAUTH[m3],'sg',markersize=7,label='AUG, ECRH only')
ax5.plot(np.logspace(-2,0,10),np.logspace(-2,0,10),'--k')
ax5.set_xlim([1.e-2,.5])
ax5.set_ylim([1.e-2,.5])
ax5.set_xlabel("$%.3f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * R^{%.2f} * P_{loss}^{%.2f}$" % pars)
ax5.set_ylabel("$\\tau_E$ [s]")
ax5.legend(loc='lower right')
plt.savefig('/home/jrwalk/graphics/Imode/AUG/tauE_Imode_free_R_powersep.pdf')

# mask off different AUG shapes, LSN vs USN
fig6 = plt.figure()
ax6 = fig6.add_subplot(111)
ax6.set_xscale('log')
ax6.set_yscale('log')
#ax6.plot(tauE_mod,tauE_all,'ob',markersize=7)
ax6.plot(tauE_cmod_mod,tauE_cmod,'ob',markersize=7,label='C-Mod')
ax6.plot(tauE_aug_mod[m4],aug.TAUTH[m4],'sr',markersize=7,label='AUG, LSN')
ax6.plot(tauE_aug_mod[m5],aug.TAUTH[m5],'sg',markersize=7,label='AUG, USN')
ax6.plot(np.logspace(-2,0,10),np.logspace(-2,0,10),'--k')
ax6.set_xlim([1.e-2,.5])
ax6.set_ylim([1.e-2,.5])
ax6.set_xlabel("$%.3f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * R^{%.2f} * P_{loss}^{%.2f}$" % pars)
ax6.set_ylabel("$\\tau_E$ [s]")
ax6.legend(loc='lower right')
plt.savefig('/home/jrwalk/graphics/Imode/AUG/tauE_Imode_free_R_shape.pdf')

# re-add kappa scaling

tau_model,errs,r2,cov = pl.fit_model(tauE_all,[.056,.68,.77,.01,2.,1.5,-.28],Ip,Bt,nebar,R,kappa,Pow)
tauE_cmod_mod = pl.linmodel(tau_model,Ip_cmod,Bt_cmod,nebar_cmod,R_cmod,kappa_cmod,Pow_cmod)
tauE_aug_mod = pl.linmodel(tau_model,abs(aug.IP)/1.e6,abs(aug.BT),aug.NEL/1.e20,aug.RGEO,aug.KAPPA,Ploss_aug(aug))

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("C-Mod plus AUG databases, include kappa scaling")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("R exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("kappa exponent = %f +/- %f" % (tau_model[5],errs[5]))
print("Ploss exponent = %f +/- %f" % (tau_model[6],errs[6]))
print("R^2 = %f" % r2)

fig7 = plt.figure()
ax7 = fig7.add_subplot(111)
ax7.set_xscale('log')
ax7.set_yscale('log')
#ax7.plot(tauE_mod,tauE_all,'ob',markersize=7)
ax7.plot(tauE_cmod_mod,tauE_cmod,'ob',markersize=7,label='C-Mod')
ax7.plot(tauE_aug_mod,aug.TAUTH,'sr',markersize=7,label='AUG')
ax7.plot(np.logspace(-2,0,10),np.logspace(-2,0,10),'--k')
ax7.set_xlim([1.e-2,.5])
ax7.set_ylim([1.e-2,.5])
ax7.set_xlabel("$%.3f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * R^{%.2f} * \\kappa^{%.2f} * P_{loss}^{%.2f}$" % pars)
ax7.set_ylabel("$\\tau_E$ [s]")
ax7.legend(loc='lower right')
plt.savefig('/home/jrwalk/graphics/Imode/AUG/tauE_Imode_free_kappa.pdf')

##################################################################

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

##################################################################

# H-factor scalings
H98_cmod = np.concatenate((sql.H,revB.h98,forB.h98))
betaN_cmod = np.concatenate((sql.betan,revB.betan,forB.betan))
HI_cmod = H_imode_new(Ip_cmod,Bt_cmod,nebar_cmod,R_cmod,Pow_cmod,tauE_cmod)
HI_aug = H_imode_new(abs(aug.IP)/1.e6,abs(aug.BT),aug.NEL/1.e20,aug.RGEO,Ploss_aug(aug),aug.TAUTH)

# nebar vs. H98, HI
fig8 = plt.figure()
ax8 = fig8.add_subplot(111)
ax8.plot(nebar_cmod,H98_cmod,'ob',markersize=7,label='C-Mod')
ax8.plot(aug.NEL/1.e20,aug.H98Y2,'sr',markersize=7,label='AUG')
ax8.plot(np.linspace(0.0,2.5,10),np.zeros(10)+1.,'--k')
ax8.set_xlabel('$\\overline{n}_e$ [$10^{20}$ m$^{-3}$]')
ax8.set_ylabel('$H_{98}$')
ax8.legend(loc='lower right')
ax8.set_xlim([0.0,2.5])
ax8.set_ylim([0.0,1.5])
plt.savefig('/home/jrwalk/graphics/Imode/AUG/nebar_H98.pdf')

fig9 = plt.figure()
ax9 = fig9.add_subplot(111)
ax9.plot(nebar_cmod,HI_cmod,'^b',markersize=7,label='C-Mod')
ax9.plot(aug.NEL/1.e20,HI_aug,'^r',markersize=7,label='AUG')
ax9.plot(np.linspace(0.0,2.5,10),np.zeros(10)+1.,'--k')
ax9.set_xlabel('$\\overline{n}_e$ [$10^{20}$ m$^{-3}$]')
ax9.set_ylabel('$H_{I-mode}$')
ax9.legend(loc='lower right')
ax9.set_xlim([0.0,2.5])
ax9.set_ylim([0.0,1.5])
plt.savefig('/home/jrwalk/graphics/Imode/AUG/nebar_HI.pdf')

# betaN vs. H98, HI
fig8 = plt.figure()
ax8 = fig8.add_subplot(111)
ax8.plot(betaN_cmod,H98_cmod,'ob',markersize=7,label='C-Mod')
ax8.plot(aug.betaN,aug.H98Y2,'sr',markersize=7,label='AUG')
ax8.plot(np.linspace(0.0,2.5,10),np.zeros(10)+1.,'--k')
ax8.set_xlabel('$\\beta_N$')
ax8.set_ylabel('$H_{98}$')
ax8.legend(loc='lower right')
ax8.set_xlim([0.4,1.6])
ax8.set_ylim([0.0,1.5])
plt.savefig('/home/jrwalk/graphics/Imode/AUG/betaN_H98.pdf')

fig9 = plt.figure()
ax9 = fig9.add_subplot(111)
ax9.plot(betaN_cmod,HI_cmod,'^b',markersize=7,label='C-Mod')
ax9.plot(aug.betaN,HI_aug,'^r',markersize=7,label='AUG')
ax9.plot(np.linspace(0.0,2.5,10),np.zeros(10)+1.,'--k')
ax9.set_xlabel('$\\beta_N$')
ax9.set_ylabel('$H_{I-mode}$')
ax9.legend(loc='lower right')
ax9.set_xlim([0.4,1.6])
ax9.set_ylim([0.0,1.5])
plt.savefig('/home/jrwalk/graphics/Imode/AUG/betaN_HI.pdf')


##################################################################

# parameter covariances

# power vs field
fig10 = plt.figure()
ax10 = fig10.add_subplot(111)
ax10.plot(Bt_cmod,Pow_cmod,'ob',markersize=7,label='C-Mod')
ax10.plot(abs(aug1.BT),Ploss_aug(aug1),'sr',markersize=7,label='AUG I-mode')
#ax10.plot(abs(aug2.BT),Ploss_aug(aug2),'^g',markersize=7,label='AUG IH')
ax10.set_xlabel('$B_T$ [T]')
ax10.set_ylabel('$P_{loss}$ [MW]')
ax10.set_xlim([1.0,6.5])
ax10.set_ylim([1,6])
ax10.legend(loc='upper left')
plt.savefig('/home/jrwalk/graphics/Imode/AUG/Bt_Ploss.pdf')

# current vs density
fig11 = plt.figure()
ax11 = fig11.add_subplot(111)
ax11.plot(Ip_cmod,nebar_cmod,'ob',markersize=7,label='C-Mod')
ax11.plot(abs(aug1.IP)/1.e6,aug1.NEL/1.e20,'sr',markersize=7,label='AUG I-mode')
#ax11.plot(abs(aug2.IP)/1.1e6,aug2.NEL/1.e20,'^g',markersize=7,label='AUG IH')
ax11.set_xlabel('$I_p$ [MA]')
ax11.set_ylabel('$\\overline{n}_e$ [$10^{20}$ m$^{-3}$]')
ax11.set_xlim([.5,1.5])
ax11.set_ylim([0,2.5])
#ax11.legend(loc='lower right')
plt.savefig('/home/jrwalk/graphics/Imode/AUG/Ip_nebar.pdf')


##################################################################

# DIII-D parameters:
R_d3d = 1.67
eps_d3d = .67/1.67
Ip_d3d = 1.5
Bt_d3d = 1.5
ne_d3d = 0.6
P_d3d = 20.0    # check this!!!
kappa_d3d = 1.75
Pth_d3d = Pthres(R_d3d,eps_d3d*R_d3d,kappa_d3d,ne_d3d,Bt_d3d)

(tau_y2_d3d,tau_y1_d3d) = tau_imode(Ip_d3d,Bt_d3d,ne_d3d,R_d3d,eps_d3d,2.*Pth_d3d)
tau98_d3d = tau98(Ip_d3d,Bt_d3d,ne_d3d,R_d3d,eps_d3d*R_d3d,kappa_d3d,2.*Pth_d3d)
print("\nDIII-D confinement:")
print("tauE I-mode y1 = %f" % tau_y1_d3d)
print("tauE I-mode y2 = %f" % tau_y2_d3d)
print("tauE ITER98 = %f" % tau98_d3d)
print("H-mode threshold power = %f" % Pth_d3d)

# JET parameters:
R_jet = 3.4
eps_jet = 0.9/3.4
Ip_jet = 3.5
Bt_jet = 3.8
ne_jet = .5
P_jet = 30.0
kappa_jet = 1.75

Pth_jet = Pthres(R_jet,eps_jet*R_jet,kappa_jet,ne_jet,Bt_jet)
(tau_y2_jet,tau_y1_jet) = tau_imode(Ip_jet,Bt_jet,ne_jet,R_jet,eps_jet,2.*Pth_jet)
tau98_jet = tau98(Ip_jet,Bt_jet,ne_jet,R_jet,eps_jet*R_jet,kappa_jet,2.*Pth_jet)
print("\nJET confinement")
print("tauE I-mode y1 = %f" % tau_y1_jet)
print("tauE I-mode y2 = %f" % tau_y2_jet)
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
(tau_y2_iter,tau_y1_iter) = tau_imode(Ip_iter,Bt_iter,ne_iter,R_iter,eps_iter,P_iter)
tau98_iter = tau98(Ip_iter,Bt_iter,ne_iter,R_iter,eps_iter*R_iter,kappa_iter,P_iter)
print("\nITER confinement")
print("tauE I-mode y1 = %f" % tau_y1_iter)
print("tauE I-mode y2 = %f" % tau_y2_iter)
print("tauE ITER98 = %f" % tau98_iter)
print("H-mode threshold power = %f" % Pth_iter)

# extrapolations!
(tau_y2_cmod,tau_y1_cmod) = tau_imode(Ip_cmod,Bt_cmod,nebar_cmod,R_cmod,eps_cmod,Pow_cmod)
(tau_y2_aug,tau_y1_aug) = tau_imode(abs(aug.IP)/1.e6,abs(aug.BT),aug.NEL/1.e20,aug.RGEO,aug.AMIN/aug.RGEO,aug.PLTH/1.e6)

fig12 = plt.figure()
ax12 = fig12.add_subplot(111)
ax12.set_xscale('log')
ax12.set_yscale('log')
ax12.plot(tau_y1_cmod,tauE_cmod,'ob',markersize=7)
ax12.plot(tau_y1_aug,aug.TAUTH,'sr',markersize=7)
ax12.plot(tau_y1_d3d,tau_y1_d3d,'*k',markersize=15)
ax12.text(tau_y1_d3d,tau_y1_d3d,'DIII-D prediction')
ax12.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_d3d,'--r')
ax12.plot(tau_y1_jet,tau_y1_jet,'*k',markersize=15)
ax12.text(tau_y1_jet,tau_y1_jet,'JET prediction')
ax12.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_jet,'--r')
ax12.plot(tau_y1_iter,tau_y1_iter,'*k',markersize=15)
ax12.text(tau_y1_iter,tau_y1_iter,'ITER prediction')
ax12.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_iter,'--r')
ax12.plot(np.logspace(-2,1,10),np.logspace(-2,1,10),'--k')
ax12.set_xlabel("$0.036 * R^{1.5} * \\varepsilon^{0.3} * I_p^{0.68} * B_t^{0.77} * \\overline{n}_e^{0.01} * P_{loss}^{-0.28}$")
ax12.set_ylabel("$\\tau_E$ [s]")
ax12.set_xlim([1.e-2,10])
ax12.set_ylim([1.e-2,10])
plt.savefig('/home/jrwalk/graphics/Imode/AUG/tauE_Imode_fixed-L-like_project.pdf')

fig13 = plt.figure()
ax13 = fig13.add_subplot(111)
ax13.set_xscale('log')
ax13.set_yscale('log')
ax13.plot(tau_y2_cmod,tauE_cmod,'ob',markersize=7)
ax13.plot(tau_y2_aug,aug.TAUTH,'sr',markersize=7)
ax13.plot(tau_y2_d3d,tau_y2_d3d,'*k',markersize=15)
ax13.text(tau_y2_d3d,tau_y2_d3d,'DIII-D prediction')
ax13.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_d3d,'--r')
ax13.plot(tau_y2_jet,tau_y2_jet,'*k',markersize=15)
ax13.text(tau_y2_jet,tau_y2_jet,'JET prediction')
ax13.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_jet,'--r')
ax13.plot(tau_y2_iter,tau_y2_iter,'*k',markersize=15)
ax13.text(tau_y2_iter,tau_y2_iter,'ITER prediction')
ax13.plot(np.logspace(-2,1,10),np.zeros(10)+tau98_iter,'--r')
ax13.plot(np.logspace(-2,1,10),np.logspace(-2,1,10),'--k')
ax13.set_xlabel("$0.056 * R^{2} * \\varepsilon^{0.5} * I_p^{0.68} * B_t^{0.77} * \\overline{n}_e^{0.01} * P_{loss}^{-0.28}$")
ax13.set_ylabel("$\\tau_E$ [s]")
ax13.set_xlim([1.e-2,10])
ax13.set_ylim([1.e-2,10])
plt.savefig('/home/jrwalk/graphics/Imode/AUG/tauE_Imode_fixed-H-like_project.pdf')
















