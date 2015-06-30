"""least-squares regression optimizer to model energy confinement time in I-mode based on SQL and IDL-saveset databases.
"""
import numpy as np
from SQLpull import *
import buildImodeDBs as b
import matplotlib.pyplot as plt
import matplotlib as mpl
import powerlaw as pl
import powerlaw_fixedsize as plf
import powerlaw_fixedsize_Lmode as plfl

#mpl.rc('text',usetex=True)
#mpl.rc('text.latex',preamble= r'\usepackage[lf]{Myriad}')
#mpl.rc('font',size=18,family='Myriad')

plt.ion()

def Pnet(obj):
    return obj.Picrf + obj.Pohm - obj.Prad - obj.dWdt

def Ploss(obj):
    return obj.Picrf + obj.Pohm - obj.dWdt

def epsilon(obj):
    return obj.a/obj.R

# load in datasets
revB,forB = b.buildDB(False)
SQLdata = SQLpull("mode='IMODE'")

# mask out obviously bad values of Prad
mr = np.where(np.logical_and(revB.prad > 0,revB.prad < 2))[0]
revB = revB[mr]
mf = np.where(np.logical_and(forB.prad > 0,forB.prad < 2))[0]
forB = forB[mf]

print("\nusing full ITER98y2 parameters -- nebar, Ip, R, a, kappa, Bt, Ploss")

#################################################################

# tau_E, SQL data only
# using full ITER98y2 scaling
# tau_model,errs,r2,cov = pl.fit_model(SQLdata.tauE,[.0562,.41,.93,1.39,.58,.78,.15,-0.69],SQLdata.nebar/1.e19,SQLdata.ip,SQLdata.R/100.,SQLdata.a/100.,SQLdata.kappa,SQLdata.bt,Ploss(SQLdata))
# tauE_mod = pl.linmodel(tau_model,SQLdata.nebar/1.e19,SQLdata.ip,SQLdata.R/100.,SQLdata.a/100.,SQLdata.kappa,SQLdata.bt,Ploss(SQLdata))

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("SQL database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("nebar exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Ip exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("R exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("a exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("kappa exponent = %f +/- %f" % (tau_model[5],errs[5]))
# print("Bt exponent = %f +/- %f" % (tau_model[6],errs[6]))
# print("Ploss exponent = %f +/- %f" % (tau_model[7],errs[7]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,SQLdata.tauE,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * \\overline{n}_e^{%.2f} * I_p^{%.2f} * R^{%.2f} * a^{%.2f} * \\kappa^{%.2f} * B_T^{%.2f} * P_{loss}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("SQL database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

#################################################################
#
# tau_E, rev-B IDL data only
# using full ITER98y2 scaling
#
# note: nebar in 10^19

# tau_model,errs,r2,cov = pl.fit_model(revB.taue,[.0562,.41,.93,1.39,.58,.78,.15,-0.69],revB.nebar*10.,-1.*revB.ip,revB.rmajor,revB.aminor,revB.kappa,-1.*revB.bt,revB.ploss)
# tauE_mod = pl.linmodel(tau_model,revB.nebar*10.,-1.*revB.ip,revB.rmajor,revB.aminor,revB.kappa,-1.*revB.bt,revB.ploss)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("SQL database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("nebar exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Ip exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("R exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("a exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("kappa exponent = %f +/- %f" % (tau_model[5],errs[5]))
# print("Bt exponent = %f +/- %f" % (tau_model[6],errs[6]))
# print("Ploss exponent = %f +/- %f" % (tau_model[7],errs[7]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,revB.taue,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * \\overline{n}_e^{%.2f} * I_p^{%.2f} * R^{%.2f} * a^{%.2f} * \\kappa^{%.2f} * B_T^{%.2f} * P_{loss}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("rev-B IDL database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

#################################################################
#
# tau_E, for-B IDL data only
# using full ITER98y2 scaling
#
# note: nebar in 10^19

# tau_model,errs,r2,cov = pl.fit_model(forB.taue,[.0562,.41,.93,1.39,.58,.78,.15,-0.69],forB.nebar*10.,forB.ip,forB.rmajor,forB.aminor,forB.kappa,forB.bt,forB.ploss)
# tauE_mod = pl.linmodel(tau_model,forB.nebar*10.,forB.ip,forB.rmajor,forB.aminor,forB.kappa,forB.bt,forB.ploss)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("SQL database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("nebar exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Ip exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("R exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("a exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("kappa exponent = %f +/- %f" % (tau_model[5],errs[5]))
# print("Bt exponent = %f +/- %f" % (tau_model[6],errs[6]))
# print("Ploss exponent = %f +/- %f" % (tau_model[7],errs[7]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,forB.taue,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * \\overline{n}_e^{%.2f} * I_p^{%.2f} * R^{%.2f} * a^{%.2f} * \\kappa^{%.2f} * B_T^{%.2f} * P_{loss}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("for-B IDL database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

#################################################################

# tau_E, SQL, revB IDL
# using full ITER98y2 scaling

# note: nebar in 10^19

# nebar = np.concatenate((SQLdata.nebar/1.e19,revB.nebar*10.))
# Ip = np.concatenate((SQLdata.ip,-1.*revB.ip))
# R = np.concatenate((SQLdata.R/100.,revB.rmajor))
# a = np.concatenate((SQLdata.a/100.,revB.aminor))
# eps = a/R
# kappa = np.concatenate((SQLdata.kappa,revB.kappa))
# Bt = np.concatenate((SQLdata.bt,-1.*revB.bt))
# Pow = np.concatenate((Ploss(SQLdata),revB.ploss))
# tauE = np.concatenate((SQLdata.tauE,revB.taue))

# tau_model,errs,r2,cov = pl.fit_model(tauE,[.0562,.41,.93,1.98,.58,.78,.15,-0.69],nebar,Ip,R,eps,kappa,Bt,Pow)
# tauE_mod = pl.linmodel(tau_model,nebar,Ip,R,a,kappa,Bt,Pow)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("SQL + rev-B IDL databases")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("nebar exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Ip exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("R exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("epsilon exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("kappa exponent = %f +/- %f" % (tau_model[5],errs[5]))
# print("Bt exponent = %f +/- %f" % (tau_model[6],errs[6]))
# print("Ploss exponent = %f +/- %f" % (tau_model[7],errs[7]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,tauE,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * \\overline{n}_e^{%.2f} * I_p^{%.2f} * R^{%.2f} * \\varepsilon^{%.2f} * \\kappa^{%.2f} * B_T^{%.2f} * P_{loss}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("SQL + rev-B IDL")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

#################################################################

# tau_E, SQL, revB, forB IDL
# using full ITER98y2 scaling

# note: nebar in 10^20

nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar,forB.nebar))
Ip = np.concatenate((SQLdata.ip,-1.*revB.ip,forB.ip))
R = np.concatenate((SQLdata.R/100.,revB.rmajor,forB.rmajor))
a = np.concatenate((SQLdata.a/100.,revB.aminor,forB.aminor))
eps = a/R
kappa = np.concatenate((SQLdata.kappa,revB.kappa,forB.kappa))
Bt = np.concatenate((SQLdata.bt,-1.*revB.bt,forB.bt))
Pow = np.concatenate((Ploss(SQLdata),revB.ploss,forB.ploss))
tauE = np.concatenate((SQLdata.tauE,revB.taue,forB.taue))

tau98_params = [.0562,.41,.93,1.98,.58,.78,.15,-0.69]

tau_model,errs,r2,cov = pl.fit_model(tauE,tau98_params,nebar,Ip,R,eps,kappa,Bt,Pow)
tauE_mod = pl.linmodel(tau_model,nebar,Ip,R,eps,kappa,Bt,Pow)

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("SQL + rev-B, for-B IDL databases")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("nebar exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Ip exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("R exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("epsilon exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("kappa exponent = %f +/- %f" % (tau_model[5],errs[5]))
print("Bt exponent = %f +/- %f" % (tau_model[6],errs[6]))
print("Ploss exponent = %f +/- %f" % (tau_model[7],errs[7]))
print("R^2 = %f" % r2)

plt.figure()
plt.plot(tauE_mod,tauE,'ob',markersize=7)
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * \\overline{n}_e^{%.2f} * I_p^{%.2f} * R^{%.2f} * \\varepsilon^{%.2f} * \\kappa^{%.2f} * B_T^{%.2f} * P_{loss}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.title("SQL + rev-B, for-B IDL")
plt.text(0.04,0.005,"$R^2 = %f$" % r2)

#################################################################

# remake figure, separating datasets

tauE_mod_SQL = pl.linmodel(tau_model,SQLdata.nebar/1.e20,SQLdata.ip,SQLdata.R/100.,SQLdata.a/SQLdata.R,SQLdata.kappa,SQLdata.bt,Ploss(SQLdata))
tauE_mod_revB = pl.linmodel(tau_model,revB.nebar,-1.*revB.ip,revB.rmajor,revB.aminor/revB.rmajor,revB.kappa,-1.*revB.bt,revB.ploss)
tauE_mod_forB = pl.linmodel(tau_model,forB.nebar,forB.ip,forB.rmajor,forB.aminor/forB.rmajor,forB.kappa,forB.bt,forB.ploss)

plt.figure()
plt.plot(tauE_mod_SQL,SQLdata.tauE,'ob',markersize=7,label='pedestal database')
plt.plot(tauE_mod_revB,revB.taue,'sr',markersize=7,label='rev-B IDL database')
plt.plot(tauE_mod_forB,forB.taue,'^g',markersize=7,label='for-B IDL database')
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * \\overline{n}_e^{%.2f} * I_p^{%.2f} * R^{%.2f} * \\varepsilon^{%.2f} * \\kappa^{%.2f} * B_T^{%.2f} * P_{loss}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.legend(loc='lower right')

#################################################################

# tau_E, SQL, revB, forB IDL
# using full ITER98y2 scaling

# note: nebar in 10^20

nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar,forB.nebar))
Ip = np.concatenate((SQLdata.ip,-1.*revB.ip,forB.ip))
R = np.concatenate((SQLdata.R/100.,revB.rmajor,forB.rmajor))
a = np.concatenate((SQLdata.a/100.,revB.aminor,forB.aminor))
eps = a/R
kappa = np.concatenate((SQLdata.kappa,revB.kappa,forB.kappa))
Bt = np.concatenate((SQLdata.bt,-1.*revB.bt,forB.bt))
Pow = np.concatenate((Pnet(SQLdata),revB.pnet,forB.pnet))
tauE = np.concatenate((SQLdata.tauE,revB.taue,forB.taue))

tau_model,errs,r2,cov = pl.fit_model(tauE,[.0562,.41,.93,1.98,.58,.78,.15,-0.69],nebar,Ip,R,eps,kappa,Bt,Pow)
tauE_mod = pl.linmodel(tau_model,nebar,Ip,R,eps,kappa,Bt,Pow)

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("SQL + rev-B, for-B IDL databases")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("nebar exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Ip exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("R exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("epsilon exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("kappa exponent = %f +/- %f" % (tau_model[5],errs[5]))
print("Bt exponent = %f +/- %f" % (tau_model[6],errs[6]))
print("Pnet exponent = %f +/- %f" % (tau_model[7],errs[7]))
print("R^2 = %f" % r2)

plt.figure()
plt.plot(tauE_mod,tauE,'ob',markersize=7)
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * \\overline{n}_e^{%.2f} * I_p^{%.2f} * R^{%.2f} * \\varepsilon^{%.2f} * \\kappa^{%.2f} * B_T^{%.2f} * P_{net}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.title("SQL + rev-B, for-B IDL")
plt.text(0.04,0.005,"$R^2 = %f$" % r2)





print("\nusing Ip, Bt, nebar, Ploss, kappa")

#################################################################

# # tau_E, SQL data only
# # model using current, field, nebar, loss power, kappa
# tau_model,errs,r2,cov = pl.fit_model(SQLdata.tauE,[0.1,1.0,0.1,1.0,0.0,0.0],SQLdata.ip,SQLdata.bt,SQLdata.nebar/1.e20,Pnet(SQLdata),SQLdata.kappa)
# tauE_mod = pl.linmodel(tau_model,SQLdata.ip,SQLdata.bt,SQLdata.nebar/1.e20,Pnet(SQLdata),SQLdata.kappa)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("SQL database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("Pnet exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("kappa exponent = %f +/- %f" % (tau_model[5],errs[5]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,SQLdata.tauE,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{net}^{%.2f} * \\kappa^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("SQL database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

# ##################################################################

# # tau_E, rev-B only
# tau_model,errs,r2,cov = pl.fit_model(revB.taue,[0.1,1.0,0.1,1.0,0.0,0.0],-1.*revB.ip,-1.*revB.bt,revB.nebar,revB.ploss,revB.kappa)
# tauE_mod = pl.linmodel(tau_model,-1.*revB.ip,-1.*revB.bt,revB.nebar,revB.ploss,revB.kappa)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("IDL database, rev-B only")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("kappa exponent = %f +/- %f" % (tau_model[5],errs[5]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,revB.taue,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f} * \\kappa^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("IDL database, rev-B only")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

## #################################################################

# # tau_E, for-B only
# tau_model,errs,r2,cov = pl.fit_model(forB.taue,[0.1,1.0,0.1,1.0,0.0,0.0],forB.ip,forB.bt,forB.nebar,forB.ploss,forB.kappa)
# tauE_mod = pl.linmodel(tau_model,forB.ip,forB.bt,forB.nebar,forB.ploss,forB.kappa)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("IDL database, for-B only")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("kappa exponent = %f +/- %f" % (tau_model[5],errs[5]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,forB.taue,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f} * \\kappa^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("IDL database, for-B only")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

##  #################################################################

#  tau_E, SQL plus rev-B IDL
# Ip = np.concatenate((SQLdata.ip,-1.*revB.ip))
# Bt = np.concatenate((SQLdata.bt,-1.*revB.bt))
# Pow = np.concatenate((Ploss(SQLdata),revB.ploss))
# nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar))
# kappa = np.concatenate((SQLdata.kappa,revB.kappa))
# tauE = np.concatenate((SQLdata.tauE,revB.taue))

# tau_model,errs,r2,cov = pl.fit_model(tauE,[.1,1.0,0.1,1.0,0.0,0.0],Ip,Bt,nebar,Pow,kappa)
# tauE_mod = pl.linmodel(tau_model,Ip,Bt,nebar,Pow,kappa)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("SQL database + IDL rev-B database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("kappa exponent = %f +/- %f" % (tau_model[5],errs[5]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,tauE,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f} * \\kappa^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("SQL database + IDL rev-B database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

#################################################################

# tau_E, SQL plus rev-B, for-B IDL
Ip = np.concatenate((SQLdata.ip,-1.*revB.ip,forB.ip))
Bt = np.concatenate((SQLdata.bt,-1.*revB.bt,forB.bt))
Pow = np.concatenate((Ploss(SQLdata),revB.ploss,forB.ploss))
nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar,forB.nebar))
kappa = np.concatenate((SQLdata.kappa,revB.kappa,forB.kappa))
tauE = np.concatenate((SQLdata.tauE,revB.taue,forB.taue))

tau_model,errs,r2,cov = pl.fit_model(tauE,[.1,1.0,0.1,1.0,0.0,0.0],Ip,Bt,nebar,Pow,kappa)
tauE_mod = pl.linmodel(tau_model,Ip,Bt,nebar,Pow,kappa)

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("SQL database + IDL rev-B, for-B database")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("kappa exponent = %f +/- %f" % (tau_model[5],errs[5]))
print("R^2 = %f" % r2)

plt.figure()
plt.plot(tauE_mod,tauE,'ob',markersize=7)
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f} * \\kappa^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.title("SQL database + IDL rev-B, for-B database")
plt.text(0.04,0.005,"$R^2 = %f$" % r2)

#################################################################

# tau_E, SQL plus rev-B, for-B IDL
Ip = np.concatenate((SQLdata.ip,-1.*revB.ip,forB.ip))
Bt = np.concatenate((SQLdata.bt,-1.*revB.bt,forB.bt))
Pow = np.concatenate((Pnet(SQLdata),revB.pnet,forB.pnet))
nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar,forB.nebar))
kappa = np.concatenate((SQLdata.kappa,revB.kappa,forB.kappa))
tauE = np.concatenate((SQLdata.tauE,revB.taue,forB.taue))

tau_model,errs,r2,cov = pl.fit_model(tauE,[.1,1.0,0.1,1.0,0.0,0.0],Ip,Bt,nebar,Pow,kappa)
tauE_mod = pl.linmodel(tau_model,Ip,Bt,nebar,Pow,kappa)

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("SQL database + IDL rev-B, for-B database")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("Pnet exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("kappa exponent = %f +/- %f" % (tau_model[5],errs[5]))
print("R^2 = %f" % r2)

plt.figure()
plt.plot(tauE_mod,tauE,'ob',markersize=7)
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{net}^{%.2f} * \\kappa^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.title("SQL database + IDL rev-B, for-B database")
plt.text(0.04,0.005,"$R^2 = %f$" % r2)








print("\nusing Ip, Bt, nebar, Ploss")

## ################################################################

# # tau_E, SQL data only
# # model using current, field, nebar, loss power
# tau_model,errs,r2,cov = pl.fit_model(SQLdata.tauE,[0.1,1.0,0.1,1.0,0.0],SQLdata.ip,SQLdata.bt,SQLdata.nebar/1.e20,Pnet(SQLdata))
# tauE_mod = pl.linmodel(tau_model,SQLdata.ip,SQLdata.bt,SQLdata.nebar/1.e20,Pnet(SQLdata))

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("SQL database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("Pnet exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,SQLdata.tauE,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{net}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("SQL database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

##################################################################

# # tau_E, rev-B only
# tau_model,errs,r2,cov = pl.fit_model(revB.taue,[0.1,1.0,0.1,1.0,0.0],-1.*revB.ip,-1.*revB.bt,revB.nebar,revB.ploss)
# tauE_mod = pl.linmodel(tau_model,-1.*revB.ip,-1.*revB.bt,revB.nebar,revB.ploss)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("rev-B IDL database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,revB.taue,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("IDL database, rev-B only")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

##################################################################

# # tau_E, for-B only
# tau_model,errs,r2,cov = pl.fit_model(forB.taue,[0.1,1.0,0.1,1.0,0.0],forB.ip,forB.bt,forB.nebar,forB.ploss)
# tauE_mod = pl.linmodel(tau_model,forB.ip,forB.bt,forB.nebar,forB.ploss)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("for-B IDL database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,forB.taue,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("IDL database, for-B only")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

#################################################################

# # tau_E, SQL plus rev-B IDL
# Ip = np.concatenate((SQLdata.ip,-1.*revB.ip))
# Bt = np.concatenate((SQLdata.bt,-1.*revB.bt))
# Pow = np.concatenate((Ploss(SQLdata),revB.ploss))
# nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar))
# tauE = np.concatenate((SQLdata.tauE,revB.taue))

# tau_model,errs,r2,cov = pl.fit_model(tauE,[.1,1.0,0.1,1.0,0.0],Ip,Bt,nebar,Pow)
# tauE_mod = pl.linmodel(tau_model,Ip,Bt,nebar,Pow)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("SQL database + rev-B IDL database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,tauE,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("SQL database + IDL rev-B database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

#################################################################

# tau_E, SQL plus rev-B, for-B IDL
Ip = np.concatenate((SQLdata.ip,-1.*revB.ip,forB.ip))
Bt = np.concatenate((SQLdata.bt,-1.*revB.bt,forB.bt))
Pow = np.concatenate((Ploss(SQLdata),revB.ploss,forB.ploss))
nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar,forB.nebar))
tauE = np.concatenate((SQLdata.tauE,revB.taue,forB.taue))

tau_model,errs,r2,cov = pl.fit_model(tauE,[.1,1.0,0.1,1.0,0.0],Ip,Bt,nebar,Pow)
tauE_mod = pl.linmodel(tau_model,Ip,Bt,nebar,Pow)

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("SQL database + rev-B, for-B IDL database")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("R^2 = %f" % r2)

plt.figure()
plt.plot(tauE_mod,tauE,'ob',markersize=7)
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.title("SQL database + IDL rev-B, for-B database")
plt.text(0.04,0.005,"$R^2 = %f$" % r2)

#################################################################

# remake figure distinguishing datasets

tauE_mod_SQL = pl.linmodel(tau_model,SQLdata.ip,SQLdata.bt,SQLdata.nebar/1.e20,Ploss(SQLdata))
tauE_mod_revB = pl.linmodel(tau_model,-1.*revB.ip,-1.*revB.bt,revB.nebar,revB.ploss)
tauE_mod_forB = pl.linmodel(tau_model,forB.ip,forB.bt,forB.nebar,forB.ploss)

plt.figure()
plt.plot(tauE_mod_SQL,SQLdata.tauE,'ob',markersize=7,label='pedestal database')
plt.plot(tauE_mod_revB,revB.taue,'sr',markersize=7,label='rev-B IDL database')
plt.plot(tauE_mod_forB,forB.taue,'^g',markersize=7,label='for-B IDL database')
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.legend(loc='lower right')

#################################################################

# tau_E, SQL plus rev-B, for-B IDL
Ip = np.concatenate((SQLdata.ip,-1.*revB.ip,forB.ip))
Bt = np.concatenate((SQLdata.bt,-1.*revB.bt,forB.bt))
Pow = np.concatenate((Pnet(SQLdata),revB.pnet,forB.pnet))
nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar,forB.nebar))
tauE = np.concatenate((SQLdata.tauE,revB.taue,forB.taue))

tau_model,errs,r2,cov = pl.fit_model(tauE,[.1,1.0,0.1,1.0,0.0],Ip,Bt,nebar,Pow)
tauE_mod = pl.linmodel(tau_model,Ip,Bt,nebar,Pow)

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("SQL database + rev-B, for-B IDL database")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("Pnet exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("R^2 = %f" % r2)

plt.figure()
plt.plot(tauE_mod,tauE,'ob',markersize=7)
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{net}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.title("SQL database + IDL rev-B, for-B database")
plt.text(0.04,0.005,"$R^2 = %f$" % r2)





print("\nusing Ip, Bt, Ploss")

#################################################################

# tau# _E, SQL data only
# # model using current, field, SOL power
# tau_model,errs,r2,cov = pl.fit_model(SQLdata.tauE,[0.1,1.0,1.0,0.0],SQLdata.ip,SQLdata.bt,Pnet(SQLdata))
# tauE_mod = pl.linmodel(tau_model,SQLdata.ip,SQLdata.bt,Pnet(SQLdata))

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("SQL database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("Pnet exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,SQLdata.tauE,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * I_p^{%.2f} * B_t^{%.2f} * P_{net}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("SQL database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

##################################################################

# tau_E# , rev-B only
# tau_model,errs,r2,cov = pl.fit_model(revB.taue,[0.1,1.0,1.0,0.0],-1.*revB.ip,-1.*revB.bt,revB.ploss)
# tauE_mod = pl.linmodel(tau_model,-1.*revB.ip,-1.*revB.bt,revB.ploss)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("IDL database, rev-B only")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("Ploss exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,revB.taue,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * I_p^{%.2f} * B_t^{%.2f} * P_{loss}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("IDL database, rev-B only")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

# #################################################
####### ###########

# # tau_E, for-B only
# tau_model,errs,r2,cov = pl.fit_model(forB.taue,[0.1,1.0,1.0,0.0],forB.ip,forB.bt,forB.ploss)
# tauE_mod = pl.linmodel(tau_model,forB.ip,forB.bt,forB.ploss)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("IDL database, for-B only")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("Ploss exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,forB.taue,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * I_p^{%.2f} * B_t^{%.2f} * P_{loss}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("IDL database, for-B only")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

#################################################################

# tau_E, SQL plus rev-B IDL
# Ip = np.concatenate((SQLdata.ip,-1.*revB.ip))
# Bt = np.concatenate((SQLdata.bt,-1.*revB.bt))
# Pow = np.concatenate((Ploss(SQLdata),revB.ploss))
# tauE = np.concatenate((SQLdata.tauE,revB.taue))

# tau_model,errs,r2,cov = pl.fit_model(tauE,[.1,1.0,1.0,0.0],Ip,Bt,Pow)
# tauE_mod = pl.linmodel(tau_model,Ip,Bt,Pow)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("SQL database + IDL rev-B database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("Ploss exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,tauE,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * I_p^{%.2f} * B_t^{%.2f} * P_{loss}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("SQL database + IDL rev-B database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

# ###############################################
##################

# tau_E, SQL plus rev-B, for-B IDL
Ip = np.concatenate((SQLdata.ip,-1.*revB.ip,forB.ip))
Bt = np.concatenate((SQLdata.bt,-1.*revB.bt,forB.bt))
Pow = np.concatenate((Ploss(SQLdata),revB.ploss,forB.ploss))
tauE = np.concatenate((SQLdata.tauE,revB.taue,forB.taue))

tau_model,errs,r2,cov = pl.fit_model(tauE,[.1,1.0,1.0,0.0],Ip,Bt,Pow)
tauE_mod = pl.linmodel(tau_model,Ip,Bt,Pow)

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("SQL database + IDL rev-B, for-B database")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("Ploss exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("R^2 = %f" % r2)

plt.figure()
plt.plot(tauE_mod,tauE,'ob',markersize=7)
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * I_p^{%.2f} * B_t^{%.2f} * P_{loss}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.title("SQL database + IDL rev-B, for-B database")
plt.text(0.04,0.005,"$R^2 = %f$" % r2)

# ###############################################
##################

# tau_E, SQL plus rev-B, for-B IDL
Ip = np.concatenate((SQLdata.ip,-1.*revB.ip,forB.ip))
Bt = np.concatenate((SQLdata.bt,-1.*revB.bt,forB.bt))
Pow = np.concatenate((Pnet(SQLdata),revB.pnet,forB.pnet))
tauE = np.concatenate((SQLdata.tauE,revB.taue,forB.taue))

tau_model,errs,r2,cov = pl.fit_model(tauE,[.1,1.0,1.0,0.0],Ip,Bt,Pow)
tauE_mod = pl.linmodel(tau_model,Ip,Bt,Pow)

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("SQL database + IDL rev-B, for-B database")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("Pnet exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("R^2 = %f" % r2)

plt.figure()
plt.plot(tauE_mod,tauE,'ob',markersize=7)
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * I_p^{%.2f} * B_t^{%.2f} * P_{net}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.title("SQL database + IDL rev-B, for-B database")
plt.text(0.04,0.005,"$R^2 = %f$" % r2)







print("\nusing Ip, Bt, Ploss/nebar")

#################################################################

# rev-B IDL
# tau_model,errs,r2,cov = pl.fit_model(revB.taue,[.1,1.0,1.0,0.0],-1.*revB.ip,-1.*revB.bt,revB.ploss/revB.nebar)
# tauE_mod = pl.linmodel(tau_model,-1.*revB.ip,-1.*revB.bt,revB.ploss/revB.nebar)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("IDL rev-B database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("P/ne exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,revB.taue,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * I_p^{%.2f} * B_t^{%.2f} * P_{loss}/\\overline{n}_e^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("IDL rev-B database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

#################################################################

# tau_E, SQL plus rev-B IDL
# Ip = np.concatenate((SQLdata.ip,-1.*revB.ip))
# Bt = np.concatenate((SQLdata.bt,-1.*revB.bt))
# Pow = np.concatenate((Ploss(SQLdata),revB.ploss))
# nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar))
# tauE = np.concatenate((SQLdata.tauE,revB.taue))

# tau_model,errs,r2,cov = pl.fit_model(tauE,[.1,1.0,1.0,0.0],Ip,Bt,Pow/nebar)
# tauE_mod = pl.linmodel(tau_model,Ip,Bt,Pow/nebar)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("SQL database + IDL rev-B database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("P/ne exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,tauE,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * I_p^{%.2f} * B_t^{%.2f} * P_{loss}/\\overline{n}_e^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("SQL database + IDL rev-B database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

#################################################################

# tau_E, SQL plus rev-B IDL
Ip = np.concatenate((SQLdata.ip,-1.*revB.ip,forB.ip))
Bt = np.concatenate((SQLdata.bt,-1.*revB.bt,forB.bt))
Pow = np.concatenate((Ploss(SQLdata),revB.ploss,forB.ploss))
nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar,forB.nebar))
tauE = np.concatenate((SQLdata.tauE,revB.taue,forB.taue))

tau_model,errs,r2,cov = pl.fit_model(tauE,[.1,1.0,1.0,0.0],Ip,Bt,Pow/nebar)
tauE_mod = pl.linmodel(tau_model,Ip,Bt,Pow/nebar)

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("SQL database + IDL rev-B, for-B database")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("Ploss/ne exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("R^2 = %f" % r2)

plt.figure()
plt.plot(tauE_mod,tauE,'ob',markersize=7)
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * I_p^{%.2f} * B_t^{%.2f} * P_{loss}/\\overline{n}_e^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.title("SQL database + IDL rev-B, forB database")
plt.text(0.04,0.005,"$R^2 = %f$" % r2)

#################################################################

# tau_E, SQL plus rev-B IDL
Ip = np.concatenate((SQLdata.ip,-1.*revB.ip,forB.ip))
Bt = np.concatenate((SQLdata.bt,-1.*revB.bt,forB.bt))
Pow = np.concatenate((Pnet(SQLdata),revB.pnet,forB.pnet))
nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar,forB.nebar))
tauE = np.concatenate((SQLdata.tauE,revB.taue,forB.taue))

tau_model,errs,r2,cov = pl.fit_model(tauE,[.1,1.0,1.0,0.0],Ip,Bt,Pow/nebar)
tauE_mod = pl.linmodel(tau_model,Ip,Bt,Pow/nebar)

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("SQL database + IDL rev-B, for-B database")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("Pnet/ne exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("R^2 = %f" % r2)

plt.figure()
plt.plot(tauE_mod,tauE,'ob',markersize=7)
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * I_p^{%.2f} * B_t^{%.2f} * P_{net}/\\overline{n}_e^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.title("SQL database + IDL rev-B, forB database")
plt.text(0.04,0.005,"$R^2 = %f$" % r2)





print("\nusing fixed R^2 sqrt(epsilon) (H-mode) size scaling, Ip, Bt, nebar, Ploss")

##################################################################

# tau_model,errs,r2,cov = plf.fit_model(revB.taue,[.1,1.0,.1,.15,0.0],revB.rmajor,revB.aminor/revB.rmajor,-1.*revB.ip,-1.*revB.bt,revB.nebar,revB.ploss)
# tauE_mod = plf.linmodel(tau_model,revB.rmajor,revB.aminor/revB.rmajor,-1.*revB.ip,-1.*revB.bt,revB.nebar,revB.ploss)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("IDL rev-B database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,revB.taue,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * R^2 * \\sqrt{\\varepsilon} * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("IDL rev-B database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

##################################################################

# Ip = np.concatenate((SQLdata.ip,-1.*revB.ip))
# Bt = np.concatenate((SQLdata.bt,-1.*revB.bt))
# Pow = np.concatenate((Ploss(SQLdata),revB.ploss))
# nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar))
# tauE = np.concatenate((SQLdata.tauE,revB.taue))
# R = np.concatenate((SQLdata.R/100.,revB.rmajor))
# a = np.concatenate((SQLdata.a/100.,revB.aminor))
# eps = a/R

# tau_model,errs,r2,cov = plf.fit_model(tauE,[.1,1.0,0.1,.15,0.0],R,eps,Ip,Bt,nebar,Pow)
# tauE_mod = plf.linmodel(tau_model,R,eps,Ip,Bt,nebar,Pow)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("SQL database + IDL rev-B database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,tauE,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * R^2 * \\sqrt{\\varepsilon} * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("SQL database + IDL rev-B database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

##################################################################

Ip = np.concatenate((SQLdata.ip,-1.*revB.ip,forB.ip))
Bt = np.concatenate((SQLdata.bt,-1.*revB.bt,forB.bt))
Pow = np.concatenate((Ploss(SQLdata),revB.ploss,forB.ploss))
nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar,forB.nebar))
tauE = np.concatenate((SQLdata.tauE,revB.taue,forB.taue))
R = np.concatenate((SQLdata.R/100.,revB.rmajor,forB.rmajor))
a = np.concatenate((SQLdata.a/100.,revB.aminor,forB.aminor))
eps = a/R

tau_model,errs,r2,cov = plf.fit_model(tauE,[.1,1.0,0.1,.15,0.0],R,eps,Ip,Bt,nebar,Pow)
tauE_mod = plf.linmodel(tau_model,R,eps,Ip,Bt,nebar,Pow)

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("SQL database + IDL rev-B, for-B database")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("R^2 = %f" % r2)

plt.figure()
plt.plot(tauE_mod,tauE,'ob',markersize=7)
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * R^2 * \\sqrt{\\varepsilon} * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.title("SQL database + IDL rev-B, for-B database")
plt.text(0.04,0.005,"$R^2 = %f$" % r2)

##################################################################

# remake figure, separating datasets

tauE_mod_SQL = plf.linmodel(tau_model,SQLdata.R/100.,epsilon(SQLdata),SQLdata.ip,SQLdata.bt,SQLdata.nebar/1.e20,Ploss(SQLdata))
tauE_mod_revB = plf.linmodel(tau_model,revB.rmajor,revB.aminor/revB.rmajor,-1.*revB.ip,-1.*revB.bt,revB.nebar,revB.ploss)
tauE_mod_forB = plf.linmodel(tau_model,forB.rmajor,forB.aminor/forB.rmajor,forB.ip,forB.bt,forB.nebar,forB.ploss)

plt.figure()
plt.plot(tauE_mod_SQL,SQLdata.tauE,'ob',markersize=7,label='pedestal database')
plt.plot(tauE_mod_revB,revB.taue,'sr',markersize=7,label='rev-B IDL database')
plt.plot(tauE_mod_forB,forB.taue,'^g',markersize=7,label='for-B IDL database')
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * R^2 * \\sqrt{\\varepsilon} * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.legend(loc='lower right')

##################################################################

# add in DIII-D, JET, AUG, ITER projection

print tau98_params

# DIII-D parameters:
R_d3d = 1.67
eps_d3d = .67/1.67
Ip_d3d = 3.0
Bt_d3d = 2.2
ne_d3d = 0.3
P_d3d = 20.0    # check this!!!
kappa_d3d = 1.75
tauE_d3d = plf.linmodel(tau_model,R_d3d,eps_d3d,Ip_d3d,Bt_d3d,ne_d3d,P_d3d)
tau98_d3d = pl.linmodel(tau98_params,ne_d3d,Ip_d3d,R_d3d,eps_d3d,kappa_d3d,Bt_d3d,P_d3d)
print("tauE DIII-D = %f" % tauE_d3d)
print("tau98 DIII-D = %f" % tau98_d3d)

# AUG parameters:
R_aug = 1.65
eps_aug = .5/1.65
Ip_aug = 1.0
Bt_aug = 2.5    # avg
ne_aug = .5
P_aug = 21.0    # check this!!!
kappa_aug = 1.75
tauE_aug = plf.linmodel(tau_model,R_aug,eps_aug,Ip_aug,Bt_aug,ne_aug,P_aug)
tau98_aug = pl.linmodel(tau98_params,ne_aug,Ip_aug,R_aug,eps_aug,kappa_aug,Bt_aug,P_aug)
print("tauE AUG = %f" % tauE_aug)
print("tau98 AUG = %f" % tau98_aug)

# JET parameters:
R_jet = 3.4
eps_jet = 0.9/3.4
Ip_jet = 4.0
Bt_jet = 3.8
ne_jet = .5
P_jet = 30.0
kappa_jet = 1.75
tauE_jet = plf.linmodel(tau_model,R_jet,eps_jet,Ip_jet,Bt_jet,ne_jet,P_jet)
tau98_jet = pl.linmodel(tau98_params,ne_jet,Ip_jet,R_jet,eps_jet,kappa_jet,Bt_jet,P_jet)
print("tauE JET = %f" % tauE_jet)
print("tau98 JET = %f" % tau98_jet)

# ITER parameters:
R_iter = 6.2
eps_iter = 2.0/6.2
Ip_iter = 15.0
Bt_iter = 5.3
ne_iter = 1.0
P_iter = 150.0
kappa_iter = 1.75
tauE_iter = plf.linmodel(tau_model,R_iter,eps_iter,Ip_iter,Bt_iter,ne_iter,P_iter)
tau98_iter = pl.linmodel(tau98_params,ne_iter,Ip_iter,R_iter,eps_iter,kappa_iter,Bt_iter,P_iter)
print("tauE ITER = %f" % tauE_iter)
print("tau98 ITER = %f" % tau98_iter)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(tauE_mod_SQL,SQLdata.tauE,'ob',markersize=7)
ax.plot(tauE_mod_revB,revB.taue,'sr',markersize=7)
ax.plot(tauE_mod_forB,forB.taue,'^g',markersize=7)
ax.plot(tauE_d3d,tauE_d3d,'*k',markersize=15)
ax.text(tauE_d3d,tauE_d3d,'DIII-D prediction')
ax.plot(tauE_aug,tauE_aug,'*k',markersize=15)
ax.text(tauE_aug,tauE_aug,'ASDEX Upgrade prediction')
ax.plot(tauE_jet,tauE_jet,'*k',markersize=15)
ax.text(tauE_jet,tauE_jet,'JET prediction')
ax.plot(tauE_iter,tauE_iter,'*k',markersize=15)
ax.text(tauE_iter,tauE_iter,'ITER prediction')
ax.plot(np.logspace(-2,1,10),np.logspace(-2,1,10),'--k')
plt.xlabel("$%.3f * R^2 * \\sqrt{\\varepsilon} * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")

##################################################################

Ip = np.concatenate((SQLdata.ip,-1.*revB.ip,forB.ip))
Bt = np.concatenate((SQLdata.bt,-1.*revB.bt,forB.bt))
Pow = np.concatenate((Pnet(SQLdata),revB.pnet,forB.pnet))
nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar,forB.nebar))
tauE = np.concatenate((SQLdata.tauE,revB.taue,forB.taue))
R = np.concatenate((SQLdata.R/100.,revB.rmajor,forB.rmajor))
a = np.concatenate((SQLdata.a/100.,revB.aminor,forB.aminor))
eps = a/R

tau_model,errs,r2,cov = plf.fit_model(tauE,[.1,1.0,0.1,.15,0.0],R,eps,Ip,Bt,nebar,Pow)
tauE_mod = plf.linmodel(tau_model,R,eps,Ip,Bt,nebar,Pow)

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("SQL database + IDL rev-B, for-B database")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("Pnet exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("R^2 = %f" % r2)

plt.figure()
plt.plot(tauE_mod,tauE,'ob',markersize=7)
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * R^2 * \\sqrt{\\varepsilon} * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{net}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.title("SQL database + IDL rev-B, for-B database")
plt.text(0.04,0.005,"$R^2 = %f$" % r2)







print("\nusing fixed R^1.5 epsilon^0.3 (L-mode) size scaling, Ip, Bt, nebar, Ploss")

##################################################################

# tau_model,errs,r2,cov = plf.fit_model(revB.taue,[.1,1.0,.1,.15,0.0],revB.rmajor,revB.aminor/revB.rmajor,-1.*revB.ip,-1.*revB.bt,revB.nebar,revB.ploss)
# tauE_mod = plf.linmodel(tau_model,revB.rmajor,revB.aminor/revB.rmajor,-1.*revB.ip,-1.*revB.bt,revB.nebar,revB.ploss)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("IDL rev-B database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,revB.taue,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * R^2 * \\sqrt{\\varepsilon} * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("IDL rev-B database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

##################################################################

# Ip = np.concatenate((SQLdata.ip,-1.*revB.ip))
# Bt = np.concatenate((SQLdata.bt,-1.*revB.bt))
# Pow = np.concatenate((Ploss(SQLdata),revB.ploss))
# nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar))
# tauE = np.concatenate((SQLdata.tauE,revB.taue))
# R = np.concatenate((SQLdata.R/100.,revB.rmajor))
# a = np.concatenate((SQLdata.a/100.,revB.aminor))
# eps = a/R

# tau_model,errs,r2,cov = plf.fit_model(tauE,[.1,1.0,0.1,.15,0.0],R,eps,Ip,Bt,nebar,Pow)
# tauE_mod = plf.linmodel(tau_model,R,eps,Ip,Bt,nebar,Pow)

# pars = tuple(tau_model)
# print("\nfitting energy confinement time tau_E")
# print("SQL database + IDL rev-B database")
# print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
# print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
# print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
# print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
# print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
# print("R^2 = %f" % r2)

# plt.figure()
# plt.plot(tauE_mod,tauE,'ob',markersize=7)
# plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
# plt.xlim([0,.06])
# plt.ylim([0,.06])
# plt.xlabel("$%.2f * R^2 * \\sqrt{\\varepsilon} * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
# plt.ylabel("$\\tau_E$ [s]")
# plt.title("SQL database + IDL rev-B database")
# plt.text(0.04,0.005,"$R^2 = %f$" % r2)

##################################################################

Ip = np.concatenate((SQLdata.ip,-1.*revB.ip,forB.ip))
Bt = np.concatenate((SQLdata.bt,-1.*revB.bt,forB.bt))
Pow = np.concatenate((Ploss(SQLdata),revB.ploss,forB.ploss))
nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar,forB.nebar))
tauE = np.concatenate((SQLdata.tauE,revB.taue,forB.taue))
R = np.concatenate((SQLdata.R/100.,revB.rmajor,forB.rmajor))
a = np.concatenate((SQLdata.a/100.,revB.aminor,forB.aminor))
eps = a/R

tau_model,errs,r2,cov = plfl.fit_model(tauE,[.1,1.0,0.1,.15,0.0],R,eps,Ip,Bt,nebar,Pow)
tauE_mod = plfl.linmodel(tau_model,R,eps,Ip,Bt,nebar,Pow)

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("SQL database + IDL rev-B, for-B database")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("Ploss exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("R^2 = %f" % r2)

plt.figure()
plt.plot(tauE_mod,tauE,'ob',markersize=7)
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * R^1.5 * \\varepsilon^0.3 * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.title("SQL database + IDL rev-B, for-B database")
plt.text(0.04,0.005,"$R^2 = %f$" % r2)

##################################################################

# remake figure, separating datasets

tauE_mod_SQL = plfl.linmodel(tau_model,SQLdata.R/100.,epsilon(SQLdata),SQLdata.ip,SQLdata.bt,SQLdata.nebar/1.e20,Ploss(SQLdata))
tauE_mod_revB = plfl.linmodel(tau_model,revB.rmajor,revB.aminor/revB.rmajor,-1.*revB.ip,-1.*revB.bt,revB.nebar,revB.ploss)
tauE_mod_forB = plfl.linmodel(tau_model,forB.rmajor,forB.aminor/forB.rmajor,forB.ip,forB.bt,forB.nebar,forB.ploss)

plt.figure()
plt.plot(tauE_mod_SQL,SQLdata.tauE,'ob',markersize=7,label='pedestal database')
plt.plot(tauE_mod_revB,revB.taue,'sr',markersize=7,label='rev-B IDL database')
plt.plot(tauE_mod_forB,forB.taue,'^g',markersize=7,label='for-B IDL database')
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * R^{1.5} * \\varepsilon^{0.3} * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.legend(loc='lower right')

##################################################################

# add in DIII-D, JET, AUG, ITER projection

print tau98_params

# DIII-D parameters:
R_d3d = 1.67
eps_d3d = .67/1.67
Ip_d3d = 3.0
Bt_d3d = 2.2
ne_d3d = 0.3
P_d3d = 20.0    # check this!!!
kappa_d3d = 1.75
tauE_d3d = plfl.linmodel(tau_model,R_d3d,eps_d3d,Ip_d3d,Bt_d3d,ne_d3d,P_d3d)
tau98_d3d = pl.linmodel(tau98_params,ne_d3d,Ip_d3d,R_d3d,eps_d3d,kappa_d3d,Bt_d3d,P_d3d)
print("tauE DIII-D = %f" % tauE_d3d)
print("tau98 DIII-D = %f" % tau98_d3d)

# AUG parameters:
R_aug = 1.65
eps_aug = .5/1.65
Ip_aug = 1.0
Bt_aug = 2.5    # avg
ne_aug = .5
P_aug = 21.0    # check this!!!
kappa_aug = 1.75
tauE_aug = plfl.linmodel(tau_model,R_aug,eps_aug,Ip_aug,Bt_aug,ne_aug,P_aug)
tau98_aug = pl.linmodel(tau98_params,ne_aug,Ip_aug,R_aug,eps_aug,kappa_aug,Bt_aug,P_aug)
print("tauE AUG = %f" % tauE_aug)
print("tau98 AUG = %f" % tau98_aug)

# JET parameters:
R_jet = 3.4
eps_jet = 0.9/3.4
Ip_jet = 4.0
Bt_jet = 3.8
ne_jet = .5
P_jet = 30.0
kappa_jet = 1.75
tauE_jet = plfl.linmodel(tau_model,R_jet,eps_jet,Ip_jet,Bt_jet,ne_jet,P_jet)
tau98_jet = pl.linmodel(tau98_params,ne_jet,Ip_jet,R_jet,eps_jet,kappa_jet,Bt_jet,P_jet)
print("tauE JET = %f" % tauE_jet)
print("tau98 JET = %f" % tau98_jet)

# ITER parameters:
R_iter = 6.2
eps_iter = 2.0/6.2
Ip_iter = 15.0
Bt_iter = 5.3
ne_iter = 1.0
P_iter = 150.0
kappa_iter = 1.75
tauE_iter = plfl.linmodel(tau_model,R_iter,eps_iter,Ip_iter,Bt_iter,ne_iter,P_iter)
tau98_iter = pl.linmodel(tau98_params,ne_iter,Ip_iter,R_iter,eps_iter,kappa_iter,Bt_iter,P_iter)
print("tauE ITER = %f" % tauE_iter)
print("tau98 ITER = %f" % tau98_iter)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(tauE_mod_SQL,SQLdata.tauE,'ob',markersize=7)
ax.plot(tauE_mod_revB,revB.taue,'sr',markersize=7)
ax.plot(tauE_mod_forB,forB.taue,'^g',markersize=7)
ax.plot(tauE_d3d,tauE_d3d,'*k',markersize=15)
ax.text(tauE_d3d,tauE_d3d,'DIII-D prediction')
ax.plot(tauE_aug,tauE_aug,'*k',markersize=15)
ax.text(tauE_aug,tauE_aug,'ASDEX Upgrade prediction')
ax.plot(tauE_jet,tauE_jet,'*k',markersize=15)
ax.text(tauE_jet,tauE_jet,'JET prediction')
ax.plot(tauE_iter,tauE_iter,'*k',markersize=15)
ax.text(tauE_iter,tauE_iter,'ITER prediction')
ax.plot(np.logspace(-2,1,10),np.logspace(-2,1,10),'--k')
plt.xlabel("$%.3f * R^{1.5} * \\varepsilon^{0.3} * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{loss}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")

##################################################################

Ip = np.concatenate((SQLdata.ip,-1.*revB.ip,forB.ip))
Bt = np.concatenate((SQLdata.bt,-1.*revB.bt,forB.bt))
Pow = np.concatenate((Pnet(SQLdata),revB.pnet,forB.pnet))
nebar = np.concatenate((SQLdata.nebar/1.e20,revB.nebar,forB.nebar))
tauE = np.concatenate((SQLdata.tauE,revB.taue,forB.taue))
R = np.concatenate((SQLdata.R/100.,revB.rmajor,forB.rmajor))
a = np.concatenate((SQLdata.a/100.,revB.aminor,forB.aminor))
eps = a/R

tau_model,errs,r2,cov = plfl.fit_model(tauE,[.1,1.0,0.1,.15,0.0],R,eps,Ip,Bt,nebar,Pow)
tauE_mod = plfl.linmodel(tau_model,R,eps,Ip,Bt,nebar,Pow)

pars = tuple(tau_model)
print("\nfitting energy confinement time tau_E")
print("SQL database + IDL rev-B, for-B database")
print("scale factor C = %f +/- %f" % (tau_model[0],errs[0]))
print("Ip exponent = %f +/- %f" % (tau_model[1],errs[1]))
print("Bt exponent = %f +/- %f" % (tau_model[2],errs[2]))
print("nebar exponent = %f +/- %f" % (tau_model[3],errs[3]))
print("Pnet exponent = %f +/- %f" % (tau_model[4],errs[4]))
print("R^2 = %f" % r2)

plt.figure()
plt.plot(tauE_mod,tauE,'ob',markersize=7)
plt.plot(np.linspace(0,.06,10),np.linspace(0,.06,10),'--k')
plt.xlim([0,.06])
plt.ylim([0,.06])
plt.xlabel("$%.3f * R^{1.5} * \\varepsilon^{0.3} * I_p^{%.2f} * B_t^{%.2f} * \\overline{n}_e^{%.2f} * P_{net}^{%.2f}$" % pars)
plt.ylabel("$\\tau_E$ [s]")
plt.title("SQL database + IDL rev-B, for-B database")
plt.text(0.04,0.005,"$R^2 = %f$" % r2)










