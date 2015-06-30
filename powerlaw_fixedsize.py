import numpy as np
from scipy.optimize import leastsq

def logmodel(param,R,eps,*args):
    """log-linear model with variable inputs.

    ARGS:
        param: list.
            list of parameter values for the model.  First entry is the scale factor, 
            with each successive value storing the exponents for the parameters.
            hard-codes R^2 sqrt(epsilon) dependence for machine size.
        R: float or array of floats.
            major radius in [m]
        eps: float or array of floats.
            aspect ratio.
        *args: tuple.
            entry method for variable parameters to the model.  Length must be len(param)-1.

    RETURNS:
        fitfunc: float or array of floats.
            log-model value for given parameters and exponents (log calculation)
    """
    # check lengths of inputs
    nparams = len(args)
    if nparams is not len(param)-1:
        raise ValueError("number of input arguments does not match parameter count.")

    fitfunc = np.log10(param[0]) + 2.*np.log10(R) + 0.5*np.log10(eps)  
    for i in range(nparams):
        fitfunc += param[i+1] * np.log10(args[i])
    return fitfunc

def linmodel(param,R,eps,*args):
    """linear model with variable-length inputs.

    ARGS:
        param: list.
            list of parameter values for the model.  First entry is the scale factor, 
            with each successive value storing the exponents for the parameters.
            hard-codes R^2 sqrt(epsilon) dependence for machine size.
        R: float or array of floats.
            major radius in [m]
        eps: float or array of floats.
            aspect ratio.
        *args: tuple.
            entry method for variable parameters to the model.  Length must be len(param)-1.

    RETURNS:
        fitfunc: float or array of floats.
            log-model value for given parameters and exponents (linear calculation).
    """
    # check lengths of inputs
    nparams = len(args)
    if nparams is not len(param)-1:
        raise ValueError("number of input arguments does not match parameter count.")

    fitfunc = param[0] * R**2 * np.sqrt(eps)
    for i in range(nparams):
        fitfunc = fitfunc * (args[i]**param[i+1])
    return fitfunc

def errfunct(param,*args):
    """error function minimized by leastsq using logmodel

    ARGS:
        param: list.
            list of parameter values for the model.  First entry is the scale factor, 
            with each successive value storing the exponents for the parameters.
        *args: tuple.
            entry method for variable number of parameters to model.  Length must be
            len(param)+2.  Last entry for *args is the ydata for comparison in calculating the residuals.
            First entry is major radius R, second entry is aspect ratio epsilon.

    RETURNS:
        resid: float or vector of floats.
            residuals of ydata versus model.
    """
    # check lengths of inputs
    nparams = len(args)
    if nparams is not len(param)+2:
        raise ValueError("number of input arguments does not match parameter count.")
    
    ydata = args[-1]
    R = args[0]
    eps = args[1]
    args = args[2:-1]
    resid = np.log10(ydata) - logmodel(param,R,eps,*args)
    return resid

def fit_model(values,guesses,R,eps,*args):
    """generates least-squares minimized model for given values modeled with variable number of modeled parameters.

    ARGS:
        values: array.
            experimental values of parameter to be modeled.
        guesses: list.
            list of initial guesses for least-squares fit.
        R: float or array of floats.
            major radius [m]
        eps: float or array of floats.
            aspect ratio.
        *args: individual inputs.
            model inputs, variable length.  Must match length of guess array.

    RETURNS:
        p1: list.
            least-squares optimized model parameters.
        err: vector.
            1-sigma errorbars for parameters.
        r2: float.
            R-squared coefficient of determination.
        cov: vector.
            covariance matrix from least-squares model.
    """
    nguesses = len(guesses)
    nparams = len(args)
    if nparams is not nguesses-1:
        raise ValueError("number of input arguments does not match parameter count.")

    args_plus_vals = (R,eps) + args + (values,)

    p1,cov,infodict,mesg,ier = leastsq(errfunct,guesses,args=args_plus_vals,full_output=True)

    # calculate R^2 value
    ss_err = (infodict['fvec']**2).sum()
    ss_tot = ((np.log10(values) - np.log10(values.mean()))**2).sum()
    r2 = 1. - (ss_err/ss_tot)

    if cov is None:
        n = len(p1)
        cov = np.zeros((n,n))

    # calculate errors of parameter estimates
    ss_err_wt = ss_err/(len(args[0]) - nguesses)
    cov_wt = cov * ss_err_wt
    errors = []
    for i in range(len(p1)):
        try:
            errors.append(np.absolute(cov_wt[i][i])**0.5)
        except:
            errors.append(0.0)
    errors = np.array(errors)

    return (p1,errors,r2,cov)





