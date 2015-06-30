# confinement
energy-confinement model for I-mode data -- generate ITER89/ITER98-like 
power-law models for I-mode energy confinement time.

Working routines are in powerlaw.py, powerlaw_fixedsize.py, and 
powerlaw_fixedsize_Lmode.py.  These allow for arbitrary parameter lists 
to be fed to the power-law regression, with powerlaw_fixedsize and 
powerlaw_fixedsize_Lmode allowing for fixed size scalings coupled to an 
otherwise-arbitrary parameter set.

Other routines callout to powerlaw scripts, running regression on parameter 
sets and displaying the result.  tauE_convert.py is a quick script to recast 
the result to dimensionless parameters.
