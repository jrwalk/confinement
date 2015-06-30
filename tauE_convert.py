import numpy as np

# inputs
alpha_I = .68
alpha_B = .77
alpha_n = .006
alpha_P = -0.28
alpha_R = 2.0
alpha_eps = 0.5

alpha_rho = (-1.5*alpha_I - 1.5*alpha_B - 3.*alpha_P - 2.*alpha_n)/(1. + alpha_P) - 1.5
alpha_beta = (.25*alpha_I + .25*alpha_B + 1.5*alpha_P + alpha_n)/(1. + alpha_P) + .25
alpha_nu = (-.25*alpha_I - .25*alpha_B - .5*alpha_P)/(1. + alpha_P) - .25
alpha_q = (-1.*alpha_I)/(1. + alpha_P)
alpha_R2 = (-.25*alpha_I - 1.25*alpha_B + .5*alpha_P - 2.*alpha_n + alpha_eps + (alpha_R + alpha_eps))/(1. + alpha_P) - 1.25
alpha_eps2 = (.5*alpha_I - 1.5*alpha_B - alpha_P - 2.*alpha_n + alpha_eps)/(1.+ alpha_P) - 1.5

print("alpha_rho = %f" % alpha_rho)
print("alpha_beta = %f" % alpha_beta)
print("alpha_nu = %f" % alpha_nu)
print("alpha_q = %f" % alpha_q)
print("alpha_R = %f" % alpha_R2)
print("alpha_eps = %f" % alpha_eps2)
