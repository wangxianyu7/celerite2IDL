import ctypes
import numpy as np
import celerite2
from celerite2 import terms

def celerite2_output(x, y, yerr, kernel_type, par1, par2, par3=0.0, par4=0.0, par5=0.0, par6=0.0):
    if kernel_type == 0:
        kernel = terms.RealTerm(a=par1, c=par2)
    elif kernel_type == 1:
        kernel = terms.ComplexTerm(a=par1, b=par2, c=par3, d=par4)
    elif kernel_type == 2:
        kernel = terms.SHOTerm(S0=par1, w0=par2, Q=par3, eps=par4)
    elif kernel_type == 3:
        kernel = terms.Matern32Term(sigma=par1, rho=par2, eps=par3)
    else:
        kernel = terms.RotationTerm(sigma=par1, period=par2, Q0=par3, dQ=par4, f=par5)
    gp = celerite2.GaussianProcess(kernel, mean=0.0)
    gp.compute(x, yerr=yerr**0.5)
    mu, var = gp.predict(y, return_var=True)
    log_likelihood = gp.log_likelihood(y)
    return log_likelihood, mu, var

# Example data
N = 100
x = np.arange(0, 10,0.1)
y = np.sin(x)
yerr = 0.1 * np.ones_like(x)
result = np.zeros(N)  # Allocate result array
print(x,y,yerr)
for i in range(5):
    for j in range(2):
        # Kernel parameters (example values)
        kernel_type = i  # RotationTerm
        output_type = j
        par1 = 1.0
        par2 = 0.5
        par3 = 0.01
        par4 = 1e-5
        par5 = 0.0


        log_likelihood_c2, mu_c2, var_c2 = celerite2_output(x, y, yerr, kernel_type, par1, par2, par3, par4, par5)
        kernel_name = ''
        if kernel_type == 0:
            kernel_name = 'RealTerm'
        elif kernel_type == 1:
            kernel_name = 'ComplexTerm'
        elif kernel_type == 2:
            kernel_name = 'SHOTerm'
        elif kernel_type == 3:
            kernel_name = 'Matern32Term'
        else:
            kernel_name = 'RotationTerm'
            
        if output_type == 0:
            print(f"{kernel_name} log-likelihood = {log_likelihood_c2}")
        else:
            print(f"{kernel_name} prediction = {mu_c2[:5]}")

