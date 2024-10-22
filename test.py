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

# Load the shared library
lib = ctypes.CDLL('./CeleriteCore.so')

# Define the argument and return types of the function
lib.compute_GP.restype = ctypes.c_int
lib.compute_GP.argtypes = [
    ctypes.c_int,                      # N
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),  # x
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),  # y
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),  # yerr
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),  # result
    ctypes.c_int,                      # kernel_type
    ctypes.c_int,                      # output_type
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double  # par1 to par6
]

# Example data
N = 100
x = np.linspace(0, 10, N)
y = np.sin(x)
yerr = 0.1 * np.ones_like(x)
result = np.zeros(N)  # Allocate result array

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

        # Call the function
        status = lib.compute_GP(
            ctypes.c_int(N),
            x,
            y,
            yerr,
            result,
            ctypes.c_int(kernel_type),
            ctypes.c_int(output_type),
            ctypes.c_double(par1),
            ctypes.c_double(par2),
            ctypes.c_double(par3),
            ctypes.c_double(par4),
            ctypes.c_double(par5)
        )

        log_likelihood_c2, mu_c2, var_c2 = celerite2_output(x, y, yerr, kernel_type, par1, par2, par3, par4, par5)
        # print("C++ Log-likelihood:", result[0], "vs Python:", log_likelihood_c2)
        if status == 0:
            # print("Function executed successfully.")
            if kernel_type == 0:
                print("Using RealTerm kernel")
            elif kernel_type == 1:
                print("Using ComplexTerm kernel")
            elif kernel_type == 2:
                print("Using SHOTerm kernel")
            elif kernel_type == 3:
                print("Using Matern32Term kernel")
            else:
                print("Using RotationTerm kernel")
                
            if output_type == 0:
                # print("Log-likelihood, C++:", result[0], "vs Python:", log_likelihood_c2)
                diff = np.abs(result[0] - log_likelihood_c2)
                if diff < 1e-10:
                    print("Log-likelihoods match within tolerance (1e-10). Passed.")
                else:
                    print("Log-likelihoods differ by:", diff)
            else:
                # print("Predicted values, C++:", result[:5], "vs Python:", mu_c2[:5])
                diff_sum = np.sum(np.abs(result - mu_c2))
                if diff_sum < 1e-10:
                    print("Predicted values match within tolerance (1e-10). Passed.")
                else:
                    print("Predicted values differ by:", diff_sum)
                pass
            print()
        else:
            print("Function failed with status code:", status)
