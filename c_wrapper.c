#include <cstdio>
#include "idl_export.h"
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include "celerite2/celerite2.h"

// Wrap the declaration with extern "C" to prevent C++ name mangling
extern "C" {
    // Forward declaration of the compute_GPRotation function
    double computeGP(
        int N, double* x, double* y, double* diag_, double* result, int kernel_type, int output_type,
        double par1, double par2, double par3, double par4, double par5
    );

    void computeGP_wrapper(int argc, void *argv[]) {
        // Confirm correct number of arguments
        if (argc != 12) {
            std::cerr << "Error: Expected 12 arguments, received " << argc << "." << std::endl;
            return;
        }
        //std::cout << "Start Receiving Arguments" << std::endl;

        // Retrieve the arguments passed from IDL
        double par1 = *(double*)argv[7];
        double par2 = *(double*)argv[8];
        double par3 = *(double*)argv[9];
        double par4 = *(double*)argv[10];
        double par5 = *(double*)argv[11];

        int N = *(IDL_LONG*)argv[0];  // Interpret as an integer directly
        int kernel_type = *(IDL_LONG*)argv[5];
        int output_type = *(IDL_LONG*)argv[6];
        
        double* x = reinterpret_cast<double*>(argv[1]);
        double* y = reinterpret_cast<double*>(argv[2]);
        double* diag_ = reinterpret_cast<double*>(argv[3]);
        double* result = reinterpret_cast<double*>(argv[4]);

        //std::cout << "output_type: " << output_type << std::endl;

        // Call the compute_GPRotation function
        double output = computeGP(
            N, x, y, diag_, result, kernel_type, output_type, par1, par2, par3, par4, par5
        );

        
        return;
    }
}

