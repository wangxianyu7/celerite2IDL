#include <iostream>
#include <Eigen/Dense>
#include <cmath>  // For M_PI
#include <celerite2/celerite2.h>

using namespace celerite2::core;

// Expose the function with C linkage for compatibility with ctypes
extern "C" int compute_GP(
    int N, double* x, double* y, double* diag_, double* result, int kernel_type, int output_type,
    double par1, double par2, double par3, double par4, double par5)
{
    // Map input arrays to Eigen vectors
    Eigen::VectorXd x_vec = Eigen::Map<Eigen::VectorXd>(x, N);
    Eigen::VectorXd y_vec = Eigen::Map<Eigen::VectorXd>(y, N);
    Eigen::VectorXd diag_vec = Eigen::Map<Eigen::VectorXd>(diag_, N);

    // Compute the diagonal (variance) from diag_
    Eigen::VectorXd diag = diag_vec.array();

    // Define RowMajorMatrix type for matrices that need to be row-major
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix;
    typedef Eigen::VectorXd Vector;

    // Initialize Y and Z as column vectors (they can remain column-major)
    Eigen::MatrixXd Y = y_vec;
    Eigen::MatrixXd Z;

    // Declare variables for celerite matrices
    Vector a, a1;
    Vector c, c1;
    RowMajorMatrix U, V, U1, V1;

    // Define the kernel based on kernel_type
    if (kernel_type == 0) {
        celerite2::RealTerm<double> kernel(par1, par2); // a, c
        std::tie(c, a, U, V) = kernel.get_celerite_matrices(x_vec, diag);
    } else if (kernel_type == 1) {
        celerite2::ComplexTerm<double> kernel(par1, par2, par3, par4); // a, b, c, d
        std::tie(c, a, U, V) = kernel.get_celerite_matrices(x_vec, diag);
    } else if (kernel_type == 2) {
        celerite2::SHOTerm<double> kernel(par1, par2, par3, par4); // S0, w0, Q, eps
        std::tie(c, a, U, V) = kernel.get_celerite_matrices(x_vec, diag);
    } else if (kernel_type == 3) {
        celerite2::Matern32Term<double> kernel(par1, par2, par3); // sigma, rho, eps
        std::tie(c, a, U, V) = kernel.get_celerite_matrices(x_vec, diag);
    } else {
        // RotationTerm: sigma, period, Q0, dQ, f, epsloc
        double sigma = par1;
        double period = par2;
        double Q0 = par3;
        double dQ = par4;
        double f = par5;
        
        // Term 1 parameters
        double amp = sigma * sigma / (1 + f);
        double Q1 = 0.5 + Q0 + dQ;
        double w1 = 4 * M_PI * Q1 / (period * std::sqrt(4 * Q1 * Q1 - 1));
        double S1 = amp / (w1 * Q1);

        // Term 2 parameters
        double Q2 = 0.5 + Q0;
        double w2 = 8 * M_PI * Q2 / (period * std::sqrt(4 * Q2 * Q2 - 1));
        double S2 = f * amp / (w2 * Q2);

        // Initialize SHOTerms and combine them into a kernel
        celerite2::SHOTerm<double> term1(S1, w1, Q1, 0.00001);
        celerite2::SHOTerm<double> term2(S2, w2, Q2, 0.00001);
        auto kernel = term1 + term2;
        
        // I found the coefficients for TermSum are not correct, so I need to manually assign the coefficients
        auto coeffs_term1 = term1.get_coefficients();
        auto [ar_term1, cr_term1, ac_term1, bc_term1, cc_term1, dc_term1] = coeffs_term1;

        auto coeffs_term2 = term2.get_coefficients();
        auto [ar_term2, cr_term2, ac_term2, bc_term2, cc_term2, dc_term2] = coeffs_term2;
        
        // Determine the sizes
        Eigen::Index nar1 = ar_term1.size();
        Eigen::Index nar2 = ar_term2.size();
        Eigen::VectorXd ar(nar1 + nar2);
        
        // Assign values using block assignments
        ar.segment(0, nar1) = ar_term1;
        ar.segment(nar1, nar2) = ar_term2;

        Eigen::Index ncr1 = cr_term1.size();
        Eigen::Index ncr2 = cr_term2.size();
        Eigen::VectorXd cr(ncr1 + ncr2);
        cr.segment(0, ncr1) = cr_term1;
        cr.segment(ncr1, ncr2) = cr_term2;
        
        Eigen::Index nac1 = ac_term1.size();
        Eigen::Index nac2 = ac_term2.size();
        Eigen::VectorXd ac(nac1 + nac2);
        ac.segment(0, nac1) = ac_term1;
        ac.segment(nac1, nac2) = ac_term2;
        
        Eigen::Index nbc1 = bc_term1.size();
        Eigen::Index nbc2 = bc_term2.size();
        Eigen::VectorXd bc(nbc1 + nbc2);
        bc.segment(0, nbc1) = bc_term1;
        bc.segment(nbc1, nbc2) = bc_term2;
        
        Eigen::Index ncc1 = cc_term1.size();
        Eigen::Index ncc2 = cc_term2.size();
        Eigen::VectorXd cc(ncc1 + ncc2);
        cc.segment(0, ncc1) = cc_term1;
        cc.segment(ncc1, ncc2) = cc_term2;
        
        Eigen::Index ndc1 = dc_term1.size();
        Eigen::Index ndc2 = dc_term2.size();
        Eigen::VectorXd dc(ndc1 + ndc2);
        dc.segment(0, ndc1) = dc_term1;
        dc.segment(ndc1, ndc2) = dc_term2;
        

        kernel.set_coefficients(ar, cr, ac, bc, cc, dc);

    
        
        
        std::tie(c, a, U, V) = kernel.get_celerite_matrices(x_vec, diag);
    }

    // Perform Cholesky factorization
    Vector d;
    RowMajorMatrix W;
    RowMajorMatrix S;
    RowMajorMatrix F;  // Workspace matrices must be row-major

    int flag = factor(x_vec, c, a, U, V, d, W, S);
    if (flag != 0) {
        std::cerr << "Factorization failed with flag: " << flag << std::endl;
        return -1;
    }

    // Initialize Z
    Z = Eigen::MatrixXd::Zero(Y.rows(), Y.cols());

    // Solve the lower triangular system
    solve_lower(x_vec, c, U, W, Y, Z, F);

    // Compute the log-likelihood
    Eigen::ArrayXd do_norm = Z.array().square().colwise() / d.array();
    double norm_sum = do_norm.sum();
    double log_det = d.array().log().sum();
    double log_likelihood = -0.5 * (log_det + N * std::log(2 * M_PI) + norm_sum);

    if (output_type == 0) { // Return log-likelihood
        result[0] = log_likelihood;
    } else {
        // Start prediction
        Vector mu = Vector::Zero(x_vec.size());
        Vector xs = x_vec;  // Copy of x_vec

        // Solve the system
        solve_lower(x_vec, c, U, W, Y, Y, F);
        Z = Y.array().colwise() / d.array();
        solve_upper(x_vec, c, U, W, Z, Z, F);

        // Get celerite matrices for xs
        if (kernel_type == 0) {
            celerite2::RealTerm<double> kernel(par1, par2);
            std::tie(c1, a1, U1, V1) = kernel.get_celerite_matrices(xs, Eigen::VectorXd::Zero(xs.size()));
        } else if (kernel_type == 1) {
            celerite2::ComplexTerm<double> kernel(par1, par2, par3, par4);
            std::tie(c1, a1, U1, V1) = kernel.get_celerite_matrices(xs, Eigen::VectorXd::Zero(xs.size()));
        } else if (kernel_type == 2) {
            celerite2::SHOTerm<double> kernel(par1, par2, par3, par4);
            std::tie(c1, a1, U1, V1) = kernel.get_celerite_matrices(xs, Eigen::VectorXd::Zero(xs.size()));
        } else if (kernel_type == 3) {
            celerite2::Matern32Term<double> kernel(par1, par2, par3);
            std::tie(c1, a1, U1, V1) = kernel.get_celerite_matrices(xs, Eigen::VectorXd::Zero(xs.size()));
        } else {
            // RotationTerm: sigma, period, Q0, dQ, f, epsloc
            double sigma = par1;
            double period = par2;
            double Q0 = par3;
            double dQ = par4;
            double f = par5;
            
            // Term 1 parameters
            double amp = sigma * sigma / (1 + f);
            double Q1 = 0.5 + Q0 + dQ;
            double w1 = 4 * M_PI * Q1 / (period * std::sqrt(4 * Q1 * Q1 - 1));
            double S1 = amp / (w1 * Q1);

            // Term 2 parameters
            double Q2 = 0.5 + Q0;
            double w2 = 8 * M_PI * Q2 / (period * std::sqrt(4 * Q2 * Q2 - 1));
            double S2 = f * amp / (w2 * Q2);

            // Initialize SHOTerms and combine them into a kernel
            celerite2::SHOTerm<double> term1(S1, w1, Q1, 0.00001);
            celerite2::SHOTerm<double> term2(S2, w2, Q2, 0.00001);
            auto kernel = term1 + term2;
            
            // I found the coefficients for TermSum are not correct, so I need to manually assign the coefficients
            auto coeffs_term1 = term1.get_coefficients();
            auto [ar_term1, cr_term1, ac_term1, bc_term1, cc_term1, dc_term1] = coeffs_term1;

            auto coeffs_term2 = term2.get_coefficients();
            auto [ar_term2, cr_term2, ac_term2, bc_term2, cc_term2, dc_term2] = coeffs_term2;
            
            // Determine the sizes
            Eigen::Index nar1 = ar_term1.size();
            Eigen::Index nar2 = ar_term2.size();
            Eigen::VectorXd ar(nar1 + nar2);
            
            // Assign values using block assignments
            ar.segment(0, nar1) = ar_term1;
            ar.segment(nar1, nar2) = ar_term2;

            Eigen::Index ncr1 = cr_term1.size();
            Eigen::Index ncr2 = cr_term2.size();
            Eigen::VectorXd cr(ncr1 + ncr2);
            cr.segment(0, ncr1) = cr_term1;
            cr.segment(ncr1, ncr2) = cr_term2;
            
            Eigen::Index nac1 = ac_term1.size();
            Eigen::Index nac2 = ac_term2.size();
            Eigen::VectorXd ac(nac1 + nac2);
            ac.segment(0, nac1) = ac_term1;
            ac.segment(nac1, nac2) = ac_term2;
            
            Eigen::Index nbc1 = bc_term1.size();
            Eigen::Index nbc2 = bc_term2.size();
            Eigen::VectorXd bc(nbc1 + nbc2);
            bc.segment(0, nbc1) = bc_term1;
            bc.segment(nbc1, nbc2) = bc_term2;
            
            Eigen::Index ncc1 = cc_term1.size();
            Eigen::Index ncc2 = cc_term2.size();
            Eigen::VectorXd cc(ncc1 + ncc2);
            cc.segment(0, ncc1) = cc_term1;
            cc.segment(ncc1, ncc2) = cc_term2;
            
            Eigen::Index ndc1 = dc_term1.size();
            Eigen::Index ndc2 = dc_term2.size();
            Eigen::VectorXd dc(ndc1 + ndc2);
            dc.segment(0, ndc1) = dc_term1;
            dc.segment(ndc1, ndc2) = dc_term2;
            

            kernel.set_coefficients(ar, cr, ac, bc, cc, dc);
            std::tie(c1, a1, U1, V1) = kernel.get_celerite_matrices(xs, Eigen::VectorXd::Zero(xs.size()));
            

        }

        // Ensure U1 and V1 are row-major
        U1 = U1.template cast<double>().eval();
        V1 = V1.template cast<double>().eval();

        // Perform matrix multiplications for predictions
        general_matmul_lower(xs, x_vec, c, U1, V, Z, mu, F);
        general_matmul_upper(xs, x_vec, c, V1, U, Z, mu, F);

        // Copy predictions to result array
        for (int i = 0; i < N; i++) {
            result[i] = mu[i];
        }
    }
    return 0;
}
