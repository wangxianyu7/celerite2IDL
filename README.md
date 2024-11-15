# celerite2

### Install Eigen and celerite2
```sh
wget -O Eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip Eigen.zip
sudo cp -r eigen-3.4.0/Eigen /usr/local/include
pip install celerite2
```
### Compile it
```
g++ -std=c++17 -shared -o CeleriteCore.so -fPIC CeleriteCore.cpp -I/path_of_celerite_cpp_code -I/path_of_Eigen
# for me, it is
g++ -std=c++17 -shared -o CeleriteCore.so -fPIC CeleriteCore.cpp -I/workspaces/celerite2IDL/c++/include -I/usr/local/include/Eigen

# with c_wrapper.c
g++ -std=c++17 -o computeGP.so -shared -fPIC c_wrapper.c CeleriteCore.cpp -I/path_of_celerite_cpp_code -I/path_of_Eigen -I/path_of_idl_export.h
# for me, it is
g++ -std=c++17 -o computeGP.so -shared -fPIC c_wrapper.c CeleriteCore.cpp -I/N/project/spinOrbit_Angles/codes/nv5/idl90/lib/EXOFASTv2/celerite2IDL/c++/include -I/usr/local/include/Eigen -I/N/project/spinOrbit_Angles/codes/nv5/idl90/external/include

```
### Test
For IDL:
```
.com computeGP.pro
.com GPTest.pro

```
Verify the results with the Celerite2 Python version:

```bash
python GPTest.py
```


_celerite_ is an algorithm for fast and scalable Gaussian Process (GP)
Regression in one dimension and this library, _celerite2_ is a re-write of the
original [celerite project](https://celerite.readthedocs.io) to improve
numerical stability and integration with various machine learning frameworks.
Documentation for this version can be found
[here](https://celerite2.readthedocs.io/en/latest/). This new implementation
includes interfaces in Python and C++, with full support for PyMC (v3 and v4)
and JAX.

This documentation won't teach you the fundamentals of GP modeling but the best
resource for learning about this is available for free online: [Rasmussen &
Williams (2006)](http://www.gaussianprocess.org/gpml/). Similarly, the
_celerite_ algorithm is restricted to a specific class of covariance functions
(see [the original paper](https://arxiv.org/abs/1703.09710) for more information
and [a recent generalization](https://arxiv.org/abs/2007.05799) for extensions
to structured two-dimensional data). If you need scalable GPs with more general
covariance functions, [GPyTorch](https://gpytorch.ai/) might be a good choice.
