# Classical Nonlinear Systems

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Qiskit](https://img.shields.io/badge/Qiskit-%236929C4.svg?logo=Qiskit&logoColor=white)
![Static Badge](https://img.shields.io/badge/paddle-quantum?style=flat&logoColor=%23FF0000&label=PaddleQuantum&link=https%3A%2F%2Fgithub.com%2FPaddlePaddle%2FQuantum)

</div>

This repository contains code to implement the classical (construction of the Hermitian form of the Fokker-Planck operator matrix) and quantum (QPE + VQSVD) subroutines for the paper ***Steady-State Statistics of Classical Nonlinear Dynamical Systems from Noisy Intermediate-Scale Quantum Devices***. 

**Authors**: Y. M. Lokare, D. Wei, L. Chan, B. M. Rubenstein, and J. B. Marston. 

The following versions of Qiskit software are needed to run numerical simulations: 
- qiskit-terra: 0.24.0
- qiskit-aer: 0.12.0
- qiskit-ibmq-provider: 0.20.2
- qiskit: 0.43.0

Version of Paddle Quantum required to run numerical simulations: **2.4.0**; refer [PaddleQuantum](https://github.com/PaddlePaddle/Quantum) for more details. 

We use the ***poetry*** package management system to manage all package dependencies. 

**Note**: The DD analysis and/or results contained in this repository are irrelevant for the present study. 
