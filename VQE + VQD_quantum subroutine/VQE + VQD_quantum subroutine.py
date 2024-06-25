# Authors: Yash Lokare

## Importing relevant libraries

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.spatial as spat
from scipy.stats import unitary_group
from scipy.stats import moment
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
from scipy.linalg import norm
import matplotlib.pyplot as plt
import math
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from qiskit import *
from qiskit import execute
from qiskit import transpiler
from qiskit import QuantumCircuit
from qiskit.circuit.add_control import add_control
from qiskit.extensions import UnitaryGate
from qiskit.circuit.reset import Reset
from qiskit_ibm_provider import IBMProvider
from qiskit.circuit.library.standard_gates import IGate, UGate, U3Gate
from qiskit.dagcircuit import DAGOpNode, DAGInNode
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
from qiskit.transpiler.passes.optimization import Optimize1qGates
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from math import fsum
from scipy.optimize import minimize
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers.models import BackendConfiguration
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.transpiler.passes import BasisTranslator
from qiskit import QuantumCircuit, IBMQ, transpile
from qiskit.circuit import Delay
from qiskit.circuit.library import XGate, YGate, ZGate, RXGate, RYGate, RZGate
from qiskit.transpiler.passes import ALAPSchedule
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.visualization import plot_histogram
from qiskit.visualization import timeline_drawer
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.providers.aer import AerSimulator

# Libraries for implementing the VQD algorithm
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import Sampler, Estimator
# from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Sampler, Session, Options
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import RealAmplitudes, TwoLocal, EfficientSU2
from qiskit.algorithms.optimizers import *
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.algorithms.eigensolvers import EigensolverResult, VQD
from qiskit_algorithms import VQE

# Import classical optimizers
from qiskit_algorithms.optimizers import SPSA, P_BFGS, COBYLA

# Import a fake backend and Qiskit simulators and/or noise libraries
from qiskit.providers.fake_provider import FakeMontreal, FakeMumbai, FakeWashington, FakeGuadalupe
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel

# Import algorithms global
from qiskit_algorithms.utils import algorithm_globals

## Save IBMQ account
IBMQ.save_account('') # your IBMQ Qiskit API token goes here

## Load IBMQ account
IBMQ.load_account()

## Importing IBMQ provider
provider = IBMQ.get_provider('ibm-q')

## Quantum (VQD) analysis
### Initialization
counts_noisy, values_noisy, steps_noisy = [], [], []
counts_noiseless, values_noiseless, steps_noiseless = [], [], []

def callback_noisy(eval_count, params, value, meta, step):
    # Function to store intermediate values from VQD (noisy simulation)

    counts_noisy.append(eval_count)
    values_noisy.append(value)
    steps_noisy.append(step)

def callback_noiseless(eval_count, params, value, meta, step):
    # Function to store intermediate values from VQD (noisy simulation)

    counts_noiseless.append(eval_count)
    values_noiseless.append(value)
    steps_noiseless.append(step)

### Running VQD
def run_VQD(matrix, k, RealAmps = True, bfgs = True, noise = True):
    # Function to implement the VQD algorithm

    dimension = matrix.shape[0]
    num_qubits = int(np.log2(dimension))

    # Define the qubit Hamiltonian
    qub_hamiltonian = SparsePauliOp.from_operator(matrix)

    # Define the circuit ansatz
    if RealAmps:
        ansatz = RealAmplitudes(num_qubits = num_qubits, reps = 4)
    else:
        ansatz = EfficientSU2(num_qubits = num_qubits, reps = 4)

    # Determine the number of variational parameters
    num_parameters = ansatz.num_parameters

    # Initialize the optimizer and the initial point
    initial_point = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

    # Initializing the estimator, sampler, and fidelity
    estimator = Estimator()
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler)

    # Get the classical optimizer
    if bfgs:
        optimizer = P_BFGS(maxfun = 1000)
    else:
        optimizer = SPSA(maxiter = 150)

    # Run the VQD simulation
    if not noise: # if False, run a noiseless simulation
        vqd = VQD(estimator = estimator, fidelity = fidelity, ansatz = ansatz, optimizer = optimizer, \
                  k = k, initial_point = initial_point, callback = callback_noiseless)
        result = vqd.compute_eigenvalues(operator = qub_hamiltonian)

        # Get the corresponding zeromode
        optimal_params = result.optimal_points
        zeromode_points = optimal_params[0]
        final_circuit = ansatz.assign_parameters(zeromode_points)
        zeromode_vqd = Statevector.from_instruction(final_circuit)

    else: # if True, run a noisy simulation
        # Get the backend
        backend = provider.get_backend('') # use your favorite backend

        # Get the noise characteristics
        noise_model = NoiseModel.from_backend(backend) # get the noise model
        coupling_map = backend.configuration().coupling_map # get the coupling map
        basis_gates = noise_model.basis_gates # get the basis gates

        # Initialize the noisy estimator for VQD
        noisy_estimator = AerEstimator(
        backend_options = {
            "coupling_map": coupling_map,
            "noise_model": noise_model,
            "basis_gates": basis_gates,
        },
        run_options = {"seed": 1, "shots": 3000},
        transpile_options = {"seed_transpiler": 1},
    )
        # Run the VQD algorithm
        vqd = VQD(estimator = noisy_estimator, fidelity = fidelity, ansatz = ansatz, optimizer = optimizer, \
                  k = k, initial_point = initial_point, callback = callback_noisy)
        result = vqd.compute_eigenvalues(operator = qub_hamiltonian)

        # Get the corresponding zeromode
        optimal_params = result.optimal_points
        zeromode_points = optimal_params[0]
        final_circuit = ansatz.assign_parameters(zeromode_points)
        zeromode_vqd = Statevector.from_instruction(final_circuit)

    return result.eigenvalues, optimal_params, zeromode_vqd, num_parameters

## Quantum (VQE) analysis
### Note: here, zeromode_nonhermitian corresponds to the zeromode of the perturbed L_{new} FPE operator

def run_VQE(matrix, zeromode_nonhermitian, noise = False):
    # Function to implement and/or run the VQE algorithm

    # Get the qubit operator
    qubit_operator = SparsePauliOp.from_operator(matrix)
    num_op_qubits = qubit_operator.num_qubits

    # Print the qubit operator
    print('The qubit operator is:')
    print(qubit_operator)
    print()

    # Initialize the ansatz, classical optimizer, and initial points
    ansatz = EfficientSU2(num_qubits = num_op_qubits, reps = 4)
    classical_optimizer = SPSA(maxiter = 150)
    initial_point = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

    # Initialize the ansatz with zeromode_nonhermitian
    ansatz.initialize(zeromode_nonhermitian, [i for i in range(num_op_qubits)])

    ## Print the number of variational parameters
    print('The number of variational parameters is:')
    print(ansatz.num_parameters)
    print()

    # Initialize random seed for the simulations
    seed = 14
    algorithm_globals.random_seed = seed

    # Run the VQE simulation
    if not noise: # if False, run a noiseless simulation

        # Initialize the Aer estimator
        noiseless_estimator = AerEstimator(
        run_options = {"seed": seed, "shots": 3000},
        transpile_options = {"seed_transpiler": seed},
    )
        # Run VQE
        vqe = VQE(estimator = noiseless_estimator, ansatz = ansatz, optimizer = classical_optimizer, \
                  initial_point = initial_point)
        result = vqe.compute_minimum_eigenvalue(operator = qubit_operator)

        # Get the corresponding zeromode
        optimal_params = result.optimal_point
        final_circuit = ansatz.assign_parameters(optimal_params)
        zeromode_vqe = Statevector.from_instruction(final_circuit)

    else: # if True, run a noisy simulation
        # Get the backend
        backend = provider.get_backend('') # use your favorite backend

        # Get the noise characteristics
        noise_model = NoiseModel.from_backend(backend) # get the noise model
        coupling_map = backend.configuration().coupling_map # get the coupling map
        basis_gates = noise_model.basis_gates # get the basis gates

        # Initialize the noisy estimator for VQE
        noisy_estimator = AerEstimator(
        backend_options = {
            "coupling_map": coupling_map,
            "noise_model": noise_model,
            "basis_gates": basis_gates,
        },
        run_options = {"seed": 1, "shots": 3000},
        transpile_options = {"seed_transpiler": 1},
    )
        # Run VQE
        vqe = VQE(estimator = noisy_estimator, ansatz = ansatz, optimizer = classical_optimizer, \
                  initial_point = initial_point)
        result = vqe.compute_minimum_eigenvalue(operator = qubit_operator)

        # Get the corresponding zeromode
        optimal_params = result.optimal_point
        final_circuit = ansatz.assign_parameters(optimal_params)
        zeromode_vqe = Statevector.from_instruction(final_circuit)

    return optimal_params, zeromode_vqe, ansatz.num_parameters