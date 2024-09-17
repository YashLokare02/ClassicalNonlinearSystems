## Importing relevant libraries

import numpy as np
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
from qiskit.circuit.add_control import add_control
from qiskit.extensions import UnitaryGate
from qiskit.circuit.reset import Reset
from math import fsum
from scipy.optimize import minimize
from qiskit.providers.models import BackendConfiguration
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit import QuantumCircuit, IBMQ, transpile
from qiskit.visualization import plot_histogram
from qiskit.visualization import timeline_drawer
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.providers.aer import AerSimulator

# Qiskit Runtime
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session, Options
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.utils import algorithms_global
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit_algorithms.optimizers import SPSA, SLSQP, SNOBFIT, IMFIL, COBYLA, BOBYQA
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp, Statevector
import csv

# Import the symbolic tools library
import sympy as sym
from sympy import symbols, Symbol

# Sampler and Estimator primitives
from qiskit_ibm_runtime import SamplerV2 as samplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# For VQE
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import Estimator as EstimatorOld
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import RealAmplitudes, TwoLocal, EfficientSU2
from qiskit_algorithms.optimizers import *
from qiskit_algorithms.state_fidelities import ComputeUncompute

## More libraries can be added, as required

## Load Qiskit Runtime
## First, load the IBM API token (unique for every user)

## Loading Qiskit Runtime Service
service = QiskitRuntimeService(channel = "ibm_quantum", token = '') # or on the cloud
backend = service.backend('') # load preferred backend

## Quantum algorithms
### QPE
### Constructing the QPE circuit

# Note: The following is the function we use to construct the QPE circuit for simulation results reported ...
# ... in the paper
def inverse_qft(qc, precision_qubits, with_inverse_barriers = True, swap = True):
    """
    Purpose: Implements inverse QFT
    Input: qc -> circuit to append the inverse QFT circuit to
           precision_qubits -> Implement
    """
    n = len(precision_qubits)

    if swap:
        for qubit in range(n//2):
            qc.swap(precision_qubits[qubit], precision_qubits[n - qubit - 1])
    if with_inverse_barriers:
        qc.barrier() ## for visualization purposes

    for target in reversed(range(n)):
        for control in reversed(range(target + 1, n)):
            k = control - target + 1
            angle = -2 * math.pi/(2**k)
            qc.cp(angle, precision_qubits[control], precision_qubits[target], label = f"R_{k}_dag") ## apply controlled phase gates
        qc.h(precision_qubits[target])

        if with_inverse_barriers and target != 0:
            qc.barrier() ## for visualization purposes

def controlled_unitary(circ, control, target_qubits, unitary):
    """
    Purpose: Implement controlled unitary gate
    Input: circ -> circuit to append the controlled unitary onto
           control -> List containing the control qubit
           target_qubits -> List containing the target qubits
           unitary -> Numpy array representing unitary matrix that will be turned into a gate
    Output: circ -> circuit with the the controlled unitary gate
    """
    U_gate = UnitaryGate(unitary)
    U_gate_ctrl = U_gate.control(1)
    targets = [control] + target_qubits
    circ.append(U_gate_ctrl, targets)

def qpe_without_DD(qpe_circ, precision_qubits, query_qubits, unitary, angles = None, control_unitary = True, \
                  with_inverse_barriers = True, measure = True):
    """
    Purpose: Construct the QPE circuit for a specific unitary operator without applying DD to the circuit
    Input: precision_qubits -> List containing the precision qubits
           query_qubits -> List containing the query qubits
           unitary -> the unitary operator used for the QPE
           control_unitary -> Boolean value representing the implementation of controlled unitary in each iteration.
                              Default is True, meaning that the power to the unitary operator is found, thus reducing
                              the number of gates needed.
    Output: qpe_circ -> the QPE Circuit
    """
    num_precision_qubits = len(precision_qubits)
    num_query_qubits = len(query_qubits)
    total_qubits = num_precision_qubits + num_query_qubits

    ## Apply Hadamard gates to all precision qubits
    qpe_circ.h(precision_qubits)

    if isinstance(angles, np.ndarray):
        assert len(angles) == len(query_qubits)
        for i in range(len(query_qubits)):
            if angles[i] != 0:
                qpe_circ.ry(angles[i] * 2, len(query_qubits) + len(precision_qubits) - 1 - i)

    for ii, qubit in enumerate(reversed(precision_qubits)):
        power = ii
        if control_unitary:
            Uexp = np.linalg.matrix_power(unitary, 2 ** power)
            controlled_unitary(qpe_circ, qubit, query_qubits, Uexp)

        else:
            for _ in range(2 ** power):
                # Apply controlled unitary operations
                controlled_unitary(qpe_circ, qubit, query_qubits, unitary)

    qpe_circ.barrier()  # for visualization purposes
    inverse_qft(qpe_circ, precision_qubits, with_inverse_barriers = with_inverse_barriers)
    qpe_circ.barrier()  # for visualization purposes

    ## Add measurements
    if measure:
        for n in range(total_qubits):
            qpe_circ.measure(n, n)

################################################################################################################################
# The following functions implement different versions of the QPE circuit

def mqpe(circ, unitary, precision_number, query_number, precision_verification_factor, query_verification_factor, \
        control_unitary = True, with_inverse_barriers = True, measure = True):

    num_precision_qubits = precision_number * precision_verification_factor
    num_query_qubits = query_number * query_verification_factor
    total_qubits = num_precision_qubits + num_query_qubits

    ## Initializing lists containing the precision and query qubits
    precision_qubits = list(np.arange(0, num_precision_qubits, precision_verification_factor))
    query_qubits = list(np.arange(num_precision_qubits, total_qubits, query_verification_factor))

    ## Applying the Quantum Phase Estimation algorithm
    qpe_without_DD(circ, precision_qubits, query_qubits, unitary, control_unitary = control_unitary, \
       with_inverse_barriers = with_inverse_barriers, measure = False)

    circ.barrier() ## for visualization purposes

    ## Adding relevant gates for post-processing
    for control in precision_qubits:
        for j in range(1, precision_verification_factor):
            circ.cx(control, control + j)

    circ.barrier() ## for visualization purposes

    for control in query_qubits:
        for j in range(1, query_verification_factor):
            circ.cx(control, control + j)

    circ.barrier() ## for visualization purposes

    ## Adding measurements
    if measure:
        for n in range(total_qubits):
            circ.measure(n, n)

def mqpe2(circ, unitary, precision_number, query_number, precision_verification_factor, query_verification_factor, \
         control_unitary = True, with_inverse_barriers = True, measure = True, \
         measure_post_first_iter = True, measure_all_post_first = False, min_power = 0, with_toffoli = False):

    ## Implement a check condition to ensure query verification exists
    if measure_post_first_iter:
        assert query_verification_factor > 1, "Cannot measure_post_first_iter if there is no query verification"

    num_precision_qubits = precision_number * precision_verification_factor
    num_query_qubits = query_number * query_verification_factor
    total_qubits = num_precision_qubits + num_query_qubits

    ## Initializing lists containing the precision and query qubits
    precision_qubits = list(np.arange(0, num_precision_qubits, precision_verification_factor))
    query_qubits = list(np.arange(num_precision_qubits, total_qubits, query_verification_factor))

    ## Add Hadamard gates to all precision qubits
    circ.h(precision_qubits) # note: precision_qubits is a list

    ## Apply controlled unitary operations and add relevant gates for post-processing
    for ii, qubit in enumerate(precision_qubits):
        power = ii + min_power

        if control_unitary:
            Uexp = np.linalg.matrix_power(unitary, 2 ** power)
            controlled_unitary(circ, qubit, query_qubits, Uexp)

        else:
            for _ in range(2**power):
                controlled_unitary(circ, qubit, query_qubits, unitary)

        if power == 0 and measure_post_first_iter:
            if measure_all_post_first:
                for control in query_qubits:
                    if with_toffoli:
                        circ.toffoli(qubit, control, control + j)
                    else:
                        circ.cx(control, control + j)

            else:
                for control in query_qubits:
                    if with_toffoli:
                        circ.toffoli(qubit, control, control + 1)
                    else:
                        circ.cx(control, control + 1)

        if measure_post_first_iter:
            if not measure_all_post_first:
                for control in query_qubits:
                    for j in range(2, query_verification_factor):
                        if with_toffoli:
                            circ.toffoli(precision_qubits[-1], control, control + j)
                        else:
                            circ.cx(control, control + j)
        else:
            for control in query_qubits:
                for j in range(1, query_verification_factor):
                    if with_toffoli:
                        circ.toffoli(precision_qubits[-1], control, control + j)
                    else:
                        circ.cx(control, control + j)

        ## Applying the inverse QFT to the circuit
        inverse_qft(circ, precision_qubits, with_inverse_barriers = with_inverse_barriers, swap = False)

        for control in precision_qubits:
            for j in range(1, query_verification_factor):
                circ.cx(control, control + j)

        circ.barrier() ## for visualization purposes

        ## Adding measurements
        if measure:
            for n in range(total_qubits):
                circ.measure(n, n)

def mqpe3(circ, unitary, precision_number, query_number, precision_verification_factor, query_verification_factor, \
         control_unitary = True, with_inverse_barriers = True, measure = True, measure_post_first_iter = True):

    ## Implement a check condition to ensure query verification exists
    if measure_post_first_iter:
        assert query_verification_factor > 1, "Cannot measure_post_first_iter if there is no query verification"

    num_precision_qubits = precision_number * precision_verification_factor
    num_query_qubits = query_number * query_verification_factor
    total_qubits = num_precision_qubits + num_query_qubits

    ## Initializing lists containing the precision and query qubits
    precision_qubits = list(np.arange(0, num_precision_qubits, precision_verification_factor))
    query_qubits = list(np.arange(num_precision_qubits, total_qubits, query_verification_factor))

    ## Add Hadamard gates to all precision qubits
    circ.h(precision_qubits) # note: precision_qubits is a list

    for ii, qubit in enumerate(precision_qubits):
        power = ii

        if control_unitary:
            Uexp = np.linalg.matrix_power(unitary, 2 ** power)
            controlled_unitary(circ, qubit, query_qubits, Uexp)
        else:
            for _ in range(2 ** power):
                controlled_unitary(circ, qubit, query_qubits, unitary)

        if power == 0 and measure_post_first_iter:
            for control in query_qubits:
                for j in range(1, query_verification_factor):
                    circ.cx(control, control + j)

    circ.barrier() ## for visualization purposes

    ## Applying inverse QFT to the circuit
    inverse_qft(circ, precision_qubits, with_inverse_barriers = with_inverse_barriers, swap = False)

    circ.barrier() ## for visualization purposes

    for control in precision_qubits:
        for j in range(1, precision_verification_factor):
            circ.cx(control, control + j)

    if not measure_post_first_iter:
        for control in query_qubits:
            for j in range(1, precision_verification_factor):
                circ.cx(control, control + j)

    circ.barrier() ## for visualization purposes

    ## Add measurements
    if measure:
        for n in range(total_qubits):
            circ.measure(n, n)

### Getting qubit arrays and controlled unitaries

def get_unitary(matrix, add_half = False):
    """
    Purpose: given a matrix, returns the unitary, hermitian matrix to be diagonalized
    Input: matrix -> the matrix to be diagonalized
    Output: U -> the unitary matrix
            nqubits -> the number of qubis needed to represent the basis of U
            dimension -> the dimension of the original matrix
    """
    assert matrix.ndim == 2, "Error: Only a matrix maybe processed"
    assert matrix.shape[0] == matrix.shape[1], "Error: Only a square matrix maybe processed"

    if np.any(np.transpose(matrix) != matrix):
        matrix_T = np.transpose(matrix)
        matrix = np.dot(matrix_T, matrix)

    ## Finding the dimension of the matrix
    dimension_hermitian = matrix.shape[0]

    ## Finding the number of qubits required to represent the matrix
    nqubits = int(np.ceil(np.log2(dimension_hermitian)))

    ## Construct the relevant matrix
    op_dim = 2 ** nqubits
    op = np.eye(op_dim)
    op[0:dimension_hermitian, 0:dimension_hermitian] = np.copy(matrix)

    if add_half:
        op = op + np.pi * np.eye(op.shape[0])

    U = la.expm(1j*op)

    # Get the dimensions of the unitary matrix
    dimension = U.shape[0]

    return U, nqubits, dimension

def get_qubits(n_precision_qubits, n_query_qubits):
    """
    Purpose: defines the precision and query qubits
    Input: n_precision_qubits -> the number of precision qubits used by the QPE
           n_query_qubits -> the number of query qubits used by the QPE
    """
    total_qubits = list(range(n_precision_qubits + n_query_qubits))

    precision_qubits = total_qubits[0:n_precision_qubits]
    query_qubits = total_qubits[n_precision_qubits:]

    return precision_qubits, query_qubits

def unitary_swap(num_query_qubits, state1, state2):
    U = np.eye(2**num_query_qubits)

    temp = np.copy(U[:, state1])
    U[:, state1] = U[:, state2]
    U[:, state2] = temp

    return U

### Post-processing QPE results

def substring(key, precision_qubits):
    short_key = ""
    for idx in precision_qubits:
        short_key = short_key + key[idx]

    return short_key

def binaryToDecimal(binary):
    """
    Purpose: Converts binary fractional to decimal fractional
    Input: binary -> binary string to be converted to decimal fractional
    Output: fracDecimal -> decimal fractional
    """
    length = len(binary)
    fracDecimal = 0
    twos = 2

    for ii in range(length):
        fracDecimal += (ord(binary[ii]) - ord("0")) / twos
        twos *= 2.0
    return fracDecimal

def get_qpe_phases(measurement_counts, precision_qubits, items_to_keep = 1):

    """
Purpose: find the phases determined by the QPE algorithm
Input: measurement_counts -> measurement results from device run
       precision_qubits -> List of qubits corresponding to the precision qubits
       items_to_keep -> number of items to return (topmost measurement counts for precision qubits)
output: phases_decimal -> the phases measured
        precision_results_dic -> contains the measurement outcomes (bit strings) and the corresponding counts
"""

    n = len(precision_qubits)
    bitstrings_precision_register = [
        substring(key, precision_qubits) for key in measurement_counts.keys()
    ]
    bitstrings_precision_register_set = set(bitstrings_precision_register)
    bitstrings_precision_register_list = list(bitstrings_precision_register_set)
    precision_results_dic = {key: 0 for key in bitstrings_precision_register_list}

    for key in measurement_counts.keys():
        counts = measurement_counts[key]
        count_key = substring(key, precision_qubits)
        precision_results_dic[count_key] += counts

    c = Counter(precision_results_dic)
    topmost = c.most_common(items_to_keep)
    phases_decimal = [binaryToDecimal(item[0]) for item in topmost]
    phases_actual = [(2 * np.pi * phase)/2 ** n for phase in phases_decimal] # extracts the actual phases

    return phases_decimal, phases_actual, precision_results_dic

"""
Purpose: given a (nonhermitian) matrix, returns a hermitian matrix
Input: L -> the matrix to be made hermitian
Output: hermitian matrix using L^dag * L
"""
## Note: this is for calcuations with the QPE algorithm
def make_hermitian(L):

    L_T = np.conj(L.T)
    return np.dot(L_T, L)

def post_process_results(results, precision_qubits):
    # Dictionary for all the measurements and their number of counts
    # Note: here, results is a result cache obtained from running the QPE algorithm ...
    # ... on the constructed circuit

    num_precision_qubits = len(precision_qubits)

    counts = results.get_counts()

    measurement_counts = {}

    for key in counts.keys():

        reversed_key = key[::-1]
        precision_key = reversed_key[:num_precision_qubits]
        reversed_query_key = reversed_key[num_precision_qubits:]
        query_key = reversed_query_key[::-1]

        new_key = precision_key + query_key

        measurement_counts[new_key] = counts[key]

    phases_decimal, phases_actual, precision_results_dic = get_qpe_phases(measurement_counts, precision_qubits, \
                                                                          2 ** num_precision_qubits)

    print('\nPhases:', phases_decimal)

    eigenvalues = [np.exp(2 * np.pi * 1j * phase) for phase in phases_decimal]

    qpe_cache = {
            "phases_decimal": phases_decimal,
            "precision_results_dic": precision_results_dic,
            "eigenvalues": eigenvalues,
            "measurement_counts": measurement_counts,
        }
    return qpe_cache

def postprocess_mqpe_results(results, precision_number, query_number, precision_verification_factor, \
                            query_verification_factor, unanimous = False):

    num_precision_qubits = precision_number * precision_verification_factor
    num_query_qubits = query_number * query_verification_factor

    counts = results.get_counts()
    measurement_counts = {}

    for key in counts_key():

        keep = True
        query_key = key[:num_query_qubits]
        reversed_precision_key = key[num_query_qubits:]
        precision_key = reversed_precision_key[::-1]
        precision_key_verified = ""

        ## Key generation
        for i in range(precision_number):
            n0 = 0
            n1 = 0
            for j in range(precision_number):
                if precision_key[i * precision_verification_factor + j] == "0":
                    n0 = n0 + 1
                else:
                    n1 = n1 + 1
            if n0 > n1:
                precision_key_verified = precision_key_verified + "0"
            else:
                precision_key_verified = precision_key_verified + "1"
            if unanimous and n0 != 0 and n1 != 0:
                keep = False

        query_key_verified = ""
        for i in range(query_number):
            n0 = 0
            n1 = 0
            for j in range(query_number):
                if query_key[i * query_verification_factor + j] == "0":
                    n0 = n0 + 1
                else:
                    n1 = n1 + 1
            if n0 > n1:
                query_key_verified = query_key_verified + "0"
            else:
                query_key_verified = query_key_verified + "1"
            if unanimous and n0 != 0 and n1 != 0:
                keep = False

        new_key = precision_key_verified + query_key_verified

        if keep:
            if new_key in measurement_counts:
                measurement_counts[new_key] = measurement_counts[new_key] + counts[key]
            else:
                measurement_counts[new_key] = counts[key]

    return measurement_counts

def find_probability(eigenvector_raw):
    """
    Purpose: Find the probability associated with each basis of an eigenvector
    Input: eigenvector_raw -> Numpy array documenting the number of times each basis is detected within the eigenvector
    Output: eigenvector_prob -> Numpy array documenting the probability of detecting each basis
    """
    count_total = np.sum(eigenvector_raw)
    eigenvector_prob = eigenvector_raw / count_total

    return eigenvector_prob

def find_amplitude(eigenvector_prob):
    """
    Purpose: Finding the probability amplitude of each basis using quantum mechanics
    Input: eigenvector_prob -> Numpy array documenting the probability that each basis is measured
    Output: eigenvector -> Numpy array representing the eigenvector
    """
    eigenvector = np.sqrt(eigenvector_prob)
    return eigenvector

def normalize_eigenvector(vector):
    """
    Purpose: Normalizes a vector such that its norm is 1
    Input: vector -> The vector to be normalized
    Output: vector -> The normalized vector
    """
    L2 = np.sum(np.square(vector))
    vector = vector / np.sqrt(L2)

    return vector

def find_eigenvector(zeromode_classic, result, num_precision_qubit, num_query_qubit, make_even = False, target_phase = 0):
    """
    Purpose: Given the results, format the count of each basis
    Input: result -> Dictionary containing the results from the ciruit
           num_query_qubit -> number of target qubits used by the QPE circuit
           num_precision_qubit -> number of precision qubits used by the QPE circuit
           target_phase -> the phase whose eigenvector we are looking for. By default set to 0
    Output: eigenvector -> the eigenvector of the phase
            total_counts -> the number of iterations that yielded the desired phase
    """
    # TODO: Generalize function to target alternative phases
    total_counts = 0

    counts = result['measurement_counts']

    assert num_query_qubit < 10, "Error: the code is only programmed for num_vec_bit < 10"

    eigenvector_raw = np.zeros((2**num_query_qubit, 1))

    eigenvalue_string = ''

    ## This part is only applicable if we are intersted in phase = 0
    for i in range(num_precision_qubit):
        eigenvalue_string = eigenvalue_string + '0'

    for i in range(2**num_query_qubit):
        bformat = '{0:0' + str(num_query_qubit) + 'b}'
        eigenvector_string = eigenvalue_string + bformat.format(i)

        if eigenvector_string in counts.keys():
            eigenvector_raw[i] = counts[eigenvector_string]
            total_counts = total_counts + counts[eigenvector_string]
        else:
            eigenvector_raw[i] = 0

        if make_even and i%2 != 0:
            eigenvector_raw[i] = 0

    eigenvector_prob = find_probability(eigenvector_raw)

    eigenvector = find_amplitude(eigenvector_prob)

    return eigenvector, eigenvector_prob, total_counts

def find_eigenvector_with_bucket(result, num_precision_qubit, num_query_qubit, bucket, make_even = False, target_phase = 0):
    """
    Purpose: Given the results, format the count of each basis
    Input: result -> Dictionary containing the results from the ciruit
           num_query_qubit -> number of target qubits used by the QPE circuit
           num_precision_qubit -> number of precision qubits used by the QPE circuit
           target_phase -> the phase whose eigenvector we are looking for. By default set to 0
    Output: eigenvector -> the eigenvector of the phase
            total_counts -> the number of iterations that yielded the desired phase
    """
    # TODO: Generalize function to target alternative phases
    total_counts = 0

    counts = result['measurement_counts']

    assert bucket.ndim == num_query_qubit, "Error: bucket dimension is incorrect"
    assert num_query_qubit < 10, "Error: the code is only programmed for num_vec_bit < 10"

    eigenvector_raw = np.zeros((2**num_query_qubit, 1))
    eigenvalue_string = ''
    reshaped = np.reshape(bucket, (2 ** num_query_qubit, ))

    ## This part is only applicable if we are intersted in phase = 0
    for i in range(num_precision_qubit):
        eigenvalue_string = eigenvalue_string + '0'

    for i in range(2**num_query_qubit):
        bformat = '{0:0' + str(num_query_qubit) + 'b}'
        eigenvector_string = eigenvalue_string + bformat.format(i)

        if eigenvector_string in counts.keys():
            eigenvector_raw[i] = counts[eigenvector_string]
            total_counts = total_counts + counts[eigenvector_string]
        else:
            eigenvector_raw[i] = 0

        if make_even and i%2 != 0:
            eigenvector_raw[i] = 0

        reshaped[i] = reshaped[i] + eigenvector_raw[i]

    bucket = np.reshape(reshaped, bucket.shape)

    eigenvector_prob = find_probability(eigenvector_raw)

    eigenvector = find_amplitude(eigenvector_prob)

    return eigenvector, total_counts

def approximate_tensor_product(bucket):
    total = np.sum(bucket)
    num_query_qubit = bucket.ndim
    angles = np.zeros((num_query_qubit, ))

    angles[0] = np.sum(bucket[0, :, :, :])
    angles[1] = np.sum(bucket[:, 0, :, :])
    angles[2] = np.sum(bucket[:, :, 0, :])
    angles[3] = np.sum(bucket[:, :, :, 0])

    angles = angles / total
    angles = np.sqrt(angles)
    angles = np.arccos(angles)

    return angles

def get_register_counts(result_cache, num_precision_qubits):
    counts = result_cache['measurement_counts']
    phase_counts = {}
    eigenvector_counts = {}

    for key in counts.keys():
        phase_measurement = key[:num_precision_qubits]
        num_counts = counts[key]
        if phase_measurement in phase_counts:
            phase_counts[phase_measurement] = phase_counts[phase_measurement] + num_counts
        else:
            phase_counts[phase_measurement] = num_counts

        eigenvector_measurement = key[num_precision_qubits:]
        if eigenvector_measurement in eigenvector_counts:
            eigenvector_counts[eigenvector_measurement] = eigenvector_counts[eigenvector_measurement] + num_counts
        else:
            eigenvector_counts[eigenvector_measurement] = num_counts
    return phase_counts, eigenvector_counts

### Further analysis of the zeromodes

def fidelity_check(qeigvals, aeigvals, runit): # runit -> round off to a certain precision
    qeigvals.sort()
    aeigvals.sort()
    qmat = np.round(qeigvals, decimals = runit)
    amat = np.round(aeigvals, decimals = runit)

    qmat = np.real(qmat) # real part of the quantum eigenvalues
    amat = np.real(amat) # real part of the classical eigenvalues

    cosine_similarity = 1 - spat.distance.cosine(qmat, amat)

    return cosine_similarity

def get_correlation(zeromode_classic, zeromode_quantum, runit):
    # Function to compute the correlation between the classical and QPE zeromodes
    # runit -> round off to a certain precision

    zeromode_classic = np.round(zeromode_classic, decimals = runit)
    zeromode_quantum = np.round(zeromode_quantum, decimals = runit)

    # Flatten the 2D arrays
    zeromode_classic = np.reshape(zeromode_classic, len(zeromode_classic))
    zeromode_quantum = np.reshape(zeromode_quantum, len(zeromode_quantum))

    # Take the real parts
    zeromode_quantum = np.real(zeromode_quantum)
    zeromode_classic = np.real(zeromode_classic)

    # Compute the correlation
    correlation_zeromode = np.corrcoef(zeromode_classic, zeromode_quantum)[0, 1]

    return correlation_zeromode

def get_overlap(zeromode_classic, zeromode_quantum, runit):
    # Function to compute the overlap between the classical and VQSVD zeromodes

    zeromode_classic = np.round(zeromode_classic, decimals = runit)
    zeromode_quantum = np.round(zeromode_quantum, decimals = runit)

    zeromode_qpe_transpose = np.real(np.transpose(zeromode_quantum))
    overlap = np.dot(zeromode_qpe_transpose, zeromode_classic)[0, 0]

    return overlap

def get_similarity(zeromode_classic, zeromode_quantum, runit):
    # Function to compute the cosine similarity between the classical and QPE zeromodes

    zeromode_classic = np.round(zeromode_classic, decimals = runit)
    zeromode_quantum = np.round(zeromode_quantum, decimals = runit)

    # Flatten the 2D arrays
    zeromode_classic = np.reshape(zeromode_classic, len(zeromode_classic))
    zeromode_quantum = np.reshape(zeromode_quantum, len(zeromode_quantum))

    # Converting to list
    zeromode_classic.tolist()
    zeromode_quantum.tolist()

    cosine_similarity_score = 1 - spat.distance.cosine(zeromode_classic, zeromode_quantum)

    return cosine_similarity_score

### Euclidean distance calculations

def euclidean_distance(zeromode_classic, zeromode_qpe, runit):
    # Function to compute the Euclidean distance between the classical and VQDSVD zeromodes

    # Rounding off
    zeromode_classic = np.round(zeromode_classic, decimals = runit)
    zeromode_qpe = np.round(zeromode_qpe, decimals = runit)

    # Convert zeromodes to lists
    zeromode_classic.tolist()
    zeromode_qpe.tolist()

    assert len(zeromode_classic) == len(zeromode_qpe), "The zeromodes must be of equal length"

    # Compute the Euclidean distance
    n = len(zeromode_classic)
    sum_vec = 0

    for i in range(n):
        sum_vec += (zeromode_classic[i] - zeromode_qpe[i]) ** 2

    return np.sqrt(sum_vec)

### Implementation

def qpe_implementation_DD(A, U, zeromode_classic, num_precision_qubits, num_query_qubits, dimension, sequence, \
                          shots, optimization_level, runit, noise, use_DD, \
                          control_unitary, use_noisy_backend = True, make_even = True):

    # Note: use_DD -> if True, apply DD to the circuit
    # use_noisy_backend -> if True, run the circuit on the IBMQ backend; else, use QASM simulator
    # make_even -> if True, set all odd orders of the Hermite polynomials to zero
    # runit -> rounding off precision

     # Get the arrays containing all precision and query qubits
    precision_qubits, query_qubits = get_qubits(num_precision_qubits, num_query_qubits)
    total_qubits = num_precision_qubits + num_query_qubits

    # Construct the circuit
    qr = QuantumRegister(total_qubits)
    cr = ClassicalRegister(total_qubits)
    circuit = QuantumCircuit(qr, cr)

    # Apply QPE to the circuit
    qpe_without_DD(circuit, precision_qubits, query_qubits, U, angles = None, control_unitary = True, \
                   with_inverse_barriers = True, measure = True)

    if use_noisy_backend:
        backend = backend
    else:
        backend = Aer.get_backend('qasm_simulator') # to run noiseless simulations, import Aer (might need to downgrade the Qiskit version)

    if noise:
        pass_manager = generate_preset_pass_manager(backend = backend, optimization_level = optimization_level, seed_transpiler = 1)
        isa_qc = pass_manager.run(circuit)

        sampler = samplerV2(backend = backend)
        job = sampler.run([(isa_qc,)], shots = shots)
        counts = job.result()[0].data.meas.get_counts()

    else:
        # Transpile and execute the circuit on the backend (might need to downgrade the Qiskit version)
        circuit = transpile(circuit, backend = backend, optimization_level = optimization_level, \
                            seed_transpiler = 1)
        job = backend.run(circuit, shots = shots)

        # Obtain all results
        result = job.result()
        counts = result.get_counts()
        time = result.time_taken

    ## Obtain quantum results
    # Get the QPE phases in decimal format
    phases_decimal, phases_actual, _ = get_qpe_phases(counts, precision_qubits, dimension) # as the measurement outcomes with the ...
    # ... highest occurrences are the most dominant eigenvectors (corresponding to U)

    eigenvalues = [np.exp(1j * phase) for phase in phases_decimal] # eigenvalues of the unitary matrix

    phases_qpe = np.array([])

    # Computing phases in Euler representation
    qpe_cache = post_process_results(result, precision_qubits)
    qpe_eigenvalues = qpe_cache['eigenvalues']

    for z in qpe_eigenvalues:
        a, phase = euler_representation(z)
        phases_qpe = np.append(phases_qpe, [phase])

    # Computing the eigenvalues of the Hermitian matrix
    hermitian_eigenvalues = [-1j * np.log(eigenvalue) for eigenvalue in eigenvalues]

    ## Compute the zeromode using singular value decomposition
    # Identify the minimum eigenvalue coming from QPE
    min_idx = np.argmin(np.real(eigenvalues) ** 2)
    min_eigenvalue = eigenvalues[min_idx]

    # Use SVD to compute the corresponding zeromode
    diag_matrix = U - min_eigenvalue * np.identity(dimension)
    _, _, solutions = la.svd(diag_matrix.conj().T)
    zeromode_qpe = solutions[-1, :] # extract the relevant solution
#     zeromode_qpe = -zeromode_qpe # per the SVD convention in Python

    # Reshape the zeromode as a column vector for better visualization and/or comparison
    zeromode_qpe = np.reshape(zeromode_qpe, (zeromode_qpe.shape[0], 1))

    # Identifying the zeromode and estimating the hit rate (with the wrong signs for the amplitudes, though)
    zeromode_qpe_sign, zeromode_prob, n_hit = find_eigenvector(zeromode_classic, qpe_cache, num_precision_qubits, \
                                                               num_query_qubits, make_even = make_even)
    hit_rate = n_hit/shots

    # Computing the overlap between the classical and QPE zeromode
    if len(zeromode_qpe_sign) == len(zeromode_classic):
        zeromode_qpe_transpose = np.real(np.transpose(zeromode_qpe_sign))
        overlap = np.dot(zeromode_qpe_transpose, zeromode_classic)[0, 0]

    # Compute the similarity between the classical zeromode and the one obtained using SVD
    similarity_score = get_similarity(zeromode_classic, zeromode_qpe, runit = runit)

    return U, result, counts, time, phases_actual, eigenvalues, qpe_eigenvalues, hermitian_eigenvalues, min_eigenvalue, phases_qpe, zeromode_qpe, zeromode_qpe_sign, zeromode_prob, hit_rate, overlap, similarity_score

## VQE implementation

def run_vqe(matrix, iteration, noise = False):
    # Function to run the VQE algorithm

    # For the 6 x 6 case, do the following:
    # matrix_new = np.zeros((8, 8)) # convert to a 3-qubit operator
    # matrix_new[:6, :6] = np.copy(matrix)

    # # Convert the lower 2 blocks to 2 x 2 identities (ensures that the 6-component zeromode is preserved)
    # matrix_new[6, 6] = 1
    # matrix_new[7, 7] = 1

    # In the following, for the 6 x 6 case, use 'matrix_new' instead of 'matrix'
    dimension = matrix.shape[0]
    num_qubits = int(np.log2(dimension))

    # Define the qubit Hamiltonian
    qub_hamiltonian = SparsePauliOp.from_operator(matrix)

    # Compute using NumPyMinimumEigensolver
    sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qub_hamiltonian)

    # Define the circuit ansatz
    ansatz = RealAmplitudes(num_qubits = num_qubits, reps = 4)

    # Initialize the optimizer and the initial point
    np.random.seed(6)
    algorithms_global.random_seed = 6 
    initial_point = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

    # Initializing the estimator
    if noise is False:
        estimator = EstimatorOld() # noiseless Estimator
    else:
        estimator = Estimator()

    # For noisy experiments, instantiate the Estimator with the 'backend' option (and import relevant backend configurations)
    # Also, initialize Options() as follows:
    # options = Options()
    # options.optimization_level = '' # set the optimization level
    # options.resilience_level = '' # set the resilience level; depending on error mitigation protocol to be employed
    # options.execution.shots = '' # set the number of measurement shots to be executed
    # options.seed_simulator = '' # set the random seed for the simulator

    # Note: resilience_level = 2 switches on ZNE and readout error mitigation
    #       resilience_level = 1 switches on TREX

    # Get the classical optimizer (or any optimizer, as desired)
    optimizer = IMFIL(maxiter = iteration)

    # Run the VQE algorithm
    # Define a class to store intermediate values
    @dataclass
    class VQELog:
        values: list
        parameters: list

        def update(self, count, parameters, mean, _metadata):
            self.values.append(mean)
            self.parameters.append(parameters)

    log = VQELog([], [])

    vqe = VQE(estimator,
          ansatz, optimizer, initial_point = initial_point, callback = log.update)
    result = vqe.compute_minimum_eigenvalue(qub_hamiltonian)

    plt.rcParams["font.size"] = 14
    plt.figure(dpi = 600)
    plt.figure(figsize=(12, 6))
    plt.plot(log.values, label="VQE")
    plt.axhline(y=sol.eigenvalue, color="tab:red", ls="--", label="Target")
    plt.legend(loc="best")
    plt.xlabel("Iteration")
    plt.ylabel(r'$\langle H \rangle$')
    plt.title("VQE convergence")
    plt.show()

    # Estimate the zeromode
    optimal_params = result.optimal_point
    final_circuit = ansatz.assign_parameters(optimal_params)
    zeromode_vqe = Statevector.from_instruction(final_circuit)

    return sol.eigenvaiue, result.optimal_value, zeromode_vqe

### Computing Hermitian deviation of the zeromode and relative errors in <x^2>

def expectation_value_matrix_squared(A, v):
    """
    Compute the expectation value of a matrix squared given an eigenvector.

    Parameters:
    A (np.ndarray): The matrix A.
    v (np.ndarray): The eigenvector v corresponding to A.

    Returns:
    float: The expectation value <v|A^2|v>.
    """
    # # Normalize the eigenvector
    # v = v / np.linalg.norm(v)

    # Compute A^2
    A_squared = np.dot(A, A)

    # Compute the expectation value <v|A^2|v>
    expectation_value = np.dot(np.transpose(v), np.dot(A_squared, v))

    return expectation_value

def compute_deviation(H, eigenvector):
    """
    Computes <H^2> - <H>^2 for a given matrix H and an eigenvector.

    Parameters:
    H (numpy.ndarray): The matrix H.
    eigenvector (numpy.ndarray): The eigenvector with respect to which the expectation values are computed.

    Returns:
    float: The value of <H^2> - <H>^2.
    """
    # # Normalize the eigenvector
    # eigenvector = eigenvector / np.linalg.norm(eigenvector)

    # # Compute <H>
    # H_expectation = np.vdot(eigenvector, H @ eigenvector)

    # # Compute <H^2>
    # H2_expectation = np.vdot(eigenvector, H @ H @ eigenvector)

    # # Compute <H^2> - <H>^2
    # variance = H2_expectation - np.abs(H_expectation)**2
    # Compute <H^2>
    H2_expectation = expectation_value_matrix_squared(matrix, zeromode)
    H_expectation = expect_value(zeromode, matrix)
    variance = H2_expectation - np.abs(H_expectation)**2

    return variance

def compute_errors(expect_classical, expect_quantum):
    # Function to compute the relative error in <x^2>

    error = np.abs(expect_classical - expect_quantum) / expect_classical
    return error

### Functions to compute the PDFs, expectation values (like <x^2>), and other relevant quantities (L1 normalization, etc.)

def get_pdf(n, x, dx, L, shift, zeromode_qpe, normalize = True, make_even = False):
    # Function to construct the ground state PDF using the classical/quantum zeromode

    if not make_even:
        eigenvector = zeromode_qpe
    else:
        eigenvector_old = zeromode_qpe
        eigenvector = np.zeros(n + 1)
        for i in range(len(eigenvector_old)):
            eigenvector[2*i] = eigenvector_old[i]

    x0 = x - shift

    # Computing the PDF
    y = np.zeros(len(x0))

    for i in range(len(x0)):
        states = state_n(nmax, x0[i], L)
        y[i] += (np.dot(states, eigenvector))

    if normalize:
        y = normalize_probability(y, dx)

    return x0, y

# Computing <x^2> using the analytic formula
# Define the symbols
z = symbols('z')
l = Symbol('l', positive = True)

# Hermite polynomial function
def hermite_poly(n, z):
    return sym.hermite(n, z)

# Define the integrand
def integrand(n, z, l):
    factor = sym.sqrt(l) / (sym.pi**(1/4) * sym.sqrt(2**n * sym.factorial(n)))

    return z**2 * factor * hermite_poly(n, z / l) * sym.exp(-z**2 / (2 * l**2))

# Computing <x^2> using the zeromode
def compute_x_squared_expectation_analytic(l_val, zeromode):
    # Function to compute <x^2> using the zeromode

    # Initialize
    results = []
    expectation_value = 0

    # Compute
    for n in range(0, '', 2): # fill in '' with the (dimension * 2  - 1) of the zeromode
        integral = sym.integrate(integrand(n, z, l), (z, -sym.oo, sym.oo))
        integral_num = integral.subs(l, l_val).evalf()
        results.append(integral_num)

    # Print the results
    print('The values of the integrals are:')
    print(results)
    print()

    # Compute <x^2>
    for i in range(len(results)):
        expectation_value += zeromode[i] * results[i]

    return results, expectation_value

# User can also use Simpson's 3/8th rule to compute <x^2>
def compute_expectation_x_squared_simpson(x, y, n):
    """
    Computes the expectation value of x^2 using Simpson's rule for numerical integration.

    Parameters:
    x (array-like): Discrete values of x (in the simulations/experiments, we define x \in [-4, 4] with a step size of 0.01).
    y (array-like): Corresponding values of the probability density function (PDF) at x.

    Returns:
    float: The expectation value of x^n.
    """
    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Compute x^2
    x_squared = x**n

    # Check if the number of intervals is even, if not make it even by truncating the last point
    if len(x) % 2 == 0:
        x = x[:-1]
        y = y[:-1]
        x_squared = x_squared[:-1]

    # Compute the integral using Simpson's rule
    h = (x[-1] - x[0]) / (len(x) - 1)
    integral = y[0] * x_squared[0] + y[-1] * x_squared[-1] + \
               4 * np.sum(y[1:-1:2] * x_squared[1:-1:2]) + \
               2 * np.sum(y[2:-2:2] * x_squared[2:-2:2])
    integral *= h / 3

    return integral

def normalize_pdf(x, y):
    """
    Normalize a discrete PDF using the L1 norm.

    Parameters:
    x (list or numpy array): x data points
    y (list or numpy array): y data points (PDF values)

    Returns:
    numpy array: normalized y values
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    # Calculate the area under the curve (integral of y over x)
    area = np.trapz(y, x)

    if area == 0:
        raise ValueError("The area under the PDF is zero, cannot normalize.")

    # Normalize the PDF
    y_normalized = y / area

    return np.abs(y_normalized)

def make_operator_even(op):
    op_new = np.zeros((op.shape[0]//2, op.shape[1]//2))

    for row in range(op_new.shape[0]):
        for col in range(op_new.shape[1]):
            op_new[row, col] = op[row*2, col * 2]

    return op_new

def compute_expectation(matrix, zeromode):
    # Function to compute <H> in the ground state

    value = np.dot(matrix, zeromode)
    expectation_value = np.dot(np.transpose(zeromode), value)

    return expectation_value
