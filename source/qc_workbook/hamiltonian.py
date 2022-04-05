import sys
import numpy as np

def tensor_product(ops):
    """Recursively apply np.kron to construct a tensor product of the operators.
    
    Args:
        ops (List): List of (2, 2) arrays.

    Returns:
        np.ndarray(shape=(2 ** nops, 2 ** nops), dtype=np.complex128): Tensor product of ops.
    """
    
    prod = 1.
    for op in ops:
        prod = np.kron(op, prod)

    return prod


def make_hamiltonian(paulis, coeffs=None):
    """Compute the Hamiltonian matrix from its Pauli decomposition.

    Args:
        paulis (List(List(str))): Terms in Pauli decomposition. Form [['i', 'x', 'z', ..], ['x', 'y', 'i', ..], ..]
            All inner lists must be of the same length (number of qubits n).
        coeffs (None or List(float)): If not None, the list must be of the same length as the outer list of paulis
            and should specify the coefficient of each term.

    Returns:
        np.ndarray(shape=(2 ** n, 2 ** n), dtype=np.complex128): The numerical Hamiltonian matrix. The first qubit
        corresponds to the least significant digit.
    """
    
    if len(paulis) == 0:
        return np.array([[0.]], dtype=np.complex128)

    qubit_nums = set(len(term) for term in paulis)
    assert len(qubit_nums) == 1, 'List of paulis must all have the same length.'

    if coeffs is None:
        coeffs = [1.] * len(paulis)

    num_qubits = qubit_nums.pop()

    # Basis matrices
    basis_matrices = {
        'i': np.array([[1., 0.], [0., 1.]], dtype=np.complex128),
        'x': np.array([[0., 1.], [1., 0.]], dtype=np.complex128),
        'y': np.array([[0., -1.j], [1.j, 0.]], dtype=np.complex128),
        'z': np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
    }

    # Start with an empty matrix
    hamiltonian = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=np.complex128)

    for iterm, term in enumerate(paulis):
        try:
            ops = list(basis_matrices[op.lower()] for op in term)
        except KeyError as err:
            sys.stderr.write('Invalid operator {} in term {}\n'.format(err.args[0], iterm))
            raise

        hamiltonian += coeffs[iterm] * tensor_product(ops)

    return hamiltonian


def diagonalized_evolution(hamiltonian, initial_state, time, num_steps=100):
    """Diagonalize the given reduced Hamiltonian and evolve the initial state by exp(-i time*hamiltonian).
    
    Args:
        hamiltonian (np.ndarray(shape=(D, D), dtype=np.complex128)): Hamiltonian matrix divided by hbar.
        initial_state (np.ndarray(shape=(D,), dtype=np.complex128)): Initial state vector.
        time (float): Evolution time.
        num_steps (int): Number of steps (T) to divide time into.
        
    Returns:
        np.ndarray(shape=(T,), dtype=float): Time points.
        np.ndarray(shape=(D, T), dtype=np.complex128): State vector as a function of time.
    """

    num_dim = hamiltonian.shape[0]
    num_qubits = np.round(np.log2(num_dim)).astype(int)

    # Create the array of time points
    time_points = np.linspace(0., time, num_steps, endpoint=True)

    ## Diagonalize the Hamiltonian
    eigvals, eigvectors = np.linalg.eigh(hamiltonian)

    ## Decompose the initial state vector into a linear combination of eigenvectors
    # Matrix eigvectors has the form [v_0 v_1 v_2 ..], where v_i . v_j = delta_ij
    # -> eigvectors^dagger @ initial_state = coefficients for the eigenvector decomposition of the initial state vector
    initial_coeff = eigvectors.T.conjugate() @ initial_state
    # Initial state as a matrix [c_0 v_0, c_1 v_1, ...] (shape (D, D))
    initial_state_matrix = eigvectors * initial_coeff

    ## Time-evolve the initial state to each time point
    # Phase at each time point (shape (D, T))
    phase = np.outer(-1.j * eigvals, time_points)
    phase_factor = np.exp(phase)
    statevectors = initial_state_matrix @ phase_factor # shape (D, T)

    return time_points, statevectors
