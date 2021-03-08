import numpy as np

def get_exact_solutions(num_spins, initial_state, omegat, num_steps=100):
    """Compute the exact solutions for Z spin expectation values at each site of the
    Heisenberg model with no external field and a single common coupling of J = (0.5 hbar omega).
    
    Args:
        num_spins (int): Number of spins
        initial_state (np.ndarray): Initial state vector (shape=(D,), dtype=np.complex128)
            where D = 2 ** num_spins
        omegat (float): omega times the full interval to time-evolve
        num_steps (int): Number of steps (T) to divide omegat into.
        
    Returns:
        np.ndarray: x values (shape=(T, n))
        np.ndarray: y values (shape=(T, n))
    """

    # x values (shape (T))
    xvalues = np.linspace(0., omegat, num_steps, endpoint=True)

    ## Compute the Hamiltonian (only the Pauli part)
    hamiltonian = np.zeros((2 ** num_spins, 2 ** num_spins), dtype=np.complex128)
    identity = np.array([[1., 0.], [0., 1.]], dtype=np.complex128)
    sigma_x = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
    sigma_y = np.array([[0., -1.j], [1.j, 0.]], dtype=np.complex128)
    sigma_z = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
    for j in range(num_spins - 1):
        xterm = 1.
        yterm = 1.            
        zterm = 1.
        for k in range(num_spins):
            if k == j or k == j + 1:
                xterm = np.kron(xterm, sigma_x)
                yterm = np.kron(yterm, sigma_y)
                zterm = np.kron(zterm, sigma_z)
            else:
                xterm = np.kron(xterm, identity)
                yterm = np.kron(yterm, identity)
                zterm = np.kron(zterm, identity)
                    
        hamiltonian += xterm + yterm + zterm

    ## Diagonalize the Hamiltonian
    eigvals, eigvectors = np.linalg.eigh(hamiltonian)

    ## Decompose the initial state vector into a linear combination of eigenvectors
    # Solving sum_i [c_i (eigv)_i] = (initv) for {c_i}
    initial_coeff = np.linalg.solve(eigvectors, initial_state)
    # Initial state as a matrix [c_0 (eigv)_0, c_1 (eigv)_1, ...] (shape (D, D))
    initial_state_matrix = eigvectors * initial_coeff

    ## Time-evolve the initial state to each time point
    # Phase at each time point (shape (D, T))
    phase = np.outer(1.j * 0.5 * eigvals, xvalues)
    phase_factor = np.exp(phase)
    state = initial_state_matrix @ phase_factor # shape (D, T)
    # Probability of seeing each bitstring at each time point
    probs = np.square(np.abs(state)) # shape (D, T)
        
    ## Compute the spin value of each bitstring
    indices = np.expand_dims(np.arange(2 ** num_spins, dtype=np.uint8), axis=1) # shape (D, 1)
    bits = np.unpackbits(indices, axis=1, count=num_spins, bitorder='little') # shape (D, num_spins)
    spinval = 1. - 2. * bits
    
    ## For each spin, Z expectation = sum_j [prob_j * spin_j]
    yvalues = probs.T @ spinval # shape (T, num_spins)

    return np.tile(np.expand_dims(xvalues, 1), (1, num_spins)), yvalues
