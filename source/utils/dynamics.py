import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .hamiltonian import tensor_product, make_hamiltonian, diagonalized_evolution

def bit_expectations_sv(time_points, statevectors):
    """Compute the bit expectation values at each time point from statevectors.
    
    Args:
        time_points (np.ndarray(shape=(T,), dtype=float)): Time points.
        statevectors (np.ndarray(shape=(D, T), dtype=np.complex128)): State vector as a function of time.

    Returns:
        np.ndarray(shape=(T, n), dtype=float): Time points tiled for each bit.
        np.ndarray(shape=(T, n), dtype=float): Bit expectation values.
    """

    num_bits = np.round(np.log2(statevectors.shape[0])).astype(int)
    if num_bits > 8:
        raise NotImplementedError('Function not compatible with number of qubits > 8')

    # Probability of seeing each bitstring at each time point
    probs = np.square(np.abs(statevectors)) # shape (D, T)

    # Unpack each index into a binary
    indices = np.expand_dims(np.arange(2 ** num_bits, dtype=np.uint8), axis=1) # shape (D, 1)
    bits = np.unpackbits(indices, axis=1, count=num_bits, bitorder='little').astype(float) # shape (D, num_bits)

    # For each bit, expectation = sum_j [prob_j * bit_j]
    y = probs.T @ bits # shape (T, num_bits)
        
    # Tile the time points to have one x array per spin
    x = np.tile(np.expand_dims(time_points, 1), (1, num_bits)) # shape (T, num_bits)
    
    return x, y


def bit_expectations_counts(time_points, counts_list, num_bits):
    """Compute the bit expectation values from experiment results.

    Args:
        time_points (np.ndarray(shape=(T,), dtype=float)): Time points.
        counts_list (List(Dict)): List (length T) of quantum experiment results, as given by Qiskit job.result().get_counts()
        num_bits (int): Number of qubits

    Returns:
        np.ndarray(shape=(nstep, num_bits), dtype=float): Time points tiled for each bit.
        np.ndarray(shape=(nstep, num_bits), dtype=float): Bit expectation values.
    """

    if num_bits > 8:
        raise NotImplementedError('Function not compatible with number of qubits > 8')

    num_steps = len(counts_list)

    x = np.tile(np.expand_dims(time_points, axis=1), (1, num_bits)) # shape (T, num_bits)
    y = np.zeros_like(x)
    
    for istep, counts in enumerate(counts_list):
        counts = counts_list[istep]

        total = 0
        for bitstring, count in counts.items():
            # 1. reverse the bitstring (last bit is the least significant)
            # 2. map all bits to integers
            # 3. convert to array
            bits = np.array(list(map(int, reversed(bitstring))), dtype=float)
            y[istep] += count * bits
            total += count
        
        y[istep] /= total

    return x, y


def insert_initial_counts(counts_list, initial_state):
    """Prepend a virtual 'counts' dictionary computed from the initial statevector to the counts list.
    
    Args:
        counts_list (List(Dict)): List of quantum experiment results, as given by Qiskit job.result().get_counts()
        initial_state (np.ndarray(shape=(2 ** num_spins), dtype=np.complex128)): Initial state vector.
    """
    
    num_bits = np.round(np.log2(initial_state.shape[0])).astype(int)

    initial_probs = np.square(np.abs(initial_state))
    fmt = '{{:0{}b}}'.format(num_bits)
    initial_counts = dict((fmt.format(idx), prob) for idx, prob in enumerate(initial_probs) if prob != 0.)
    counts_list.insert(0, initial_counts)
    
    
def plot_heisenberg_spins(counts_list, num_spins, initial_state, omegadt, add_theory_curve=False, spin_component='z'):
    """Compute the expectation value of the Z(/X/Y) component of each spin in the Heisenberg model from the quantum
    measurement results.

    Args:
        counts_list (List(Dict)): List of quantum experiment results, as given by Qiskit job.result().get_counts()
        num_spins (int): Number of spins in the system.
        initial_state (np.ndarray(shape=(2 ** num_spins), dtype=np.complex128)): Initial state vector.
        omegadt (float): Hamiltonian parameter (H = -0.5 hbar omega sum_j [xx + yy + zz]) times time step.
        add_theory_curve (bool): If True, compute the exact (non-Trotter) solution.
        spin_component (str): Spin component to plot. Values 'x', 'y', or 'z'. Only affects the theory curve.
    """
    
    # Number of steps
    num_steps = len(counts_list)

    # Figure and axes for the plot
    fig, ax = plt.subplots(1, 1)
    legend_items = []
    legend_labels = []

    if add_theory_curve:
        # Construct the numerical Hamiltonian matrix from a list of Pauli operators
        paulis = list()
        for j in range(num_spins - 1):
            paulis.append(list('x' if k in (j, j + 1) else 'i' for k in range(num_spins)))
            paulis.append(list('y' if k in (j, j + 1) else 'i' for k in range(num_spins)))            
            paulis.append(list('z' if k in (j, j + 1) else 'i' for k in range(num_spins)))

        hamiltonian = make_hamiltonian(paulis)
        
        # Compute the statevector as a function of time from Hamiltonian diagonalization
        time_points, statevectors = diagonalized_evolution(-0.5 * hamiltonian, initial_state, omegadt * num_steps)

        spin_basis_change = None
        if spin_component == 'x':
            spin_basis_change = np.array([[1., 1.], [1., -1.]], dtype=np.complex128) * np.sqrt(0.5)
        elif spin_component == 'y':
            spin_basis_change = np.array([[1., -1.j], [-1.j, 1.]], dtype=np.complex128) * np.sqrt(0.5)

        if spin_basis_change is not None:
            basis_change = tensor_product([spin_basis_change] * num_spins)
            statevectors = basis_change @ statevectors            

        x, y = bit_expectations_sv(time_points, statevectors)

        # Convert the bit expectations ([0, 1]) to spin expectations ([1, -1])
        y = 1. - 2. * y

        # Plot
        lines = ax.plot(x, y)
        colors = list(line.get_color() for line in lines)
        
        dummy_line = mpl.lines.Line2D([0], [0])
        dummy_line.update_from(lines[0])
        dummy_line.set_color('black')
        legend_items.append(dummy_line)
        legend_labels.append('exact')
    else:
        colors = None

    # Time points
    time_points = np.linspace(0., num_steps * omegadt, num_steps + 1, endpoint=True)
    
    # Prepend the initial "counts" to the experiment results
    insert_initial_counts(counts_list, initial_state)

    # Compute the bit expectation values from the counts
    x, y = bit_expectations_counts(time_points, counts_list, num_spins)

    # Convert the bit expectations ([0, 1]) to spin expectations ([1, -1])
    y = 1. - 2. * y

    # Plot
    markers = ax.plot(x, y, 'o')
    if colors is not None:
        for marker, color in zip(markers, colors):
            marker.set_color(color)
    
    legend_items += markers
    legend_labels += ['bit%d' % i for i in range(num_spins)]
    ax.legend(legend_items, legend_labels)
    
    ax.set_xlabel(r'$\omega t$')
    ax.set_ylabel(r'$\langle S_z \rangle$')
