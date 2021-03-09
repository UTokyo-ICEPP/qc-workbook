import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .hamiltonian import tensor_product, make_hamiltonian, diagonalized_evolution

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
    M = len(counts_list)

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
        time_points, statevectors = diagonalized_evolution(-0.5 * hamiltonian, initial_state, omegadt * M)

        spin_basis_change = None
        if spin_component == 'x':
            spin_basis_change = np.array([[1., 1.], [1., -1.]], dtype=np.complex128) * np.sqrt(0.5)
        elif spin_component == 'y':
            spin_basis_change = np.array([[1., -1.j], [-1.j, 1.]], dtype=np.complex128) * np.sqrt(0.5)

        if spin_basis_change is not None:
            basis_change = tensor_product([spin_basis_change] * num_spins)
            statevectors = basis_change @ statevectors            

        # Probability of seeing each bitstring at each time point
        probs = np.square(np.abs(statevectors)) # shape (D, T)

        # Compute the spin value of each bitstring
        indices = np.expand_dims(np.arange(2 ** num_spins, dtype=np.uint8), axis=1) # shape (D, 1)
        bits = np.unpackbits(indices, axis=1, count=num_spins, bitorder='little') # shape (D, num_spins)
        spinval = 1. - 2. * bits

        # For each spin, Z expectation = sum_j [prob_j * spin_j]
        y = probs.T @ spinval # shape (T, n)
        
        # Tile the time points to have one x array per spin
        x = np.tile(np.expand_dims(time_points, 1), (1, num_spins)) # shape (T, n)
        
        lines = ax.plot(x, y)
        colors = list(line.get_color() for line in lines)
        
        dummy_line = mpl.lines.Line2D([0], [0])
        dummy_line.update_from(lines[0])
        dummy_line.set_color('black')
        legend_items.append(dummy_line)
        legend_labels.append('exact')
    else:
        colors = None
        
    # [[0., 0., ...], [omegadt, omegadt, ...], ..., [M*omegadt, M*omegadt, ...]] x values for each spin
    x = np.tile(np.expand_dims(np.linspace(0., omegadt * M, M + 1, endpoint=True), 1), (1, num_spins))
    y = np.zeros_like(x)

    # Initial state expectation values
    initial_probs = np.square(np.abs(initial_state))
    indices = np.expand_dims(np.arange(2 ** num_spins, dtype=np.uint8), axis=1) # shape (D, 1)
    bits = np.unpackbits(indices, axis=1, count=num_spins, bitorder='little') # shape (D, num_spins)
    spinval = 1. - 2. * bits

    y[0] = np.sum(spinval.T * initial_probs, axis=1)

    for istep in range(M):
        counts = counts_list[istep]

        total = 0
        for bitstring, count in counts.items():
            # 1. reverse the bitstring (last bit is the least significant)
            # 2. map all bits to integers
            # 3. compute spin = (1 - 2*bit) <- bit = 0 corresponds to spin +1
            spinval = 1 - np.array(list(map(int, reversed(bitstring))), dtype=float) * 2
            y[istep + 1] += count * spinval
            total += count
        
        y[istep + 1] /= total

    markers = ax.plot(x, y, 'o')
    if colors is not None:
        for marker, color in zip(markers, colors):
            marker.set_color(color)
    
    legend_items += markers
    legend_labels += ['bit%d' % i for i in range(num_spins)]
    ax.legend(legend_items, legend_labels)
    
    ax.set_xlabel(r'$\omega t$')
    ax.set_ylabel(r'$\langle S_z \rangle$')
