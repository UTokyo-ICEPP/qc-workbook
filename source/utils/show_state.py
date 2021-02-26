import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer, execute

def show_state(circuit, amp_norm=None, phase_norm=(np.pi, '\pi'), register_sizes=None, terms_per_row=8, binary=False, draw=True, return_fig=False):
    """Print the quantum state of the circuit in latex markdown.
    
    Args:
        circuit (QuantumCircuit): The circuit.
        amp_norm (None or tuple): If not None, specify the normalization of the amplitudes by
            (numeric devisor, unit in latex)
        phase_norm (None or tuple): If not None, specify the normalization of the phases by
            (numeric devisor, unit in latex)
        register_sizes (None or array_like): If not None, specify the sizes of the registers as
            a list of ints.
        terms_per_row (int): Number of terms to show per row.
        binary (bool): Show ket indices in binary.
        draw (bool): Call draw('mpl') on the circuit.
    """
    
    # Run the circuit in statevector_simulator and obtain the final state statevector
    simulator = Aer.get_backend('statevector_simulator')
    simulator.set_options(method='statevector_gpu')
    statevector = execute(circuit, simulator).result().data()['statevector']

    # Absolute value and the phase of the amplitudes
    absamp = np.abs(statevector)
    logamp = np.zeros_like(statevector)
    np.log(statevector, out=logamp, where=(absamp > 0.))
    phase = logamp.imag
    
    # Set the numerical tolerance for various comparisons
    tolerance = 1.e-3
        
    # List to be concatenated into the final latex string
    str_rows = [[]]
    str_terms = str_rows[0]
    
    # Ket format template
    ket_template = ' |'
    if register_sizes is not None:
        if binary:
            slots = ['{{:0{}b}}'.format(s) for s in reversed(register_sizes)]
        else:
            slots = ['{}'] * len(register_sizes)
    else:
        if binary:
            slots = ['{{:0{}b}}'.format(circuit.num_qubits)]
        else:
            slots = ['{}']

    ket_template += ':'.join(slots) + r'\rangle'
  
    # Preprocessing
    indices = np.asarray(absamp > tolerance).nonzero()[0]
    absamp = absamp[indices]
    phase = phase[indices]

    if amp_norm is not None:
        absamp /= amp_norm[0]
        rounded_amp = np.round(absamp).astype(int)
        amp_is_int = np.asarray(np.abs(absamp - rounded_amp) < tolerance, dtype=bool)
        
    nonzero_phase = np.asarray(np.abs(phase) > tolerance, dtype=bool)
    phase = np.where(phase > 0., phase, phase + 2. * np.pi)
    reduced_phase = phase / np.pi
    rounded_reduced_phase = np.round(reduced_phase).astype(int)
    phase_is_pi_multiple = np.asarray(np.abs(reduced_phase - rounded_reduced_phase) < tolerance, dtype=bool)
    semireduced_phase = reduced_phase * 2.
    rounded_semireduced_phase = np.round(semireduced_phase).astype(int)
    phase_is_halfpi_multiple = np.asarray(np.abs(semireduced_phase - rounded_semireduced_phase) < tolerance, dtype=bool)
    
    if phase_norm is not None:
        phase /= phase_norm[0]
        rounded_phase = np.round(phase).astype(int)
        phase_is_int = np.asarray(np.abs(phase - rounded_phase) < tolerance, dtype=bool)
        
    if register_sizes is not None:
        register_sizes = np.array(register_sizes)
        cumul_register_sizes = np.roll(np.cumsum(register_sizes), 1)
        cumul_register_sizes[0] = 0
        register_indices = np.tile(np.expand_dims(indices, axis=1), (1, register_sizes.shape[0]))
        register_indices = np.right_shift(register_indices, cumul_register_sizes)
        register_indices = np.mod(register_indices, np.power(2, register_sizes))
        
    # Sign of each term (initialized to 0 so the first term doesn't get a + in front)
    sign = 0

    # Stringify each term
    for iterm, idx in enumerate(indices):
        # Latex string for this term
        basis_unsigned = ''

        # Write the amplitude
        a = absamp[iterm]

        if amp_norm is None:
            # No amplitude normalization -> just write as a raw float
            basis_unsigned += '{:.2f}'.format(a)
        else:
            # With amplitude normalization
            if amp_is_int[iterm]:
                # Amplitude is integer * norm
                if rounded_amp[iterm] != 1:
                    basis_unsigned += '{:d}'.format(rounded_amp[iterm])
            else:
                # Otherwise float * norm
                basis_unsigned += '{:.2f}'.format(a)
                
        # Write the phase
        if nonzero_phase[iterm]:
            # First check if the phase is a multiple of pi or pi/2
            if phase_is_pi_multiple[iterm]:
                if rounded_reduced_phase[iterm] % 2 == 1:
                    sign = -1

            elif phase_is_halfpi_multiple[iterm]:
                twopbypi = np.round(2. * pbypi).astype(int)
                if rounded_semireduced_phase[iterm] % 2 == 1:
                    basis_unsigned += 'i'
                if rounded_semireduced_phase[iterm] % 4 == 3:
                    sign = -1

            else:
                p = phase[iterm]
                
                basis_unsigned += 'e^{'
                if phase_norm is None:
                    # No phase normalization -> write as exp(raw float * i)
                    basis_unsigned += '{:.2f}'.format(p)
                else:
                    # With phase normalization -> write as exp(divident * norm * i)
                    if phase_is_int[iterm]:
                        if rounded_phase[iterm] != 1:
                            basis_unsigned += r'{:d} \cdot '.format(rounded_phase[iterm])
                    else:
                        basis_unsigned += r'{:.2f} \cdot '.format(p)

                    basis_unsigned += phase_norm[1]

                basis_unsigned += r' i}'

        if register_sizes is not None:
            basis_unsigned += ket_template.format(*reversed(register_indices[iterm]))
        else:
            basis_unsigned += ket_template.format(idx)

        if sign == 1:
            term = ' + ' + basis_unsigned
        elif sign == -1:
            term = ' - ' + basis_unsigned
        else:
            term = basis_unsigned
            
        str_terms.append(term)

        if len(str_terms) == terms_per_row:
            str_rows.append([])
            str_terms = str_rows[-1]
            
        # All terms except the very first are signed
        sign = 1

    # Remove empty row
    if len(str_rows[-1]) == 0:
        str_rows.pop()
        
    num_rows = len(str_rows)
        
    if amp_norm is not None:
        str_rows[0].insert(0, r'{} \left('.format(amp_norm[1]))

        if num_rows != 1:
            str_rows[0].append(r'\right.')
            str_rows[-1].insert(0, r'\left.')

        str_rows[-1].append(r'\right)')
 
    if draw:
        circuit_height = 12. * ((circuit.depth() - 1) // 70 + 1)
        fig = plt.figure(figsize=[20., circuit_height + 0.5 * num_rows])
        gs = fig.add_gridspec(2, 1, height_ratios=(circuit_height, 0.5 * num_rows))
        ax = fig.add_subplot(gs[0])
        circuit.draw('mpl', style={'dpi': '300'}, fold=70, ax=ax)
        ax = fig.add_subplot(gs[1])
    else:
        fig = plt.figure(figsize=[10., 0.5 * num_rows])
        ax = fig.add_subplot()

    ax.axis('off')
    for irow, str_terms in enumerate(str_rows):
        ax.text(0.5, 1. / num_rows * (num_rows - irow - 1), '${}$'.format(''.join(str_terms)), fontsize='x-large', ha='center')

    if return_fig:
        return fig