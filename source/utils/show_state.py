import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer, execute

def show_state(circuit, amp_norm=None, phase_norm=(np.pi, '\pi'), global_phase=None, register_sizes=None, terms_per_row=8, binary=False, state_label=None, draw=True, return_fig=False, gpu=True):
    """Print the quantum state of the circuit in latex markdown.
    
    Args:
        circuit (QuantumCircuit): The circuit.
        amp_norm (None or tuple): If not None, specify the normalization of the amplitudes by
            (numeric devisor, unit in latex)
        phase_norm (None or tuple): If not None, specify the normalization of the phases by
            (numeric devisor, unit in latex)
        global_phase (None or float or str): If not None, specify the phase to factor out by
            numeric offset or 'mean'
        register_sizes (None or array_like): If not None, specify the sizes of the registers as
            a list of ints.
        terms_per_row (int): Number of terms to show per row.
        binary (bool): Show ket indices in binary.
        state_label (None or str): If not None, prepend '|`state_label`> = ' to the printout
        draw (bool): Call draw('mpl') on the circuit.
        return_fig (bool): Returns the mpl Figure object.
        gpu (bool): Use statevector_gpu if available.
    """
    
    # Run the circuit in statevector_simulator and obtain the final state statevector
    simulator = Aer.get_backend('statevector_simulator')
    if gpu:
        try:
            simulator.set_options(method='statevector_gpu')
        except:
            simulator.set_options(method='statevector')

    statevector = execute(circuit, simulator).result().data()['statevector']
    
    if draw:
        circuit.draw('mpl', style={'dpi': '300'}, fold=70)
    
    fig = plt.figure(figsize=[10., 0.5])
    ax = fig.add_subplot()
        
    row_texts = show_statevector(statevector, amp_norm=amp_norm, phase_norm=phase_norm, global_phase=global_phase, register_sizes=register_sizes, terms_per_row=terms_per_row, binary=binary, state_label=state_label, ax=ax)
    
    fig.set_figheight(0.5 * len(row_texts))

    if return_fig:
        return fig

    
def show_statevector(statevector, amp_norm=None, phase_norm=(np.pi, '\pi'), global_phase=None, register_sizes=None, terms_per_row=8, binary=False, state_label=None, ax=None):
    """Print the quantum state of the circuit in latex markdown.
    
    Args:
        statevector (np.ndarray(*, dtype=np.complex128)): Statevector.
        amp_norm (None or tuple): If not None, specify the normalization of the amplitudes by
            (numeric devisor, unit in latex)
        phase_norm (None or tuple): If not None, specify the normalization of the phases by
            (numeric devisor, unit in latex)
        global_phase (None or float or str): If not None, specify the phase to factor out by
            numeric offset or 'mean'
        register_sizes (None or array_like): If not None, specify the sizes of the registers as
            a list of ints.
        terms_per_row (int): Number of terms to show per row.
        binary (bool): Show ket indices in binary.
        state_label (None or str): If not None, prepend '|`state_label`> = ' to the printout
        ax (None or mpl.Axes): Axes object. A new axes is created if None.
        
    Returns:
        List(str): Latex text (enclosed in $$) for each line.
    """
    
    log2_shape = np.log2(statevector.shape[0])
    assert log2_shape == np.round(log2_shape), 'Invalid statevector'
    num_qubits = np.round(log2_shape).astype(int)
    
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
            slots = ['{{:0{}b}}'.format(num_qubits)]
        else:
            slots = ['{}']

    ket_template += ':'.join(slots) + r'\rangle'
  
    # Preprocessing
    indices = np.asarray(absamp > tolerance).nonzero()[0]
    absamp = absamp[indices]
    phase = phase[indices]

    phase_offset = 0.
    if global_phase is not None:
        if global_phase == 'mean':
            phase_offset = np.mean(phase)
        else:
            phase_offset = global_phase

        phase -= phase_offset

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
            if np.abs(a - 1.) > tolerance:
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

    if amp_norm is not None or phase_offset != 0.:
        str_rows[0].insert(0, r'\left(')

        if num_rows != 1:
            str_rows[0].append(r'\right.')
            str_rows[-1].insert(0, r'\left.')

        str_rows[-1].append(r'\right)')

    if phase_offset != 0.:
        if phase_norm is not None:
            phase_offset /= phase_norm[0]
            rounded_phase_offset = np.round(phase_offset).astype(int)
            if np.abs(phase_offset - rounded_phase_offset) < tolerance:
                if rounded_phase_offset == 1:
                    phase_value_expr = phase_norm[1]
                else:
                    phase_value_expr = '{:d} \cdot {}'.format(rounded_phase_offset, phase_norm[1])
            else:
                phase_value_expr = '{:.2f} \cdot {}'.format(phase_offset, phase_norm[1])
        else:
            phase_value_expr = '{:.2f}'.format(phase_offset)
            
        str_rows[0].insert(0, 'e^{{{} i}}'.format(phase_value_expr))

    if amp_norm is not None:
        str_rows[0].insert(0, amp_norm[1])
        
    if state_label is not None:
        str_rows[0].insert(0, r'| {} \rangle = '.format(state_label))

    if ax is None:
        fig = plt.figure(figsize=[10., 0.5 * num_rows])
        ax = fig.add_subplot()
        
    row_texts = list('${}$'.format(''.join(str_terms)) for str_terms in str_rows)
        
    ax.axis('off')
    for irow, row_text in enumerate(row_texts):
        ax.text(0.5, 1. / num_rows * (num_rows - irow - 1), row_text, fontsize='x-large', ha='center')

    return row_texts
