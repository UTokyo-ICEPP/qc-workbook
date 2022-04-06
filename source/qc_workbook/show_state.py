from typing import Tuple, List, Union, Optional
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from qiskit import Aer, transpile, QuantumCircuit

def show_state(
    statevector: Union[QuantumCircuit, np.ndarray],
    amp_norm: Optional[Tuple[float, str]] = None,
    phase_norm: Tuple[float, str] = (np.pi, '\pi'),
    global_phase: Optional[Union[float, str]] = None,
    register_sizes: Optional['array_like'] = None,
    terms_per_row: int = 8,
    binary: bool = False,
    state_label: Optional[str] = None,
    ax: Optional[mpl.axes.Axes] = None
) -> Union[None, mpl.figure.Figure]:
    """Show the quantum state of the circuit as text in a matplotlib Figure.
    
    Args:
        statevector: Input statevector or a QuantumCircuit whose final state is to be extracted.
        amp_norm: Specification of the normalization of amplitudes by (numeric devisor, unit in latex).
        phase_norm: Specification of the normalization of phases by (numeric devisor, unit in latex).
        global_phase: Specification of the phase to factor out. Give a numeric offset or 'mean'.
        register_sizes: Specification of the sizes of registers as a list of ints.
        terms_per_row: Number of terms to show per row.
        binary: Show ket indices in binary.
        state_label: If not None, prepend '|`state_label`> = ' to the printout.
        ax: Axes object. A new axes is created if None.
        
    Returns:
        The newly created figure object if ax is None.
    """
    lines = statevector_expr(statevector, amp_norm=amp_norm, phase_norm=phase_norm,
                             global_phase=global_phase, register_sizes=register_sizes,
                             terms_per_row=terms_per_row, binary=binary, state_label=state_label)

    row_texts = list(f'${line}$' for line in lines)
    
    fig = None
    if ax is None:
        fig = plt.figure(figsize=[10., 0.5 * len(lines)])
        ax = fig.add_subplot()
        
    ax.axis('off')
    
    num_rows = len(row_texts)
    
    for irow, row_text in enumerate(row_texts):
        ax.text(0.5, 1. / num_rows * (num_rows - irow - 1), row_text, fontsize='x-large', ha='center')

    if fig is not None:
        return fig
    

def statevector_expr(
    statevector: Union[np.ndarray, QuantumCircuit],
    amp_norm: Optional[Tuple[float, str]] = None,
    phase_norm: Tuple[float, str] = (np.pi, '\pi'),
    global_phase: Optional[Union[float, str]] = None,
    register_sizes: Optional['array_like'] = None,
    terms_per_row: int = 0,
    binary: bool = False,
    amp_format: str = '.3f',
    phase_format: str = '.2f',
    state_label: Union[str, None] = r'\text{final}'
) -> Union[str, List[str]]:
    """Compose the LaTeX expressions for a statevector.

    Args:
        statevector: Input statevector or a QuantumCircuit whose final state is to be extracted.
        amp_norm: Specification of the normalization of amplitudes by (numeric devisor, unit in latex).
        phase_norm: Specification of the normalization of phases by (numeric devisor, unit in latex).
        global_phase: Specification of the phase to factor out. Give a numeric offset or 'mean'.
        register_sizes: Specification of the sizes of registers as a list of ints.
        terms_per_row: Number of terms to show per row. If 0 (default), a single string is returned.
        binary: Show ket indices in binary.
        amp_format: Format for the numerical value of the amplitude absolute values.
        phase_format: Format for the numerical value of the phases.
        state_label: If not None, prepend '|`state_label`> = ' to the printout.
        
    Returns:
        LaTeX expression string (if terms_per_row <= 0) or a list of expression lines.
    """
    ## If a QuantumCircuit is passed, extract the statevector
    
    if isinstance(statevector, QuantumCircuit):
        # Run the circuit in statevector_simulator and obtain the final state statevector
        simulator = Aer.get_backend('statevector_simulator')

        circuit = transpile(statevector, backend=simulator)
        statevector = np.asarray(simulator.run(circuit).result().data()['statevector'])
    
    ## Setup
    
    log2_shape = np.log2(statevector.shape[0])
    assert log2_shape == np.round(log2_shape), 'Invalid statevector'
    num_qubits = np.round(log2_shape).astype(int)
    
    # Set the numerical tolerance for various comparisons
    tolerance = 1.e-3
   
    # Ket format template
    ket_template = ' |'
    if register_sizes is not None:
        if binary:
            slots = [f'{{:0{s}b}}' for s in reversed(register_sizes)]
        else:
            slots = ['{}'] * len(register_sizes)
    else:
        if binary:
            slots = [f'{{:0{num_qubits}b}}']
        else:
            slots = ['{}']

    ket_template += ':'.join(slots) + r'\rangle'
    
    # Amplitude format template
    amp_format = f'{{:{amp_format}}}'
    
    # Phase format template
    phase_format = f'{{:{phase_format}}}'
  
    ## Preprocess the statevector
    
    # Absolute value and the phase of the amplitudes
    absamp = np.abs(statevector)
    phase = np.angle(statevector)
    
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
        
    ## Compile the LaTeX expressions
        
    # List to be concatenated into the final latex string
    lines = []
    str_terms = []
    
    # Pre- and Post-expressions
    pre_expr = ''
    post_expr = ''
    
    if state_label is not None:
        pre_expr += fr'| {state_label} \rangle = '
    
    if phase_offset != 0.:
        if phase_norm is None:
            phase_value_expr = phase_format.format(phase_offest)
        else:
            phase_offset /= phase_norm[0]
            rounded_phase_offset = np.round(phase_offset).astype(int)
            if np.abs(phase_offset - rounded_phase_offset) < tolerance:
                if rounded_phase_offset == 1:
                    phase_value_expr = phase_norm[1]
                else:
                    phase_value_expr = fr'{rounded_phase_offset:d} \cdot {phase_norm[1]}'
            else:
                phase_value_expr = phase_format.format(phase_offset) + fr' \cdot {phase_norm[1]}'
            
        pre_expr += f'e^{{{phase_value_expr} i}}'
        
    if amp_norm is not None:
        pre_expr += amp_norm[1]
        
    if amp_norm is not None or phase_offset != 0.:
        pre_expr += r'\left('
        post_expr += r'\right)'
        
    # Sign of each term
    sign = ''

    # Stringify each term
    for iterm, idx in enumerate(indices):
        # Latex string for this term
        basis_unsigned = ''

        # Write the amplitude
        a = absamp[iterm]

        if amp_norm is None:
            # No amplitude normalization -> just write as a raw float
            if np.abs(a - 1.) > tolerance:
                basis_unsigned += amp_format.format(a)
        else:
            # With amplitude normalization
            if amp_is_int[iterm]:
                # Amplitude is integer * norm
                if rounded_amp[iterm] != 1:
                    basis_unsigned += f'{rounded_amp[iterm]:d}'
            else:
                # Otherwise float * norm
                basis_unsigned += amp_format.format(a)
                
        # Write the phase
        if nonzero_phase[iterm]:
            # First check if the phase is a multiple of pi or pi/2
            if phase_is_pi_multiple[iterm]:
                if rounded_reduced_phase[iterm] % 2 == 1:
                    sign = ' - '

            elif phase_is_halfpi_multiple[iterm]:
                if rounded_semireduced_phase[iterm] % 2 == 1:
                    basis_unsigned += 'i'
                if rounded_semireduced_phase[iterm] % 4 == 3:
                    sign = ' - '

            else:
                p = phase[iterm]
                
                basis_unsigned += 'e^{'
                if phase_norm is None:
                    # No phase normalization -> write as exp(raw float * i)
                    basis_unsigned += phase_format.format(p)
                else:
                    # With phase normalization -> write as exp(divident * norm * i)
                    if phase_is_int[iterm]:
                        if rounded_phase[iterm] != 1:
                            basis_unsigned += fr'{rounded_phase[iterm]:d} \cdot '
                    else:
                        basis_unsigned += phase_format.format(p) + r' \cdot '

                    basis_unsigned += phase_norm[1]

                basis_unsigned += ' i}'

        if register_sizes is not None:
            basis_unsigned += ket_template.format(*reversed(register_indices[iterm]))
        else:
            basis_unsigned += ket_template.format(idx)

        str_terms.append(sign + basis_unsigned)
        
        sign = ' + '

        if terms_per_row > 0 and len(str_terms) == terms_per_row:
            # The line is full - concatenate the terms and append to the line list
            lines.append(''.join(str_terms))
            str_terms = []

    if len(str_terms) != 0:
        lines.append(''.join(str_terms))
        
    lines[0] = pre_expr + lines[0]
    lines[-1] += post_expr
        
    if len(lines) > 1 and (amp_norm is not None or phase_offset != 0.):
        lines[0] += r'\right.'
        lines[-1] = r'\left. ' + lines[-1]
    
    if terms_per_row > 0:
        return lines
    else:
        return ''.join(lines)
