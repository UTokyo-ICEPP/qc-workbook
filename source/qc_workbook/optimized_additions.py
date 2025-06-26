import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

def optimized_additions(
    n1: int,
    n2: int,
    measure: bool = True,
    barrier: bool = False
) -> QuantumCircuit:
    """This function sets up a circuit that performs 2^{n1+n2} additions in parallel, with operations
    optimized for a linear topology (all CNOTs between adjacent qubits).
    """

    n3 = np.ceil(np.log2((2 ** n1) + (2 ** n2) - 1)).astype(int)

    reg1 = QuantumRegister(n1, 'r1')
    reg2 = QuantumRegister(n2, 'r2')
    reg3 = QuantumRegister(n3, 'r3')

    circuit = QuantumCircuit(reg1, reg2, reg3, ClassicalRegister(n1 + n2 + n3, 'meas'))

    # Set all registers to equal superpositions
    for iq in range(n1 + n2 + n3):
        # Hadamard
        circuit.rz(np.pi * 0.5, iq)
        circuit.sx(iq)
        circuit.rz(np.pi * 0.5, iq)

    # Smallest unit of phi
    dphi = 2. * np.pi / (2 ** n3)

    # Advance the phase of the output register to create a state
    #     1/sqrt(2^(n1+n2)) sum_{a} sum_{b} sum_{k} exp(2pi i (a + b) k / 2^n3) |a>|b>|k>
    # Register placement changes from (input1, input2, output) to (output, input1, input2) as a result

    # Loop over input registers backwards
    offset = n1
    for nctrl in [n2, n1]:
        for logical_ictrl in reversed(range(nctrl)):
            # Actual control qubit number (moves as we swap the input and output qubits)
            ictrl = logical_ictrl + offset

            # Loop over qubits in the output register
            for logical_itarg in range(n3):
                # Actual target qubit number (always adjacent to the control)
                itarg = ictrl + 1

                if logical_ictrl + logical_itarg < n3:
                    # CP[2pi / 2^n3 * 2^(logical_ictrl + logical_itarg)]
                    #
                    # -----+-----     --Rz(phi/2)--+-------------------+--
                    #      |       ~               |                   |    (up to a global phase)
                    # ---P(phi)--     --Rz(phi/2)--X--Rz(2pi - phi/2)--X--

                    halfphi = dphi * (2 ** (logical_ictrl + logical_itarg)) * 0.5
                    circuit.rz(halfphi, ictrl)
                    circuit.rz(halfphi, itarg)
                    circuit.cx(ictrl, itarg)
                    circuit.rz(2. * np.pi - halfphi, itarg)
                    # Cancelling out the last cx^1_2 with the cx^1_2 of the swap below
                    #circuit.cx(ictrl, itarg)
                else:
                    # When logical_ictrl + logical_itarg >= n3, cp(phi) results in a global phase
                    # -> Just swap ictrl and itarg
                    circuit.cx(ictrl, itarg)

                # Swap ictrl and itarg. First cx is cancelled out
                #circuit.cx(ictrl, itarg)
                circuit.cx(itarg, ictrl)
                circuit.cx(ictrl, itarg)

                if barrier:
                    circuit.barrier()

                # Shift the control qubit
                ictrl += 1

        offset = 0

    if barrier:
        circuit.barrier()

    # Inverse Fourier transform on the output register to create a state
    #     1/sqrt(2^(n1+n2)) sum_{a} sum_{b} |a>|b>|a+b>
    # Using QFT with inherent swapping (does not require the swap at the front)

    # Loop through the "logical" control qubit
    # Acutal qubit location shifts to be always adjacent to the target
    for logical_ictrl in reversed(range(n3)):
        ictrl = n3 - 1
        # Hadamard
        circuit.rz(np.pi * 0.5, ictrl)
        circuit.sx(ictrl)
        circuit.rz(np.pi * 0.5, ictrl)

        if barrier:
            circuit.barrier()

        for logical_itarg in reversed(range(logical_ictrl)):
            # Target is always next to the control
            itarg = ictrl - 1

            # CP[2pi / 2^n3 * 2^(n3 - 1 - dq)]  where dq = logical_ictrl - logical_itarg

            halfphi = dphi * (2 ** (n3 - 1 - (logical_ictrl - logical_itarg))) * 0.5
            circuit.rz(-halfphi, ictrl)
            circuit.rz(-halfphi, itarg)
            circuit.cx(ictrl, itarg)
            circuit.rz(-2. * np.pi + halfphi, itarg)
            # Canceling out the adjacent cxs
            #circuit.cx(ictrl, itarg)

            #circuit.cx(ictrl, itarg)
            circuit.cx(itarg, ictrl)
            circuit.cx(ictrl, itarg)

            if barrier:
                circuit.barrier()

            # Shift the control qubit
            ictrl -= 1

    if measure:
        # Map onto classical register in the original order of qubits
        for iq in range(n1 + n2):
            circuit.measure(n3 + iq, iq)
        for iq in range(n3):
            circuit.measure(iq, n1 + n2 + iq)

    return circuit
