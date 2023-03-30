"""Utility functions for qc-workbook."""

from typing import Callable, Optional, Union, Tuple
import collections
import numpy as np

def operational_backend(
    min_qubits: int = 0,
    min_qv: int = 0,
    qubits: Optional[int] = None,
    qv: Optional[int] = None
) -> Callable[['Backend'], bool]:
    """Returns a filter function that flags operational backends.

    The returned function can be passed to provider.backends(). Operational backends are those
    that satisfy all of the following criteria:

    - Not a simulator
    - `backend.status().operational` is True
    - If `qubits` is set, the number of qubits matches the value exactly. If not and if `min_qubits` is set,
      the number of qubits is greater than or equal to the given value.
    - If `qv` is is set, quantum volume matches the value exactly. If not and if `min_qv` is set,
      quantum volume is greater than or equal to the given value.

    Args:
        min_qubits: Minimum number of qubits.
        min_qv: Minimum value of quantum volume.
        qubits: Exact number of qubits.
        qv: Exact value of quantum volume.

    Returns:
        A function that takes a Backend object and returns True if the backend is operational.
    """

    def backend_filter(backend):
        if not backend.status().operational:
            return False

        config = backend.configuration()

        if config.simulator:
            return False

        if qubits is not None:
            if config.n_qubits != qubits:
                return False
        else:
            if config.n_qubits < min_qubits:
                return False

        if qv is not None:
            if config.quantum_volume != qv:
                return False
        else:
            if config.quantum_volume < min_qv:
                return False

        return True

    return backend_filter


def find_best_chain(
    backend: 'Backend',
    length: int,
    return_error_prod: bool = False
) -> Union[Tuple[int], Tuple[Tuple[int, float, float]]]:
    """Find a chain of qubits with the smallest product of CNOT and measurement errors.

    Args:
        backend: IBMQ backend.
        length: Length of the chain.
        return_error_prod: If True, returns log(prod(gate error)) and log(prod(readout error)) along with the qubit IDs.

    Returns:
        Qubit IDs (int) of the best chain, or a tuple containing the tuple of qubit IDs, log(prod(gate error)),
        and log(prod(readout error)).
    """

    # Put the couplings into a dict for convenience
    couplings = collections.defaultdict(list)
    for pair in backend.configuration().coupling_map:
        couplings[pair[0]].append(pair[1])

    # Recursive function to form a list of chains given a starting qubit
    def make_chains(qubit, chain=tuple()):
        chain += (qubit,)

        if len(chain) == length:
            return [chain]

        chains = []
        for neighbor in couplings[qubit]:
            if neighbor in chain:
                continue

            chains += make_chains(neighbor, chain)

        return chains

    # Get all chains starting from all qubits
    chains = []
    for qubit in range(backend.configuration().n_qubits):
        chains += make_chains(qubit)

    # Find the chain with the smallest error (CX and readout) product
    prop = backend.properties()

    min_log_gate_error = 0.
    min_log_readout_error = 0.
    best_chain = None
    for chain in chains:
        log_gate_error = sum(np.log(prop.gate_error('cx', [q1, q2])) for q1, q2 in zip(chain[:-1], chain[1:]))
        log_readout_error = sum(np.log(prop.readout_error(q)) for q in chain)

        if log_gate_error + log_readout_error < min_log_gate_error + min_log_readout_error:
            min_log_gate_error = log_gate_error
            min_log_readout_error = log_readout_error
            best_chain = chain

    if return_error_prod:
        return best_chain, min_log_gate_error, min_log_readout_error
    else:
        return best_chain
