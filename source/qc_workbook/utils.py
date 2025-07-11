"""Utility functions for qc-workbook."""

from typing import Callable, Optional
import collections
import numpy as np
from qiskit.providers.backend import BackendV2 as Backend

def operational_backend(
    min_qubits: int = 0,
    min_qv: int = 0,
    qubits: Optional[int] = None,
    qv: Optional[int] = None
) -> Callable:

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
            if config.quantum_volume is None or config.quantum_volume != qv:
                return False
        elif min_qv > 0:
            if config.quantum_volume is None or config.quantum_volume < min_qv:
                return False

        return True

    return backend_filter


def find_best_chain(backend: Backend, length: int, return_error_prod: bool = False):
    """Find a chain of qubits with the smallest product of CNOT and measurement errors."""
    # Bidirectional neighbors list
    neighbors = collections.defaultdict(set)
    for qubit in range(backend.num_qubits):
        for nq in backend.coupling_map.neighbors(qubit):
            neighbors[qubit].add(nq)
            neighbors[nq].add(qubit)

    # Recursive function to form a list of chains given a starting qubit
    def make_chains(qubit, chain=()):
        chain += (qubit,)

        if len(chain) == length:
            return [chain]

        chains = []
        for neighbor in neighbors[qubit]:
            if neighbor in chain:
                continue

            chains += make_chains(neighbor, chain)

        return chains

    # Get all chains starting from all qubits
    chains = []
    for qubit in range(backend.num_qubits):
        chains += make_chains(qubit)

    # Find the chain with the smallest error (CX and readout) product
    entangling_gate = 'ecr' if 'ecr' in backend.basis_gates else 'cz'

    min_log_gate_error = 0.
    min_log_readout_error = 0.
    best_chain = None
    for chain in chains:
        log_gate_error = 0.
        for q1, q2 in zip(chain[:-1], chain[1:]):
            try:
                err = backend.target[entangling_gate][(q1, q2)].error
            except KeyError:
                err = backend.target[entangling_gate][(q2, q1)].error
            log_gate_error += np.log(err)

        log_readout_error = sum(np.log(backend.target['measure'][(q,)].error) for q in chain)

        if log_gate_error + log_readout_error < min_log_gate_error + min_log_readout_error:
            min_log_gate_error = log_gate_error
            min_log_readout_error = log_readout_error
            best_chain = chain

    if return_error_prod:
        return best_chain, min_log_gate_error, min_log_readout_error
    else:
        return best_chain
