# Tested with python 3.8.12, qiskit 0.34.2, numpy 1.22.2
import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit

def make_grover_circuit(n_qubits):
    
    grover_circuit = QuantumCircuit(n_qubits)

    grover_circuit.h(range(n_qubits))

    # オラクルを作成して、回路に実装
    oracle = QuantumCircuit(n_qubits)

    # N=45のオラクル
    ##################
    ### EDIT BELOW ###
    ##################
    oracle.x(1)
    oracle.x(4)
    oracle.h(n_qubits-1)
    oracle.mct(list(range(n_qubits-1)), n_qubits-1)
    oracle.h(n_qubits-1)
    oracle.x(1)
    oracle.x(4)
    ##################
    ### EDIT ABOVE ###
    ##################
    oracle_gate = oracle.to_gate()
    oracle_gate.name = "U_w"
    
    def diffuser(n):
        qc = QuantumCircuit(n)

        qc.h(range(n))

        ##################
        ### EDIT BELOW ###
        ##################
        qc.rz(2*np.pi, n-1)
        qc.x(list(range(n)))
        
        # multi-controlled Zゲート
        qc.h(n-1)
        qc.mct(list(range(n-1)), n-1)
        qc.h(n-1)
        
        qc.x(list(range(n)))        
        ##################
        ### EDIT ABOVE ###
        ##################
    
        qc.h(range(n))
        
        U_s = qc.to_gate()
        U_s.name = "U_s"
        return U_s


    grover_circuit.append(oracle_gate, list(range(n_qubits)))
    grover_circuit.append(diffuser(n_qubits), list(range(n_qubits)))
    grover_circuit.measure_all()
    #grover_circuit.decompose().draw('mpl')

    return grover_circuit
