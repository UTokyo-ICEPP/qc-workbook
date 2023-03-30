# Tested with python 3.8.12, qiskit 0.34.2, numpy 1.22.2
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def make_vqe_circuit(
    num_qubits: int,
    num_layers: int,
    num_pararameters: int
) -> QuantumCircuit:

    # n = 3   # 量子ビット数
    # nl = 2  # レイヤー数
    # npar = n*2*nl   # パラメータ数

    qc = QuantumCircuit(num_qubits)
    param_list = ParameterVector('param_list', num_pararameters)
    for i in range(num_layers):
        qc.ry(param_list[6*i], 0)
        qc.ry(param_list[6*i+1], 1)
        qc.ry(param_list[6*i+2], 2)
        qc.rz(param_list[6*i+3], 0)
        qc.rz(param_list[6*i+4], 1)
        qc.rz(param_list[6*i+5], 2)
        #qc.cnot(0, 1)
        #qc.cnot(1, 2)


    return qc
