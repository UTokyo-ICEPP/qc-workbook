# Tested with python 3.8.12, qiskit 0.34.2, numpy 1.22.2
import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def make_vqe_circuit(n,nl,npar):

    # n = 3   # 量子ビット数
    # nl = 2  # レイヤー数
    # npar = n*2*nl   # パラメータ数

    qc = QuantumCircuit(n)
    param_list = ParameterVector('param_list',npar)
    for i in range(nl):
        qc.ry(param_list[6*i], 0)
        qc.ry(param_list[6*i+1], 1)
        qc.ry(param_list[6*i+2], 2)
        qc.rz(param_list[6*i+3], 0)
        qc.rz(param_list[6*i+4], 1)
        qc.rz(param_list[6*i+5], 2)
        #qc.cnot(0, 1)
        #qc.cnot(1, 2)

        
    return qc
