---
jupytext:
  notebook_metadata_filter: all
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.10.6
---

# 【Exercise】Finding Operations of Bit Reversing Board

Here we consider a problem of converting a given number to another using a hypothetical board "Bit Reversing Board" and Grove's algorithm.

+++ {"pycharm": {"name": "#%% md\n"}}

## Problem Setup

In {doc}`Grover's algorithm <grover>`, we considered the problem of finding 45 from the list containing $N=2^6$ elements ($=[0,1,2,\cdots,63]$) is considered. In this exercise, we extend this 6-qubit search problem, as follows. 

We have a board in our hand, and we write down a number in binary format. For example, 45 is written as $101101(=45)$ in the board with 6 slots. 

```{image} figs/grover_kadai1.png
:alt: grover_kadai1
:width: 500px
:align: center
```

This board has a property that when one *pushes down* the bit of a certain digit, that bit and neighboring bits are *reversed*. For example, in the case of 45, if one pushes down the second bit from the highest digit, the number is changed to $010101(=21)$.

```{image} figs/grover_kadai2.png
:alt: grover_kadai2
:width: 500px
:align: center
```

In this exercise, we attemp to convert a certain number, say 45, to another number, e.g, 13 using this board. In particular, we want to find a **sequence (= the order of pushing down the bits) of the smallest number of bit operations** to reach the desired number.

```{image} figs/grover_kadai3.png
:alt: grover_kadai3
:width: 500px
:align: center
```

+++ {"pycharm": {"name": "#%% md\n"}}

## Hint

There are many approaches to tackle the problem, but we can consider a quantum circuit with 3 quantum registers and 1 classical register.

- Quantum register to store a number on the board = *board*
- Quantum register to store a pattern of pushing down the board = *flip*
- Quantum register to store a bit whose phase is flipped when the number on the board is equal to the desired one = *oracle*
- Clasical register to hold the measurement result = *result*

You could try the followings using this circuit:

1. Set 45 as an initial state on the board register.
2. Create the superposition of *all possible patterns of pushing down the 6 qubits* in the flip register for a 6-bit number problem. 
3. Implement quantum gates to reverse bits in the board register for individual bit operations. 
4. Construct unitary to flip phase of the oracle register when the number on the board register is what we want.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
# Tested with python 3.8.12, qiskit 0.34.2, numpy 1.22.2
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import AerSimulator
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
# Consider 6-qubit search problem
n = 6  # Number of qubits

# Number of Grover iterations = Closest integer to pi/4*sqrt(2**6)
niter = 6

# Registers
board = QuantumRegister(n)   # Register to store the board number
flip = QuantumRegister(n)   # Register to store the pattern of pushing down the board
oracle = QuantumRegister(1)   # Register for phase flip when the board number is equal to the desired one.
result = ClassicalRegister(n)   # Classical register to hold the measurement result
```

Complete the following cell.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
qc = QuantumCircuit(board, flip, oracle, result)

##################
### EDIT BELOW ###
##################

# Write down the qc circuit here.

##################
### ABOVE BELOW ###
##################
```

Run the code using simulator and check the result. Among the results of bit sequences, those with 10 highest occurrences are displayed below as the final score.  

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
# Run on simulator
backend = AerSimulator()
qc_tr = transpile(qc, backend=backend)
job = backend.run(qc_tr, shots=8000)
result = job.result()
count = result.get_counts()

score_sorted = sorted(count.items(), key=lambda x:x[1], reverse=True)
final_score = score_sorted[0:10]

print('Final score:')
print(final_score)
```

+++ {"pycharm": {"name": "#%% md\n"}}

**Items to submit**
- Quantum circuit to solve the problem.
- Result demonstrating that the pattern to convert 45 to 13 is successfully found.
