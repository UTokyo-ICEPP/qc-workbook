---
jupytext:
  notebook_metadata_filter: all
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
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
  version: 3.10.12
---

+++ {"pycharm": {"name": "#%% md\n"}}

# Performing Database Search

+++

In this unit, we'll introduce **Grover's algorithm**{cite}`grover_search,nielsen_chuang_search` and consider the problem of searching for an answer in an unstructured database using this algorithm. After that, We will implement Grover's algorithm using Qiskit.

```{contents} Contents
---
local: true
---
```

$\newcommand{\ket}[1]{| #1 \rangle}$
$\newcommand{\bra}[1]{\langle #1 |}$
$\newcommand{\braket}[2]{\langle #1 | #2 \rangle}$

+++

## Introduction

In order to realize quantum advantage over classical computation, one has to exploit quantum algorithm that can take advantage of quantum properties. One example of such algorithms is Grover's algorithm. Grover's algorithm is suited for **searching in unstructured database**, and it has been proven that Grover's algorithm can find solutions with fewer computational resources than those required for classical counterparts. This algorithm is based on a method known as **amplitude amplification** and is widely used as a subroutine in various quantum algorithms.

+++

(database)=
## Searching for Unstructured Data

Let us consider that there is a list consisting of $N$ elements, and we want to find one element $w$ in that list. To find $w$ using a classical computer, out would have to query the list $N$ times in a worst case, or on average $N/2$ times. With Grover's algorithm, it is known that $w$ can be found by querying about $\sqrt{N}$ times. That it, Glover's algorithm allows one to search for unstructured data quadratically faster than the classical calculation.

+++

(grover)=
## Grover's Algorithm

Here we will consider $n$ qubits, and the list composed of every possible computational basis states. In other words, the list contains $N=2^n$ elements, composed of $\ket{00\cdots00}$, $\ket{00\cdots01}$, $\cdots$, $\ket{11\cdots11}$ ($\ket{0}$, $\ket{1}$, $\cdots$, $\ket{N-1}$ in decimal form).

+++

(grover_phaseoracle)=
### Introduction of Phase Oracle

The Grover's algorithm is characterized by the existence of phase oracle, which changes phase of a certain state. First, let us consider a phase oracle $U$ defined as $U\ket{x}=(-1)^{f(x)}\ket{x}$, that is, when acting on the state $\ket{x}$, it shifts the phase by $-1^{f(x)}$ with a certain function $f(x)$. If we consider the function $f(x)$ to be like below:

$$
f(x) = \bigg\{
\begin{aligned}
&1 \quad \text{if} \; x = w \\
&0 \quad \text{else} \\
\end{aligned}
$$

this leads to the oracle (denoted as $U_w$) which inverts the phase for the answer $w$ that we want:

$$
U_w:\begin{aligned}
&\ket{w} \to -\ket{w}\\
&\ket{x} \to \ket{x} \quad \forall \; x \neq w
\end{aligned}
$$

If we use a matrix form, the $U_w$ can be written as as $U_w=I-2\ket{w}\bra{w}$. Furthermore, if we think of another function $f_0(x)$:

$$
f_0(x) = \bigg\{
\begin{aligned}
&0 \quad \text{if} \; x = 0 \\
&1 \quad \text{else} \\
\end{aligned}
$$

then we can get unitary $U_0$ that inverts phases for all the states except 0.

$$
U_0:\begin{aligned}
&\ket{0}^{\otimes n} \to \ket{0}^{\otimes n}\\
&\ket{x} \to -\ket{x} \quad \forall \; x \neq 0
\end{aligned}
$$

Here the matrix form of $U_0$ is given as $U_0=2\ket{0}\bra{ 0}^{\otimes n}-I$.

+++

(grover_circuit)=
### Structure of Quantum Circuit

The structure of the circuit used to implement Grover's algorithm is shown below. Starting with the $n$-qubit initial state $\ket{0}$, a uniform superposition state is created first by applying Hadamard gates to every qubits. Then, the operator denoted as $G$ is applied repeatedly.

```{image} figs/grover.png
:alt: grover
:width: 600px
:align: center
```

$G$ is a unitary operator called **Grover Iieration** and consists of the following four steps.

```{image} figs/grover_iter.png
:alt: grover_iter
:width: 550px
:align: center
```

The $U_w$ and $U_0$ are oracles that invert the phase of an answer $w$ and the phase of all states other than 0, respectively, as introduced above.

Together with the Hadamard operator at the beginning of the circuit, we will look at steps involved in a single Grover iteration in detail below.

```{image} figs/grover_iter1.png
:alt: grover_iter1
:width: 600px
:align: center
```

+++

(grover_superposition)=
### Creation of Superposition State

First, a uniform superposition state is produced by applying Hadamard gates to initial state $\ket{0}^{\otimes n}$ of the $n$-qubit circuit.

$$
\ket{s} = H^{\otimes n}\ket{0}^{\otimes n} = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}\ket{x}
$$

This state is denoted as $\ket{s}$.

+++

(grover_geometry)=
### Geometrical Representation

Let's view this $\ket{s}$ state geometrically. First, consider a two-dimensional plane created by the superposition state $\ket{s}$ and the state $\ket{w}$, which is what we are trying to find. Since the state $\ket{w^{\perp}}$, which is orthogonal to $\ket{w}$, can be expressed as $\ket{w^{\perp}}:=\frac{1}{\sqrt{N-1}}\sum_{x \neq w}\ket{x}$, it corresponds to the axis orthogonal to $\ket{w}$ in this 2D plane. Therefore, $\ket{w^{\perp}}$ and $\ket{w}$ can be regarded as orthonormal basis states, i.e, $\ket{w^{\perp}}=\begin{bmatrix}1\\0\end{bmatrix}$, $\ket{w}=\begin{bmatrix}0\\1\end{bmatrix}$.

In short, $\ket{s}$ can be expressed as a linear combination of the two vectors ($\ket{w^{\perp}}$, $\ket{w}$) on this 2D plane.
$$
\begin{aligned}
\ket{s}&=\sqrt{\frac{N-1}{N}}\ket{w^{\perp}}+\frac1{\sqrt{N}}\ket{w}\\
&=: \cos\frac\theta2\ket{w^{\perp}}+\sin\frac\theta2\ket{w}\\
&= \begin{bmatrix}\cos\frac\theta2\\\sin\frac\theta2\end{bmatrix}
\end{aligned}
$$

In above equations, the amplitude of $\ket{w}$ is $\frac1{\sqrt{N}}$ and the amplitude of $\ket{w^{\perp}}$ is $\sqrt{\frac{N-1}{N}}$ because we want to find only one answer. If we define $\theta$ to fulfill $\sin\frac\theta2=\frac1{\sqrt{N}}$, then the $\theta$ is expressed as

$$
\theta=2\arcsin\frac{1}{\sqrt{N}}
$$

The $\ket{s}$ state on ($\ket{w^{\perp}}$, $\ket{w}$) plane is depicted as follows.

```{image} figs/grover_rot1.png
:alt: grover_rot1
:width: 300px
:align: center
```

+++

(grover_oracle)=
### Application of Oracle

Next, we will apply the oracle $U_w$ oracle to $\ket{s}$. This oracle can be expressed on this plane as $U_w=I-2\ket{w}\bra{ w}=\begin{bmatrix}1&0\\0&-1\end{bmatrix}$. This indicates that the action of $U_w$ is equivalent to the inversion of $\ket{s}$ with respect to the $\ket{w^{\perp}}$ axis (see figure below), hence reversing the phase of $\ket{w}$.

```{image} figs/grover_rot2.png
:alt: grover_rot2
:width: 300px
:align: center
```

+++

(grover_diffuser)=
### Application of Diffuser

Next is the application of $H^{\otimes n}U_0H^{\otimes n}$, and this operation is called Diffuser. Since $U_0=2\ket{0}\bra{0}^{\otimes n}-I$, if we define $U_s$ to be $U_s \equiv H^{\otimes n}U_0H^{\otimes n}$, then it is expressed as

$$
\begin{aligned}
U_s &\equiv H^{\otimes n}U_0H^{\otimes n}\\
&=2H^{\otimes n}\ket{0}^{\otimes n}\bra{0}^{\otimes n}H^{\otimes n}-H^{\otimes n}H^{\otimes n}\\
&=2\ket{s}\bra{ s}-I\\
&=\begin{bmatrix}\cos\theta&\sin\theta\\\sin\theta&-\cos\theta\end{bmatrix}
\end{aligned}
$$

This means that the diffuser $U_s$ is an operator that inverts $U_w\ket{s}$ with respect to $\ket{s}$ (see figure below).

```{image} figs/grover_rot3.png
:alt: grover_rot3
:width: 300px
:align: center
```

In summary, the Grover iteration $G=U_sU_w$ is written as

$$
\begin{aligned}
G&=U_sU_w\\
&= \begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
\end{aligned}
$$

and is equivalent to rotating the $\ket{s}$ towards $\ket{w}$ by the angle $\theta$ (figure below).

```{image} figs/grover_rot4.png
:alt: grover_rot4
:width: 300px
:align: center
```

This correspondence between $G$ and the $\theta$ rotation means that if $G$ is applied $r$ times, $\ket{s}$ is rotated by $r\theta$. After that, the state of $\ket{s}$ becomes

$$
G^r\ket{s}=\begin{bmatrix}\cos\frac{2r+1}{2}\theta\\\sin\frac{2r+1}{2}\theta\end{bmatrix}
$$

This indicates that $\ket{s}$ would need to be rotated $r$ times so that $\frac{2r+1}2\theta\approx\frac{\pi}2$ to reach the desired answer of $\ket{w}$. If each rotation angle $\theta$ is small enough, then $\sin\frac\theta2=\frac{1}{\sqrt{N}}\approx\frac\theta2$, hence $r\approx\frac\pi4\sqrt{N}$. Now we have shown that {\cal O}(\sqrt{N})$ operations would allow us to reach the desired answer $\ket{w}$, meaning that it is quadratically faster than the classical calculation.

Let's look at the diffuser's role a bit more. We can think of a certain state $\ket{\psi}$ and assume that it is written as a superposition of some bases $\ket{k}$ with amplitude $a_k$, $\ket{\psi}:=\sum_k a_k\ket{k}$. When we apply the diffuser to this state,

$$
\begin{aligned}
\left( 2\ket{s}\bra{ s} - I \right)\ket{\psi}&=\frac2N\sum_i\ket{i}\cdot\sum_{j,k}a_k\braket{j}{k}-\sum_k a_k\ket{k}\\
&= 2\frac{\sum_i a_i}{N}\sum_k\ket{k}-\sum_k a_k\ket{k}\\
&= \sum_k \left( 2\langle a \rangle-a_k \right)\ket{k}
\end{aligned}
$$

$\langle a \rangle\equiv\frac{\sum_i a_i}{N}$ is an average of the amplitudes. If you consider the amplitude $a_k$ of the state $\ket{k}$ to be expressed in the form of deviation from the average, $a_k=\langle a \rangle-\Delta$, then this equation could be better understood. That is, the amplitude will become $2\langle a \rangle-a_k=\langle a \rangle+\Delta$ after applying the diffuser. This means that the action of diffuser corresponds to the inversion of the amplitudes with respect to the average $\langle a \rangle$.

+++

(grover_amp)=
### Visualize Amplitude Amplification

Let us visualize how the amplitude of the state corresponding to correct answer is amplified.

First, the Hadamard gates applied at the beginning of the circut create a superposition of all the computational basis states with equal amplitudes ((1) in the figure below). The horizontal axis shows $N$ computational basis states, and the vertical axis shows the magnitude of the amplitude for each basis state and it is $\frac{1}{\sqrt{N}}$ for all the states (the average is shown by a dotted red line).

Next, applying the oracle $U_w$ inverts the phase of $\ket{w}$, making the amplitude to $-\frac{1}{\sqrt{N}}$ ((2) of the figure). In this state, the average amplitude is $\frac{1}{\sqrt{N}}(1-\frac2N)$, which is lower than that in state (1).

Last, the diffuser is applied to the state and all the amplitudes are reversed with respect to the average ((3) of the figure). This increases the amplitude of $\ket{w}$ and decreases the amplitudes of all other basis states. As seen in the figure, the amplitude of $\ket{w}$ state is amplified roughly three times. Repeating this process will further increase the amplitude of $\ket{w}$, so we can expect that we will have higher chance of getting the correct answer.

```{image} figs/grover_amp.png
:alt: grover_amp
:width: 800px
:align: center
```

+++

(grover_multidata)=
### Searching for Multiple Data

We have only considered so far searching for a single data. What if we consider finding multiple data? For example, we want to find $M$ data $\{w_i\}\;(i=0,1,\cdots,M-1)$ from a sample of $N=2^n$ data. Just as before, we can discuss this situation on a two-dimensional plance with the state we want to find, $\ket{w}$, and its orthogonal state, $\ket{w^{\perp}}$.

$$
\begin{aligned}
&\ket{w}:=\frac{1}{\sqrt{M}}\sum_{i=0}^{M-1}\ket{w_i}\\
&\ket{w^{\perp}}:=\frac{1}{\sqrt{N-M}}\sum_{x\notin\{w_0,\cdots,w_{M-1}\}}\ket{x}
\end{aligned}
$$

$\ket{s}$ can be expressed on this plane as follows:

$$
\begin{aligned}
\ket{s}&=\sqrt{\frac{N-M}{N}}\ket{w^{\perp}}+\sqrt{\frac{M}{N}}\ket{w}\\
&=: \cos\frac\theta2\ket{w^{\perp}}+\sin\frac\theta2\ket{w}\\
\end{aligned}
$$

If the amplitude $\sqrt{\frac{M}{N}}$ of the state $\ket{w}$ is defined as $\sin\frac\theta2$, then the angle $\theta$ is $\theta=2\arcsin\sqrt{\frac{M}{N}}$. Compared to the case of finding a single data, the angle rorated by the single Grover iteration is $\sqrt{M}$ times larger. As a result, we can reach the answer by a smaller number of rotations, $r\approx\frac\pi4\sqrt{\frac{N}{M}}$, than the single-data case.

+++ {"pycharm": {"name": "#%% md\n"}}

(imp)=
## Implementation of Grover's Algorithm（Case of $N=2^6$）

Now let's try to solve database search problem by implementing Grover's algorithm.

This exercise is to find a single answer "45" in a list containing $N=2^6$ elements of $[0,1,2,\cdots,63]$ (Of course, you could look for another number and feel free to do that later if you like). That is, we try to find $\ket{45}=\ket{101101}$ using a 6-qubit quantum circuit.

+++

(imp_qiskit)=
### Qiskit Implementation

Set up the environment first.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
# Tested with python 3.8.12, qiskit 0.34.2, numpy 1.22.2
import matplotlib.pyplot as plt
import numpy as np

# Import qiskit packages
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.accounts import AccountNotFoundError

# Original module for this workbook
from qc_workbook.utils import operational_backend
```

Prepare a 6-qubit circuit `grover_circuit`.

A quantum circuit to perform a single Grover iteration will be something like below, and please write a quantum circuit that implements gates outlined in red (phase oracle and the unitary corresponding to $2\ket{0}\bra{0}-I$ of the diffuser).

```{image} figs/grover_6bits_45.png
:alt: grover_6bits_45
:width: 600px
:align: center
```

Implement the oracle after generating a uniform superposition state $\ket{s}$.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
tags: [remove-output]
---
Nsol = 45
n = 6

grover_circuit = QuantumCircuit(n)

grover_circuit.h(range(n))

# Create the oracle and implement it in the circuit
oracle = QuantumCircuit(n)

##################
### EDIT BELOW ###
##################

#oracle.?

##################
### EDIT ABOVE ###
##################

oracle_gate = oracle.to_gate()
oracle_gate.name = "U_w"
print(oracle)

grover_circuit.append(oracle_gate, list(range(n)))
grover_circuit.barrier()
```

**Answer**

````{toggle}

```{code-block} python

##################
### EDIT BELOW ###
##################

oracle.x(1)
oracle.x(4)
oracle.h(n-1)
oracle.mcx(list(range(n-1)), n-1)
oracle.h(n-1)
oracle.x(1)
oracle.x(4)

##################
### EDIT ABOVE ###
##################
```

````

Next, implement the diffuser circuit.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
def diffuser(n):
    qc = QuantumCircuit(n)

    qc.h(range(n))

    ##################
    ### EDIT BELOW ###
    ##################

    #qc.?

    ##################
    ### EDIT ABOVE ###
    ##################

    qc.h(range(n))

    #print(qc)
    U_s = qc.to_gate()
    U_s.name = "U_s"
    return U_s

grover_circuit.append(diffuser(n), list(range(n)))
grover_circuit.measure_all()
grover_circuit.decompose().draw('mpl')
```

**Answer**

````{toggle}

```{code-block} python
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
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)

    qc.x(list(range(n)))

    ##################
    ### EDIT ABOVE ###
    ##################

    qc.h(range(n))

    #print(qc)
    U_s = qc.to_gate()
    U_s.name = "U_s"
    return U_s
```

````


(imp_simulator)=
### Experiment with Simulator

Once you have implemented the circuit, run the simulator and make a plot of the results. To make the results easy to understand, the measured bitstring is converted to integers before making the plot.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
tags: [remove-output]
---
simulator = AerSimulator()
grover_circuit = transpile(grover_circuit, backend=simulator)
results = simulator.run(grover_circuit, shots=1024).result()
answer = results.get_counts()

# Plot the values along the horizontal axis in integers
def show_distribution(answer):
    n = len(answer)
    x = [int(key,2) for key in list(answer.keys())]
    y = list(answer.values())

    fig, ax = plt.subplots()
    rect = ax.bar(x,y)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.3f}'.format(height/sum(y)),
                        xy=(rect.get_x()+rect.get_width()/2, height),xytext=(0,0),
                        textcoords="offset points",ha='center', va='bottom')
    autolabel(rect)
    plt.ylabel('Probabilities')
    plt.show()

show_distribution(answer)
```

If the circuit is implemented correctly, you will see that the state $\ket{101101}=\ket{45}$ is measured with high probability.

However, as discussed above, a single Grover iteration will produce incorrect answers with non-negligible probabilities in the search of $N=2^6$ elements. Later, we will see if repeating Grover iteration can produce correct answers with higher probabilities.

+++

(imp_qc)=
### Experiment with Quantum Computer

Before attempting multiple Grover iterations, let us first run a single Grover iteration on a quantum computer.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
# Specify an instance here if you have access to multiple
# instance = 'hub-x/group-y/project-z'
instance = None

try:
    service = QiskitRuntimeService(channel='ibm_quantum', instance=instance)
except AccountNotFoundError:
    service = QiskitRuntimeService(channel='ibm_quantum', token='__paste_your_token_here__', instance=instance)

# Find the backend with the shortest queue
backend = service.least_busy(min_num_qubits=6, filters=operational_backend())

print(f'Jobs will run on {backend.name}')
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
# Execute the circuit on the backend with the highest level of availability. Monitor job execution in the queue.

grover_circuit = transpile(grover_circuit, backend=backend, optimization_level=3)
job = backend.run(grover_circuit, shots=1024)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
# Calculation results
results = job.result()
answer = results.get_counts(grover_circuit)
show_distribution(answer)
```

As you can see, the results are much worse than what we got with the simulator. Unfortunately, this is a typical result of running a quantum circuit of Grover's search algorithm on the present quantum computer as it is. We can however expect that the quality of the results will be improved by employing {ref}`error mitigation <measurement_error_mitigation>` techniques.

+++ {"pycharm": {"name": "#%% md\n"}}

(imp_simulator_amp)=
### Confirm Amplitude Amplification

Here we will see how the amplitude is amplified by running the Grover's iteration multiple times using simulator.

For example, the circuit to execute the Grover's iteration three times is prepared and executed.

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
# Repetition of Grover's iteration
Niter = 3

grover_circuit_iterN = QuantumCircuit(n)
grover_circuit_iterN.h(range(n))
for I in range(Niter):
    grover_circuit_iterN.append(oracle_gate, list(range(n)))
    grover_circuit_iterN.append(diffuser(n), list(range(n)))
grover_circuit_iterN.measure_all()
grover_circuit_iterN.draw('mpl')
```

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
grover_circuit_iterN_tr = transpile(grover_circuit_iterN, backend=simulator)
results = simulator.run(grover_circuit_iterN_tr, shots=1024).result()
answer = results.get_counts()
show_distribution(answer)
```

+++ {"pycharm": {"name": "#%% md\n"}}

You will see that the correct answer of $\ket{45}$ appears with higher probability.

Next, we make a plot of showing the correlation between the number of Grover's iterations and how many times we observe the correct answer. Here the Grover's iteration is repated 10 times.

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
x = []
y = []

# Repeating Grover's iteration 10 times
for Niter in range(1,11):
    grover_circuit_iterN = QuantumCircuit(n)
    grover_circuit_iterN.h(range(n))
    for I in range(Niter):
        grover_circuit_iterN.append(oracle_gate, list(range(n)))
        grover_circuit_iterN.append(diffuser(n), list(range(n)))
    grover_circuit_iterN.measure_all()

    grover_circuit_iterN_tr = transpile(grover_circuit_iterN, backend=simulator)
    results = simulator.run(grover_circuit_iterN_tr, shots=1024).result()
    answer = results.get_counts()

    x.append(Niter)
    y.append(answer[format(Nsol,'b').zfill(n)])

plt.clf()
plt.scatter(x,y)
plt.xlabel('N_iterations')
plt.ylabel('# of correct observations (1 solution)')
plt.show()
```

+++ {"pycharm": {"name": "#%% md\n"}}

The result will show that the correct answer is observed with the highest probability when repeating the Grover's iteration $5\sim6$ times. Please check if this is consistent with what we analytically obtained above as the most probable number of iterations to get a correct answer.

+++ {"pycharm": {"name": "#%% md\n"}}

Exercise : Consider the case of a single answer. Examine the most probable number of iterations obtained using simulator and the size of the list, $N$, when $N$ varies from $N=2^4$ to $N=2^{10}$.

+++ {"pycharm": {"name": "#%% md\n"}}

(imp_simulator_multi)=
### Case of Searching for Multiple Data

Now we consider searching for multiple data from the list. Let us modify the circuit to be able to find two integers $x_1$ and $x_2$, and make a plot of the number of Grover's iterations versus how many times we observe the correct answer.

For example, $x_1=45$ and $x_2=26$,

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
N1 = 45
N2 = 26

oracle_2sol = QuantumCircuit(n)

# 45
oracle_2sol.x(1)
oracle_2sol.x(4)
oracle_2sol.h(n-1)
oracle_2sol.mcx(list(range(n-1)), n-1)
oracle_2sol.h(n-1)
oracle_2sol.x(1)
oracle_2sol.x(4)

# 26
oracle_2sol.x(0)
oracle_2sol.x(2)
oracle_2sol.x(5)
oracle_2sol.h(n-1)
oracle_2sol.mcx(list(range(n-1)), n-1)
oracle_2sol.h(n-1)
oracle_2sol.x(0)
oracle_2sol.x(2)
oracle_2sol.x(5)

oracle_2sol_gate = oracle_2sol.to_gate()
oracle_2sol_gate.name = "U_w(2sol)"
print(oracle_2sol)

x = []
y = []
for Niter in range(1,11):
    grover_circuit_2sol_iterN = QuantumCircuit(n)
    grover_circuit_2sol_iterN.h(range(n))
    for I in range(Niter):
        grover_circuit_2sol_iterN.append(oracle_2sol_gate, list(range(n)))
        grover_circuit_2sol_iterN.append(diffuser(n), list(range(n)))
    grover_circuit_2sol_iterN.measure_all()
    #print('-----  Niter =',Niter,' -----------')
    #print(grover_circuit_2sol_iterN)

    grover_circuit_2sol_iterN_tr = transpile(grover_circuit_2sol_iterN, backend=simulator)
    results = simulator.run(grover_circuit_2sol_iterN_tr, shots=1024).result()
    answer = results.get_counts()
    #show_distribution(answer)

    x.append(Niter)
    y.append(answer[format(N1,'06b')]+answer[format(N2,'06b')])

plt.clf()
plt.scatter(x,y)
plt.xlabel('N_iterations')
plt.ylabel('# of correct observations (2 solutions)')
plt.show()
```

+++ {"pycharm": {"name": "#%% md\n"}}

In this case, the number of iterations to have the highest probability is smaller than that in the case of single answer. Is this what you expected, right?

```{code-cell} ipython3

```
