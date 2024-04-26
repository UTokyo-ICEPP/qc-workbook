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

# Confirming the violation of the CHSH inequality

+++

In this first exercise, we will confirm that the device that we call a quantum computer indeed exhibits quantum mechanical behavior &mdash; entanglement, in particular. You will be introduced to the concepts of quantum mechanics and the fundamentals of quantum computing through this exercise.

```{contents} Contents
---
local: true
---
```

$\newcommand{\ket}[1]{|#1\rangle}$
$\newcommand{\rmI}{\mathrm{I}}$
$\newcommand{\rmII}{\mathrm{II}}$
$\newcommand{\rmIII}{\mathrm{III}}$
$\newcommand{\rmIV}{\mathrm{IV}}$

+++

## Is this really a quantum computer?

The aim of this workbook is to familiarize yourself with quantum computers (QCs), but until just a few years ago, QCs were something that only existed in science fiction. However, now we are told that they are available as real computational resources over the cloud &mdash; but are these devices really QCs? How can we check?

The fundamental way that QCs work is that they manipulate a physical system, composed of elements such as superconducting resonators or cold atoms, so that the results of computations are expressed in the quantum state of the system. In other words, a device can be called a quantum computer only when it contains a physical system whose quantum state can be manipulated in certain ways and also be retained for a long period of time. There must also be algorithms that translate abstract calculations into tangible operations on the physical system. The algorithm portion will be introduced little by little throughout this workbook. Here, let's confirm that our "QC" really operates a quantum mechanical system.

+++

## The CHSH inequality

One way to experimentally verify that a system behaves quantum mechanically is to test the violation of the CHSH inequality{cite}`PhysRevLett.23.880,nielsen_chuang_epr`. To put it briefly, the CHSH inequality is an inequality of specific observables in a two-body system that is satisfied unless there are quantum-specific phenomena such as entanglement. Said more straightforwardly, if you measure these observables in a device and their values violate the CHSH inequality, the device may actually be using quantum phenomena.

Normally, this type of experiment would require a highly sophisticated lab setup (involving laser, non-linear crystals, cold atoms, etc.). But with a cloud-based QC, all that's needed is a web browser. In this workbook, you will use Jupyter Notebooks to write Python programs and run them on <a href="https://quantum.ibm.com/" target="_blank">IBM Quantum</a> devices.

+++

## The basic structure of Qiskit

[Qiskit](https://qiskit.org/), a Python library provided by IBM, is used for programming IBM quantum computers. The basic procedure for using Qiskit is as follows:

1. Decide on the number of quantum bits to use.
1. Apply quantum computation operations (gates) to the quantum bits to create a quantum circuit.
1. Execute the circuit and produce calculation results. Here, there are two options:
   - Send the circuit to the actual QC device and get the results back.
   - Simulate the circuit on your computer.
1. Analyze the calculation results.

You will perform this process below, while being introduced to various important concepts. In this exercise, you will only use the actual device. Please refer to the {doc}first assignment <nonlocal_correlations> for information on simulating circuits.

+++

## Quantum bits and quantum registers

**Quantum bits**, or **qubits**, are the fundamental elements that make up the quantum computer. They are the smallest possible unit of quantum information. A number of quantum bits gathered together is referred to as a quantum register.

A quantum register in a quantum computer is always in one "state." Following a common practice used in physics, the states of quantum registers are referred to as "kets" and denoted with a symbol like $\ket{\psi}$[^mixed_state]. If you are unfamiliar with quantum mechanics, this notation method may look intimidating, but the ket itself is merely a symbol, so there is no need to be overly concerned. You could also write it as $\psi$ without the enclosure, or even use the 🔱 emoji if you want. Anything works!

What's important is that two **basis states** can be defined for each qubit. Conventionally, the two basis states are labled as $\ket{0}$ and $\ket{1}$, and form what is called a computational basis. Any state of a qubit can then be expressed through a *superposition* of the two computational basis states, using complex numbers $\alpha$ and $\beta$:

$$
\alpha \ket{0} + \beta \ket{1}
$$

In quantum mechanics, the coefficients $\alpha$ and $\beta$ are called the probability amplitudes, or simply **amplitudes**. Again, the actual formatting used is not particularly significant. The states could just as well be written as $[\alpha, \beta]$[^complexarray].

Another way to look at this is that a single qubit carries an amount of information equivalent to two complex numbers. However, there is a caveat: Due to the rules of quantum mechanics, the amplitudes $\alpha$ and $\beta$ must satisfy the following requirement.

$$
|\alpha|^2 + |\beta|^2 = 1
$$

Furthermore, the overall complex phase of a quantum state is not physically meaningful. In other words, for an arbitrary real number $\theta$,

$$
\alpha \ket{0} + \beta \ket{1} \sim e^{i\theta} (\alpha \ket{0} + \beta \ket{1}),
$$

where $\sim$ indicates that the two sides represent the same quantum state.

A single complex number can be written using two real numbers, so $\alpha$ and $\beta$ together would appear to have the same amount of information as four real numbers. However, due to these two constraints, the actual degree of freedom is 4-2=2. Another expression of the state of a qubit that makes the number of degrees of freedom more explicit is

$$
e^{-i\phi/2}\cos\frac{\theta}{2}\ket{0} + e^{i\phi/2}\sin\frac{\theta}{2}\ket{1},
$$

which is sometimes called the Bloch sphere notation.

Things get more interesting when there are multiple qubits. For example, if there are two, each has a $\ket{0}, \ket{1}$ computational basis, so the overall state is a superposition parametrized by four complex numbers,

$$
\alpha \ket{0}\ket{0} + \beta \ket{0}\ket{1} + \gamma \ket{1}\ket{0} + \delta \ket{1}\ket{1}.
$$

The "products" of the computational basis states of the two qubits, $\ket{0}\ket{0}, \ket{0}\ket{1}, \ket{1}\ket{0}$, and $\ket{1}\ket{1}$, are the computational basis states of this quantum register. Their abbreviated notations are $\ket{00}, \ket{01}, \ket{10}$, and $\ket{11}$.

The rules of quantum mechanics regarding the amplitudes in this case are

$$
|\alpha|^2 + |\beta|^2 + |\gamma|^2 + |\delta|^2 = 1
$$

and

$$
\alpha \ket{00} + \beta \ket{01} + \gamma \ket{10} + \delta \ket{11} \sim e^{i\theta} (\alpha \ket{00} + \beta \ket{01} + \gamma \ket{10} + \delta \ket{11}).
$$

There are only two constraints, regardless of the number of qubits.

In other words, a register with $n$ qubits has $2^n$ basis states with a complex amplitude for each, so *the amount of information that it carries is equivalent to $2 \times 2^n - 2$ real numbers*. This is why the word "exponential" often appears in the discussion of quantum calculation.

There is another frequently used notation for the computational basis states of quantum registers. One could look at the string of 0/1s that appears in the ket as a binary number and express it with the corresponding decimal number. For example, the four-qubit register states $\ket{0000}$ and $\ket{1111}$ can be expressed as $\ket{0}$ and $\ket{15}$, respectively. However, in this case, care must be taken to clearly indicate which qubit, the leftmost or the rightmost, corresponds to the least significant bit (LSB). Whether $\ket{0100}$ becomes $\ket{4}$ (the rightmost qubit is the LSB) or $\ket{2}$ (the leftmost is the LSB) depends on which convention is used. In this workbook, in accordance with the definition used in Qiskit, we take the rightmost qubit to be the LSB. Furthermore, we want to make the first qubit of the register correspond to the LSB. Therefore, when expressing a computational basis state with an array of kets or 0/1s, the register's qubits will be arranged from right to left.

Qiskit has a quantum register class, whose instance is created as:
```{code-block} python
from qiskit import QuantumRegister
register = QuantumRegister(4, 'myregister')
```
i.e., by specifying the number of qubits (four in this case) and the name (`'myregister'`). By default, all qubits will be in the $\ket{0}$ state. The register object is not very useful by itself. Instead, they are usually used as parts in quantum circuits, which are introduced below.

[^mixed_state]: Strictly speaking, the state of a register can be expressed with kets only when the register is not entangled with other registers, but we will skip the details here.
[^complexarray]: In the simulations of QCs on classical computers, the states of the quantum registers are expressed with arrays of complex numbers, which corresponds well with this notation.

+++

### Quantum gates, quantum circuits, and measurement

All quantum calculation that we will consider in this textbook amount to generating specific states in a quantum register and then using its amplitude.
However, you can't simply conjure up whatever quantum state you want. Instead, complex states are created by combining, in order, simple operations with defined patterns (such as swapping $\ket{0}$ and $\ket{1}$, updating the phase angle $\phi$ in Bloch sphere representation, etc.). These simple operations are generally referred to as quantum **gates**, and programs that specify types and sequences of these gates are called quantum **circuits**.

Qiskit represents quantum circuits using `QuantumCircuit` objects.
```{code-block} python
from qiskit import QuantumCircuit, QuantumRegister
register = QuantumRegister(4, 'myregister')
circuit = QuantumCircuit(register)
```

The quantum circuit object above is defined with a fixed number of qubits but nothing else, and requires gates to be added. For example, the following is used to apply a Hadamard gate (explained below) to the second quantum bit of a register.
```{code-block} python
circuit.h(register[1])
```

Now we know how a state can be generated on a register. For the next step, we deliberately used the vague expression "using its amplitude" above, because there are many ways to use the amplitude. However, no matter the method, the quantum register is always **measured**. Measurement is the only way to obtain information from a quantum computer. In Qiskit, qubits of a circuit is measured like so:
```{code-block} python
circuit.measure_all()
```

Measurement is like "peeking" at the state of the quantum register. Practically, what happens in each measurement operation is that we simply obtain a binary value (0 or 1) for each qubit. In other words, even if the measured register is in a quantum state expressed as a complex superposition of $2^n$ computational basis states, a bit sequence that corresponds to one such computational basis is output upon measurement. What's more, when a quantum bit is measured, its state is fixed to the computational basis state that it was measured to be in, and the complex superposition is lost.

How do we know exactly which bit sequence will be obtained in a measurement? Actually, except in special cases, we don't know. You can repeat the state preparation and measurement using the exact same circuit, and the bit sequence will be decided randomly each time. However, there does exist a law of quantum mechanics that governs this randomness, and it is that **the probability of obtaining a specific bit sequence is given by the square of the norm of the amplitude of the corresponding computational basis**. In other words, when the state of an $n$-bit register is $\sum_{j=0}^{2^n-1} c_j \ket{j}$, the probability of bit string $|c_k|^2$ being obtained is $|c_k |^2$.

+++

### Analysis of the result of quantum computation

If you repeatedly run a circuit including a measurement and record the frequency with which each bit sequence occurs, you can estimate the values $|c_j|^2$ $(j=0, 1, \dots, 2^n-1)$. For example, if you ran a two-qubit circuit 1000 times and measured "00" 246 times, "01" 300 times, "10" 103 times, and "11" 351 times, the estimates would be $|c_0|^2=0.24 \pm 0.01$, $|c_1|^2=0.30 \pm 0.01$, $|c_2|^2=0.11 \pm 0.01$, and $|c_3|^2=0.35 \pm 0.01$ with statistical uncertainties. However, obviously what you will have determined is only the absolute value of $c_j$. There is simply no way of knowing the complex phase. That may be unsatisfying, but that's how you obtain information from quantum computers.

Conversely, the true essence of quantum algorithm design lies in skillfully designing circuits that exploits the exponentially large state space internally and yet produces meaningful results through the limited method of measurement. For example, ideally, if the answer to some calculation is the integer $k$, and the final state of a circuit used to perform that calculation is simply $\ket{k}$, then you will know the answer after a single measurement (corresponding to the special case mentioned above). Even if the final state is a superposition $\sum_{j=0}^{2^n-1} c_j \ket{j}$, if $|c_k| \gg |c_{j \neq k}|$, the answer can be determined with a high likelihood after several measurements. The phase estimation algorithm introduced in {doc}`shor` is a good example of such a case.

Aside from these special cases, quantum computation will invariably involve large numbers of circuit trials ("shots"). The number of shots is an important parameter to be specified whenever a circuit is passed to a quantum computer or a simulator for execution.

+++

(common_gates)=
### Common gates

In superconducting quantum computers such as the IBM systems that we will use, the only gates that are availble for building a circuit are single-qubit gates (gates that act on a single qubit) and a few two-qubit gates. That does not sound enough to realize various quantum states needed for computation. However, it has been mathematically proven that certain sets of single-qubit and two-qubit gates can be composed to generate states on a $n$-qubit register that approximates any $n$-qubit quantum states to arbitrary precision. A subset of the gates introduced below indeed forms such a *universal gate set*. In other words, knowing the functions of the gates below will be sufficient to design quantum circuits for any computation tasks!

#### Single-qubit gates

The following gates are often used with single qubit operations. (`i` and `j` in the code are the qubit numbers.)

```{list-table}
:header-rows: 1
* - Gate name
  - Explanation
  - Qiskit code
* - $X$
  - Switches $\ket{0}$ and $\ket{1}$.
  - `circuit.x(i)`
* - $Z$
  - Multiplies the amplitude of $\ket{1}$ by -1.
  - `circuit.z(i)`
* - $H$（Hadamard gate）
  - Applies the following transformation to each computational basis.
    ```{math}
    H\ket{0} = \frac{1}{\sqrt{2}} (\ket{0} + \ket{1}) \\
    H\ket{1} = \frac{1}{\sqrt{2}} (\ket{0} - \ket{1})
    ```
     (When using ket notation to indicate that a gate is applied to a quantum state, the symbol of the gate is written to the left of the ket.)
     For example, for $\ket{\psi} = \alpha\ket{0} + \beta\ket{1}$, the Hadamard gate acts as

    ```{math}
    \begin{align}
    H\ket{\psi} & = \alpha \frac{1}{\sqrt{2}} (\ket{0} + \ket{1}) + \beta \frac{1}{\sqrt{2}} (\ket{0} - \ket{1}) \\
                & = \frac{1}{\sqrt{2}} (\alpha + \beta) \ket{0} + \frac{1}{\sqrt{2}} (\alpha - \beta) \ket{1}.
    \end{align}
    ```
  - `circuit.h(i)`
* - $R_{y}$
  - Takes a parameter $\theta$ and applies the following transformation to each computational basis.
    ```{math}
    R_{y}(\theta)\ket{0} = \cos\frac{\theta}{2}\ket{0} + \sin\frac{\theta}{2}\ket{1} \\
    R_{y}(\theta)\ket{1} = -\sin\frac{\theta}{2}\ket{0} + \cos\frac{\theta}{2}\ket{1}
    ```
  - `circuit.ry(theta, i)`
* - $R_{z}$
  - Takes a parameter $\phi$ and applies the following transformation to each computational basis.
    ```{math}
    R_{z}(\phi)\ket{0} = e^{-i\phi/2}\ket{0} \\
    R_{z}(\phi)\ket{1} = e^{i\phi/2}\ket{1}
  - `circuit.rz(phi, i)`
```

Let's create a circuit in Qiskit that applies an $H$, an $R_y$, and an $X$ gate sequentially to the 0th bit of a 2-qubit register and then measures it.

```{code-cell} ipython3
:tags: [remove-output]

# First, import all the necessary python modules
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.accounts import AccountNotFoundError
# qc_workbook is the original module written for this workbook
# If you encounter an ImportError, edit the environment variable PYTHONPATH or sys.path
from qc_workbook.utils import operational_backend

print('notebook ready')
```

```{code-cell} ipython3
circuit = QuantumCircuit(2) # You can also create a circuit by specifying the number of bits, without using a register
circuit.h(0) # In that case, directly specify the number of the quantum bit for the gate, not register[0]
circuit.ry(np.pi / 2., 0) #　θ = π/2
circuit.x(0)
# Measurement is always needed to get an output
circuit.measure_all()

print(f'This circuit has {circuit.num_qubits} qubits and {circuit.size()} operations')
```

The reason the last print statement says "5 operations" despite there only being three gates is that each quantum bit measurement is also counted as an operation.

As an exercise, let's look at what happens to the 0th bit through these gate operations. Since the gates are applied from the left in the ket notation (as opposed to being appended to the right in the circuit visualization), we will calculate $X R_y(\pi/2) H \ket{0}$.

$$
\begin{align}
X R_y\left(\frac{\pi}{2}\right) H \ket{0} & = X R_y\left(\frac{\pi}{2}\right) \frac{1}{\sqrt{2}}(\ket{0} + \ket{1}) \\
& = \frac{1}{\sqrt{2}} X \left[\left(\cos\left(\frac{\pi}{4}\right)\ket{0} + \sin\left(\frac{\pi}{4}\right)\ket{1}\right) + \left(-\sin\left(\frac{\pi}{4}\right)\ket{0} + \cos\left(\frac{\pi}{4}\right)\ket{1}\right)\right] \\
& = \frac{1}{\sqrt{2}} X \frac{1}{\sqrt{2}} \left[\left(\ket{0} + \ket{1}\right) + \left(-\ket{0} + \ket{1}\right)\right] \\
& = X \ket{1} \\
& = \ket{0}
\end{align}
$$

We see that these operations eventually take us back to the $\ket{0}$ state.

#### Two-qubit gates

Two-qubit gates on superconducting quantum computers are typically realized as *controlled gates* for reasons related to how qubits are implemented on such devices. In a controlled gate, one of the two qubits is called the control and the other is called the target, and a nontrivial (not identitiy) operation is applied to the target qubit only for computational bases where the value of the control qubit is 1.

More concretely, a controlled-$U$ gate $C^i_j[U]$, where $U$ is some single-qubit gate and $i$ and $j$ are the indices of the control and target qubits, acts as

$$
\begin{align}
C^i_j[U](\ket{0}_i\ket{0}_j) & = \ket{0}_i\ket{0}_j \\
C^i_j[U](\ket{0}_i\ket{1}_j) & = \ket{0}_i\ket{1}_j \\
C^i_j[U](\ket{1}_i\ket{0}_j) & = \ket{1}_iU\ket{0}_j \\
C^i_j[U](\ket{1}_i\ket{1}_j) & = \ket{1}_iU\ket{1}_j.
\end{align}
$$

Of the $X$, $Z$, $H$, $R_y$, and $R_z$ gates introduced in the previous section, all except the $H$ gate commonly appear in the controlled form in various circuits. Particularly, $C[X]$, also known as CX or CNOT, is often used as a basic element in quantum calculation.

```{list-table}
:header-rows: 1
* - Gate name
  - Explanation
  - Qiskit code
* - $C^i_j[X]$, CX, CNOT
  - Performs the operation of gate $X$ on bit $j$ for a computational basis in which the value of bit $i$ is 1.
  - `circuit.cx(i, j)`
* - $C^i_j[Z]$
  - Reverses the sign of the computational basis when the values of bits $i$ and $j$ are 1.
  - `circuit.cz(i, j)`
* - $C^i_j[R_{y}]$
  - Obtains parameter $\theta$ and performs the operation of gate $R_y$ on bit $j$ for a computational basis in which the value of bit $i$ is 1.
  - `circuit.cry(theta, i, j)`
* - $C^i_j[R_{z}]$
  - Obtains parameter $\phi$ and performs the operation of gate $R_z$ on bit $j$ for a computational basis in which the value of bit $i$ is 1.
  - `circuit.crz(phi, i, j)`
```

Let's write a quantum circuit where controlled gates are used to a 2-bit register to make the squared amplitudes of computational bases $\ket{0}$, $\ket{1}$, $\ket{2}$, and $\ket{3}$ have the ratio $1:2:3:4$. Furthermore, let's use a $C_1^0[Z]$ gate to flip the sign of the amplitude of only $\ket{3}$.

```{code-cell} ipython3
theta1 = 2. * np.arctan(np.sqrt(7. / 3.))
theta2 = 2. * np.arctan(np.sqrt(2.))
theta3 = 2. * np.arctan(np.sqrt(4. / 3))

circuit = QuantumCircuit(2)
circuit.ry(theta1, 1)
circuit.ry(theta2, 0)
circuit.cry(theta3 - theta2, 1, 0) # C[Ry] 1 is the control and 0 is the target
circuit.cz(0, 1) # C[Z] 0 is the control and 1 is the target (in reality, for C[Z], the results are the same regardless of which the control is)

circuit.measure_all()

print(f'This circuit has {circuit.num_qubits} qubits and {circuit.size()} operations')
```

This is a little complex, but let's follow the computation steps in order. First, given the definitions of angles $θ_1$,$θ_2$, and $θ_3$, the following relationships are satisfied.

$$
\begin{align}
R_y(\theta_1)\ket{0} & = \sqrt{\frac{3}{10}} \ket{0} + \sqrt{\frac{7}{10}} \ket{1} \\
R_y(\theta_2)\ket{0} & = \sqrt{\frac{1}{3}} \ket{0} + \sqrt{\frac{2}{3}} \ket{1} \\
R_y(\theta_3 - \theta_2)R_y(\theta_2)\ket{0} & = R_y(\theta_3)\ket{0} = \sqrt{\frac{3}{7}} \ket{0} + \sqrt{\frac{4}{7}} \ket{1}.
\end{align}
$$

Therefore,

$$
\begin{align}
& C^1_0[R_y(\theta_3 - \theta_2)]R_{y1}(\theta_1)R_{y0}(\theta_2)\ket{0}_1\ket{0}_0 \\
= & C^1_0[R_y(\theta_3 - \theta_2)]\left(\sqrt{\frac{3}{10}} \ket{0}_1 + \sqrt{\frac{7}{10}} \ket{1}_1\right) R_y(\theta_2)\ket{0}_0\\
= & \sqrt{\frac{3}{10}} \ket{0}_1 R_y(\theta_2)\ket{0}_0 + \sqrt{\frac{7}{10}} \ket{1}_1 R_y(\theta_3)\ket{0}_0 \\
= & \sqrt{\frac{3}{10}} \ket{0}_1 \left(\sqrt{\frac{1}{3}} \ket{0}_0 + \sqrt{\frac{2}{3}} \ket{1}_0\right) + \sqrt{\frac{7}{10}} \ket{1}_1 \left(\sqrt{\frac{3}{7}} \ket{0}_0 + \sqrt{\frac{4}{7}} \ket{1}_0\right) \\
= & \sqrt{\frac{1}{10}} \ket{00} + \sqrt{\frac{2}{10}} \ket{01} + \sqrt{\frac{3}{10}} \ket{10} + \sqrt{\frac{4}{10}} \ket{11},
\end{align}
$$

where the $R_y$ gates that are applied to bits 0 and 1 in the last line are denoted as $R_{y0}$ and $R_{y1}$.

When $C[Z]$ is applied at the end, only the sign of $\ket{11}$ is reversed.

+++

### Visualizing quantum circuits

There is a standard way of visualizing quantum circuits. With Qiskit, you can use the `draw()` method of the QuantumCircuit to automatically draw circuit diagrams.

```{code-cell} ipython3
circuit.draw('mpl')
```

Here, the `'mpl'` is used to draw the circuit diagram in color, using the matplotlib library. Some operating environments may not support this. In those cases, use `draw()` with no argument. The result will not be as visually appealing as the circuit diagrams produced by `mpl`, but the content will be the same.

```{code-cell} ipython3
circuit.draw()
```

Circuit diagrams are read from left to right. The two horizontal solid lines represent, from top to bottom, quantum bits 0 and 1. The squares on top of the lines are the gates. The boxes at the end, with arrows extending downwards, represent measurements. The vertical lines with circles at their ends extending from the 1 bit gate represent control. The double line at the very bottom corresponds to the "classical register", which is where the measurement results of qubits 0 and 1 are recorded.

+++

## Building the circuits for CHSH test

Let us now get to the main part of this exercise. The CHSH inequality is an inequality involving four observables of a two-body system, so we will prepare four 2-bit circuits. Each will represent the Bell state $1/\sqrt{2}(\ket{00} + \ket{11})$. The Bell state is one in which "the states of both of the quantum bits are neither $\ket{0}$ nor $\ket{1}$." In other words, despite the fact that the overall state is pure, its individual parts are not. In situations such as this, we say that the two quantum bits are entangled. The existence of entanglement is an extremely important feature of quantum mechanics.

Let's create a Bell state by combining Hadamard gates and CNOT gates. We will defer detailed explanations to {doc}`nonlocal_correlations` and will just say that we build circuits I, II, III, and IV. In circuits I and III, we apply an $R_y(-\pi/4)$ to qubit 1 right before the measurement, while in circuits II and IV an $R_y(-3\pi/4)$ is applied instead. Additionally, circuits III and IV have an $R_y(-\pi/2)$ gate on qubit 0 before the measurement. The circuits are appended to a list named `circuits` to be passed to the quantum computer at the end.

```{code-cell} ipython3
circuits = []

# Circuit I - H, CX[0, 1], Ry(-π/4)[1]
circuit = QuantumCircuit(2, name='circuit_I')
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(-np.pi / 4., 1)
circuit.measure_all()
# Append to list
circuits.append(circuit)

# Circuit II - H, CX[0, 1], Ry(-3π/4)[1]
circuit = QuantumCircuit(2, name='circuit_II')
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(-3. * np.pi / 4., 1)
circuit.measure_all()
# Append to list
circuits.append(circuit)

# Circuit III - H, CX[0, 1], Ry(-π/4)[1], Ry(-π/2)[0]
circuit = QuantumCircuit(2, name='circuit_III')
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(-np.pi / 4., 1)
circuit.ry(-np.pi / 2., 0)
circuit.measure_all()
# Append to list
circuits.append(circuit)

# Circuit IV - H, CX[0, 1], Ry(-3π/4)[1], Ry(-π/2)[0]
circuit = QuantumCircuit(2, name='circuit_IV')
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(-3. * np.pi / 4., 1)
circuit.ry(-np.pi / 2., 0)
circuit.measure_all()
# Append to list
circuits.append(circuit)

# draw() can accept a matplotlib Axes object as an argument, to which the circuit will be drawn
# This is useful when visualizing multiple circuits from a single Jupyter cell
fig, axs = plt.subplots(2, 2, figsize=[12., 6.])
for circuit, ax in zip(circuits, axs.reshape(-1)):
    circuit.draw('mpl', ax=ax)
    ax.set_title(circuit.name)
```

Let's calculate the probability of basis states $\ket{00}$, $\ket{01}$, $\ket{10}$, and $\ket{11}$ appearing in the 2-bit register of each circuit.

The state of circuit 1 is as follows:

$$
\begin{align}
R_{y1}\left(-\frac{\pi}{4}\right) C^0_1[X] H_0 \ket{0}_1\ket{0}_0 = & R_{y1}\left(-\frac{\pi}{4}\right) \frac{1}{\sqrt{2}} (\ket{0}_1\ket{0}_0 + \ket{1}_1\ket{1}_0) \\
= & \frac{1}{\sqrt{2}} \big[(c\ket{0}_1 - s\ket{1}_1)\ket{0}_0 + (s\ket{0}_1 + c\ket{1}_1)\ket{1}_0\big]\\
= & \frac{1}{\sqrt{2}} (c\ket{00} + s\ket{01} - s\ket{10} + c\ket{11}),
\end{align}
$$

where for notational simplicity we have set $c = \cos(\pi/8)$ and $s = \sin(\pi/8)$.
Therefore the probabilities $P^{\rmI}_{l} \, (l=00,01,10,11)$ for circuit I are

$$
P^{\rmI}_{00} = P^{\rmI}_{11} = \frac{c^2}{2} \\
P^{\rmI}_{01} = P^{\rmI}_{10} = \frac{s^2}{2}.
$$

Likewise, the state for circuit II is as follows:

```{math}
:label: eqn-circuit1
R_{y1}\left(-\frac{3\pi}{4}\right) \frac{1}{\sqrt{2}} (\ket{0}_1\ket{0}_0 + \ket{1}_1\ket{1}_0) = \frac{1}{\sqrt{2}} (s\ket{00} + c\ket{01} - c\ket{10} + s\ket{11}),
```

which yields probabilities $P^{\rmII}_{l}$ of

$$
P^{\rmII}_{00} = P^{\rmII}_{11} = \frac{s^2}{2} \\
P^{\rmII}_{01} = P^{\rmII}_{10} = \frac{c^2}{2}.
$$

For circuit III,

$$
\begin{align}
& R_{y1}\left(-\frac{\pi}{4}\right) R_{y0}\left(-\frac{\pi}{2}\right) \frac{1}{\sqrt{2}} (\ket{0}_1\ket{0}_0 + \ket{1}_1\ket{1}_0) \\
= & \frac{1}{\sqrt{2}} \left[ \frac{1}{\sqrt{2}} (c\ket{0}_1 - s\ket{1}_1) (\ket{0}_0 - \ket{1}_0) + \frac{1}{\sqrt{2}} (s\ket{0}_1 + c\ket{1}_1) (\ket{0}_0 + \ket{1}_0) \right] \\
= & \frac{1}{2} \big[ (s+c)\ket{00} + (s-c)\ket{01} - (s-c)\ket{10} + (s+c)\ket{11} \big],
\end{align}
$$

and the probabilities $P^{\rmIII}_{l}$ are

$$
P^{\rmIII}_{00} = P^{\rmIII}_{11} = \frac{(s + c)^2}{4} \\
P^{\rmIII}_{01} = P^{\rmIII}_{10} = \frac{(s - c)^2}{4}.
$$

Finally, the state and probabilities $P^{\rmIV}_l$ for circuit IV are:

$$
\begin{align}
& R_{y1}\left(-\frac{3\pi}{4}\right) R_{y0}\left(-\frac{\pi}{2}\right) \frac{1}{\sqrt{2}} (\ket{0}_1\ket{0}_0 + \ket{1}_1\ket{1}_0) \\
= & \frac{1}{2} \big[ (s+c)\ket{00} - (s-c)\ket{01} + (s-c)\ket{10} + (s+c)\ket{11} \big],
\end{align}
$$

$$
P^{\rmIV}_{00} = P^{\rmIV}_{11} = \frac{(s + c)^2}{4} \\
P^{\rmIV}_{01} = P^{\rmIV}_{10} = \frac{(s - c)^2}{4}.
$$

For each circuit, we define the difference between the probability $P^{i}_{00} + P^{i}_{11}$ of observing the same value for bits 0 and 1 (even parity) and the probability $P^{i}_{01} + P^{i}_{10}$ of observing different values (odd parity) as $C^i$.

$$
C^{\rmI} = c^2 - s^2 = \cos\left(\frac{\pi}{4}\right) = \frac{1}{\sqrt{2}} \\
C^{\rmII} = s^2 - c^2 = -\frac{1}{\sqrt{2}} \\
C^{\rmIII} = 2sc = \sin\left(\frac{\pi}{4}\right) = \frac{1}{\sqrt{2}} \\
C^{\rmIV} = 2sc = \frac{1}{\sqrt{2}}.
$$

Therefore, combining these, the value of $S = C^{\rmI} - C^{\rmII} + C^{\rmIII} + C^{\rmIV}$ is $2\sqrt{2}$.

Actually, if there is no entanglement, the value of this observable $S$ is known to not exceed 2. For example, if the state before the $R_y$ gates is not a Bell state but a mixed state where there is a $\frac{1}{2}$ chance that the value is $\ket{00}$ and a $\frac{1}{2}$ chance that it is $\ket{11}$, we get

$$
C^{\rmI} = \frac{1}{\sqrt{2}} \\
C^{\rmII} = -\frac{1}{\sqrt{2}} \\
C^{\rmIII} = 0 \\
C^{\rmIV} = 0
$$

and therefore $S = \sqrt{2} < 2$. This is the CHSH inequality.

Now let's check if the IBM "quantum computer" really generates entangled states by using the four circuits above to calculate the value of $S$.

+++

## Executing the circuits on a quantum device

We first connect to and authenticate with the IBM cloud infrastructure. If you are running on IBM Quantum Experience (a Jupyter Lab environment under the IBM Quantum website) or if you are in a local environment with the the {ref}`authentcation information already saved to disk <install_token>`, a line like
```{code-block} python
service = QiskitRuntimeService(channel='ibm_quantum')
```
is all you need. Otherwise, you need to pass your `authentication token <install_token>` to the constructor of `QiskitRuntimeService`.

```{code-cell} ipython3
:tags: [remove-output, raises-exception]

# Specify an instance if you have access to multiple (e.g. premium access plan）
# instance = 'hub-x/group-y/project-z'
instance = None

try:
    service = QiskitRuntimeService(channel='ibm_quantum', instance=instance)
except AccountNotFoundError:
    service = QiskitRuntimeService(channel='ibm_quantum', token='__paste_your_token_here__', instance=instance)
```

Once authentication has been completed, choose the quantum computer you wish to use (called a "backend").

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

# Find the backend that is operational and has the shortest job queue
backend = service.least_busy(filters=operational_backend())

print(f'Jobs will run on {backend.name}')
```

Use the `transpile()` function and the `run()` method of the backend object to submit your circuits to the backend. Transpilation will be explained in {ref}`transpilation`, so just think of it as a technical necessity for now. As explained above, we specify the number of shots when submitting a circuit with the `run()` method. The maximum allowed number of shots per job depends on the backend and your access plan.

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

# max_shots = the maximum number of allowed shots for this backend with the access parameters
shots = min(backend.max_shots, 2000)
print(f'Running four circuits, {shots} shots each')

circuits = transpile(circuits, backend=backend)
# Execute each circuit for `shots` times
job = backend.run(circuits, shots=shots)
```

This will send the circuits to the backend as a job, which is added to the queue. The job execution results is checked using the job object, returned by the `run()` method.

IBM Quantum backends are used by many users around the world, so in some cases there may be many jobs in the queue and it may take a long time for your job to be executed. The length of the queue for each backend can be seen at right on the <a href=https://quantum.ibm.com/services/resources" target="_blank">IBM Quantum Experience website under the Compute Resources tab</a>. Clicking one of the backends will display details about the backend.

Use the <a href="https://quantum.ibm.com/jobs" target="_blank">Jobs tab</a> to see the status of jobs you have submitted.

+++

## Analysis of the result

Calling the `result()` method for a job object will block further execution of the program until the job is complete and a result arrives. The execution results will be returned as an object. Use the `get_counts()` method on this object to obtain histogram data on how many times each bit sequence was observed in the form of a Python dict.

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

result = job.result()

# List to collect the histogram data from the four circuits
counts_list = []

# Extracting the bit sequence counts from the result object
for idx in range(4):
    # get_counts(i) returns the histogram data for circuit i
    counts = result.get_counts(idx)
    # Append to list
    counts_list.append(counts)

print(counts_list)
```

```{code-cell} ipython3
:tags: [remove-cell]

# To be ignored (dummy cell)
try:
    counts_list
except NameError:
    counts_list = [
        {'00': 3339, '01': 720, '10': 863, '11': 3270},
        {'00': 964, '01': 3332, '10': 3284, '11': 612},
        {'00': 3414, '01': 693, '10': 953, '11': 3132},
        {'00': 3661, '01': 725, '10': 768, '11': 3038}
    ]

    shots = 8192
```

This information can be visualized using Qiskit's `plot_histogram` function.

```{code-cell} ipython3
fig, axs = plt.subplots(2, 2, sharey=True, figsize=[12., 8.])
for counts, circuit, ax in zip(counts_list, circuits, axs.reshape(-1)):
    plot_histogram(counts, ax=ax)
    ax.set_title(circuit.name)
    ax.yaxis.grid(True)
```

Since the expected values are $c^2/2 = (s + c)^2/4 = 0.427$, $s^2/2 = (s - c)^2 / 4 = 0.073$, so the probability was fairly close to the mark.

The reality is that current quantum computers still have various types of noise and errors, so calculation results sometimes deviate from theoretical values beyond statistical uncertainties. There are methods for mitigating specific errors to some degree, but there is no way to prevent all errors. Current quantum computers are known as *Noisy intermediate-scale quantum (NISQ)* devices. This "noisy" aspect is very evident even in simple experiments like this.

To use NISQ devices effectively requires robust circuits that produce meaningful results despite noise and errors. A great deal of attention is being turned to methods for achieving this, such as optimization using the variational quantum circuits introduced in {doc}`vqe`.

Let us finish this section by confirming the violation of the CHSH inequality. We determine the value of $S$ by calculating $C^{\rmI}$, $C^{\rmII}$, $C^{\rmIII}$, and $C^{\rmIV}$.

```{code-cell} ipython3
# Convert the counts dictionary to numpy arrays
c_value = np.zeros(4, dtype=float)

for ic, counts in enumerate(counts_list):
    # Use counts.get('00', 0) instead of counts['00']
    c_value[ic] = counts.get('00', 0) + counts.get('11', 0) - counts.get('01', 0) - counts.get('10', 0)

# Normalize the sums of counts to get the expectation values
c_value /= shots

s_value = c_value[0] - c_value[1] + c_value[2] + c_value[3]

print('C:', c_value)
print('S =', s_value)
if s_value > 2.:
    print('Yes, we are using a quantum computer!')
else:
    print('Armonk, we have a problem.')
```

$S$ is, indeed, greater than 2.
