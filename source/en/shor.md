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

# Learning the Integer Factoring Algorithm

+++

In this exercise, you will be learning about **Shor's algorithm**. You may have heard this name before, as Shor's algorithm{cite}`shor,nielsen_chuang_qft_app` is one of the most famous quantum algorithms. After learning the method called **quantum phase estimation**, on which Shor's algorithm is based, we will then introduce each step of Shor's algorithm, together with actual examples. Lastly, we will use Qiskit to implement Shor's algorithm and exercise integer factoring.

```{contents} Contents
---
local: true
---
```

$\newcommand{\ket}[1]{|#1\rangle}$
$\newcommand{\modequiv}[3]{#1 \equiv #2 \pmod{#3}}$
$\newcommand{\modnequiv}[3]{#1 \not\equiv #2 \pmod{#3}}$

(shor_introduction)=
## Introduction

One of the most widely known examples of how the capabilities of quantum computation far surpass classical computation is Shor's algorithm. The problem this algorithm seeks to address is to break down a large positive number into two prime numbers. The problem itself is a simple one. However, there are no known classical computation algorithms that can efficiently performe integer factoring, and as the number in question grows larger, the amount of calculation involved is believed to **grow exponentially**.  Using Shor's algorithm is believed to make it possible to solve this problem in **polynomial time** (generally speaking, if an algorithm can solve a problem with a computation time that increases polynomially with respect to the problem size, the algorithm is considered efficient).

The difficulty involved in performing integer factoring with classical calculation is the basis of the encryption technologies that are currently widely in use. Therefore, if it were possible to realize an exponentially-fast Shor's algorithm using a quantum computer, it could result in the leakage of confidential information. This is why so much attention is being paid to Shor's algorithm.

+++

(qpe)=
## Quantum Phase Estimation

First, let's learn about **quantum phase estimation** or QPE, on which Shor's algorithm is based. If you understand Shor's algorithm, you will realize that the heart of the algorithm is basically QPE itself. To understand QPE, it is essential that you understand the **quantum Fourier transform** or QFT. For more information about QFT, please refer to task 7 of [this exercise](circuit_from_scratch) or to reference material[1].

Since the QPE is a very important technique, it is widely used, not only in Shor's algorithm but also in various algorithms as a subroutine.

The question QPE seeks to address is
"Given a unitary operation $U$ and an eigenvector $\ket{\psi}$ that satisfies $U\ket{\psi}=e^{2\pi i\theta}\ket{\psi}$, what is the phase $\theta$ of the eigenvalue $e^{2\pi i\theta}$?"

+++

(qpe_1qubit)=
### Single-Qubit QPE
First, let us consider a quantum circuit like the one in the figure below. Here, the upper qubit is $\ket{0}$ and the lower qubit is $U$'s eigenvector $\ket{\psi}$.

```{image} figs/qpe_1qubit.png
:alt: qpe_1qubit
:width: 300px
:align: center
```

In this case, the quantum states in each of steps 1 to 3 of the quantum circuit can be expressed as follows.

- Step 1 : $\frac{1}{\sqrt{2}}(\ket{0}\ket{\psi}+\ket{1}\ket{\psi})$
- Step 2 : $\frac{1}{\sqrt{2}}(\ket{0}\ket{\psi}+\ket{1} e^{2\pi i\theta}\ket{\psi})$
- Step 3 : $\frac{1}{2}\left[(1+e^{2\pi i\theta})\ket{0}+(1-e^{2\pi i\theta})\ket{1}\right]\ket{\psi}$

If we measure the upper qubit in this state, there is a $|(1+e^{2\pi i\theta})/2|^2$ probability that the value is 0 and a $|(1-e^{2\pi i\theta})/2|^2$ probability that it is 1. In other words, we can determine the value of $\theta$ from these probabilities. However, when the value of $\theta$ is small ($\theta\ll1$), there is an almost 100% probability that the measured value will be 0 and an almost 0% probability that it will be 1. Therefore, in order to measure small discrepancies from 100% or 0%, we will have to perform measurements many times. This does not make for a particularly superior approach.

```{hint}
This controlled-$U$ gate corresponds to the "oracle" of Shor's algorithm. The inverse quantum Fourier transform comes (QFT) after that, and it is an $H$ gate for one-qubit case ($H=H^\dagger$). In other words, a single-qubit QFT is an $H$ gate itself.
```

Returning to the topic at hand, let us see if there any way to determine the phase more accurately, using only a small number of measurements.

+++ {"pycharm": {"name": "#%% md\n"}}

(qpe_nqubit)=
### $n$-Qubit QPE
Let us think of a quantum circuit with the upper register expanded to $n$ qubits (as shown in the figure below).

```{image} figs/qpe_wo_iqft.png
:alt: qpe_wo_iqft
:width: 500px
:align: center
```

As a result of this, $U$ is repeatedly applied to the lower registers, but the key is that the $U$ is applied $2^x$ times where $x$ runs from 0 through $n-1$. To understand what this means, let's check that $U^{2^x}\ket{\psi}$ can be written as shown below (this may be obvious).

$$
\begin{aligned}
U^{2^x}\ket{\psi}&=U^{2^x-1}U\ket{\psi}\\
&=U^{2^x-1}e^{2\pi i\theta}\ket{\psi}\\
&=U^{2^x-2}e^{2\pi i\theta2}\ket{\psi}\\
&=\cdots\\
&=e^{2\pi i\theta2^x}\ket{\psi}
\end{aligned}
$$

If we trace the quantum states of this quantum circuit in the same fashion using steps 1, 2, ... $n+1$, we find the following.

- Step 1 : $\frac{1}{\sqrt{2^n}}(\ket{0}+\ket{1})^{\otimes n}\ket{\psi}$
- Step 2 : $\frac{1}{\sqrt{2^n}}(\ket{0}+e^{2\pi i\theta2^{n-1}}\ket{1})(\ket{0}+\ket{1})^{\otimes n-1}\ket{\psi}$
- $\cdots$
- Step $n+1$ : $\frac{1}{\sqrt{2^n}}(\ket{0}+e^{2\pi i\theta2^{n-1}}\ket{1})(\ket{0}+e^{2\pi i\theta2^{n-2}}\ket{1})\cdots(\ket{0}+e^{2\pi i\theta2^0}\ket{1})\ket{\psi}$

If we look closely at the state of the $n$-qubit register after step $n+1$, we will see that it is equivalent to the state with QFT where $j$ was replaced with $2^n\theta$. Thus, if we apply an inverse Fourier transform $\rm{QFT}^\dagger$ to this $n$-qubit state, we will be able to obtain the state $\ket{2^n\theta}$!  Measuring this state, we can determine $2^n$\theta$ -- that is, phase $\theta$ (multiplied by $2^n$) of the eigenvalue. This is how the QPE works (see figure below).

(qpe_nqubit_fig)=
```{image} figs/qpe.png
:alt: qpe
:width: 700px
:align: center
```

However, generally speaking, there is not guarantee that $2^n \theta$ will be an integer. See the {ref}`supplementary information page <nonintegral_fourier>` for information about performing inverse Fourier transformation on non-integer values.

+++ {"pycharm": {"name": "#%% md\n"}}

(qpe_imp)=
## Implementation of Example QPE Problem

Next, let's try to implement QPE using a simple quantum circuit.

First, it is necessary to prepare an eigenstate $\ket{\psi}$ and a unitary operator $U$ that satisfy $U\ket{\psi}=e^{2\pi i\theta}\ket{\psi}$. Here we consider $S$ gate (phase $\sqrt{Z}$ gate) as $U$. Since $S\ket{1}=e^{i\pi/2}\ket{1}$ with $\ket{1}=\begin{pmatrix}0\\1\end{pmatrix}$, $\ket{1}$ is an eigenvector of $S$ gate and $e^{i\pi/2}$ is its eigenvalue. This means that $\theta=1/4$ for $U=S$ because QPE allows us to estimate phase $\theta$ of the eigenvalue $e^{2\pi i\theta}$. We will confirm this using a quantum circuit.

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
from fractions import Fraction
import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.accounts import AccountNotFoundError

# Modules for this workbook
from qc_workbook.utils import operational_backend
```

+++ {"pycharm": {"name": "#%% md\n"}}

We create a quantum circuit consisting of a single-qubit register for an eigenvetor $\ket{1}$ and a 3-qubit register for phase estimation. $\ket{1}$ is prepared by applying a Pauli-$X$ to $\ket{0}$, and then the controlled-$S$ gate is applied $2^x$ times for QPE.

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
n_meas = 3

# Register used to obtain phase
qreg_meas = QuantumRegister(n_meas, name='meas')
# Register used to hold eigenvector
qreg_aux = QuantumRegister(1, name='aux')
# Classical register written by the output of phase estimation
creg_meas = ClassicalRegister(n_meas, name='out')

# Create quantum circuit from above registers
qc = QuantumCircuit(qreg_meas, qreg_aux, creg_meas)

# Initialize individudal registers
qc.h(qreg_meas)
qc.x(qreg_aux)

# This is the phase value that we want to get with QPE
angle = np.pi / 2

# Replace (Controlled-S)^x with CP(xπ/2) because S = P(π/2)
for x, ctrl in enumerate(qreg_meas):
    qc.cp(angle * (2 ** x), ctrl, qreg_aux[0])
```

+++ {"pycharm": {"name": "#%% md\n"}}

Then, we measure qubits after applying an inverse QFT to the register for phase estimation.

Write an **inverse circuit of QFT** based on {ref}`this workbook <fourier_addition>`. The `qreg` argument is an object of the measurement register.

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
def qft_dagger(qreg):
    """Circuit for inverse quantum fourier transform"""
    qc = QuantumCircuit(qreg)

    ##################
    ### EDIT BELOW ###
    ##################

    #qc.?

    ##################
    ### EDIT ABOVE ###
    ##################

    qc.name = "QFT^dagger"

    return qc

qc.barrier()
qc.append(qft_dagger(qreg_meas), qargs=qreg_meas)
qc.barrier()
qc.measure(qreg_meas, creg_meas)
qc.draw('mpl')
```

**Solution**

````{toggle}

Use Inverse QFT in `setup_addition` funton of {ref}`fourier_addition`

```{code-block} python
def qft_dagger(qreg):
    """Circuit for inverse quantum fourier transform"""
    qc = QuantumCircuit(qreg)

    ##################
    ### EDIT BELOW ###
    ##################

    for j in range(qreg.size // 2):
        qc.swap(qreg[j], qreg[-1 - j])

    for itarg in range(qreg.size):
        for ictrl in range(itarg):
            power = ictrl - itarg - 1
            qc.cp(-2. * np.pi * (2 ** power), ictrl, itarg)

        qc.h(itarg)

    ##################
    ### EDIT ABOVE ###
    ##################

    qc.name = "QFT^dagger"
    return qc
```

````

```{code-cell} ipython3
:tags: [remove-input, remove-output]

## Cell for text

def qft_dagger(qreg):
    qc = QuantumCircuit(qreg)

    for j in range(qreg.size // 2):
        qc.swap(qreg[j], qreg[-1 - j])

    for itarg in range(qreg.size):
        for ictrl in range(itarg):
            power = ictrl - itarg - 1
            qc.cp(-2. * np.pi * (2 ** power), ictrl, itarg)

        qc.h(itarg)

    qc.name = "IQFT"
    return qc

qreg_meas = QuantumRegister(n_meas, name='meas')
qreg_aux = QuantumRegister(1, name='aux')
creg_meas = ClassicalRegister(n_meas, name='out')

qc = QuantumCircuit(qreg_meas, qreg_aux, creg_meas)

qc.h(qreg_meas)
qc.x(qreg_aux)

angle = np.pi / 2

for x, ctrl in enumerate(qreg_meas):
    qc.cp(angle * (2 ** x), ctrl, qreg_aux[0])

qc.append(qft_dagger(qreg_meas), qargs=qreg_meas)
qc.measure(qreg_meas, creg_meas)
```

+++ {"pycharm": {"name": "#%% md\n"}}

(qpe_imp_simulator)=
### Experiment using Simulator

The probability distribution of the measured outcome is produced using simulator.

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
simulator = AerSimulator()
shots = 2048
qc_tr = transpile(qc, backend=simulator)
results = simulator.run(qc_tr, shots=shots).result()
answer = results.get_counts()

def show_distribution(answer):
    n = len(answer)
    x = [int(key, 2) for key in list(answer.keys())]
    y = list(answer.values())

    fig, ax = plt.subplots()
    rect = ax.bar(x,y)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height/sum(y):.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 0),
                        textcoords="offset points", ha='center', va='bottom')
    autolabel(rect)
    plt.ylabel('Probabilities')
    plt.show()

show_distribution(answer)
```

+++ {"pycharm": {"name": "#%% md\n"}}

Here the answer is 2 in decimal number. Since the measured result should be $\ket{2^n\theta}$, $\theta=2/2^3=1/4$, thus we successfully obtain the correct $\theta$.

The quantum circuit used here is simple, but is a good staring point to explore the behaviour of QPE circuit. Please examine the followings, for example:
- Here we examined the case of $S=P(\pi/2)$ (except global phase). What happens if we insted use $P(\phi)$ gate with the $\phi$ value varying within $0<\phi<\pi$?
- You will see that the precision of estimated phase value gets worse depending on the choice of $\phi$. How can you improve the precision?
- We used $\ket{1}$ as it is eigenvector of $S$ gate, but what happens if we use a state different from $\ket{1}$? Please examine the case when the state is linearly dependent of the eigenvector.

+++ {"pycharm": {"name": "#%% md\n"}}

(qpe_imp_real)=
### Experiment with Quantum Computer

Finally, we will run the circuit on quantum computer and check results. We can use the least busy machine by using the following syntax.

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
# Running on real IBM device
# Specify an instance here if you have access to multiple
# instance = 'hub-x/group-y/project-z'
instance = None

try:
    service = QiskitRuntimeService(channel='ibm_quantum', instance=instance)
except AccountNotFoundError:
    service = QiskitRuntimeService(channel='ibm_quantum', token='__paste_your_token_here__', instance=instance)

backend = service.least_busy(min_num_qubits=4, filters=operational_backend())
print(f"least busy backend: {backend.name}")
```

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
# Execute circuit on the least busy backend. Monitor jobs in the queue.
qc_tr = transpile(qc, backend=backend, optimization_level=3)
job = backend.run(qc_tr, shots=shots)
```

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
# Results
results = job.result()
answer = results.get_counts()
show_distribution(answer)
```

(shor_algo)=
## Shor's Algorithm

Very well, let's move on to the main topic, Shor's algorithm. Shor's algorithm attempts to break down a positive composite number $N$ into a product of two prime numbers $N=qp$.

First, let's review the method of notation used for integer remainders. Consider the following string of integer values $x$. If they are divided, for example, by 3, then the values of the remainder $y$ will be as shown below.

|x|0|1|2|3|4|5|6|7|8|9|
|-|-|-|-|-|-|-|-|-|-|-|
|y|0|1|2|0|1|2|0|1|2|0|

Let us write this as $\modequiv{x}{y}{3}$ (if $k$ is an integer value of 0 or greater, this can also be written as $x=3k+y$).

Shor's algorithm is composed of multiple steps of computation and they can be written in the form of flowchart below. The steps in black are computed using classical calculation, and those in blue are computed using a quantum computer. You might be wondering why we use quantum calculation only for a part of the algorithm. This is because that blue part is difficult to calcualte classically. that is, the basic idea is to use classical calculation for the steps that can be efficiently processed classically, and quantum calculation for those that cannot be done classically. Later on, it will become clear why classical calculation has difficulty in performing the blue part.

(shor_algo_fig)=
```{image} figs/shor_flow.png
:alt: shor_flow
:width: 500px
:align: center
```

+++ {"pycharm": {"name": "#%% md\n"}}

(factoring_example)=
### Example of Integer Factoring

As a simple example, let us consider the factoring of $N=15$ using this algorithm.

For example, imagine that we have selected $a=7$ as a coprime number to 15. Dividing $7^x$ by 15, the remainder $y$ is as follows.

|x|0|1|2|3|4|5|6|$\cdots$|
|-|-|-|-|-|-|-|-|-|
|y|1|7|4|13|1|7|4|$\cdots$|

As you can see, the smallest (non-trivial) value for $r$ that meets the condition $\modequiv{7^r}{1}{15}$ is 4.  Since $r=4$ is an even number, we can define $\modequiv{x}{7^{4/2}}{15}$, that leads to $x=4$. $x+1 = \modnequiv{5}{0}{15}$, so the following is true.

$$
\{p,q\}=\{\gcd(5,15), \gcd(3,15)\}=\{5,3\}
$$

Thus, we managed to obtain the result $15=5\times3$!

+++

(shor_circuit)=
### Quantum Circuit

Next, let's look at a quantum circuit to perform integer factoring of $N=15$. It might look like we're jumping right to the answer, but below is the structure of the circuit itself.

(shor_circuit_fig)=
```{image} figs/shor.png
:alt: shor
:width: 700px
:align: center
```

The top 4 qubits comprise the measurement register, and the bottom 4 the work register. Each register has 4 qubits ($n=4$) because they suffice to express 15 (the binary notation of 15 is $1111_2$). All the qubits are initialized to $\ket{0}$, and the state of the measurement (work) register is represented as $\ket{x}$ ($\ket{w}$). $U_f$ is the oracle given below:

```{image} figs/shor_oracle2.png
:alt: shor_oracle2
:width: 300px
:align: center
```

and it outputs the state $\ket{w\oplus f(x)}$ in the work register (this will be explained in detail later). Let us define the function $f(x)$ as $f(x) = a^x \bmod N$.

As done above, let us check the quantum states of the circuit through steps 1 through 5. First, in step 1, we generate an equal superposition of computational basis states in the measurement register. Let's write each computational basis state as an integer between 0 and 15.

- Step 1 :$\frac{1}{\sqrt{2^4}}\left[\sum_{j=0}^{2^4-1}\ket{j}\right]\ket{0}^{\otimes 4} = \frac{1}{4}\left[\ket{0}+\ket{1}+\cdots+\ket{15}\right]\ket{0}^{\otimes 4}$

After applying the oracle $U_f$, given the definition of the oracle, the state is as follows.

- Step 2 :

$$
\begin{aligned}
&\frac{1}{4}\left[\ket{0}\ket{0 \oplus (7^0 \bmod 15)}+\ket{1}\ket{0 \oplus (7^1 \bmod 15)}+\cdots+\ket{15}\ket{0 \oplus (7^{15} \bmod 15)}\right]\\
=&\frac{1}{4}\left[\ket{0}\ket{1}+\ket{1}\ket{7}+\ket{2}\ket{4}+\ket{3}\ket{13}+\ket{4}\ket{1}+\cdots+\ket{15}\ket{13}\right]
\end{aligned}
$$

After step 2, we measure the work register. $\ket{w}$ is $\ket{7^x \bmod 15}$, that is, either one of the four states $\ket{1}$, $\ket{7}$, $\ket{4}$ and $\ket{13}$. Let us assume, for example, that the measurement result was 13. In that case, the state of the measurement register would be:

- Step 3 :$\frac{1}{2}\left[\ket{3}+\ket{7}+\ket{11}+\ket{15}\right]$

Next, an inverse QFT $\rm{QFT}^\dagger$ is applied to the measurement register. The inverse QFT converts $\ket{j} \to \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1}e^{\frac{-2\pi ijk}{N}}\ket{k}$.

- Step 4 :

$$
\begin{aligned}
&\frac{1}{2}\mathrm{QFT}^\dagger\left[\ket{3}+\ket{7}+\ket{11}+\ket{15}\right]\\
=&\frac{1}{2}\frac1{\sqrt{2^4}}\sum_{k=0}^{2^4-1}\left[e^{\frac{-2\pi i\cdot3k}{2^4}}+e^{\frac{-2\pi i\cdot7k}{2^4}}+e^{\frac{-2\pi i\cdot11k}{2^4}}+e^{\frac{-2\pi i\cdot15k}{2^4}}\right]\ket{k}\\
=&\frac{1}{8}\left[4\ket{0}+4i\ket{4}-4\ket{8}-4i\ket{12}\right]
\end{aligned}
$$

Here, the key is that only the states $\ket{0}$, $\ket{4}$, $\ket{8}$ and $\ket{12}$ appear. Here the interference between quantum states is exploited to reduce the amplitudes of incorrect solutions.

- Step 5 :Last, we measure the measurement bit, and find that 0, 4, 8, and 12 each occur with a 1/4 probability.
-
You may have anticipated, but the signs of repetition is becoming apparent because $7^x \bmod 15$ is calculated in step 2.

+++

(shor_measurement)=
### Analysis of Measurement Results

Let's think about what these measurement results mean. Given the similarity between the Shor's algorithm {ref}`circuit <shor_circuit_fig>` and the $n$-qubit QPE {ref}`circuit <qpe_nqubit_fig>`, we can natually hypothesize that both function in the same way (supplementary explanation is provided below). If that is the case, the measurement register should represent $2^4=16$ times the phase $\theta$ of eigenvalue $e^{2\pi i\theta}$. If we get, e.g., 4, from the measurement register, the phase $\theta$ will be $\theta=4/16=0.25$. What does this value mean?

As a quantum circuit for Shor's algorithm, we have so far used $\ket{w}=\ket{0}^{\otimes n}$ as the initial state and an oracle $U_f$ that acts as $U_f\ket{x}\ket{w}=\ket{x}\ket{w\oplus f(x)}$ $(f(x) = a^x \bmod N)$. To implement this $U_f$, let us consider the following unitary operator $U$.

```{math}
:label: U_action
U\ket{m} =
\begin{cases}
\ket{am \bmod N)} & 0 \leq m \leq N - 1 \\
\ket{m} & N \leq m \leq 2^n-1
\end{cases}
```

This unitary satisfies the following relation:

$$
U^{x}\ket{1} = U^{x-1} \ket{a \bmod N} = U^{x-2} \ket{a^2 \bmod N} = \cdots = \ket{a^x \bmod N}
$$

Therefore, we can use the $U$ to implement $U_f\ket{x}\ket{0}$ where $w=0$.

$$
\begin{aligned}
U_f\ket{x}\ket{0}&=\ket{x}\ket{0 \oplus (a^x \bmod N)}\\
&=\ket{x}\ket{a^x \bmod N}\\
&=\ket{x} U^x \ket{1}
\end{aligned}
$$

Here, we'll consider the state $\ket{\psi_s}$ as defined follows, with $s$ being an integer within $0 \leq s \leq r-1$:

$$
\ket{\psi_s} \equiv \frac{1}{\sqrt{r}}\sum_{k=0}^{r-1}e^{-2\pi isk/r}\ket{a^k \bmod N}
$$

With this state $\ket{\psi_s}$, we can derive

$$
\frac{1}{\sqrt{r}}\sum_{s=0}^{r-1}\ket{\psi_s}=\ket{1}.
$$

At the same time, we can see that the $\ket{\psi_s}$ is an eigenvector of the operator $U$ and has an eigenvalue $e^{2\pi is/r}$:

$$
U\ket{\psi_s}=e^{2\pi is/r}\ket{\psi_s}
$$

In other words, the operation of performing an oracle $U_f$ in Shor's algorithm is equivalent to applying the unitary $U$ $x$ times to $\ket{1}$, which is the superposition of the eigenvectors $\ket{\psi_s}$ of eigenvalue $e^{2\pi is/r}. If you compare this with the {ref}`QPE circuit<qpe_nqubit_fig>`, it is obvious that they are essentially doing the same thing. Since the inverse QFT is performed after these operations, the entire operation corresponds exactly to QPE.

Recall that the phase value determined using QPE is $\theta$ of the eigenvalue $e^{2\pi i\theta}$ for the unitary $U$ and eigenvector $\ket{\psi}$ that satisfy $U\ket{\psi}=e^{2\pi i\theta}\ket{\psi}$. From these, you would see that the phase $\theta$ derived from Shor's algorithm is (an integer multiple of) $s/r$.

+++

(continued_fractions)=
### Continued Fraction Expansion

Through the above, we understood that the phase obtained by the measurement is $\theta \approx s/r$. In order to derive an order $r$ from these results, we will use **continued-fraction expansion** but the details will be left to other references (P. 230, Box 5.3 of {cite}`nielsen_chuang_qft_app` describes the continued fraction algorithm). This method allows us to determine $s/r$ as the closet fracton to $\theta$.

For example, we get $r=4$ if $\theta=0.25$ (though we might have $r=8$ at small frequency). If you can get this far, then you can use classical calculation to break it down into two prime numbers (See {ref}`here<factoring_example>`).

+++

(modular_exponentiation)=
### Modular Exponentiation

Let's explore the operation of oracle $U_f\ket{x}\ket{w}=\ket{x}\ket{w\oplus f(x)}$ a bit further. By using the binary representation of $x$:

$$
x=(x_{n-1}x_{n-2}\cdots x_0)_2 = 2^{n-1}x_{n-1}+2^{n-2}x_{n-2}+\cdots+2^0x_0,
$$

we can express $f(x) = a^x \bmod N$ as follows.

$$
\begin{aligned}
f(x) & = a^x \bmod N \\
 & = a^{2^{n-1}x_{n-1}+2^{n-2}x_{n-2}+\cdots+2^0x_0} \bmod N \\
 & = a^{2^{n-1}x_{n-1}}a^{2^{n-2}x_{n-2}}\cdots a^{2^0x_0} \bmod N
\end{aligned}
$$

That is, this function can be implemented using unitary operations as shown below.

```{image} figs/shor_oracle.png
:alt: shor_oracle
:width: 600px
:align: center
```

Comparing this circuit with $n$-qubit QPE {ref}`circuit<qpe_nqubit_fig>`, you'll immediately see that this circuit implements $U^{2^x}$ operations of the QPE. Applying $a^x \bmod N$, controlled by each qubit of the 1st register, to the contents of the second register (corresponding to the bottom wire in the diagram above) to implement the $U^{2^x}$ operations of QPE is called modular exponentiation.

(shor_imp)=
## Implementation of Shor's algorithm

We will now switch to code implementation of Shor's algorithm.

+++ {"pycharm": {"name": "#%% md\n"}}

(shor_imp_period)=
### Order Finding

First, let's look into the algorithm for determining the order (period) of repetitions.

With a positive integer $N$, let's investigate the behavior of function $f(x) = a^x \bmod N$. In [Shor's algorithm](shor_algo_fig), $a$ is a positive integer that is smaller than $N$ and $a$ and $N$ are coprime. The order $r$ is the smallest non-zero integer that satisfies $\modequiv{a^r}{1}{N}$.
The graph shown below is an example of this function. The arrow between the two points indicates the periodicity.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
N = 35
a = 3

# Calculate the data to be plotted
xvals = np.arange(35)
yvals = [np.mod(a**x, N) for x in xvals]

# Use matplotlib to perform plotting
fig, ax = plt.subplots()
ax.plot(xvals, yvals, linewidth=1, linestyle='dotted', marker='x')
ax.set(xlabel='$x$', ylabel=f'${a}^x$ mod {N}',
       title="Example of Periodic Function in Shor's Algorithm")

try: # Plot r on the graph
    r = yvals[1:].index(1) + 1
except ValueError:
    print('Could not find period, check a < N and have no common factors.')
else:
    plt.annotate(text='', xy=(0, 1), xytext=(r, 1), arrowprops={'arrowstyle': '<->'})
    plt.annotate(text=f'$r={r}$', xy=(r / 3, 1.5))
```

(shor_imp_oracle)=
### Oracle Implementation

Below we aim at factoring $N=15$. As explained above, we implement the oracle $U_f$ by repeating the unitary $U\ket{m}=\ket{am \bmod N}$ $x$ times.

For this practice task, please implement the function `c_amod15` that executes $C[U^{2^l}] \ket{z} \ket{m}=\ket{z} \ket{a^{z 2^{l}} m \bmod 15} \; (z=0,1)$ below (`c_amod15` returns the entire controlled gate, but here you should write $U$ that is applied to the target register).

Consider the argument `a` which is an integer smaller than 15 and is coprime to 15. In general, when $a = N-1$, $r=2$ because $\modequiv{a^2}{1}{N}$. Therefore, $a^{r/2} = a$ and $\modequiv{a + 1}{0}{N}$, meaning that such `a` cannot be used for Shor's algorithm. This would require the value of `a` to be less than or equal to 13.

Such unitary operation will need a complicated circuit if it should work for general values of $a$ and $N${cite}`shor_oracle`, but it can be implemented with a few lines of code if the problem is restricted to $N=15$.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
def c_amod15(a, l):
    """mod 15-based control gate"""

    if a not in [2, 4, 7, 8, 11, 13]:
        raise ValueError("'a' must be 2, 4, 7, 8, 11, or 13")

    U = QuantumCircuit(4)

    ##################
    ### EDIT BELOW ###
    ##################

    #if a == 2:
    #    ...
    #elif a == 4:
    #    ...
    #    ...

    ##################
    ### EDIT ABOVE ###
    ##################

    # Repeating U 2^l times
    U_power = U.repeat(2 ** l)

    # Convert U_power to gate
    gate = U_power.to_gate()
    gate.name = f"{a}^{2 ** l} mod 15"

    # Convert gate to controlled operation
    c_gate = gate.control()
    return c_gate
```

+++ {"pycharm": {"name": "#%% md\n"}}

**Solution**

````{toggle}
First we consider the case of `a=2, 4, 8`. Representing $m$ as a binary number:

```{math}
m=\sum_{j=0}^{3} 2^j m_j \, (m_j=0,1)
```

```{math}
:label: ammod15
am \bmod 15 = \left( \sum_{j=0}^{3} 2^{j+\log_2 a} m_j \right) \bmod 15
```

Note $15 = 2^4 - 1$. In general, for nutaral numbers $n, m$ and the smallest natural number $p$ that satisfies $n-pm < m$, the following relation holds (the proof is simple):

```{math}
2^n \bmod (2^m - 1) = 2^{n-pm}
```

Therefore, $2^{j+\log_2 a} \bmod 15$ takes a value of $1, 2, 4, 8$ once and only once for $j=0, 1, 2, 3$.

For $m \leq 14$, since the sum of the modulo 15 of individual terms inside the parentheses on the righthand side of Equation {eq}`ammod15` never exceeds 15,

```{math}
am \bmod 15 = \sum_{j=0}^{3} (2^{j+\log_2 a} \bmod 15) m_j,
```

it turns out that we can multiply by $a$ and take a modulo 15 for each bit independently. If we write out the values of $2^{j+\log_2 a} \bmod 15$:

|       | $j=0$ | $j=1$ | $j=2$ | $j=3$ |
|-------|-------|-------|-------|-------|
| $a=2$ | 2     |  4    | 8     | 1     |
| $a=4$ | 4     |  8    | 1     | 2     |
| $a=8$ | 8     |  1    | 2     | 4     |

This action can be implemented using a cyclic bit shift. For example, `a=2` should be as follows:

```{math}
\begin{align}
0001 & \rightarrow 0010 \\
0010 & \rightarrow 0100 \\
0100 & \rightarrow 1000 \\
1000 & \rightarrow 0001
\end{align}
```

We can implement this using SWAP gates into a quantum circuit.

```{code-block} python
    ##################
    ### EDIT BELOW ###
    ##################

    if a == 2:
        # Applying SWAP gates from higher qubits in this order
        U.swap(3, 2)
        U.swap(2, 1)
        U.swap(1, 0)
    elif a == 4:
        # Bit shift by skipping one bit
        U.swap(3, 1)
        U.swap(2, 0)
    elif a == 8:
        # Applying from lower qubits
        U.swap(1, 0)
        U.swap(2, 1)
        U.swap(3, 2)

    ##################
    ### EDIT ABOVE ###
    ##################
```

A good thing about using SWAP gates in this way is that Equation {eq}`U_action` is correctly realized because the $U$ does not change state in the register for $m=15$. This is however not very crucial when using this function below, because $\ket{15}$ does not appear in the work register.

How about `a=7, 11, 13`? Again, we can exploit some uniqueness of the number 15. Noting that $7 = 15 - 8$, $11 = 15 - 4 and $13 = 15 - 2$,

```{math}
\begin{align}
7m \bmod 15 & = (15 - 8)m \bmod 15 = 15 - (8m \bmod 15) \\
11m \bmod 15 & = (15 - 4)m \bmod 15 = 15 - (4m \bmod 15) \\
13m \bmod 15 & = (15 - 2)m \bmod 15 = 15 - (2m \bmod 15),
\end{align}
```

we can see that the circuit we want is the one that subtracts the results of `a=2, 4, 8` from 15. Subtracting from 15 for a 4-bit register corresponds to reversing all the bits, that is, applying $X$ gates. Therefore, what we need finally is something like below:

```{code-block} python
    ##################
    ### EDIT BELOW ###
    ##################

    if a in [2, 13]:
        # Applying SWAP gates from higher qubits in this order
        U.swap(3, 2)
        U.swap(2, 1)
        U.swap(1, 0)
    elif a in [4, 11]:
        # Bit shift by skipping one bit
        U.swap(3, 1)
        U.swap(2, 0)
    elif a in [8, 7]:
        # Applying from lower qubits
        U.swap(1, 0)
        U.swap(2, 1)
        U.swap(3, 2)

    if a in [7, 11, 13]:
        U.x([0, 1, 2, 3])

    ##################
    ### EDIT ABOVE ###
    ##################
```

````

```{code-cell} ipython3
---
editable: true
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
slideshow:
  slide_type: ''
tags: [remove-input, remove-output]
---
# Cell for textbook generation

def c_amod15(a, l):
    U = QuantumCircuit(4, name='U')

    if a in [2, 13]:
        U.swap(3, 2)
        U.swap(2, 1)
        U.swap(1, 0)
    elif a in [4, 11]:
        U.swap(3, 1)
        U.swap(2, 0)
    elif a in [8, 7]:
        U.swap(1, 0)
        U.swap(2, 1)
        U.swap(3, 2)

    if a in [7, 11, 13]:
        U.x([0, 1, 2, 3])

    U_power = U.repeat(2 ** l)

    gate = U_power.to_gate()
    gate.name = f"{a}^{2 ** l} mod 15"
    c_gate = gate.control()
    return c_gate
```

+++ {"pycharm": {"name": "#%% md\n"}}

(shor_imp_circuit)=
### Implementation of Entire Circuit

Let's use 8 qubits for the measurement register.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
# Coprime to 15
a = 7

# Number of measurement bits (precision of phase estimation)
n_meas = 8

# Register used to obtain phase
qreg_meas = QuantumRegister(n_meas, name='meas')
# Register to hold eigenvector
qreg_aux = QuantumRegister(4, name='aux')
# Classical register written by the output of phase estimation
creg_meas = ClassicalRegister(n_meas, name='out')

# Create quantum circuit from above registers
qc = QuantumCircuit(qreg_meas, qreg_aux, creg_meas)

# Initialize individual registers
qc.h(qreg_meas)
qc.x(qreg_aux[0])

# Apply controlled-U gate
for l, ctrl in enumerate(qreg_meas):
    qc.append(c_amod15(a, l), qargs=([ctrl] + qreg_aux[:]))

# Apply inverse QFT
qc.append(qft_dagger(qreg_meas), qargs=qreg_meas)

# Measure the circuit
qc.measure(qreg_meas, creg_meas)
qc.draw('mpl')
```

Execute the circuit using simulator and check the results.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
qc = transpile(qc, backend=simulator)
results = simulator.run(qc, shots=2048).result()
answer = results.get_counts()

show_distribution(answer)
```

+++ {"pycharm": {"name": "#%% md\n"}}

(shor_imp_ana)=
### Analysis of Measured Results
Let's get phase from the output results.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
rows, measured_phases = [], []
for output in answer:
    decimal = int(output, 2)  # Converting to decimal number
    phase = decimal / (2 ** n_meas)
    measured_phases.append(phase)
    # Save these values
    rows.append(f"{decimal:3d}      {decimal:3d}/{2 ** n_meas} = {phase:.3f}")

# Print the results
print('Register Output    Phase')
print('------------------------')

for row in rows:
    print(row)
```

From the phase information, you can determine $s$ and $r$ using the continued fraction expansion. You can use the built-in Python `fractions` module to convert fractions into `Fraction` objects.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
rows = []
for phase in measured_phases:
    frac = Fraction(phase).limit_denominator(15)
    rows.append(f'{phase:10.3f}      {frac.numerator:2d}/{frac.denominator:2d} {frac.denominator:13d}')

# Print the results
print('     Phase   Fraction   Guess for r')
print('-------------------------------------')

for row in rows:
    print(row)
```

Using the `limit_denominator` method, we obtain the fraction closest to the phase value, for which the denominator is smaller than a specific value (15 in this case).

From the measurement results, you can see that the two values (64 and 192) provide the correct answer of $r=4$.
