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

# 【Exercise】Spectral Decomposition with Phase Estimation

```{contents} 目次
---
local: true
---
```

$\newcommand{\ket}[1]{|#1\rangle}$

+++

## Estimation of Energy Spectra

In physics and chemistry, determining energy eigenvalues (spectra) and corresponding eigenstates of a system is an extremely important task, an example of which appears in a {doc}`later exercise<vqe>`. Deriving the energy spectrum of a system is equivalent to determining the Hamiltonian that governs the system and diagonalizing it.

However, as we saw in the {doc}`previous exercise<dynamics_simulation>`, the number of dimensions in a typical quantum system is extremely large, and we cannot properly perform the inverse matrix calculations that are the key to diagonalizing the Hamiltonian. In the exercise, we found that even in that case, if the form of the Hamiltonian allows us to perform efficient Suzuki-Trotter decomposition, we can simulate time evolution of the system using a quantum computer. However, in this simulation, we did not use energy eigenvalues and eigenstates of the system explicitly.

In fact, we can numerically determine the energy eigenvalues{cite}`Aspuru-Guzik1704` by combining the simulation of time evolution with the phase estimation method that we discussed in {doc}`shor`. This approach could be even extended to investigate corresponding eigenstates. In this assignment, we will consider Heisenberg model with an external magnetic field and attempt the decomposition of energy spectra using phase estimation technique.

+++

## Reconsideration of Heisenberg Model

The Hamiltonian of the Heisenberg model, introduced in the previous section, is as follows.

$$
H = -J \sum_{j=0}^{n-2} (\sigma^X_{j+1}\sigma^X_{j} + \sigma^Y_{j+1}\sigma^Y_{j} + \sigma^Z_{j+1} \sigma^Z_{j}) \quad (J > 0)
$$

This Hamiltonian represents a system composed of particles with spins, lined up in one dimensional space, that interact between adjacent particles. In this system, the interactions will lower the energy when the directions of the spins are aligned. Therefore, the lowest energy will be achived when all the spin directions are aligned. 

In this assignment, we will apply an external magnetic field to the system. When there is an external magnetic field, the energy will be lowered when the spin is aligned with the magnetic field. Therefore, if we apply the external magnetic field along the +$Z$ direction, the Hamiltonian is as follows.

$$
H = -J \sum_{j=0}^{n-1} (\sigma^X_{j+1}\sigma^X_{j} + \sigma^Y_{j+1}\sigma^Y_{j} + \sigma^Z_{j+1} \sigma^Z_{j} + g \sigma^Z_j)
$$

This Hamiltonian has one more difference from the one considered previously. For the previous case, we considered the boundary condition that the spins located at the end of the chain interacted only spins at "inner side" of the chain by taking the sum of the spins from $j=0$ to $n-2$. This time the sum is taken all the way to $n-1$. This represents "periodic boundary condition" (the spins on a circle, not on a straight line) by treating $\sigma^{X,Y,Z}_n$ as equal to $\sigma^{X,Y,Z}_0$.

Let's look into the eigenvalues and eigenstates of such Hamiltonian for a specific example. We consider the simplest case of $n=2$ and $g=0$, and derive the true answers by exact diagonalization. 

```{code-cell} ipython3
:tags: [remove-output]

# First, import all necessary modules
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
# Workbook-specific modules
from qc_workbook.show_state import show_state

print('notebook ready')
```

```{code-cell} ipython3
# Number of spins
n_s = 2
# Coupling parameter
J = 1.
# External field / J
g = 0.

# Construct the Hamiltonian matrix
paulis = list()
coeffs = list()

xx_template = 'I' * (n_s - 2) + 'XX'
yy_template = 'I' * (n_s - 2) + 'YY'
zz_template = 'I' * (n_s - 2) + 'ZZ'

for j in range(n_s):
    paulis.append(xx_template[j:] + xx_template[:j])
    paulis.append(yy_template[j:] + yy_template[:j])
    paulis.append(zz_template[j:] + zz_template[:j])
    coeffs += [-J] * 3

    if g != 0.:
        paulis.append('I' * (n_s - j - 1) + 'Z' + 'I' * j)
        coeffs.append(-J * g)

hamiltonian = SparsePauliOp(paulis, coeffs).to_matrix()

# Diagonalize and obtain the eigenvalues and vectors
eigvals, eigvectors = np.linalg.eigh(hamiltonian)

# Print the eigenvectors
for i in range(eigvals.shape[0]):
    show_state(eigvectors[:, i], binary=True, state_label=r'\phi_{} (E={}J)'.format(i, eigvals[i]))
```

In the last part, the [`show_state` function](https://github.com/UTokyo-ICEPP/qc-workbook/tree/master/source/qc_workbook/show_state.py) was used to show the eigenvalues and eigenvectors. We can see that there are three independent eigenvectors that correspond to the lowest energy state (eigenvalue $-2J$). Therefore, an arbitrary linear combination of these eigenvectors is also the lowest energy state. The excited state (eigenvalue $6J$) is $1/\sqrt{2} (-\ket{01} + \ket{10})$.

+++

## Spectral Decomposition with Phase Estimation

Let's now begin the main part of this exercise. In the figure shown below at {doc}`shor`, what can you conclude when $U$ is a time evolution operator $U_H(-\tau)$ for a certain time $\tau$ for a Hamiltonian $H$?

```{image} figs/qpe_wo_iqft.png
:alt: qpe_wo_iqft
:width: 500px
:align: center
```

Below, we will refer to the upper register in the figure above (initial state $\ket{0}$) as "readout" register R, and the lower register (initial state $\ket{\psi}$) as "state" register S. The number of bits for the registers R and S are $n_R$ and $n_S$, respectively. Furthermore, note that the bit corresponding to the lowest order in the readout register is written at the bottom of the figure, and this is opposite to the notation used in Qiskit.

Now, we will break the Hamiltonian $H$ into a constant $\hbar \omega$ (in energy dimension) and the dimensionless Hermitian operator $\Theta$.

$$
H = \hbar \omega \Theta.
$$

Here, the $\omega$ can be chosen arbitrary. If the $\omega$ is selected to be $x$-times larger, we can simply multiply the $\Theta$ by $1/x$. In practice, as we see later, we will choose tge $\omega$ such that the absolute value of $\Theta$'s eigenvalue is slightly smaller than 1. The formula can thus be rewritten as follows.

$$
U_H(-\tau) \ket{\psi} = \exp\left(i\omega \tau \Theta\right) \ket{\psi}
$$

If the operator correspondting to the circuit in the figure is denoted by $\Gamma$, we arrive at the following.

$$
\Gamma \ket{0}_R \ket{\psi}_S = \frac{1}{\sqrt{2^{n_R}}} \sum_{j=0}^{2^{n_R} - 1} \exp\left(i j \omega \tau \Theta\right) \ket{j}_R \ket{\psi}_S
$$

Then, we apply an inverse Fourier transform to this state, as done in the exercise,

$$
\text{QFT}^{\dagger}_R \Gamma \ket{0}_R \ket{\psi}_S = \frac{1}{2^{n_R}} \sum_{k=0}^{2^{n_R} - 1} \sum_{j=0}^{2^{n_R} - 1} \exp(i j \omega \tau \Theta) \exp\left(-\frac{2 \pi i j k}{2^{n_R}}\right) \ket{k}_R \ket{\psi}_S.
$$

So far we have simply stated that $\tau$ is a certain time, but actually it can take any value. Now let us fix the $\tau$ value to be $\omega \tau = 2 \pi$. This leads to the following.

$$
\text{QFT}^{\dagger}_R \Gamma \ket{0}_R \ket{\psi}_S = \frac{1}{2^{n_R}} \sum_{k=0}^{2^{n_R} - 1} \sum_{j=0}^{2^{n_R} - 1} \exp\left[\frac{2 \pi i j}{2^{n_R}} \left(2^{n_R} \Theta - k\right)\right] \ket{k}_R \ket{\psi}_S
$$

Therefore, if $\ket{\psi}$ can be written as

```{math}
:label: spectral_decomposition
\ket{\psi} = \sum_{m=0}^{2^{n_S} - 1} \psi_m \ket{\phi_m}
```

using the eigenvectors of $\Theta$ $\{\ket{\phi_m}\}$, then the $\text{QFT}^{\dagger}_R \Gamma \ket{0}_R \ket{\psi}_S$ can be written with the corresponding eigenvalues $\{\theta_m\}$ as  

```{math}
:label: spectrum_estimation_final
\begin{align}
\text{QFT}^{\dagger}_R \Gamma \ket{0}_R \ket{\psi}_S & = \frac{1}{2^{n_R}} \sum_{k=0}^{2^{n_R} - 1} \sum_{j=0}^{2^{n_R} - 1} \sum_{m=0}^{2^{n_S} - 1} \psi_m \exp\left[\frac{2 \pi i j}{2^{n_R}} (\kappa_m - k)\right] \ket{k}_R \ket{\phi_m}_S \\
& = \sum_{k=0}^{2^{n_R} - 1} \sum_{m=0}^{2^{n_S} - 1} \psi_m f(\kappa_m - k) \ket{k}_R \ket{\phi_m}_S.
\end{align}
```

Here, the second equation is written using the function $f(\kappa_m - k)$ defined as $f(\kappa_m - k) := \frac{1}{2^{n_R}} \sum_{j} \exp \left[2 \pi i j (\kappa_m - k) / 2^{n_R}\right]$.

Finally we are going to measure the state, and then multiply $\theta_m = 2^{-n_R} \kappa_m$, estimated from the measured bitstrings in the R register, by $\hbar \omega$ to determine the energy eigenvalue of $H$.

You might find it difficult to digest what we actually did because some new *ad-hoc* parameters $\omega$ and $\tau$ were introduced. Let us now look at the problem from different perspective. Eventually, what we did above was the following when a Hamiltonian $H$ was provided:

1. Normalize $H$ such that the eigenvalue is $\lesssim 1$, or the absolute value is $\lesssim \frac{1}{2}$ if the eigenvalue could be negative (record the normalization constant). 
2. Perform phase estimation of $U = \exp(-2 \pi i \Theta)$ with the normalized operator as $\Theta$. 
3. Obtain the energy eigenvalue by multiplying the eigenvalues of $\Theta$ obtained from phase estimation by the normalization constant in Step 1.

By doing above, we determine the eigenvalues of $\Theta$ so that the eigenvalues from the readout register will not cause any {ref}`overflow <signed_binary>`.

It might look contradictory to select a normalization constant so that the eigenvalue can take a specific value in the problem of eigenvalue determination. However, as we touched on in {doc}`dynamics_simulation`, the Hamiltonian expressed in quantum computing can all be decomposed into a linear combination of products of basis state operators ${I, \sigma^X, \sigma^Y, \sigma^Z}$. Since the eigenvalues of products of individual basis state operators are $\pm 1$, if the Hamiltonian $H$ is decomposed into the product of the basis state operators $\sigma_k$ and the energy coefficient $h_k$, as follows:

$$
H = \sum_{k} h_k \sigma_k
$$

then the absolute value of the eigenvalue of $H$ is at most $\sum_{k} |h_k|$. Therefore, even for a completely unknown Hamiltonian, we can take $\hbar \omega = 2 \sum_{k} |h_k|$ as the normalization constant. If it turns out that the maximum eigenvalue is smaller from the spectral estimation, we can simply adjust the normalization constant and perform the calculation again.

The same logic can be used to decide the number of bits of the readout register R. Now let us think about an absolute value of the smallest non-zero eigenvalue. The smallest value is equal to or greater than:

$$
\mu = \min_{s_k = \pm 1} \left| \sum_{k} s_k h_k \right|
$$

In principle, we will have to examine $2^{L}$ combinations (where $L$ is the number of Hamiltonian terms) to determine this value. But, since practical Hamiltonians do not have too many terms, it is likely that this can be handled with a reasonable amount of computation. Let us set the number of bits in register R, $n_R$, to be able to read out $2^{n_R}\mu/(\hbar \omega)$. 

We have seen so far that the normalization constant and the size of the readout register can be determined easily. A problem resides in the state $\ket{\psi}$ which the operator $U$ is applied to. If we want to know the $m$-th excited energy of eigenvalues, $\psi_m \neq 0$ must be true in Equation {eq}`spectral_decomposition`. In special cases we may have an idea of the eigenvector before performing spectral estimation, but it is obvious that for general Hamiltonians we cannot prepare such states for an arbitrary value of $m$. 

On the other hand, for the lowest energy $\hbar \omega \theta_0$, we can evaluate it at relatively good precision by approximating the lowest energy state by using techniques in {doc}`vqe` and setting the obtained state as the input to the S register. Therefore, the above method can, in principle, be used to completely decompose the energy spectra, but in practice it is most commonly used to determine the lowest energy and its eigenvector accurately.

+++

## Exercise 1: Implement Spectrum Estimation and Comparison with Exact Solutions

Let us now derive the energy spectra of the Hamiltonian for the Heisenberg model using phase estimation.  

We will use Suzuki-Trotter decomposition to calculate $U_H(-\tau)$ on a quantum computer. Refer to {doc}`dynamics_simulation` to implement the rotation gates of $ZZ$, $XX$ and $YY$.

The next cell defines a function that returns a quantum circuit composed of Suzuki-Trotter steps of the Hamiltonian evolution. The argument `num_steps` specifies the number of Suzuki-Trotter steps. 

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
def trotter_twopi_heisenberg(state_register, energy_norm, g, num_steps):
    """Return a function that implements a single Trotter step for the Heisenberg model.

    The Heisenberg model Hamiltonian is
    H = -J * sum_of_sigmas = hbar*ω * Θ

    The returned circuit implements a negative time evolution
    U = exp(-i H*(-τ)/hbar)
    where τ = 2π / ω, which leads to
    U = exp(i 2π Θ).

    Because we employ the Suzuki-Trotter decomposition, the actual circuit corresponds to
    U = [exp(i 2π/num_steps Θ)]^num_steps.

    Args:
        state_register (QuantumRegister): Register to perform the Suzuki-Trotter simulation.
        energy_norm (float): J/(hbar*ω).
        g (float): External field strength relative to the coupling constant J.
        num_steps (float): Number of steps to divide the time evolution of ωτ=2π.

    Returns:
        QuantumCircuit: A quantum circuit implementing the Trotter simulation of the Heisenberg
        model.
    """
    circuit = QuantumCircuit(state_register, name='ΔU')

    n_spins = state_register.size
    step_size = 2. * np.pi / num_steps

    # Implement the circuit corresponding to exp(i*step_size*Θ) below, where Θ is defined by
    # Θ = -J/(hbar*ω) * sum_of_sigmas = -energy_norm * sum_of_sigmas
    ##################
    ### EDIT BELOW ###
    ##################

    # circuit.?

    ##################
    ### EDIT ABOVE ###
    ##################

    circuit = circuit.repeat(num_steps)
    circuit.name = 'U'

    return circuit
```

In the next cell, the algorithm of spectral estimation is implemented. This function returns a quantum circuit that takes state register, readout register and the time-evolution circuit as arguments and performs phase estimation.  

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
def spectrum_estimation(state_register, readout_register, u_circuit):
    """Perform a spectrum estimation given a circuit containing state and readout registers and a callable implementing
    a single Trotter step.

    Args:
        state_register (QuantumRegister): State register.
        readout_register (QuantumRegister): Readout register.
        u_circuit (QuantumCircuit): A circuit implementing U_H(-2π/ω).

    Returns:
        QuantumCircuit: A circuit implementing the spectrum estimation of the given Hamiltonian.
    """
    circuit = QuantumCircuit(state_register, readout_register, name='Spectrum estimation')

    # Set the R register to an equal superposition
    circuit.h(readout_register)

    # Apply controlled-U operations to the circuit
    for iq, qubit in enumerate(readout_register):
        # Repeat the 2π evolution by 2^iq and convert it to a controlled gate
        controlled_u_gate = u_circuit.repeat(2 ** iq).to_gate().control(1)

        # Append the controlled gate specifying the control and target qubits
        circuit.append(controlled_u_gate, qargs=([qubit] + state_register[:]))

    circuit.barrier()

    # Inverse QFT
    for iq in range(readout_register.size // 2):
        circuit.swap(readout_register[iq], readout_register[-1 - iq])

    dphi = 2. * np.pi / (2 ** readout_register.size)

    for jtarg in range(readout_register.size):
        for jctrl in range(jtarg):
            power = jctrl - jtarg - 1 + readout_register.size
            circuit.cp(-dphi * (2 ** power), readout_register[jctrl], readout_register[jtarg])

        circuit.h(readout_register[jtarg])

    return circuit
```

In this exercise, we examine the case of $n=3$ and $g=0$ for which the exact solutions were derived above. Since we already know the energy eigenvalues this time, the normalization constant of the Hamiltonian is set $\hbar \omega = 16J$ so that the output state from the readout register becomes simple. In this case, the readout result has sign and the maximum absolute value is $2^{n_R} (6/16)$, therefore the overflow can be avoided by taking $n_R = 1 + 3$.

In the next cell, the parameters of the simulation and phase estimation are set. 

```{code-cell} ipython3
## Physics model parameter
g = 0.

## Spectrum estimation parameters
# Hamiltonian normalization
energy_norm = 1. / 16. # J/(hbar*ω)
# Number of steps per 2pi evolution
# Tune this parameter to find the best balance of simulation accuracy versus circuit depth
num_steps = 6
# Register sizes
n_state = 2
n_readout = 4

## Registers
state_register = QuantumRegister(n_state, 'state')
readout_register = QuantumRegister(n_readout, 'readout')
```

Let us check whether the function is properly defined above.

```{code-cell} ipython3
:tags: [remove-output]

u_circuit = trotter_twopi_heisenberg(state_register, energy_norm, g, num_steps)
u_circuit.draw('mpl')
```

```{code-cell} ipython3
:tags: [remove-output]

se_circuit = spectrum_estimation(state_register, readout_register, u_circuit)
se_circuit.draw('mpl')
```

Let us prepare a function that sets the following state:

```{math}
:label: two_qubit_init
\frac{1}{2}\ket{00} - \frac{1}{\sqrt{2}}\ket{01} + \frac{1}{2} \ket{11} = \frac{1}{2} \ket{\phi_0} + \frac{1}{2} \ket{\phi_1} + \frac{1}{2} \ket{\phi_2} + \frac{1}{2} \ket{\phi_3}
```

as the initial state of the state register. Here the $\ket{\phi_i}$ are four exact solutions of the eigenvectors we initially determined.

```{code-cell} ipython3
:tags: [remove-output]

def make_initial_state(state_register, readout_register):
    circuit = QuantumCircuit(state_register, readout_register)

    # Set the initial state of the state vector to (1/2)|00> - (1/sqrt(2))|01> + (1/2)|11>
    ##################
    ### EDIT BELOW ###
    ##################

    #circuit.?

    ##################
    ### EDIT ABOVE ###
    ##################

    return circuit


init_circuit = make_initial_state(state_register, readout_register)
init_circuit.draw('mpl')
```

Finally, everything is combined.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
tags: [remove-output]
---
u_circuit = trotter_twopi_heisenberg(state_register, energy_norm, g, num_steps)
se_circuit = spectrum_estimation(state_register, readout_register, u_circuit)

circuit = make_initial_state(state_register, readout_register)
circuit.compose(se_circuit, inplace=True)
circuit.measure_all()
circuit.draw('mpl')
```

Execute the circuit with simulator and get the output histogram.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
tags: [remove-output]
---
# Run the circuit in simulator and plot the histogram
simulator = AerSimulator()
circuit = transpile(circuit, backend=simulator)
job = simulator.run(circuit, shots=10000)
result = job.result()
counts = result.get_counts(circuit)
plot_histogram(counts)
```

Since the initial state of the state register is given in Equation {eq}`two_qubit_init`, the final state of the circuit should be:

$$
\frac{1}{2} \ket{-2}_{R} \ket{00}_{S} - \frac{1}{2\sqrt{2}} \ket{-2}_{R} \left( \ket{01}_{S} + \ket{10}_{S} \right) + \frac{1}{2} \ket{-2}_{R} \ket{11}_{S} - \frac{1}{2\sqrt{2}} \ket{6}_{R} \left( \ket{01}_{S} - \ket{10}_{S} \right)
$$

Is the output histogram consistent with this state?

**Items to submit**:

- Completed `make_trotter_step_heisenberg` function 
- Completed quantum circuit to initialize the state register 
- Histogram of the results of spectrum esimation and its explanation

+++

## Exercise 2: Examine the Behaviour of Non-trivial States

Next, let's determine all energy spectra of the Heisenberg model with $n=4$ as a function of $g$. We can do an exact diagonalization, as done above, because $n=4$, but we will solely rely on quantum computation here. 

In order to know all the energy eignvalues, we will need to elaborate on the initial states of $S$. But, since we do not have any prior knowledge, we will take a strategy of exhausitve search. That is, we will repeat spectral estimation for each of the computational basis states $\ket{0}$ to $\ket{15}$ as input, and determine the entire spectrum by combining all the results.

What kind of information can we get if the spectrum estimation is performed for all the computational basis states? In Equation {eq}`spectrum_estimation_final`, if $\ket{\psi} = \ket{l}$ $(l=0,\dots,2^{n_S} - 1)$ and

```{math}
:label: phim_decomposition
\ket{l} = \sum_{m=0}^{2^{n_S} - 1} c^l_m \ket{\phi_m}
```

then, fhe final state of this circuit will be:

$$
\sum_{k=0}^{2^{n_R} - 1} \sum_{m=0}^{2^{n_S} - 1} c^l_m f(\kappa_m - k) \ket{k}_R \ket{\phi_m}_S
$$

When Eqaation {eq}`phim_decomposition` holds, the following Equation also holds[^unitarity].

```{math}
:label: l_decomposition
\ket{\phi_m} = \sum_{l=0}^{2^{n_S} - 1} c^{l*}_m \ket{l}
```

Therefore, if we measure the final state of the circuit in the computational basis states of R and S, the resulting probability of obtaining $k, h$, denoted as $P_l(k, h)$, is

$$
P_l(k, h) = \left| \sum_{m=0}^{2^{n_S} - 1} c^l_m c^{h*}_m f(\kappa_m - k) \right|^2
$$

となります。$c^l_m$の値がわからなくても、これらの分布から
Let's think how we can obtain: 

$$
P(k) = \frac{1}{2^{n_S}} \sum_{m=0}^{2^{n_S} - 1} |f(\kappa_m - k)|^2
$$

from these distributions, even if we do not know the value of $c_m^l$.

Since the distribution of $|f(\kappa_m - k)|$ has a sharp peak near $\kappa_m$, we will be able to observe $m$ peaks (though they could be partially overlapped) by making plots of $P(k)$ with respect to $k$. From these peaks, we can calculate the energy eigenvalues.

For example, the $P(k)$ distribution for $n=2$, $g=0$, $\hbar \omega = 20J and $n_R=4$ is as follows (Note that unlike exercise 1, $\kappa_m$ is not an integer because $\hbar \omega = 20J$). In this plot, we set $P(k - 2^{n_R}) = P(k)$ and show the range of $-2^{n_R - 1} \leq k < 2^{n_R - 1}$ to visualize the negative eigenvalues. 


```{image} figs/spectrum_estimation_example.png
:alt: spectrum_estimation_example
:width: 500px
:align: center
```

Let's create plots like this by taking $n=4$ and incrementing the $g$ value by 0.1 from 0 to 0.5.

First, a function that takes computational bases and the $g$ values as arguments and returns the probability distribution of the final state is defined. We will use state vector simulator by default to avoid statistical errors due to finite sampling caused by using an ordinary shot-based simulator.

[^unitarity]: This is because both $\{\ket{l}\}$ and $\{\ket{\phi_m}\}$ span orthonormal basis states for the state register (transformation matrices are unitary).

```{code-cell} ipython3
def get_spectrum_for_comp_basis(
    n_state: int,
    n_readout: int,
    l: int,
    energy_norm: float,
    g: float,
    shots: int = 0
) -> np.ndarray:
    """Compute and return the distribution P_l(k, h) as an ndarray.

    Args:
        n_state: Size of the state register.
        n_readout: Size of the readout register.
        l: Index of the initial-state computational basis in the state register.
        energy_norm: Hamiltonian normalization.
        g: Parameter g of the Heisenberg model.
        shots: Number of shots. If <= 0, statevector simulation will be used.
    """

    # Define the circuit
    state_register = QuantumRegister(n_state, 'state')
    readout_register = QuantumRegister(n_readout, 'readout')
    circuit = QuantumCircuit(state_register, readout_register)

    # Initialize the state register
    for iq in range(n_state):
        if ((l >> iq) & 1) == 1:
            circuit.x(state_register[iq])

    u_circuit = trotter_twopi_heisenberg(state_register, energy_norm, g, num_steps)
    se_circuit = spectrum_estimation(state_register, readout_register, u_circuit)

    circuit.compose(se_circuit, inplace=True)

    # Extract the probability distribution as an array of shape (2 ** n_readout, 2 ** n_state)
    if shots <= 0:
        circuit.save_statevector()

        simulator = AerSimulator(method='statevector')
        circuit = transpile(circuit, backend=simulator)
        job = simulator.run(circuit)
        result = job.result()
        statevector = result.data()['statevector']

        # Convert the state vector into a probability distribution by taking the norm-squared
        probs = np.square(np.abs(statevector)).reshape((2 ** n_readout, 2 ** n_state))
        # Clean up the numerical artifacts
        probs = np.where(probs > 1.e-6, probs, np.zeros_like(probs))

    else:
        circuit.measure_all()

        # Run the circuit in simulator and plot the histogram
        simulator = AerSimulator()
        circuit = transpile(circuit, backend=simulator)
        job = simulator.run(circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        probs = np.zeros((2 ** n_readout, 2 ** n_state), dtype=float)

        for bitstring, count in counts.items():
            readout = int(bitstring[:n_readout], 2)
            state = int(bitstring[n_readout:], 2)

            probs[readout, state] = count

        probs /= np.sum(probs)

    # probs[k, h] = P_l(k, h)
    return probs
```

We decide the number of bits in the readout register here. We take $\hbar \omega = 8(3 + |g|)J$ because the number of spins is 4. When $g=0$, the expected smallest absolute value of the eigenvalues of $\Theta$ is $1/24$. But, in fact the smallest value is expected to be $n=4$ times that value, that is, $1/6$, due to the symmetry of the system. Since we only consider $|g| \ll 1$, the external magnetic field is treated as a perturbation, and $n_R=5$ is chosen so that $2^{n_R} / 6$ is sufficiently greater than 1. 

Now we have decided the circuit parameters. We will then define a function that calls the `get_spectrum_for_comp_basis` function with $g$ as an argument for $2^n$ computational basis states, and the function is executed for $g=0$ (this will take some time).


```{code-cell} ipython3
:tags: [remove-output]

n_state = 4
n_readout = 5
energy_norm = 1. / 24.

g_values = np.linspace(0., 0.5, 6, endpoint=True)

spectra = np.empty((g_values.shape[0], 2 ** n_readout), dtype=float)

def get_full_spectrum(g):
    """Compute and return the distribution P(k) for a value of g.
    """

    spectrum = np.zeros(2 ** n_readout, dtype=float)

    for l in range(2 ** n_state):
        probs = get_spectrum_for_comp_basis(n_state, n_readout, l, energy_norm, g)
        print('Computed spectrum for g = {:.1f} l = {:d}'.format(g, l))

        ##################
        ### EDIT BELOW ###
        ##################

        ##################
        ### EDIT ABOVE ###
        ##################

    return spectrum

# roll(spectrum, 2^{n_R-1}) => range of k is [-2^{n_R}/2, 2^{n_R}/2 - 1]
spectra[0] = np.roll(get_full_spectrum(0.), 2 ** (n_readout - 1))
```

Let's make a plot of the resulting $P(k)$ distribution by converting $k$ to the energy.

```{code-cell} ipython3
:tags: [remove-output]

plt.plot(np.linspace(-0.5 / energy_norm, 0.5 / energy_norm, 2 ** n_readout), spectra[0], 'o')
plt.xlabel('E/J')
plt.ylabel('P(E)')
```

Next, we will execute the same function for $g=0.1$, 0.2, 0.3, 0.4 and 0.5, and make plot for the relation between the energy eigenvalues of the system and $g$ from each spectrum.

```{code-cell} ipython3
:tags: [remove-output]

for i in range(1, g_values.shape[0]):
    spectra[i] = np.roll(get_full_spectrum(g_values[i]), 2 ** (n_readout - 1))
```

```{code-cell} ipython3
:tags: [remove-output]

energy_eigenvalues = np.empty((g_values.shape[0], 2 ** n_state))

# Extract the energy eigenvalues from spectra and fill the array
##################
### EDIT BELOW ###
##################

#energy_eigenvalues[ig, m] = E_ig_m

##################
### EDIT ABOVE ###
##################

plt.plot(g_values, energy_eigenvalues)
```

**Items to submit**:

- Explanation of how to derive $P(k)$ from $P_l(k, h)$ and its implementation into the `get_full_spectrum` function
- Work out how to extract energy eigenvalues from $P(k)$ and the code to implement the method (if you come up with completely different method for extracting the energy eigenvalues, it's fine to submit that as well)
- Plot of the relation between 16 energy eigenvalues and $g$


**HInt**:

(Regarding the derivation of $P(k)$) If you look at Equations {eq}`phim_decomposition` and {eq}`l_decomposition`, you will see that the following equation holds.

$$
\sum_{l=0}^{2^{n_S} - 1} c^l_m c^{l*}_n = \delta_{mn}
$$

Here $\delta_{mn}$ is a Kronecker's $\delta$, which is 1 for $m=n$ and 0 otherwise.
