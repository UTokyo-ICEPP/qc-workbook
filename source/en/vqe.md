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

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-input", "remove-output"]}

# Variational Principle and Variational Quantum Eigensolver

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-input", "remove-output"]}

In this exercise, we will learn the basic concepts of variational method and the computation based on variational quantum algorithm. In particular, we will focus on **quantum-classical hybrid** approach of variational quantum algorithm where quantum and classical computations are combined. As a concrete implementation of such hybrid algorithm, we discuss an algorithm called **variational quantum eigensolver**, that enables us to approximately calculate eigenvalues of a physical system.

```{contents} Contents
---
local: true
---
```

$\newcommand{\ket}[1]{| #1 \rangle}$
$\newcommand{\bra}[1]{\langle #1 |}$
$\newcommand{\braket}[2]{\langle #1 | #2 \rangle}$
$\newcommand{\expval}[3]{\langle #1 | #2 | #3 \rangle}$

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-input", "remove-output"]}

## Introduction
For physical systems described by a Hermitian matrix, finding the smallest eigenvalue of the matrix is an important technique for many applications. For example, in chemistry calculation the smallest eigenvalue of a Hamiltonian that characterizes the system of, e.g, a molecule is the ground state energy of the molecule. A method called **"Quantum Phase Estimation"** (QPE) can be used to find the smallest eigenvalue (see this {doc}`exercise<spectrum_estimation>`), but it is generally known that the QPE requires a deep quantum circuit to solve pratical problems, preventing it from using it on NISQ machines. Therefore, the **Variational Quantum Eigensolver** (VQE) was proposed insteadd to estimate the ground state energy of a molecule with much shallower circuits{cite}`vqe`.

First, let's formally express the relation that forms the basis of VQE. Given a Hermitian matrix $H$ with an unknown minimum eigenvalue $\lambda_{min}$ associated with the eigenstate $\ket{\psi_{min}}$, VQE allows us to determine an approximated value $\lambda_{\theta}$ of the lowest energy bound $\lambda_{min}$ of the system.

In other words, it corresponds to determining the smallest $\lambda_{\theta}$ value that satisfies the following:

$$
\lambda_{min} \le \lambda_{\theta} \equiv \expval{ \psi(\theta)}{H}{\psi(\theta) }
$$

where $\ket{\psi(\theta)}$ is an eigenstate associated with the eigenvalue $\lambda_{\theta}$, and $\theta$ is a parameter. The idea is to obtain the state $\ket{\psi(\theta)} \equiv U(\theta)\ket{\psi}$ that approximates $\ket{\psi_{min}}$ by applying a parameterized unitary $U(\theta)$ to a certain initial state $\ket{\psi}$. The optimized value of the parameter $\theta$ is determined by iterating classical calculation such that the expectation value $\expval{\psi(\theta)}{H}{\psi(\theta)}$ is minimized.

+++

## Variational Method in Quantum Mechanics

### Background

VQE is based on **variational method** in quantum mechanics. To better understand the variational method, a fundamental mathematical background is provided first.

An eigenvector $\ket{\psi_i}$ of a matrix $A$ and its eigenvalue $\lambda_i$ satisfies $A \ket{\psi_i} = \lambda_i \ket{\psi_i}$. When the $H$ is an Hermitian ($H = H^{\dagger}$), the eigenvalue of $H$ is a real number ($\lambda_i = \lambda_i^*$) according to the spectral theorem. As any experimentally measured quantity is real number, we usually consider a Hermitian matrix to form the Hamiltonian of a physical system. Moreover, $H$ may be expressed as follows:

$$
H = \sum_{i = 1}^{N} \lambda_i \ket{\psi_i} \bra{ \psi_i }
$$

where $\lambda_i$ is the eigenvalue associated with eigenvector $\ket{\psi_i}$. Since the expectation value of an observable $H$ for a quantum state $\ket{\psi}$ is given by

$$
\langle H \rangle_{\psi} \equiv \expval{ \psi }{ H }{ \psi },
$$

substituting the $H$ above into this equation results in the following.

$$
\begin{aligned}
\langle H \rangle_{\psi} = \expval{ \psi }{ H }{ \psi } &= \bra{ \psi } \left(\sum_{i = 1}^{N} \lambda_i \ket{\psi_i} \bra{ \psi_i }\right) \ket{\psi}\\
&= \sum_{i = 1}^{N} \lambda_i \braket{ \psi }{ \psi_i} \braket{ \psi_i }{ \psi} \\
&= \sum_{i = 1}^{N} \lambda_i | \braket{ \psi_i }{ \psi} |^2
\end{aligned}
$$

The last equation shows that the expectation value of $H$ in a given state $\ket{\psi}$ can be expressed as a linear combination of (the squared absolute value of) the inner products of the eigenvector $\ket{\psi_i}$ and $\ket{\psi}$, weighted by the eigenvalue $\lambda_i$. Since $| \braket{ \psi_i }{ \psi} |^2 \ge 0$, it is obvious that the following holds:

$$
\lambda_{min} \le \langle H \rangle_{\psi} = \expval{ \psi }{ H }{ \psi } = \sum_{i = 1}^{N} \lambda_i | \braket{ \psi_i }{ \psi} |^2
$$

The above equation is known as **variational method** (also referred to as **variational principle**). It shows that if we can take an *appropriate* wavefunction, the smallest eigenvalue can be apprroximately obtained as the lower bound to the expectation value of the Hamiltonian $H$ (though how we can take such wavefunction is unknown at this point). Moreover, the expectation value of the eigenstate $\ket{\psi_{min}}$ is given by $\expval{ \psi_{min}}{H}{\psi_{min}} = \expval{ \psi_{min}}{\lambda_{min}}{\psi_{min}} = \lambda_{min}$.

### Approximation of Ground States
When the Hamiltonian of a system is described by the Hermitian matrix $H$, the ground state energy of that system is the smallest eigenvalue associated with $H$. Selecting an arbitrary wavefunction $\ket{\psi}$ (called an **ansatz**) as an initial guess of $\ket{\psi_{min}}$, we calculate the expectation value of the Hamiltonian $\langle H \rangle_{\psi}$ under the state. The key to the variational method lies in iteratively performing calculations while updating the wavefunction to make the expectation value smaller, thereby approaching the ground state energy of the Hamiltonian.

+++

(vqa)=
## Variational Quantum Algorithm

First, we will look at **Variational Quantum Algorithm** (VQA), which the VQE is based on.

+++

### Variational Quantum Circuit
To implement the variational method on a quantum computer, we need a mechanism to update the ansatz. As we know, quantum gates can be used to update quantum states. VQE also uses quantum gates, but does so through the use of a parameterized quantum circuit with a certain structure (called a **variational quantum circuit**). Such a circuit is sometimes called a **variational form**, and the entire circuit is often represented as a unitary operation $U(\theta)$ ($\theta$ is a parameter, often becomes a vector with multiple parameters).

When a variational form is applied to an initial state $\ket{\psi}$ (such as the standard state $\ket{0}$), it generates an output state $\ket{\psi(\theta)} \equiv U(\theta)\ket{\psi}$. The VQE attemps to optimize parameter $\theta$ for $\ket{\psi(\theta)}$ such that the expectation value $\expval{ \psi(\theta)}{H}{\psi(\theta)}$ gets close to $\lambda_{min}$. This parameter optimization is envisioned as being performed using classical computation. In this sense, VQE is a typical **quantum-classical hybrid algorithm**.

When constructing variational forms, it is possible to choose variational forms with specific structures based on the domain knowledge of the question to be answered. There are also variational forms that are not domain-specific and can be applied to a wide range of problems (such as $R_X$, $R_Y$, and other rotation gates). Later on, we will look at an assignment in which you will apply VQE to a high energy physics experiment. That assignment implements variational forms that use $R_Y$ and controlled $Z$ gates.

+++

### Simple Variational Form
When constructing a variational form, we have to consider the balance between two competing objectives. If the number of parameters is increased, our $n$-qubit variational form might be able to generate any possible state $\ket{\psi}$ with $2^{n+1}-2$ real degrees of freedom. However, in order to optimize the parameters, we would like the variational form to use as few as possible. Increasing the number of gates that use rotation angles as parameters generally makes the computation more susceptible to noise. From that point of view, one could desire to generate states using as few parameters (and gates) as possible.

Consider the case where $n=1$. The Qiskit $U$ gate (not to confuse with $U(\theta)$) notation used above!) takes three parameters of $\theta$、$\phi$ and $\lambda$, and represents the following transformation.

$$
U(\theta, \phi, \lambda) = \begin{pmatrix}\cos\frac{\theta}{2} & -e^{i\lambda}\sin\frac{\theta}{2} \\ e^{i\phi}\sin\frac{\theta}{2} & e^{i\lambda + i\phi}\cos\frac{\theta}{2} \end{pmatrix}
$$

If the initial state of the variational form is taken to be $\ket{0}$, only the first column of the above matrix acts to transform the state, and the two parameters $\theta$ and $\phi$ of the matrix allow us to produce an arbitrary single qubit state. Therefore, this variational form is known to be **universal**. However, this universality brings a subtle issue: when one tries to use this variational form to generate some quantum states and calculate expectation values of a given Hamiltonian, the generated states could be any states other than the eigenstates of the Hamiltonian. In other words, whether the VQE can find efficiently the smallest eigenvalue or not depends crucially on how we can appropriately optimize the parameters while avoiding such vast majority of unwanted quantum states.

+++

### Parameter Optimization
Once you select a parameterized variational form, then you need to optimize the parameters based on the variational method to minimize the expectation value of the target Hamiltonian. The parameter optimization process involves various challenges. For example, quantum hardware has a variety of noises, so there is no guarantee that measuring the energy in this state will return the correct answer. This may result in deviation from the correct objective function value, thus preventing the parameters from being updated properly. Additionally, depending on optimization methods (**"optimizers"**), the number of objective function evaluations generally increases with the number of parameters, making it even more susceptible to noise. An appropriate optimizer must therefore be selected by considering the requirements of the application.

The most typical optimization strategy is based on **gradient descent**, in which each parameter is updated in the direction yielding the largest decrease in energy. Because the gradient is calculated for each parameter, the greater the number of parameters to optimize the greater the number of objective function evaluations. This allows the algorithm to quickly find a local minimum in the search space. However, this optimization strategy often gets stuck at local minima, preventing the algorithm from reaching the global minimum. The gradient descent is intuitive and easy to understand, but it is considered difficult to perform accurate calculation of gradient descent on a present NISQ computer (the gradient-based optimization is introduced later when implementing VQE).

As an optimizer suitable for a noisy quantum computer, the *Simultaneous Perturbation Stochastic Approximation optimizer* (**SPSA**){cite}`bhatnagar_optimization` has been proposed. The SPSA attempts to approximate the gradient of the objective function with only two measurements. In addition, the SPSA concurrently changes all the parameters in a random fashion, in contrast to gradient descent where each parameter is changed independently. Because of this, the SPSA is generally recommended as the optimizer when utilizing VQE.

When evaluating an objective function on a noise-less quantum computer (e.g, when executing on a statevector simulator), a wide variety of optimizers, e.g, those included in Python's <a href="https://www.scipy.org/scipylib/index.html" target="_blank">SciPy</a> package, may be used. In this exercise, we will use one of the optimizers supported in Qiskit, namely the *Constrained Optimization by Linear Approximation optimizer* (**COBYLA**). The COBYLA performs the evaluation of objective function only once (that is, independently of the number of parameters in a circuit). Therefore, if one wants to minimize the number of evaluations under a noise-free situation, the COBYLA tends to be recommended. However, since the performance of optimizer depends crucially on implementation of the VQE algorithm and execution environment, it is important to study it beforehand and select the most appropriate optimizer for problem in hand.

+++

### Example of Variational Form
We now try to perform parameter optimization using a single-qubit variational form composed of $U$ gate. Since a single-qubit state is determined (up to global phase) by using the expectation values of $\langle X \rangle$, $\langle Y \rangle$ and $\langle Z \rangle$ for the observables $X$, $Y$ and $Z$, respectively, one may optimize $\theta$ and $\phi$ so that the expectation values $\langle X \rangle_{\theta, \phi}$, $\langle Y \rangle_{\theta, \phi}$ and $\langle Z \rangle_{\theta, \phi}$ for $X$, $Y$ and $Z$ under the state $\ket{\psi(\theta, \phi)}$ become equal to the corresponding expectation values $\langle X \rangle_0$, $\langle Y \rangle_0$ and $\langle Z \rangle_0$ under $\ket{\psi_0}$.

Therefore, the problem is to minimize the following objective function.

$$
L(\theta, \phi) = [\langle X \rangle_{\theta, \phi} - \langle X \rangle_0]^2 + [\langle Y \rangle_{\theta, \phi} - \langle Y \rangle_0]^2 + [\langle Z \rangle_{\theta, \phi} - \langle Z \rangle_0]^2
$$

```{image} figs/vqe_u3.png
:alt: vqe_u3
:width: 400px
:align: center
```

[^actually_exact]: Since $U(\theta, \phi, 0)$ is universal for a single qubit, it can in principle be exact, not approximation.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import BackendEstimator
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_aer import AerSimulator
```

First, the function to randomly generate the target state vector and the function to calculate the expectation values of $X$, $Y$ and $Z$ from the state vector are defined. The state vector is represented using Statevector class in Qiskit, and SparsePauliOp class is used for Pauli operators.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
rng = np.random.default_rng(999999)

# Function to create a random state vector with nq qubits.
def random_statevector(nq):
    # Randomly generate 2^nq complex numbers
    data = rng.random(2 ** nq) + 1.j * rng.random(2 ** nq)
    # Normalization
    data /= np.sqrt(np.sum(np.square(np.abs(data))))

    return Statevector(data)

# Example: U(π/3, π/6, 0)|0>
statevector = Statevector(np.array([np.cos(np.pi / 6.), np.exp(1.j * np.pi / 6.) * np.sin(np.pi / 6.)]))
for pauli in ['X', 'Y', 'Z']:
    op = SparsePauliOp(pauli)
    print(f'<{pauli}> = {statevector.expectation_value(op).real}')
```

Next, define a variational form. Here the angles of $U$ gate are parameterized using Parameter objects in Qiskit. The Parameter object can be substituted for real values that will be put later.

```{code-cell} ipython3
:tags: [remove-output]

theta = Parameter('θ')
phi = Parameter('φ')

ansatz_1q = QuantumCircuit(1)
ansatz_1q.u(theta, phi, 0., 0)
```

We use the `assign_parameters` method of the circuit to assign values to Parameter object.

```{code-cell} ipython3
# Parameter value is unknown
ansatz_1q.draw('mpl')
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input, remove-output]
---
# Assign π/3 and π/6 to theta and phi
ansatz_1q.assign_parameters({theta: np.pi / 3., phi: np.pi / 6.}, inplace=False).draw('mpl')
```

Define the circuit to measure the expectation values of $X$, $Y$ and $Z$ under the state of the variational form.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input, remove-output]
---
circuits = {}

# Basis change with H gate for <X>
circuits['X'] = ansatz_1q.copy()
circuits['X'].h(0)
circuits['X'].measure_all()

# Basis change with Sdg and H gates for <Y>
circuits['Y'] = ansatz_1q.copy()
circuits['Y'].sdg(0)
circuits['Y'].h(0)
circuits['Y'].measure_all()

# No basis change for <Z>
circuits['Z'] = ansatz_1q.copy()
circuits['Z'].measure_all()
```

Now define the function to execute each circuit with `run()` method of the backend and calculate the expectation values from the results.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input, remove-output]
---
backend = AerSimulator()

def circuit_expval(circuit, param_vals):
    bound_circuit = circuit.assign_parameters({theta: param_vals[0], phi: param_vals[1]}, inplace=False)

    bound_circuit_tr = transpile(bound_circuit, backend=backend)
    # shots is defined outside the function
    job = backend.run(bound_circuit_tr, shots=shots)
    counts = job.result().get_counts()

    return (counts.get('0', 0) - counts.get('1', 0)) / shots

# Example: U(π/3, π/6, 0)|0>
shots = 10000
param_vals = [np.pi / 3., np.pi / 6.]
for pauli in ['X', 'Y', 'Z']:
    print(f'<{pauli}> = {circuit_expval(circuits[pauli], param_vals)}')
```

The objective function to be minimized is defined here.

```{code-cell} ipython3
def objective_function(param_vals):
    loss = 0.
    for pauli in ['X', 'Y', 'Z']:
        # target_state_1q is defined outside the function
        op = SparsePauliOp(pauli)
        target = target_state_1q.expectation_value(op).real
        current = circuit_expval(circuits[pauli], param_vals)
        loss += (target - current) ** 2

    return loss

# Called every optimization step. Store objective function values in a list.
def callback_function(param_vals):
    # losses is defined outside the function
    losses.append(objective_function(param_vals))
```

The function to calculate the fidelity $|\langle \psi_0 | \psi(\theta, \phi) \rangle|^2$ between the initial state and the target state after optimization is defined. If the optimization is perfect, this function value is exactly one.

```{code-cell} ipython3
def fidelity(ansatz, param_vals, target_state):
    # Can get the list of circuit parameters with circuit.parameters
    parameters = ansatz.parameters

    param_binding = dict(zip(parameters, param_vals))
    opt_ansatz = ansatz.assign_parameters(param_binding, inplace=False)

    # Statevector can be generated from quantum circuit (the final state obtaind by applying the circuit to |0>)
    circuit_state = Statevector(opt_ansatz)

    return np.square(np.abs(target_state.inner(circuit_state)))
```

Finally, an instance of the COBYLA optimizer is created and used to run the algorithm.

```{code-cell} ipython3
# Maximum number of steps in COBYLA
maxiter = 500
# Convergence threshold of COBYLA (the smaller the better approximation)
tol = 0.0001
# Number of shots in the backend
shots = 1000

# Intance
optimizer = COBYLA(maxiter=maxiter, tol=tol, callback=callback_function)
```

```{code-cell} ipython3
:tags: [remove-input]

# Cell for text

import os
if os.getenv('JUPYTERBOOK_BUILD') == '1':
    del optimizer
```

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

# Target state
target_state_1q = random_statevector(1)

# Choose theta and phi randomly within [0, π) and [0, 2π), respectively
init = [rng.uniform(0., np.pi), rng.uniform(0., 2. * np.pi)]

# Perform optimization
losses = list()
min_result = optimizer.minimize(objective_function, x0=init)
```

```{code-cell} ipython3
:tags: [remove-input]

# Cell for text

import pickle
if os.getenv('JUPYTERBOOK_BUILD') == '1':
    with open('data/vqe_results_1q.pkl', 'rb') as source:
        min_result, losses = pickle.load(source)
```

Make a plot of the objective function values during the optimization process.

```{code-cell} ipython3
plt.plot(losses);
```

```{raw-cell}
From the returned value `min_result` of `optimizer.minimize()`, we can obtain various information about the optimization process such as the number of calls of objective function and the number of steps required to reach the minimum. In particular, we calculate the fidelity using the optimized parameters from `min_result.x`.
```

```{code-cell} ipython3
fidelity(ansatz_1q, min_result.x, target_state_1q)
```

```{raw-cell}
Since the number of shots in finite, the optimized parameters do not exactly coincide with the exact solutions due to statistical uncertainty. Check the level of agreement by changing the number of shots and steps.
```

#### Using Estimator

For variational quantum algorithm including VQE, the optimization loop in which the parameters in variational form are substituted and the expectation valus of observables are calculated appear frequently. Therefore, it is recommended to use Estimator class, in which this process is automated and various error mitigation techniques can be adopted (though not used here). In the exercise below, we will use BackendEstimator to perform calculation using a specific backend.

```{code-cell} ipython3
# Create instance of BackendEstimator
estimator = BackendEstimator(backend)

# Define observable using SparsePauliOp objects
observables = [SparsePauliOp('X'), SparsePauliOp('Y'), SparsePauliOp('Z')]

param_vals = [np.pi / 3., np.pi / 6.]

# Pass variational form, observable and parameter values to run()
# Since there are three observables, 3 ansatz_1q and 3 param_values are prepared accordingly
job = estimator.run([ansatz_1q] * 3, observables, [param_vals] * 3, shots=10000)
result = job.result()
result.values
```

Define objective function using Estimator.

```{code-cell} ipython3
observables_1q = [SparsePauliOp('X'), SparsePauliOp('Y'), SparsePauliOp('Z')]

def objective_function_estimator(param_vals):
    target = np.array(list(target_state_1q.expectation_value(op).real for op in observables_1q))

    job = estimator.run([ansatz_1q] * len(observables_1q), observables_1q, [param_vals] * len(observables_1q), shots=shots)
    current = np.array(job.result().values)

    return np.sum(np.square(target - current))

def callback_function_estimator(param_vals):
    # losses is defined outside the function
    losses.append(objective_function_estimator(param_vals))
```

Optimize the objective function defined above.

```{code-cell} ipython3
# Maximum number of steps in COBYLA
maxiter = 500
# Convergence threshold of COBYLA (the smaller the better approximation)
tol = 0.0001
# Number of shots in the backend
shots = 1000

# Instance
optimizer = COBYLA(maxiter=maxiter, tol=tol, callback=callback_function_estimator)
```

```{code-cell} ipython3
:tags: [remove-input]

# Cell for text

if os.getenv('JUPYTERBOOK_BUILD') == '1':
    del optimizer
```

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

# Target state
target_state_1q = random_statevector(1)

# Choose theta and phi randomly within [0, π) and [0, 2π), respectively
init = [rng.uniform(0., np.pi), rng.uniform(0., 2. * np.pi)]

# Perform optimization
losses = []
min_result = optimizer.minimize(objective_function_estimator, x0=init)
```

```{code-cell} ipython3
:tags: [remove-input]

# Cell for text

if os.getenv('JUPYTERBOOK_BUILD') == '1':
    with open('data/vqe_result_1q_estimator.pkl', 'rb') as source:
        min_result = pickle.load(source)
```

```{code-cell} ipython3
fidelity(ansatz_1q, min_result.x, target_state_1q)
```

### Introducing Entanglement

Next, let us extend this problem to 2-qubit problem. Two-qubit pure state has degrees of freedom of 6 real values, but here we consider 15 observables to determine the most generic two-qubit state, with the expectation values of

$$
\langle O_1 O_2 \rangle \quad (O_1, O_2 = I, X, Y, Z; O_1 O_2 \neq II).
$$

Here $I$ is an identity operator.

The functions `random_statevector` and `pauli_expval` regarding the target states can be used as they are. First, consider a very simple variational form of $U$ gates attached on 2 qubits each, and define the objective function.

```{code-cell} ipython3
:tags: [remove-output]

# Create 4-element ParameterVector because the number of parameters is 4.
params = ParameterVector('params', 4)

ansatz_2q = QuantumCircuit(2)
ansatz_2q.u(params[0], params[1], 0., 0)
ansatz_2q.u(params[2], params[3], 0., 1)
```

```{code-cell} ipython3
paulis_1q = ['I', 'X', 'Y', 'Z']
paulis_2q = list(f'{op1}{op2}' for op1 in paulis_1q for op2 in paulis_1q if (op1, op2) != ('I', 'I'))
observables_2q = list(SparsePauliOp(pauli) for pauli in paulis_2q)

def objective_function_2q(param_vals):
    # target_state_2q is defined outside the function
    target = np.array(list(target_state_2q.expectation_value(op).real for op in observables_2q))

    job = estimator.run([ansatz_2q] * len(observables_2q), observables_2q, [param_vals] * len(observables_2q), shots=shots)
    current = np.array(job.result().values)

    return np.sum(np.square(target - current))

def callback_function_2q(param_vals):
    # losses is defined outside the function
    losses.append(objective_function_2q(param_vals))
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
# Maximum number of steps in COBYLA
maxiter = 500
# Convergence threshold of COBYLA (the smaller the better approximation)
tol = 0.0001
# Number of shots in the backend
shots = 1000

# Instance
optimizer = COBYLA(maxiter=maxiter, tol=tol, callback=callback_function_2q)

# Target state
target_state_2q = random_statevector(2)

# Parameter initial values
init = rng.uniform(0., 2. * np.pi, size=4)
```

```{code-cell} ipython3
:tags: [remove-input]

# Cell for text

if os.getenv('JUPYTERBOOK_BUILD') == '1':
    del optimizer
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
# Perform optimization
losses = list()
min_result = optimizer.minimize(objective_function_2q, x0=init)
```

```{code-cell} ipython3
:tags: [remove-input]

# Cell for text

if os.getenv('JUPYTERBOOK_BUILD') == '1':
    with open('data/vqe_result_2q.pkl', 'rb') as source:
        min_result = pickle.load(source)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
fidelity(ansatz_2q, min_result.x, target_state_2q)
```

You'll see that the results are not good compared to the case of single qubit. How can we improve?

+++

**One solution: Introduction of entanglement in the variational form**

```python
ansatz_2q = QuantumCircuit(2)
ansatz_2q.u(params[0], params[1], 0., 0)
ansatz_2q.u(params[2], params[3], 0., 1)
ansatz_2q.cx(0, 1)
```

Let us see what happens.

+++

A generic 2-qubit state is usually entangled, so one could natually expect the accuracy to be improved by introucing 2-qubit gates in the variational form. For example, we can see this most clearly when we want to produce the Bell state ([Confirming the violation of the CHSH inequality](https://utokyo-icepp.github.io/qc-workbook/chsh_inequality.html#id14)).

If you replace

```python
target_state_2q = random_statevector(2)
```

with

```python
target_state_2q = Statevector(np.array([1., 0., 0., 1.], dtype=complex) / np.sqrt(2.))
```

and execute, what do you get? You will observe a large difference with and without adding entanglment.

Play by extending the circuit to 3 qubits, e.g, by targeting the GHZ state ([Creating simple circuit from scratch](https://utokyo-icepp.github.io/qc-workbook/circuit_from_scratch.html#ghz))

```python
target_state_3q = Statevector(np.array([1.] + [0.] * 6 + [1.], dtype=complex) / np.sqrt(2.))
```

+++ {"pycharm": {"name": "#%% md\n"}}

(vqe)=
## Variational Quantum Eigensolver

Now, let us try to implement a simple VQE algorithm.

(param_shift)=
### Parameter Shift Rule
Before diving into VQE implementation, the optimization technique based on the gradient of objective function, calld **Parameter Shift Rule**, is explainedd. For certain types of quantum circuits, the gradient of objective function is known to be exactly calculablle. The classical optimizer can optimize the parameters using the values of the gradient.

First, we consider a parameterized unitary $U({\boldsymbol \theta})=\prod_{j=1}^LU_j(\theta_j)$ to derive gradients using parameter shift rule. The $U_j(\theta_j)$ is a unitary with parameter $\theta_j$, and can take the form of $U_j(\theta_j)=\exp(-i\theta_jP_j/2)$ with $\theta_j$ as an angle of Pauli operator $P_j\in\{X,Y,Z\}$. Evolving the initial state $\rho$ by $U({\boldsymbol \theta})$, the expectation value $\langle M({\boldsymbol \theta})\rangle$ for the observable $M$ under the evolved state is

$$
\langle M({\boldsymbol \theta})\rangle=\text{Tr}\left[MU({\boldsymbol \theta})\rho U({\boldsymbol \theta})^\dagger\right] = \text{Tr}\left[MU_{L:1}\rho U_{L:1}^\dagger\right]
$$

Here $U_{l:m}:=\prod_{j=m}^lU_j(\theta_j)$. The gradient of this expectation value for parameter $\theta_j$ is

$$
\frac{\partial}{\partial\theta_j}\langle M({\boldsymbol \theta})\rangle=\text{Tr}\left[M\frac{\partial U_{L:1}}{\partial\theta_j}\rho U_{L:1}^\dagger\right]+\text{Tr}\left[MU_{L:1}\rho\frac{\partial U_{L:1}^\dagger}{\partial\theta_j}\right]
$$

Since $P_j^\dagger=P_j$,

$$
\begin{aligned}
\frac{\partial U_{L:1}}{\partial\theta_j} &= U_L\ldots U_{j+1}\frac{\partial U_j}{\partial\theta_j}U_{j-1}\ldots U_1=-\frac{i}{2}U_{L:j}P_jU_{j-1:1} \\
\frac{\partial U_{L:1}^\dagger}{\partial\theta_j} &=\frac{i}{2}U_{j-1:1}^\dagger P_jU_{L:j}^\dagger
\end{aligned}
$$

holds. From this, one can get

$$
\frac{\partial}{\partial\theta_j}\langle M({\boldsymbol \theta})\rangle=-\frac{i}{2}\text{Tr}\left[MU_{L:j}\left[P_j,U_{j-1:1}\rho U_{j-1:1}^\dagger\right]U_{L:j}^\dagger\right].
$$

Since $P_j$ is a Pauli operator, $U_j(\theta_j)=\exp(-i\theta_jP_j/2)=\cos(\theta_j/2)I-i\sin(\theta_j/2)P_j$ ($I$ is identity), thus $U(\pm\pi/2)=(1/\sqrt{2})(I\mp iP_j)$. Therefore,

$$
U_j\left(\frac{\pi}{2}\right)\rho U_j^\dagger\left(\frac{\pi}{2}\right)-U_j\left(-\frac{\pi}{2}\right)\rho U_j^\dagger\left(-\frac{\pi}{2}\right) = \frac12\left(I-iP_j\right)\rho\left(I+iP_j^\dagger\right)-\frac12\left(I+iP_j\right)\rho\left(I-iP_j^\dagger\right) = -i[P_j,\rho].
$$

When applying this result to the above equation of $\partial\langle M({\boldsymbol \theta})\rangle/\partial\theta_j$,

$$
\begin{aligned}
\frac{\partial}{\partial\theta_j}\langle M({\boldsymbol \theta})\rangle &=-\frac{i}{2}\text{Tr}\left[MU_{L:j}[P_j,U_{j-1:1}\rho U_{j-1:1}^\dagger]U_{L:j}^\dagger\right] \\
&= \frac12\text{Tr}\left[MU_{L:j+1}U_j\left(\theta_j+\frac{\pi}{2}\right)U_{j-1:1}\rho U_{j-1:1}^\dagger U_j^\dagger\left(\theta_j+\frac{\pi}{2}\right) U_{L:j+1}^\dagger-MU_{L:j+1}U_j\left(\theta_j-\frac{\pi}{2}\right)U_{j-1:1}\rho U_{j-1:1}^\dagger U_j^\dagger\left(\theta_j-\frac{\pi}{2}\right) U_{L:j+1}^\dagger)\right] \\
&= \frac12\left[\left\langle M\left({\boldsymbol \theta}+\frac{\pi}{2}{\boldsymbol e}_j\right)\right\rangle - \left\langle M\left({\boldsymbol \theta}-\frac{\pi}{2}{\boldsymbol e}_j\right)\right\rangle\right]
\end{aligned}
$$

holds where ${\boldsymbol e}_j$ is a vector with 1 for the $j$-th element and 0 otherwise.

From this, it turns out that the gradient of the expectation value $\langle M({\boldsymbol \theta})\rangle$ with respect to parameter $\theta_j$ can be obtained as a difference between the two expectation values with the $\theta_j$ values shifted by $\pm\pi/2$. This is the parameter shift rule.

+++ {"pycharm": {"name": "#%% md\n"}}

(vqe_imp)=
### VQE Implementation
Moving on to a simple VQE implementation with parameter shift rule. The problem is to determine the parameters of ansatz by minimizing the expectation value of a certain observable with VQE.

The circuit contains only $R_YR_Z$ gates and the observable is $ZXY$, a tensor product of Pauli $Z$, $X$ and $Y$.

Actully the parameter shift rule can be implemented in a single line using ParamShiftEstimatorGradient API in Qiskit (if you are interested in gradient calculation, you could write the circuit to calculate the expectation values with each parameter shifted by $\pm\pi/2$ and compare the gradient from the API wtih the difference of the two expectation values). The parameter optimization is performed using Conjugate Descent (CG) and Gradient Descent optimizers, both based on gradient descent, and is compared with COBYLA.

Finally, the energies obtained using VQE are compared with the true lowest energy from exact diagonalization.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input, remove-output]
---
from qiskit_algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import CG, GradientDescent
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
```

```{code-cell} ipython3
# Definition of ansatz

num_qubits = 3   # number of qubits
num_layers = 2  # number of layers

ansatz = QuantumCircuit(num_qubits)

# Parameter array with length 0
theta = ParameterVector('θ')

# Add one element to the array and return the edded element
def new_theta():
    theta.resize(len(theta) + 1)
    return theta[-1]

for _ in range(num_layers):
    for iq in range(num_qubits):
        ansatz.ry(new_theta(), iq)

    for iq in range(num_qubits):
        ansatz.rz(new_theta(), iq)

    #for iq in range(num_qubits - 1):
    #    ansatz.cx(iq, iq + 1)

ansatz.draw('mpl')
```

```{code-cell} ipython3
# Observable
obs = SparsePauliOp('ZXY')

# Initial values of parameters
init = rng.uniform(0., 2. * np.pi, size=len(theta))

# Gradient from parameter shift rule using estimator object
grad = ParamShiftEstimatorGradient(estimator)

# VQE with Conjugate Gradient methof
optimizer_cg = CG(maxiter=200)
vqe_cg = VQE(estimator, ansatz, optimizer_cg, gradient=grad, initial_point=init)

# VQE with Gradient Descent method
optimizer_gd = GradientDescent(maxiter=200)
vqe_gd = VQE(estimator, ansatz, optimizer_gd, gradient=grad, initial_point=init)

# VQE with COBYLA method
optimizer_cobyla = COBYLA(maxiter=300)
vqe_cobyla = VQE(estimator, ansatz, optimizer_cobyla, initial_point=init)

# Solver with Exact Diagonalization
ee = NumPyMinimumEigensolver()
```

```{code-cell} ipython3
:tags: [remove-input]

# Cell for text

if os.getenv('JUPYTERBOOK_BUILD') == '1':
    del obs
```

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

result_vqe_cg = vqe_cg.compute_minimum_eigenvalue(obs)
result_vqe_gd = vqe_gd.compute_minimum_eigenvalue(obs)
result_vqe_cobyla = vqe_cobyla.compute_minimum_eigenvalue(obs)
result_ee = ee.compute_minimum_eigenvalue(obs)
```

```{code-cell} ipython3
:tags: [remove-input]

with open('data/vqe_results.pkl', 'rb') as source:
    result_ee, result_vqe_cobyla, result_vqe_cg, result_vqe_gd = pickle.load(source)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
print('Result:')
print(f'  Exact      = {result_ee.eigenvalue}')
print(f'  VQE(COBYLA) = {result_vqe_cobyla.optimal_value}')
print(f'  VQE(CG)    = {result_vqe_cg.optimal_value}')
print(f'  VQE(GD)    = {result_vqe_gd.optimal_value}')
```

+++ {"pycharm": {"name": "#%% md\n"}}

The VQE with COBYLA likely returns a result very close to the exact answer (= -1.0). The VQE with gradient-based optimizers also often works well, but sometime returns very bad answers depending on the initial values of the parameters.

You could play with different ansatzes, changing the observables and parameters.
