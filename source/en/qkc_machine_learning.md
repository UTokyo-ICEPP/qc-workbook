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

# 【Exercise】Classification of New Physics with Quantum Kernel

+++ {"pycharm": {"name": "#%% md\n"}}

We have so far looked at the possibility of using the {doc}`quantum circuit learning<vqc_machine_learning>` technique to search for new physics, which is a quantum-classical hybrid algorithm. Here in this exercise, we consider the method of utilizing **Quantum Kernels**, an alternative approach for quantum machine learning. In particular, we attempt to apply the **Support Vector Machine**{cite}`quantum_svm` based on quantum kernels to the same problem of new physics search.

```{contents} Contents
---
local: true
---
```

$\newcommand{\ket}[1]{| #1 \rangle}$
$\newcommand{\expval}[3]{\langle #1 | #2 | #3 \rangle}$

+++ {"pycharm": {"name": "#%% md\n"}}

(q_kernel)=
## Quantum Kernel

For quantum circuit learning based on variational quantum algorithm, the following feature map $U_{\text{in}}(x_i)$ was considered.

$$
U_{\text{in}}(x_i) = \prod_j R_j^Z(\cos^{-1}(x^2))R_j^Y(\sin^{-1}(x))
$$

Applying the feature map to the initial state $\ket{0}^{\otimes n}$, the quantum state $\ket{\phi(x_i)}=U_{\text{in}}(x_i)\ket{0}^{\otimes n}$ is prepared from the input data.
The quantum kernel is defined as the (square of the absolute value of) inner product of this state $\langle\phi(x_j)|\phi(x_i)\rangle$:

$$
K(x_i,x_j):=|\langle\phi(x_j)|\phi(x_i)\rangle|^2=|\langle0^{\otimes n}|U_{\text{in}}^\dagger(x_j)U_{\text{in}}(x_i)|0^{\otimes n}\rangle|^2
$$

The quantum kernel provides a measure of how close the two states of $\ket{\phi(x_i)}$ and $\ket{\phi(x_j)}$, are or how much they are overlapped to each other.

+++ {"pycharm": {"name": "#%% md\n"}}

(q_kernel_imp)=
### Estimation of Quantum Kernel

In order to evaluate quantum kernel, it is necessary to calculate $K(x_i,x_j)=|\langle\phi(x_j)|\phi(x_i)\rangle|^2$ for all the pairs of $\{x_i,x_j\}$ in the training dataset. For the calculation of quantum kernel, one could often hear the term *kernel trick*. In the context of quantum computation, this is generally referred to that the kernel function $K(x_i,x_j)$ is calculated without explicitly using the coordinate values in Hilbert space. One way of doing this is to construct the following circuit:

```{image} figs/qke_circuit.png
:alt: qke_circuit
:width: 700px
:align: center
```

Assume that the input data $x_i$ and $x_j$ are encoded into the circuit and the output state is measured in $Z$ basis. From the definition of quantum kernel, the probability of measuring 0 for all the qubits (i.e, obtaining $0^n$ bitstring after the measurement) provides the $K(x_i,x_j)$ value. By repeating this for all the pairings of input data, the elements of kernel function are determined. Since the kernel is determined from the measurement of $0^n$ bitstring, a statistical uncertainty of ${\cal O}(1/\sqrt{N})$ ($N$ is the number of measurements) is associated with the kernel function.

+++ {"pycharm": {"name": "#%% md\n"}}

(svm)=
## Support Vector Machine

Kernel matrix has been obtained in steps so far. Now the kernel matrix will be incorporated into support vector machine to perform a 2-class classification task.

### Two-Class Linear Separation Problem

First, let us look at what the two-class linear separation problem is. Consider the training data $\{(\mathbf{X}_i,y_i)\}\:(i=1,\ldots,N)$ with $N$ samples, where $\mathbf{X}_i \in \mathbb{R}^d$ is an input and $y_i\in\{+1,-1\}$ is the label for the input. A separation problem means that what we aim at is to define a border in the space of input data $\mathbb{R}^d$ and separate the space into the region populated by data with label $+1$ and the region with label $-1$. When this border is a hyperplane, the problem is called linear separation problem. Here the hyperplane corresponds to, for a certain $\mathbf{w}\in\mathbb{R}^d, b \in \mathbb{R}$,

$$
\{\mathbf{X}| \mathbf{X} \in \mathbb{R}^d, \: \mathbf{w}\cdot\mathbf{X}+b=0\}
$$

A vector $\mathbf{w}$ is orthogonal to this hyperplane. Defining the norm of this vector as $\lVert \mathbf{w} \rVert$, $b/\lVert \mathbf{w} \rVert$ corresponds to the signed distance between the hyperplane and the origin (taken to be positive towards $\mathbf{w}$).

Since a hyperplane is simple and hence a special set of points, there is a case where the training data cannot be separated by the hyperplane, depending on the data distribution. Whether such separation is possible or not is equivalent to whether $(\mathbf{w},b)$ that satisfies

```{math}
:label: linear_separation
S_i(\mathbf{w}, b) := y_i(\mathbf{w}\cdot\mathbf{X}_i+b) \geq 1,\:\:\:\forall i=1,\ldots,N
```
exists or not. This equation can be interpreted as follows: the $\mathbf{w} \cdot \mathbf{X}_i + b$ in parentheses is the signed distance between the data point $X_i$ and hyperplane $(\mathbf{w},b)$, multiplied by $\lVert \mathbf{w} \rVert$. When this quantity is multiplied by $y_i$ and it is larger than 1, this means the data points with $y_i=1(-1)$ are in the positive (negative) region with respect to the hyperplane, and every point in the space is at least $1/\lVert \mathbf{w} \rVert$ distant from the hyperplane.

The purpose of machine learning is to construct a model based on training data and predict for unseen data with the trained model. For the present separation problem, $(\mathbf{w}, b)$ corresponds to the model, and the label prediction for unseen input $X$ is given by

```{math}
:label: test_data_label
y = \mathrm{sgn}(\mathbf{w} \cdot \mathbf{X} + b)
```

where $\mathrm{sgn}(z)$ is the sign of $z \in \mathbb{R}$. In this setup, we assume that a model which separates the training data the most "strongly" can predict for unseen data the most accurately. "Strongly" separating means that the distance between the hyperplane and all training data points, $1/\lVert \mathbf{w} \rVert$, is large. For linearly separable training data, $(\mathbf{w}, b)$ that satisfies Eq.{eq}`linear_separation` is not unique, and the model with the smallest $\lVert \mathbf{w} \rVert$ is going to be the best one.

For training data that cannot be separated linearly, we can also think of a problem where a model tries to separate "as much data as possible" in a similar fashion. In this case, the training corresponds to looking for $\mathbf{w}$ and $b$ that make $\lVert \mathbf{w} \rVert$ as small as possible and $\sum_{i} S_i(\mathbf{w}, b)$ as large as possible, and this can be achieved by minimizing the following objective function:

```{math}
:label: primal_1
f(\mathbf{w}, b) = \frac{1}{2} \lVert \mathbf{w} \rVert^2 + C \sum_{i=1}^{N} \mathrm{max}\left(0, 1 - S_i(\mathbf{w}, b)\right)
```

Here the coefficient $C>0$ is a hyperparameter that controls which of the two purposes is preferred and to what extent it is. The second term ignores the data points that have the $S_i$ value greater than 1 in $\mathrm{max}$ function (sufficiently distant from the hyperplane). The data points that are not ignored, i.e, the data near the separating hyperplane or wrongly separated data with $\{\mathbf{X}_i | S_i < 1\}$, are called "support vector". Which data point will be a support vector depends on the values of $\mathbf{w}$ and $b$, but once the parameters that minimize the function $f$ are determined, only the corresponding support vector is used to predict for unseen input data (more details of how it is used are discussed later). This machine learning model is called support vector machine.

+++ {"pycharm": {"name": "#%% md\n"}}

### Dual Formulation

Next, we consider a "dual formulation" of this optimization problem. A dual form can be obtained by defining a Lagrangian with constraints in an optimization problem and representing the values at stationary points as a function of undetermined multipliers. The introduction of constraints is carried out using the method of Karush-Kuhn-Tucker (KKT) conditions, which is a generalization of the method of Lagrange multipliers. The Lagrange multiplier allows only equality constraints while the method of KKT conditions is generalized to allow inequality constraints.

Let us first re-write Eq.{eq}`primal_1` by introducing parameters $\xi_i$ instead of using the $\mathrm{max}$ function:

$$
\begin{align}
F(\mathbf{w}, b, \{\xi_i\}) & = \frac{1}{2} \lVert \mathbf{w} \rVert^2 + C \sum_{i=1}^{N} \xi_i \\
\text{with} & \: \xi_i \geq 1 - S_i, \: \xi_i \geq 0 \quad \forall i
\end{align}
$$

When the $\mathbf{w}$, $b$ and $\{\xi_i\}$ that minimize $F$ by using the constraints on the second line, please confirm if the function $f$ is also minimized.

The Lagrangian of this optimization problem is given as follows by introducing non-negative mutlipliers $\{\alpha_i\}$ and $\{\beta_i\}$:

```{math}
:label: lagrangian
L(\mathbf{w}, b, \{\xi_i\}; \{\alpha_i\}, \{\beta_i\}) = \frac{1}{2} \lVert \mathbf{w} \rVert^2 + C \sum_{i=1}^{N} \xi_i - \sum_{i=1}^{N} \alpha_i \left(\xi_i + S_i(\mathbf{w}, b) - 1\right) - \sum_{i=1}^{N} \beta_i \xi_i
```

At stationary points, the following equations hold.

```{math}
:label: stationarity
\begin{align}
\frac{\partial L}{\partial \mathbf{w}} & = \mathbf{w} - \sum_i \alpha_i y_i \mathbf{X}_i = 0 \\
\frac{\partial L}{\partial b} & = -\sum_i \alpha_i y_i = 0 \\
\frac{\partial L}{\partial \xi_i} & = C - \alpha_i - \beta_i = 0
\end{align}
```

Therefore, by substituting these relations into Eq.{eq}`lagrangian`, the dual objection function

```{math}
:label: dual
\begin{align}
G(\{\alpha_i\}) & = \sum_{i} \alpha_i - \frac{1}{2} \sum_{ij} \alpha_i \alpha_j y_i y_j \mathbf{X}_i \cdot \mathbf{X}_j \\
\text{with} & \sum_i \alpha_i y_i = 0, \: 0 \leq \alpha_i \leq C \quad \forall i
\end{align}
```

can be obtained. Therefore, the dual formulation of the problem is to find $\{\alpha_i\}$ that maximizes the $G$. In addition, the optimized solutions $\mathbf{w}^*$, $b^*$ and $\{\xi^*_i\}$ in the main formulation and the solutions \{\alpha^*_i\}$ in the dual formulation have the following relations (complementarity conditions):

```{math}
:label: complementarity
\begin{align}
\alpha^*_i (\xi^*_i + S_i(\mathbf{w}^*, b^*) - 1) & = 0 \\
\beta^*_i \xi^*_i = (C - \alpha^*_i) \xi^*_i & = 0
\end{align}
```

+++ {"pycharm": {"name": "#%% md\n"}}

### Relation with Kernel Matrix

Even at this point, probably the relation between kernel matrix and the linear separation by support vector machine is not clear at first glance, but there is a hint in the dual formulation. The $\mathbf{X}_i \cdot \mathbf{X}_j$ in Eq.{eq}`dual` is the inner product of input vectors in the input space of $\mathbb{R}^d$. However, since the parameter $\mathbf{w}$ does not appear in the dual formulation, the problem is still valid even if $\mathbf{X}_i$ is taken to be an element in another linear space $V$. In fact, this part of Eq.{eq}`dual` is not even an inner product of vectors. Considering an element $x_i$ in some (not necessarily linear) space $D$, we can think of a function $K$ that represents a "distance" between two elements of $x_i$ and $x_j$:

$$
K: \: D \times D \to \mathbb{R}
$$

Then, most generally, the support vector machine can be defined as a problem to maximize the following objective function in terms of training data $\{(x_i, y_i) \in D \times \mathbb{R}\} \: (i=1,\ldots,N)$:

```{math}
:label: dual_kernel
\begin{align}
G(\{\alpha_i\}) & = \sum_{i} \alpha_i - \frac{1}{2} \sum_{ij} \alpha_i \alpha_j y_i y_j K(x_i, x_j) \\
\text{with} & \sum_i \alpha_i y_i = 0, \: \alpha_i \geq 0 \quad \forall i
\end{align}
```

The kernel function defined above corresponds exactly to this distance function $K(x_i, x_j)$. Now it becomes clear how the kernel function is incorporated into the support vector machine.

Looking further into the complementarity conditions of Eq.{eq}`complementarity`, it turns out that the optimized parameters $\alpha^*_i$, $\xi^*_i$ and $S^*_i$ ($S^*_i := S_i(\mathbf{w}^*, b^*)$) can have only values that satisfy either one of the following three conditions:

- $\alpha^*_i = C, \xi^*_i = 1 - S^*_i \geq 0$
- $\alpha^*_i = 0, \xi^*_i = 0$
- $0 < \alpha^*_i < C, \xi^*_i = 0, S^*_i = 1$

In particular, when $S^*_i > 1$, $\alpha^*_i = 0$. This indicates that the summation in Eq.{eq}`dual_kernel` can be taken over all $i$'s with $S^*_i \leq 1$, that is, the points in the support vector.

Finally, let us look at how the label of unseen data $x$ is predicted when the support vector machine represented in the kernel form is trained (i.e, when the $\{\alpha_i\}$ that maximizes $G$ are found). In the original main formulation of the problem, the label is given by Eq.{eq}`test_data_label`. When substituting the first equation of Eq.{eq}`stationarity` into it,

$$
y = \mathrm{sgn}\left(\sum_{i\in \mathrm{s.v.}} \alpha^*_i y_i K(x_i, x) + b^*\right)
$$

is obtained. Here $\alpha^*_i$ is the optimized parameter that maximizes $G$, and the summation is taken over $i$'s in the support vector. The optimized parameter $b^*$ can be obtained by solving

$$
y_j \left(\sum_{i\in \mathrm{s.v.}} \alpha^*_i y_i K(x_i, x_j) + b^*\right)= 1
$$

for data points $j$ that satisfy $S^*_j = 1$.

+++ {"pycharm": {"name": "#%% md\n"}}

(qsvm_imp)=
## Application to New Physics Search

Let us now move onto the problem we considered {doc}`here <vqc_machine_learning>` and see how we can use the quantum support vector machine.

The preparation of the dataset is the same as before.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import TwoLocal, ZFeatureMap, ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
# Read out variables from input file
df = pd.read_csv("data/SUSY_1K.csv",
                 names=('isSignal', 'lep1_pt', 'lep1_eta', 'lep1_phi', 'lep2_pt', 'lep2_eta',
                        'lep2_phi', 'miss_ene', 'miss_phi', 'MET_rel', 'axial_MET', 'M_R', 'M_TR_2',
                        'R', 'MT2', 'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos_theta_r1'))

# NUmber of input features used in the training
feature_dim = 3  # dimension of each data point

# Sets of 3, 5 and 7 input features
if feature_dim == 3:
    selected_features = ['lep1_pt', 'lep2_pt', 'miss_ene']
elif feature_dim == 5:
    selected_features = ['lep1_pt', 'lep2_pt', 'miss_ene', 'M_TR_2', 'M_Delta_R']
elif feature_dim == 7:
    selected_features = ['lep1_pt', 'lep1_eta', 'lep2_pt', 'lep2_eta', 'miss_ene', 'M_TR_2', 'M_Delta_R']

# Number of events in the training and testing samples
train_size = 20
test_size = 20

df_sig = df.loc[df.isSignal==1, selected_features]
df_bkg = df.loc[df.isSignal==0, selected_features]

# Creation of the samples
df_sig_train = df_sig.values[:train_size]
df_bkg_train = df_bkg.values[:train_size]
df_sig_test = df_sig.values[train_size:train_size + test_size]
df_bkg_test = df_bkg.values[train_size:train_size + test_size]
# The first (last) train_size events are signal (background) events that (do not) contain SUSY particles
train_data = np.concatenate([df_sig_train, df_bkg_train])
# The first (last) test_size events are signal (background) events that (do not) contain SUSY particles
test_data = np.concatenate([df_sig_test, df_bkg_test])

# Label
train_label = np.zeros(train_size * 2, dtype=int)
train_label[:train_size] = 1
test_label = np.zeros(train_size * 2, dtype=int)
test_label[:test_size] = 1

mms = MinMaxScaler((-1, 1))
norm_train_data = mms.fit_transform(train_data)
norm_test_data = mms.transform(test_data)
```

+++ {"pycharm": {"name": "#%% md\n"}}

(problem1)=
### Exercise 1

Select feature map and implement it as a quantum circuit object named `feature_map`. You could use the existing classes such as `ZFeatureMap` and `ZZFeatureMap` as done in {doc}`vqc_machine_learning`, or make an empty `QuantumCircuit` object and write a circuit by hand using `Parameter` and `ParameterVector`.

You could choose any number of qubits, but the `FidelityQuantumKernel` class used later seems to work better when the number of qubits is equal to the number of input features.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
##################
### EDIT BELOW ###
##################

# In case of writing a circuit from scratch
input_features = ParameterVector('x', feature_dim)
num_qubits = feature_dim
feature_map = QuantumCircuit(num_qubits)
# ...

##################
### EDIT ABOVE ###
##################
```

+++ {"pycharm": {"name": "#%% md\n"}}

(problem2)=
### Exercise 2

Create a circuit named with `manual_kernel` to calculate a kernel matrix from the feature map determined above. There is an API (`FidelityQuantumKernel` class) to do this automatically in Qiskit, but here please try to start with an empty `QuantumCircuit` object and write a parameterized circuit with the feature map.

**Hint 1**

A QuantumCircuit object ican be added into another QuantumCircuit by doing

```python
circuit.compose(another_circuit, inplace=True)
```
If `inplace=True` is omitted, the `compose` method just returns a new circuit object, instead of addint the `circuit` into `another_circuit`.

**Hint 2**

The QuantumCircuit class contains a method to return an inverse circuit called `inverse()`.

**Hint 3**

Please be careful about the parameter set of `manual_kernel`. If we create a `manual_kernel` from the `feature_map` or a simple copy of the `feature_map`, the `manual_kernel` will contain only parameters used in the `feature_map`.

A parameter set in a circuit can be replaced with another parameter set by doing, for example,

```python
current_parameters = circuit.parameters
new_parameters = ParameterVector('new_params', len(current_parameters))
bind_params = dict(zip(current_parameters, new_parameters))
new_circuit = circuit.assign_parameters(bind_params, inplace=False)
```

In this case, the `new_circuit` is parameterized with `new_parameters`.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
manual_kernel = QuantumCircuit(feature_map.num_qubits)

##################
### EDIT BELOW ###
##################

##################
### EDIT ABOVE ###
##################

manual_kernel.measure_all()
```

+++ {"pycharm": {"name": "#%% md\n"}}

Execute the created circuit with simulator to calculate the probability of measuring 0 for all qubits, $|\langle0^{\otimes n}|U_{\text{in}}^\dagger(x_1)U_{\text{in}}(x_0)|0^{\otimes n}\rangle|^2$.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
sampler = Sampler()

first_two_inputs = np.concatenate(norm_train_data[:2]).flatten()

job = sampler.run(manual_kernel, parameter_values=first_two_inputs, shots=10000)
# quasi_dists[0]がmanual_kernelの測定結果のcountsから推定される確率分布
fidelity = job.result().quasi_dists[0].get(0, 0.)
print(f'|<φ(x_0)|φ(x_1)>|^2 = {fidelity}')
```

Let us do the same thing using the `FidelityQuantumKernel` class.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
# FidelityQuantumKernel creates internally an instance of Sample class automatically.
q_kernel = FidelityQuantumKernel(feature_map=feature_map)

bind_params = dict(zip(feature_map.parameters, norm_train_data[0]))
feature_map_0 = feature_map.bind_parameters(bind_params)
bind_params = dict(zip(feature_map.parameters, norm_train_data[1]))
feature_map_1 = feature_map.bind_parameters(bind_params)

qc_circuit = q_kernel.fidelity.create_fidelity_circuit(feature_map_0, feature_map_1)
qc_circuit.decompose().decompose().draw('mpl')
```

+++ {"pycharm": {"name": "#%% md\n"}, "tags": ["raises-exception", "remove-output"]}

We can easily visualize the contents of kernel matrix with the `FidelityQuantumKernel` class. Let us make plots of the kernel matrix obtained from the training data alone, and that from the training and testing data.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
matrix_train = q_kernel.evaluate(x_vec=norm_train_data)
matrix_test = q_kernel.evaluate(x_vec=norm_test_data, y_vec=norm_train_data)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(np.asmatrix(matrix_train), interpolation='nearest', origin='upper', cmap='Blues')
axs[0].set_title("training kernel matrix")
axs[1].imshow(np.asmatrix(matrix_test), interpolation='nearest', origin='upper', cmap='Reds')
axs[1].set_title("validation kernel matrix")
plt.show()
```

+++ {"pycharm": {"name": "#%% md\n"}, "tags": ["raises-exception", "remove-output"]}

At the end, we attempt to perform classification with support vector machine implemented in sklearn package. Please check how the classification accuracy varies when changing the dataset size or feature maps.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
qc_svc = SVC(kernel='precomputed') # Default valuye of hyperparameter C is 1
qc_svc.fit(matrix_train, train_label)

train_score = qc_svc.score(matrix_train, train_label)
test_score = qc_svc.score(matrix_test, test_label)

print(f'Precomputed kernel: Classification Train score: {train_score*100}%')
print(f'Precomputed kernel: Classification Test score:  {test_score*100}%')
```

+++ {"pycharm": {"name": "#%% md\n"}, "tags": ["raises-exception", "remove-output"]}

**Items to submit**
- Explanation of the selected feature map and the code (Exercise 1).
- Quantum circuit to calculate kernel matrix and the result of $K(x_0, x_1)$ obtained using the circuit (Exercise 2).
- Comparison with results from quantum machine learning using variational quantum circuit in {doc}`this workboo <vqc_machine_learning>`.
   - Can we observe any systematic difference in classification accuracy when comparing the two methods in the same conditions (input features, dataset size, feature map)? Vary the conditions and discuss the observed behavior.
   - If one is systematically worse than the other, how can we improve the worse one? When the datasize is small, it is likely that the over-fitting occurs, i.e, the performance for the testing data is worse than that for the training data. Discuss if/how we can improve the classification performance for the testing data while reducing the effect of over-fitting.
