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
varInspector:
  cols:
    lenName: 16
    lenType: 16
    lenVar: 40
  kernels_config:
    python:
      delete_cmd_postfix: ''
      delete_cmd_prefix: 'del '
      library: var_list.py
      varRefreshCmd: print(var_dic_list())
    r:
      delete_cmd_postfix: ') '
      delete_cmd_prefix: rm(
      library: var_list.r
      varRefreshCmd: 'cat(var_dic_list()) '
  types_to_exclude: [module, function, builtin_function_or_method, instance, _Feature]
  window_display: false
---

# Search for New Physics with Quantum Machine Learning

+++

In this exercise, we will first learn the basics of **Quantum Machine Learning** (QML), which is a typical application of quantum-classical hybrid algorithm. After that, we will explore, as an example, application of quantum machine learning to **search for new particles in particle physics experiment**. The QML technique that we learn here is called **Quantum Circuit Learning** (QCL) {cite}`quantum_circuit_learning` developed as an extension of Variational Quantum Algorithm. 

```{contents} Contents
---
local: true
---
```

$\newcommand{\ket}[1]{| #1 \rangle}$
$\newcommand{\expval}[3]{\langle #1 | #2 | #3 \rangle}$

+++

## Introduction <a id='introduction'></a>

In recent years, **Deep Learning** (DL) has gained considerable attention in the field of machine learning (ML). A generic deep learning model uses multiple hidden layers in the architecture of **neural networks** (NN)) and attempts to learn complex relationship between 
input and output. The successfully learned DL model is capable of predicting outout for unseen input data. The QML algorithm that we study in this unit is based on a variational quantum circuit that replaces neural network in a ML model. In other words, instead of adjusting trainable parameters for each neuron in neural networks, the QML model learns the input-output relation by adjusting the parameters of variational quantum circuit such as angles of rotation gates. 
A strength of quantum computer lies in the fact that it can represent exponentially large Hilbert space using finite number of qubits. If we can leverage this strength, the QML might be able to learn complex, high-dimensional correlations between data, providing a unique strength compared to conventional classical ML. 

The QML model is in general capable of representing wide range of functions using a polynomial number of quantum gates if the circuit is sufficiently deep. However, the QML in quantum-classical hybrid architecture is essentially heuristic and has no mathematical guarantee that it is superior to classical ML in terms of computational complexity (except for specific cases). In particular, in the era of *Noisy Intermediate-Scale Quantum* (NISQ) computer with hardware noise, it is not clear whether the QML has any advantage over classical ML, and therefore it is an active area of research. Since the quantum-classical hybrid QML is suitable for NISQ computer, an early QCL algorithm was implemented into IBM Quantum hardware by IBM team in March 2019, and the result was published in a paper{cite}`quantum_svm`. 

+++

## Machine Learneing and Deep Learning <a id='ml'></a>

Machine learning could be (very broadly) described as a series of processes that is provided data and returns predictions from the data. For example, imagine that we have data consisting of two variables, $\mathbf{x}$ and $\mathbf{y}$ (both are vectors made of elements $(x_i, y_i)$ with $i$ as the element index), and consider machine learning for determining the relation between the variables. Let us think of a function $f$ which takes $x_i$ as an input variable. Then, this machine learning problem corresponds to determining the function $f$ from the data so that the output of the function $\tilde{y_i}=f(x_i)$ is as close to $\tilde{y}_i\simeq y_i$ as possible. In general, this function $f$ has parameters other than the input variable $x$, denoted by $\mathbf{w}$ here. Therefore, this ML problem is to determine the function $f=f(x,\mathbf{w}^*)$ and optimized parameter $\mathbf{w}^*$ by adjusting the $\mathbf{w}$ such that $y_i\simeq\tilde{y}_i$.

One of the most popular methods to approximate the function $f$ is to use artificial neural networks that model the neural structure of the brain. The basic structure of neural networks is shown below. The circles indicate structural units (neurons), and the arrows indicate the flow of information between connected neurons. Neural networks have many different structures, but shown in the figure is the basic one, with the output from neurons in a layer becoming input to neurons in the next layer. In addition to the input layer which accepts inputs $x$ and the output layer that outputs $\tilde{y}$, the intermediate layers called hidden layers are often considered. Such neural network model is collectively referred to as deep neural networks.

```{image} figs/neural_net.png
:alt: var_circuit
:width: 500px
:align: center
```

Let us look at the mathematical model of neural networks. Denoting the $j$-th unit in the $l$-th layer as $u_j^l$, if the $u_j^l$ takes $n$ inputs $o_k^{l-1}$ ($k=1,2,\cdots n$) from the units in the preceding $(l-1)$-th layer, the output from the unit $u_j^l$ is expressed as follows by applying weights $w_k^j$ to the inputs $o_k^{l-1}$: 

$$
o_j^l=g\left(\sum_{k=1}^n o_k^{l-1}w_k^l\right)
$$

This can be shown in the figure below.

```{image} figs/neuron.png
:alt: var_circuit
:width: 350px
:align: center
```

The function $g$ is called activation function, and gives a non-linear output with respect to the input. Sigmoid function or ReLU (Rectified Linear Unit) is often used as activation function.

To determine the function $f(x,\mathbf{w}^*)$, we need to optimize the parameter $\mathbf{w}$ and this process is called learning. For this reason, another function $L(\mathbf{w})$ to quantify the difference between the output $\tilde{y}$ and the target variable $y$, generally called "loss function" or "cost function", is necessary.

$$
L(\mathbf{w}) = \frac{1}{N}\sum_{i=1}^N L(f(x_i,\mathbf{w}),y_i)
$$

Here $N$ is the number of $(x_i, y_i)$ data points. We want to determine the parameter $\mathbf{w}^*$ that minimizes the loss function $L(\mathbf{w})$, and this can be done using the method called gradient descent. 
In the gradient descent method, one attempts to calculate the partial derivative of the loss function, $\Delta_w L(\mathbf{w})$, for each parameter $w$ and update the parameter so that the loss function "decreases" as follows:

$$
w'=w-\epsilon\Delta_w L(\mathbf{w}),
$$

where $w$ and $w'$ are parameters before and after being updated, respectively. The $\epsilon\:(>0)$ is a parameter known as a learning rate, and this typically needs to be given by hand.

+++

## Quantum Circuit Learning<a id='qml'></a>

The QML algorithm based on variational quantum circuit generally takes the following steps to construct learning model, implement it into a quantum circuit and execute: ed to perform computation.

1. Prepare the training data $\{(\mathbf{x}_i, y_i)\}$. The $\mathbf{x}_i$ is the input data vector and $y_i$ is the true value of the input data (e.g, teacher label) ($i$ stands for the index of training data sample). 
2. Create a circuit $U_{\text{in}}(\mathbf{x})$ (called **feature map**) to encode the input data $\mathbf{x}$, then producing the input state $\ket{\psi_{\text{in}}(\mathbf{x}_i)} = U_{\text{in}}(\mathbf{x}_i)\ket{0}$, in which the $\mathbf{x}_i$ information is embedded. 
3. Generate the output state $\ket{\psi_{\text{out}}(\mathbf{x}i,\boldsymbol{\theta})} = U(\boldsymbol{\theta})\ket{\psi{\text{in}}(\mathbf{x}_i)}$ by applying a parametrized unitary $U(\boldsymbol{\theta})$ (**variational form**) with parameter $\boldsymbol{\theta}$. 
4. Measure some **observable** under the output state and get the measurement outcome $O$. For example, consider the expectation value of a Pauli $Z$ operator for the first qubit, $\langle Z_1\rangle = \expval{\psi_{\text{out}}}{Z_1}{\psi_{\text{out}}}$. 
5. Introduce some function $F$ and obtain $F(O)$ as the model output $y(\mathbf{x}_i,\boldsymbol{\theta})$. 
6. Define a **cost function** $L(\boldsymbol{\theta})$ to quantify the gap between the true value $y_i$ and the output $y(\mathbf{x}_i,\boldsymbol{\theta})$ and calculate it using a classical computer.
7. Update the parameter $\boldsymbol{\theta}$ so that the $L(\boldsymbol{\theta})$ gets smaller.
8. Repeat steps 3 through 7 to minimize the cost function and obtain the optimized parameter $\boldsymbol{\theta^*}$. 
9. Obtain the **prediction of the model** as $y(\mathbf{x},\boldsymbol{\theta^*})$ after training.

```{image} figs/var_circuit.png
:alt: var_circuit
:width: 700px
:align: center
```


Let us implement the quantum machine learning algorithm by following these steps. First, the required libraries are imported.

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
import pandas as pd
from IPython.display import clear_output
from sklearn.preprocessing import MinMaxScaler

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import TwoLocal, ZFeatureMap, ZZFeatureMap
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.algorithms.classifiers import VQC
#from qiskit.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit_ibm_runtime import Session, Sampler as RuntimeSampler
from qiskit_ibm_runtime.accounts import AccountNotFoundError
```

## Simple Example<a id='example'></a>

When an input $\{x_i\}$ and the output $y_i=f(x_i)$ of a known function $f$ are provided as data, we attempt to approximately obtain the function $f$ from the data. For the function $f$, let us consider $f(x)=x^3$.

### Preparation of Training Data<a id='func_data'></a>

First, let us prepare the training data. After randomly selecting `num_x_train` samples of data in the range between $x_{\text{min}}$ and $x_{\text{max}}$, a noise sampled from a normal distribution is added. The `nqubit` is the number of qubits and the `c_depth` is the number of layers in the variational form (explained later).

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
random_seed = 0
rng = np.random.default_rng(random_seed)

# Define the number of qubits, the circuit depth, the number of training samples, etc.
nqubit = 3
nlayer = 5
x_min = -1.
x_max = 1.
num_x_train = 30
num_x_validation = 20

# Define the function
func_to_learn = lambda x: x ** 3

# Training data
x_train = rng.uniform(x_min, x_max, size=num_x_train)
y_train = func_to_learn(x_train)

# Apply noise with a normal distribution to the function
mag_noise = 0.05
y_train_noise = y_train + rng.normal(0., mag_noise, size=num_x_train)

# Testing data
x_validation = rng.uniform(x_min, x_max, size=num_x_validation)
y_validation = func_to_learn(x_validation) + rng.normal(0., mag_noise, size=num_x_validation)
```

### Embedding Input Data<a id='func_state_preparation'></a>

Next, we create a circuit $U_{\text{in}}(x_i)$ to embed the input data $x_i$ into the initial state $\ket{0}^{\otimes n}$ (feature map). First, by following the reference{cite}`quantum_circuit_learning`, the circuit $U_{\text{in}}(x_i)$ is defined using rotation gates around the $Y$ axis, $R_j^Y(\theta)=e^{-i\theta Y_j/2}$, and those around the $Z$ axis, $R_j^Z(\theta)=e^{-i\theta Z_j/2}$:

$$
U_{\text{in}}(x_i) = \prod_j R_j^Z(\cos^{-1}(x^2))R_j^Y(\sin^{-1}(x))
$$

By applying the $U_{\text{in}}(x_i)$ to the standard zero state, the input data $x_i$ is encoded into a quantum state $\ket{\psi_{\text{in}}(x_i)}=U_{\text{in}}(x_i)\ket{0}^{\otimes n}$.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
u_in = QuantumCircuit(nqubit, name='U_in')
x = Parameter('x')

for iq in range(nqubit):
    # parameter.arcsin() returns arcsin(v) when the parameter is assigned a value v
    u_in.ry(x.arcsin(), iq)
    # Similarly for arccos
    u_in.rz((x * x).arccos(), iq)

u_in.bind_parameters({x: x_train[0]}).draw('mpl')
```

### Tranforming State using Variational Form<a id='func_variational_form'></a>

#### Creating Ansatz Circuit $U(\boldsymbol{\theta})$
Next, we will create the variational quantum circuit $U(\boldsymbol{\theta})$ to be optimized. This is done in three steps as follows.

1. Place 2-qubit gates to create entanglement
2. Place single-qubit rotation gates
3. Create a variational quantum circuit $U(\boldsymbol{\theta})$ by alternating single- and 2-qubit gates from 1 and 2


#### 2-Qubit Gate
We will use controlled-$Z$ gates ($CZ$) to entangle qubits, increasing the expressibility of the circuit model.

#### Rotation Gate and $U(\boldsymbol{\theta})$
Using entangling gates $U_{\text{ent}}$ of $CZ$ and single-qubit rotation gates on $j$-th qubit ($j \:(=1,2,\cdots n)$) in the $l$-th layer, 

$$
U_{\text{rot}}(\theta_j^l) = R_j^Y(\theta_{j3}^l)R_j^Z(\theta_{j2}^l)R_j^Y(\theta_{j1}^l)
$$

the $U(\boldsymbol{\theta})$ is constructed. In this exercise, we use the $U_{\text{rot}}$ first, then the combination of $U_{\text{ent}}$ and U_{\text{rot}}$ $d$ times. Therefore, the $U(\boldsymbol{\theta})$ takes the form of

$$
U\left(\{\theta_j^l\}\right) = \prod_{l=1}^d\left(\left(\prod_{j=1}^n U_{\text{rot}}(\theta_j^l)\right) \cdot U_{\text{ent}}\right)\cdot\prod_{j=1}^n U_{\text{rot}}(\theta_j^0)
$$

The $U(\boldsymbol{\theta})$ contains $3n(d+1)$ parameters, and they are all randomly initialized within the range of $[0, 2\pi]$.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
u_out = QuantumCircuit(nqubit, name='U_out')

# Parameters with 0 length
theta = ParameterVector('θ', 0)

# Fundtion to add new element to theta and return the last parameter
def new_theta():
    theta.resize(len(theta) + 1)
    return theta[-1]

for iq in range(nqubit):
    u_out.ry(new_theta(), iq)

for iq in range(nqubit):
    u_out.rz(new_theta(), iq)

for iq in range(nqubit):
    u_out.ry(new_theta(), iq)

for il in range(nlayer):
    for iq in range(nqubit):
        u_out.cz(iq, (iq + 1) % nqubit)

    for iq in range(nqubit):
        u_out.ry(new_theta(), iq)

    for iq in range(nqubit):
        u_out.rz(new_theta(), iq)

    for iq in range(nqubit):
        u_out.ry(new_theta(), iq)

print(f'{len(theta)} parameters')

theta_vals = rng.uniform(0., 2. * np.pi, size=len(theta))

u_out.bind_parameters(dict(zip(theta, theta_vals))).draw('mpl')
```

### Measurement and Model Output<a id='func_measurement'></a>

モデルの出力（予測値）として、状態$\ket{\psi_{\text{out}}(\mathbf{x},\boldsymbol{\theta})}=U(\boldsymbol{\theta})\ket{\psi_{\text{in}}(\mathbf{x})}$の元で最初の量子ビットを$Z$基底で測定した時の期待値を使うことにします。つまり$y(\mathbf{x},\boldsymbol{\theta}) = \langle Z_0(\mathbf{x},\boldsymbol{\theta}) \rangle = \expval{\psi_{\text{out}}(\mathbf{x},\boldsymbol{\theta})}{Z_0}{\psi_{\text{out}}(\mathbf{x},\boldsymbol{\theta})}$です。
As an output of the model (prediction value), we take the expectation value of Pauli $Z$ operator on the first qubit under the state $\ket{\psi_{\text{out}}(\mathbf{x},\boldsymbol{\theta})}=U(\boldsymbol{\theta})\ket{\psi_{\text{in}}(\mathbf{x})}$. 
That means y(\mathbf{x},\boldsymbol{\theta}) = \langle Z_0(\mathbf{x},\boldsymbol{\theta}) \rangle = \expval{\psi_{\text{out}}(\mathbf{x},\boldsymbol{\theta})}{Z_0}{\psi_{\text{out}}(\mathbf{x},\boldsymbol{\theta})}$.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
model = QuantumCircuit(nqubit, name='model')

model.compose(u_in, inplace=True)
model.compose(u_out, inplace=True)

bind_params = dict(zip(theta, theta_vals))
bind_params[x] = x_train[0]

model.bind_parameters(bind_params).draw('mpl')
```

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
# Use Estimator class
estimator = Estimator()

# Calculate y value from the given parameters and x value
def yvals(param_vals, x_vals=x_train):
    circuits = list()
    for x_val in x_vals:
        circuits.append(model.bind_parameters({x: x_val}))

    # Observable = IIZ (the first qubit from right is 0-th qubit)
    observable = SparsePauliOp('I' * (nqubit - 1) + 'Z')

    # shots is defined outside the function
    job = estimator.run(circuits, [observable] * len(circuits), [param_vals] * len(circuits), shots=shots)

    return np.array(job.result().values)

def objective_function(param_vals):
    return np.sum(np.square(y_train_noise - yvals(param_vals)))

def callback_function(param_vals):
    # losses is defined outside the function
    losses.append(objective_function(param_vals))

    if len(losses) % 10 == 0:
        print(f'COBYLA iteration {len(losses)}: cost={losses[-1]}')
```

The mean squared error of the model prediction $y(x_i, \theta)$ and the true values $y_i$ is used as the cost function $L$.

Let us execute the circuit and check the results.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
# Maximum number of steps in COBYLA
maxiter = 50
# Convergence threshold of COBYLA (the smaller the better approximation)
tol = 0.05
# Number of shots in the backend
shots = 1000


optimizer = COBYLA(maxiter=maxiter, tol=tol, callback=callback_function)
```

```{code-cell} ipython3
:tags: [remove-input]

# Cell for text
import os
if os.getenv('JUPYTERBOOK_BUILD') == '1':
    del objective_function
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
initial_params = rng.uniform(0., 2. * np.pi, size=len(theta))

losses = list()
min_result = optimizer.minimize(objective_function, initial_params)
```

```{code-cell} ipython3
:tags: [remove-input]

# Cell for text

if os.getenv('JUPYTERBOOK_BUILD') == '1':
    import pickle

    with open('data/qc_machine_learning_xcube.pkl', 'rb') as source:
        min_result, losses = pickle.load(source)
```

Make a plot of cost functon values.

```{code-cell} ipython3
plt.plot(losses)
```

+++ {"jupyter": {"outputs_hidden": false}, "pycharm": {"name": "#%%\n"}}

Check the output of the trained model with optimized parameters and the inputs taken uniformly between x_min and x_max. 

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
x_list = np.linspace(x_min, x_max, 100)

y_pred = yvals(min_result.x, x_vals=x_list)

# Results
plt.plot(x_train, y_train_noise, "o", label='Training Data (w/ Noise)')
plt.plot(x_list, func_to_learn(x_list), label='Original Function')
plt.plot(x_list, np.array(y_pred), label='Predicted Function')
plt.legend();
```

Please check the results. You will see that the original function $f(x)=x^3$ is approximately derived from the training data with noise.

In order to converge optimization steps quickly, the maximum number of calls (`maxiter`) is set to 50 and the tolerance for the accuracy (`tol`) is set to 0.05 for COBYLA optimizer. Check how the accuracy if the `maxiter` is raised or `tol` is lowered. Note however that if the `maxiter` is taken too large and the `tol` too small, the calculation will take a very long time.

+++

## 素粒子現象の探索への応用<a id='susy'></a>

次の実習課題では、素粒子物理の基本理論（**標準模型**と呼ばれる）を超える新しい理論の枠組みとして知られている「**超対称性理論**」（*Supersymmetry*、略してSUSY）で存在が予言されている新粒子の探索を考えてみます。
In the next assignment, we will be considering the search for new particles whose existence is predicted by **Supersymmetry** (SUSY), a new theoretical framework which goes beyond the fundamental theory of particle physics (**Standard Model**).

左下の図は、グルーオン$g$が相互作用してヒッグス粒子$h$を作り、それが2つのSUSY粒子$\chi^+\chi^-$に崩壊する過程を示しています。$\chi^+$粒子はさらに崩壊し、最終的には$\ell^+\ell^-\nu\nu\chi^0\chi^0$という終状態に落ち着くとします。右下の図は標準模型で存在が知られている過程を表していて、クォーク$q$と反クォーク$\bar{q}$が相互作用して$W$ボソン対を作り、それが$\ell^+\ell^-\nu\nu$に崩壊しています。
The figure at bottom left shows the process in which a gluon, $g$, creates a Higgs boson, $h$, through mutual interaction, which then breaks down into two SUSY particles, $\chi^+\chi^-$. The $\chi^+$ particle then breaks down further, ultimately producing a stable final state consisting of $\ell^+\ell^-\nu\nu\chi^0\chi^0$. The figure at bottom right shows a process known to exist under the Standard Model, in which a quark, $q$, and another quark, $\bar{q}$̅, mutually interact, creating $W$ bosons which break down into $\ell^+\ell^-\nu\nu$.

```{image} figs/susy_bg.png
:alt: susy_bg
:width: 700px
:align: center
```
(Figures courtesy of reference material{cite}`dl_susy`)

左と右の過程を比べると、終状態の違いは$\chi^0\chi^0$が存在しているかどうかだけですね。この$\chi^0$という粒子は検出器と相互作用しないと考えられているので、この二つの過程の違いは（大雑把に言うと）実際の検出器では観測できないエネルギーの大きさにしかなく、探索することが難しい問題と考えることができます。以上のような状況で、この二つの物理過程を量子回路学習で分類できるかどうかを試みます。
If you compare the processes in the left and right figures, you can see that the only difference in the end state is the presence or absence of the $\chi^0\chi^0$. This $\chi^0$ particle is not believed to interact with the detector, so (broadly speaking), the differences between these processes lie only in the amount of energy that cannot be observed by the detector, which makes it very difficult to find these $\chi^0$ particles. Let's see, based on these conditions, if we can distinguish between these physical processes using quantum machine learning.

+++

### 学習データの準備<a id='susy_data'></a>

学習に用いるデータは、カリフォルニア大学アーバイン校（UC Irvine）の研究グループが提供する[機械学習レポジトリ](https://archive.ics.uci.edu/ml/index.php)の中の[SUSYデータセット](https://archive.ics.uci.edu/ml/datasets/SUSY)です。このデータセットの詳細は文献{cite}`dl_susy`に委ねますが、ある特定のSUSY粒子生成反応と、それに良く似た特徴を持つ背景事象を検出器で観測した時に予想される信号（運動学的変数）をシミュレートしたデータが含まれています。
The data used in this section will be the [SUSY data set](https://archive.ics.uci.edu/ml/datasets/SUSY) contained in the [machine learning repository](https://archive.ics.uci.edu/ml/index.php) supplied by a research group in the University of California, Irvine (UC Irvine). We'll leave the details of this data set up to reference material{cite}`dl_susy`, but we will point out that the data includes simulated data of signals (kinematic variables) predicted when a detector observes specific SUSY particle generation reactions and background phenomena with similar characteristics.

探索に役立つ運動学的変数をどう選ぶかはそれ自体が大事な研究トピックですが、ここでは簡単のため、前もって役立つことを経験上知っている変数を使います。以下で、学習に使う運動学的変数を選んで、その変数を指定したサンプルを訓練用とテスト用に準備します。
Selecting the kinematic variables that assist in the search for particles is, itself, an important research topic, but here, for simplicity's sake, we will assume that this is a given which we have determined is useful through prior experience. Below, we will select the kinematic variables to use for learning and prepare samples which specify those variables, for use in training and testing.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
# Read the variables from the file
df = pd.read_csv("data/SUSY_1K.csv",
                 names=('isSignal', 'lep1_pt', 'lep1_eta', 'lep1_phi', 'lep2_pt', 'lep2_eta',
                        'lep2_phi', 'miss_ene', 'miss_phi', 'MET_rel', 'axial_MET', 'M_R', 'M_TR_2',
                        'R', 'MT2', 'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos_theta_r1'))

# Number of variables to use in learning
feature_dim = 3  # dimension of each data point

# Set of variables when using 3, 5, or 7 variables
if feature_dim == 3:
    selected_features = ['lep1_pt', 'lep2_pt', 'miss_ene']
elif feature_dim == 5:
    selected_features = ['lep1_pt', 'lep2_pt', 'miss_ene', 'M_TR_2', 'M_Delta_R']
elif feature_dim == 7:
    selected_features = ['lep1_pt', 'lep1_eta', 'lep2_pt', 'lep2_eta', 'miss_ene', 'M_TR_2', 'M_Delta_R']

# Phenomenon used in learning: "training" samples are used for training, "testing" samples are used for testing
train_size = 20
test_size = 20

df_sig = df.loc[df.isSignal==1, selected_features]
df_bkg = df.loc[df.isSignal==0, selected_features]

# Generate samples
df_sig_train = df_sig.values[:train_size]
df_bkg_train = df_bkg.values[:train_size]
df_sig_test = df_sig.values[train_size:train_size + test_size]
df_bkg_test = df_bkg.values[train_size:train_size + test_size]
# 最初のtrain_size事象がSUSY粒子を含む信号事象、残りのtrain_size事象がSUSY粒子を含まない背景事象
train_data = np.concatenate([df_sig_train, df_bkg_train])
# 最初のtest_size事象がSUSY粒子を含む信号事象、残りのtest_size事象がSUSY粒子を含まない背景事象
test_data = np.concatenate([df_sig_test, df_bkg_test])

# one-hotベクトル（信号事象では第1次元の第0要素が1、背景事象では第1次元の第1要素が1）
train_label_one_hot = np.zeros((train_size * 2, 2))
train_label_one_hot[:train_size, 0] = 1
train_label_one_hot[train_size:, 1] = 1

test_label_one_hot = np.zeros((test_size * 2, 2))
test_label_one_hot[:test_size, 0] = 1
test_label_one_hot[test_size:, 1] = 1

#datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
#datapoints_tr, class_to_label_tr = split_dataset_to_data_and_labels(training_input)

mms = MinMaxScaler((-1, 1))
norm_train_data = mms.fit_transform(train_data)
norm_test_data = mms.transform(test_data)
```

+++ {"pycharm": {"name": "#%% md\n"}}

### 量子状態の生成<a id='susy_state_preparation'></a>

次は特徴量マップ$U_{\text{in}}(\mathbf{x}_i)$の作成ですが、ここでは参考文献{cite}`quantum_svm`に従い、
Next, we create the feature map, $U_{\text{in}}(\mathbf{x}_i)$. Here, as indicated in reference material{cite}`quantum_svm`, we set it as follows.

$$
U_{\phi_{\{k\}}}(\mathbf{x}_i)=\exp\left(i\phi_{\{k\}}(\mathbf{x}_i)Z_k\right)
$$

あるいは

$$
U_{\phi_{\{l,m\}}}(\mathbf{x}_i)=\exp\left(i\phi_{\{l,m\}}(\mathbf{x}_i)Z_lZ_m\right)
$$

とします（$k$、$l$、$m$は入力値$\mathbf{x}_i$のベクトル要素の添字）。この特徴量マップは、パウリZ演算子の形から前者をZ特徴量マップ、後者をZZ特徴量マップと呼ぶことがあります。ここで$\phi_{\{k\}}(\mathbf{x}_i)=x_i^{(k)}$（$x_i^{(k)}$は$\mathbf{x}_i$の$k$番目要素）、$\phi_{\{l,m\}}(\mathbf{x}_i)=(\pi-x_i^{(l)})(\pi-x_i^{(m)})$（$x_i^{(l,m)}$は$\mathbf{x}_i$の$l,m$番目要素）と決めて、入力値$\mathbf{x}_i$を量子ビットに埋め込みます。Z特徴量マップは入力データの各要素を直接量子ビットに埋め込みます（つまり$\phi_{\{k\}}(\mathbf{x}_i)$は1入力に対して1量子ビットを使う）。ZZ特徴量マップは実際はZ特徴量マップを含む形で使うことが多いため、$\phi_{\{l,m\}}(\mathbf{x}_i)$の場合も$\phi_{\{k\}}(\mathbf{x}_i)$と同数の量子ビットに対して$(l,m)$を循環的に指定して埋め込むことになります。ZZ特徴量マップでは量子ビット間にエンタングルメントを作っているため、古典計算では難しい特徴量空間へのマッピングになっていると考えられます。

($k$ indicates the vector element of input value $\mathbf{x}_i$) Here, $\phi_{\{k\}}(\mathbf{x}_i)=x_i^{(k)}$（$x_i^{(k)}$ ($x_i^k$ is the $k$-th element of $x_i$), and input value $\mathbf{x}_i$ is embedded in the $k$-th quantum bit. 

この$U_\phi(\mathbf{x}_i)$にアダマール演算子を組み合わせることで、全体として、Z特徴量マップは

This U_\phi(\mathbf{x}_i)$ can be combined with a Hadamard operator, producing the following.

$$
U_{\text{in}}(\mathbf{x}_i) = U_{\phi}(\mathbf{x}_i)H^{\otimes n},\:\:U_{\phi}(\mathbf{x}_i) = \exp\left(i\sum_{k=1}^nx_i^{(k)}Z_k\right)
$$

ZZ特徴量マップは

$$
U_{\text{in}}(\mathbf{x}_i) = U_{\phi}(\mathbf{x}_i)H^{\otimes n},\:\:U_{\phi}(\mathbf{x}_i) = \exp\left(i\sum_{k=1}^n(\pi-x_i^{(k)})(\pi-x_i^{(k\%n+1)})Z_kZ_{k\%n+1}\right)\exp\left(i\sum_{k=1}^nx_i^{(k)}Z_k\right)
$$

となります。$U_{\phi}(\mathbf{x}_i)H^{\otimes n}$を複数回繰り返すことでより複雑な特徴量マップを作ることができるのは、上の例の場合と同じです。

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
#feature_map = ZFeatureMap(feature_dimension=feature_dim, reps=1)
feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=1, entanglement='circular')
feature_map.decompose().draw('mpl')
```

### 変分フォームを使った状態変換<a id='susy_variational_form'></a>

変分量子回路$U(\boldsymbol{\theta})$は上の初歩的な例で用いた回路とほぼ同じですが、回転ゲートとして
Variational quantum circuit $U(\boldsymbol{\theta})$ is roughly the same as the circuit used in the basic example above, but it uses the following rotation gates.

$$
U_{\text{rot}}(\theta_j^l) = R_j^Z(\theta_{j2}^l)R_j^Y(\theta_{j1}^l)
$$

を使います。上の例では$U(\boldsymbol{\theta})$を自分で組み立てましたが、Qiskitにはこの$U(\boldsymbol{\theta})$を実装するAPIがすでに準備されているので、ここではそれを使います。
In the example above, we assembled $U(\boldsymbol{\theta})$を ourselves, but Qiskit includes as one of its standard API an API for implementing $U(\boldsymbol{\theta})$を, so we will use that here.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
ansatz = TwoLocal(num_qubits=feature_dim, rotation_blocks=['ry', 'rz'], entanglement_blocks='cz', entanglement='circular', reps=3)
#ansatz = TwoLocal(num_qubits=feature_dim, rotation_blocks=['ry'], entanglement_blocks='cz', entanglement='circular', reps=3)
ansatz.decompose().draw('mpl')
```

### 測定とモデル出力<a id='susy_measurement'></a>

測定やパラメータの最適化、コスト関数の定義も初歩的な例で用いたものとほぼ同じです。QiskitのVQCというクラスを用いるので、プログラムはかなり簡略化されています。
The measurement and parameter optimization and the definition of the cost function are largely the same as those in the basic example above. We will use Qiskit's API, so the program itself will be greatly simplified.

VQCクラスでは、特徴量マップと変分フォームを結合させ、入力特徴量とパラメータ値を代入し、測定を行い、目的関数を計算し、パラメータのアップデートを行う、という一連の操作を内部で行なってしまいます。測定を行うのに使用するのはSamplerというクラスで、これはEstimatorと同様の働きをしますが、後者が観測量の期待値を計算するのに対し、前者はすべての量子ビットをZ基底で測定した結果のビット列の確率分布を出力します。VQCではこの分布を利用して分類を行います。今回は2クラス分類なので、入力の各事象に対して、q0の測定値が0である確率が、それが信号事象である確率（モデルの予測）に対応します。

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
tags: [remove-output]
---
# 上のEstimatorと同じく、バックエンドを使わずシミュレーションを簡略化したSampler
sampler = Sampler()

# 実機で実行する場合
# instance = 'ibm-q/open/main'

# try:
#     service = QiskitRuntimeService(channel='ibm_quantum', instance=instance)
# except AccountNotFoundError:
#     service = QiskitRuntimeService(channel='ibm_quantum', token='__paste_your_token_here__',
#                                    instance=instance)

# backend_name = 'ibm_washington'
# session = Session(service=service, backend=backend_name)

# sampler = RuntimeSampler(session=session)

maxiter = 300

optimizer = COBYLA(maxiter=maxiter, disp=True)

objective_func_vals = []
# Draw the value of objective function every time when the fit() method is called
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    #print('obj_func_eval =',obj_func_eval)

    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(objective_func_vals)
    plt.show()

vqc = VQC(num_qubits=feature_dim,
          feature_map=feature_map,
          ansatz=ansatz,
          loss="cross_entropy",
          optimizer=optimizer,
          callback=callback_graph,
          sampler=sampler)
```

```{code-cell} ipython3
:tags: [remove-input]

# テキスト作成用のセル - わざと次のセルでエラーを起こさせる
if os.getenv('JUPYTERBOOK_BUILD') == '1':
    del objective_func_vals
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
vqc.fit(norm_train_data, train_label_one_hot)

# 実機で実行している（RuntimeSamplerを使っている）場合
# session.close()
```

```{code-cell} ipython3
:tags: [remove-input]

# テキスト作成用のセルなので無視してよい
with open('data/vqc_machine_learning_susycost.pkl', 'rb') as source:
    fig = pickle.load(source)

with open('data/vqc_machine_learning_susyresult.pkl', 'rb') as source:
    vqc._fit_result = pickle.load(source)

print('''   Return from subroutine COBYLA because the MAXFUN limit has been reached.

   NFVALS =  300   F = 7.527796E-01    MAXCV = 0.000000E+00
   X = 2.769828E+00   2.690659E+00   2.209438E+00  -6.544334E-01   1.791658E+00
       5.891710E-01  -4.081896E-02   5.725357E-01   1.662590E+00  -5.402978E-01
       2.029238E+00   1.562530E-01   1.229339E+00   1.596802E+00   2.692169E-01
      -5.828812E-01   1.891610E+00   7.527701E-01  -6.604400E-01   1.587293E+00
       1.194641E+00   1.505717E-01''')
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
train_score = vqc.score(norm_train_data, train_label_one_hot)
test_score = vqc.score(norm_test_data, test_label_one_hot)

print(f'--- Classification Train score: {train_score} ---')
print(f'--- Classification Test score:  {test_score} ---')
```

+++ {"pycharm": {"name": "#%% md\n"}}

この結果を見てどう思うでしょうか？機械学習を知っている方であれば、この結果はあまり良いようには見えませんね。。訓練用のデータでは学習ができている、つまり信号とバックグラウンドの選別ができていますが、テスト用のサンプルでは選別性能が悪くなっています。これは「過学習」を起こしている場合に見られる典型的な症状で、訓練データのサイズに対して学習パラメータの数が多すぎるときによく起こります。
What do you think of these results? If you're familiar with machine learning, you would probably say that, even speaking charitably, there's no way that these results could be called "good." The training data has been learned -- in other words, the signal selection efficiency (True Positive Rate) is higher than the background selection efficiency (False Positive Rate). However, the ROC curve for the testing data sample is almost a diagonal line and shows no signs of learning whatsoever. This is typical of "overlearning," and often happens when the number of learning parameters is too large for the amount of training data.

試しに、データサンプルの事象数を50や100に増やして実行し、結果を調べてみてください（処理時間は事象数に比例して長くなるので要注意）。
Try increasing the number of data samples to 50 or 100 and compare the results (note that the processing time will increase proportionally with the number of data samples).