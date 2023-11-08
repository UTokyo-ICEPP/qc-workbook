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

+++ {"pycharm": {"name": "#%% md\n"}}

# Performing Database Search 

+++

In this unit, we'll introduce **Grover's algorithm**{cite}`grover_search,nielsen_chuang_search` and consider the problem of searching for an answer in an unstructured database using this algorithm. After that, We will implement Grover's algorithm using Qiskit. 

```{contents} 目次
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

The structure of the circuit used to implement Grover's algorithm is shown below. Starting with the $n$-qubit initial state $\ket{0}$, an equal superposition state is created first by applying Hadamard gates to every qubits. Then, the operator denoted as $G$ is applied repeatedly.

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

First, an equal superposition state is produced by applying Hadamard gates to initial state $\ket{0}^{\otimes n}$ of the $n$-qubit circuit.

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
### 振幅増幅を可視化する
グローバーアルゴリズムで振幅がどのように増幅されるのか、実際目で見てみることにします。
Let's see how the Grover algorithm amplifies the amplitude with our own eyes.

まず最初のHadamard変換で、全ての計算基底が等しい振幅を持つ重ね合わせ状態を生成します（下図の1）。横軸は$N$個の計算基底、縦軸は各基底の振幅の大きさを表しており、全ての基底が$\frac{1}{\sqrt{N}}$の大きさの振幅を持っています（振幅の平均を赤破線で表示）。
First, the initial Hadamard conversion creates a superposition of computational basis states, all with equal amplitudes (no. 1 in the figure below). The horizontal axis shows $N$ computational basis states and the vertical axis shows the size of the amplitude for each basis state. All of the basis states have an amplitude of $\frac{1}{\sqrt{N}}$ (the average of the amplitudes is shown with a dotted red line). 

次にオラクル$U_w$を適用すると、$\ket{w}$の位相が反転し、振幅が$-\frac{1}{\sqrt{N}}$になります（下図の2）。この状態での振幅の平均は$\frac{1}{\sqrt{N}}(1-\frac2N)$になり、(1)の状態より低くなります。
Next, if we apply oracle $U_w$, the phase of $\ket{w}$ is reversed and the amplitude becomes $-\frac{1}{\sqrt{N}}$ (no. 2 in the figure below). In this state, the average amplitude is $\frac{1}{\sqrt{N}}(1-\frac2N)$, which is lower than it was in state (1). 

最後にDiffuserを適用すると、平均に関して振幅を反転します（下図の3）。その結果、$\ket{w}$の振幅が増幅され、$\ket{w}$以外の基底の振幅は減少します。1回のグローバーの反復操作で、$\ket{w}$の振幅が約3倍程度増幅することも図から見てとれます。この操作を繰り返し実行すれば$\ket{w}$の振幅がさらに増幅されるため、正しい答えを得る確率が増加していくだろうということも予想できますね。
Last, when we apply the diffuser, the amplitude is reversed with respect to the average (no. 3 in the figure below). This amplifies the amplitude of $\ket{w}$ and decreases the amplitude of basis states other than $\ket{w}$. As the figure shows, a single Grover iteration amplified the amplitude of $\ket{w}$ roughly three-fold. Repeating this process will amplify the amplitude of $\ket{w}$ even further, so we can expect a greater likelihood of the correct answer being produced. 

```{image} figs/grover_amp.png
:alt: grover_amp
:width: 800px
:align: center
```

+++

(grover_multidata)=
### 複数データの検索
今までは検索するデータが一つだけの場合を考えてきましたが、このセクションの最後に複数のデータを検索する場合を考察してみましょう。例えば、$N=2^n$個のデータから$M$個のデータ$\{w_i\}\;(i=0,1,\cdots,M-1)$を探すケースです。これまでと同様に、求める状態$\ket{w}$とそれに直行する状態$\ket{w^{\perp}}$
So far, we have only thought about searching for a single item of data. At the end of this section, we will consider finding several items of data. For example, this would include finding $M$ items of data $\{w_i\}\;(i=0,1,\cdots,M-1)$ from $N=2^n$ items of data. Just as before, we will discuss this in terms of a plane formed by the state we seek to discover, $\ket{w}$, and an orthogonal state, $\ket{w^{\perp}}$. 

$$
\begin{aligned}
&\ket{w}:=\frac{1}{\sqrt{M}}\sum_{i=0}^{M-1}\ket{w_i}\\
&\ket{w^{\perp}}:=\frac{1}{\sqrt{N-M}}\sum_{x\notin\{w_0,\cdots,w_{M-1}\}}\ket{x}
\end{aligned}
$$

が張る2次元平面の上で、同様の議論を進めることができます。$\ket{s}$はこの平面上で
$\ket{s}$ can be expressed on this plane by the following. 

$$
\begin{aligned}
\ket{s}&=\sqrt{\frac{N-M}{N}}\ket{w^{\perp}}+\sqrt{\frac{M}{N}}\ket{w}\\
&=: \cos\frac\theta2\ket{w^{\perp}}+\sin\frac\theta2\ket{w}\\
\end{aligned}
$$

と表現でき、$\ket{w}$の振幅$\sqrt{\frac{M}{N}}$を$\sin\frac\theta2$と定義すると、角度$\theta$は$\theta=2\arcsin\sqrt{\frac{M}{N}}$になります。答えが一つのケースと比べて、角度は$\sqrt{M}$倍大きく、1回のグローバーの反復操作でより大きく回転することになります。その結果、より少ない$r\approx\frac\pi4\sqrt{\frac{N}{M}}$回の回転操作で答えに到達することが可能になることが分かります。
If we define the amplitude $\sqrt{\frac{M}{N}}$ of $\ket{w}$ as $\sin\frac\theta2$, then angle $\theta$ is $\theta=2\arcsin\sqrt{\frac{M}{N}}$. Compared to the case in which there was only one answer, the angle is $\sqrt{M}$ times larger, and the rotation of a single Grover iteration is larger. As a result, as you can see, we can arrive at the answer with a shorter rotation process, performing $r\approx\frac\pi4\sqrt{\frac{N}{M}}$ rotations. 

+++ {"pycharm": {"name": "#%% md\n"}}

(imp)=
## アルゴリズムの実装（$N=2^6$の場合）
ではここから、実際にグローバーアルゴリズムを実装してデータベースの検索問題に取り掛かってみましょう。
Now, let's actually implement Grover's algorithm to solve a database search problem. 

ここで考える問題は、$N=2^6$個の要素を持つリスト（$=[0,1,2,\cdots,63]$）から、一つの答え"45"を見つけるグローバーアルゴリズムの実装です（もちろんこの数はなんでも良いので、後で自由に変更して遊んでみてください）。つまり6量子ビットの量子回路を使って、$\ket{45}=\ket{101101}$を探す問題です。
In this problem, we're going to use Grover's algorithm to find the number "45" in a list containing $N=2^6$ elements ($=[0,1,2,\cdots,63]$) (of course, we could look for any number, so feel free later to try it out with other numbers). In other words, we will use a 6-quantum bit quantum circuit to find $\ket{45}=\ket{101101}$. 

+++

(imp_qiskit)=
### Qiskitでの実装

まず必要な環境をセットアップします。
Always set up the environment first. 

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

# Import Qiskit-related packages
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from qiskit_aer import AerSimulator
from qiskit_ibm_provider import IBMProvider, least_busy
from qiskit_ibm_provider.accounts import AccountNotFoundError

# ワークブック独自のモジュール
from qc_workbook.utils import operational_backend
```

6量子ビットの回路`grover_circuit`を準備します。
Let's prepare a 6-quantum bit `grover_circuit` circuit. 

グローバー反復を一回実行する量子回路は以下のような構成になりますが、赤枠で囲んだ部分（オラクルとDiffuserの中の$2\ket{0}\bra{0}-I$の部分）を実装する量子回路を書いてください。
Below is a quantum circuit that performs a single Grover iteration. Draw the quantum circuits used in the areas outlined in red (the oracle and the $2\ket{0}\bra{0}-I$ portion of the diffuser). 

```{image} figs/grover_6bits_45.png
:alt: grover_6bits_45
:width: 600px
:align: center
```

一様な重ね合わせ状態$\ket{s}$を生成した後に、オラクルを実装します。
Implement the oracle after a uniform superposition state $\ket{s}$ is generated. 

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

**解答**

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

次に、Diffuser用の回路を実装します。
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

**解答**

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
### シミュレータでの実験

回路の実装ができたら、シミュレータで実行して結果をプロットしてみます。結果が分かりやすくなるように、測定したビット列を整数にしてからプロットするようにしてみます。
Once you have implemented the circuit, run the simulator and plot the results. To make the results easy to understand, convert the measured bit sequences into integers before plotting them. 

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

正しく回路が実装できていれば、$\ket{101101}=\ket{45}$の状態を高い確率で測定できる様子を見ることができるでしょう。
If you implement the circuit correctly, you will see that you can measure the state $\ket{101101}=\ket{45}$ with a high probability of success. 

しかし、上での議論からも分かるように、$N=2^6$の探索では、一回のグローバー反復では正しくない答えも無視できない確率で現れてきます。グローバーの反復を複数回繰り返すことで、正しい答えがより高い確率で得られることをこの後見ていきます。
However, as you saw in the discussion above, in searches of $N=2^6$ sequences, Grover iterations also produce incorrect answers with a probability that cannot be ignored. In the assignment, we will look at repeating Grover iterations multiple times to produce the correct answer with a greater probability of success. 

+++

(imp_qc)=
### 量子コンピュータでの実験

反復を繰り返す前に、まずは一回のグローバー反復を量子コンピュータで実行してみましょう。

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
# Implementation on a quantum computer
instance = 'ibm-q/open/main'

try:
    provider = IBMProvider(instance=instance)
except IBMQAccountCredentialsNotFound:
    provider = IBMProvider(token='__paste_your_token_here__', instance=instance)

backend_list = provider.backends(filters=operational_backend(min_qubits=6))
backend = least_busy(backend_list)
print("least busy backend: ", backend)
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
job_monitor(job, interval=2)
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

シミュレータに比べると結果は非常に悪いですね。。。残念ながら、今の量子コンピュータをそのまま使うとこういう結果になってしまいます。しかし、{ref}`エラー緩和 <measurement_error_mitigation>`等のテクニックを使うことである程度改善することはできます。
As you can see, the results are far worse than those produced by the simulator. Unfortunately, these are the results that are produced when we run the circuit as-is on a modern quantum computer. However, we can improve these results somewhat by using techniques such as {ref}`error mitigation<measurement_error_mitigation>`.  

+++ {"pycharm": {"name": "#%% md\n"}}

(imp_simulator_amp)=
### 振幅増幅を確認する

では次に、グローバーのアルゴリズムを繰り返し使うことで、振幅が増幅していく様子をシミュレータを使って見てみましょう。

例として、上で作ったグローバー反復を3回実行する量子回路を作って実行してみます。

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
# 繰り返しの回数
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

正しい答え$\ket{45}$をより高い確率で測定できていることが分かりますね。

では次に、実装した回路を繰り返し実行して、求める解を観測した回数と反復した回数との相関関係を図にしてみます。

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
x = []
y = []

# 例えば10回繰り返す
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

この図から、グローバー反復を5~6回程度繰り返すことで、正しい答えを最も高い確率で測定できることが分かりますね。計算で求めた検索に必要な反復回数と一致しているかどうか、確認してみてください。

+++ {"pycharm": {"name": "#%% md\n"}}

問題：解が一つの場合で、探索リストのサイズを$N=2^4$から$N=2^{10}$まで変えた時に、測定で求めた最適な反復回数が$N$とどういう関係になっているのか調べてください。

+++ {"pycharm": {"name": "#%% md\n"}}

(imp_simulator_multi)=
### 複数解の探索の場合

では次に、複数の解を探索する問題に進んでみましょう。2つの整数$x_1$と$x_2$を見つける問題へ量子回路を拡張して、求める解を観測した回数と反復した回数との相関関係を図にしてみます。

例えば、$x_1=45$と$x_2=26$の場合は

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

複数解の場合、確率が最大になる反復回数が単一解の場合より減っていますね。予想と合っているでしょうか？
