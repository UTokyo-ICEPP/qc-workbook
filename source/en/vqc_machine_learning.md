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

# 量子機械学習を使った新しい素粒子現象の探索

+++

この実習では、**量子・古典ハイブリッドアルゴリズム**の応用である**量子機械学習**の基本的な実装を学んだのち、その活用例として、**素粒子実験での新粒子探索**への応用を考えます。ここで学ぶ量子機械学習の手法は、変分量子アルゴリズムの応用として提案された、**量子回路学習**と呼ばれる学習手法{cite}`quantum_circuit_learning`です。
In this exercise, you will first learn the basics of implementing **quantum machine learning**, which applies hybrid quantum-classical algorithms. Then, as an application example, you will explore applying quantum machine learning to the **search for new particles in elementary particle experiments**. The quantum machine learning methods you will learn here are learning methods which use variational quantum circuits, proposed from the perspective of using quantum computers to improve the capabilities of classical machine learning{cite}`quantum_circuit_learning`.

```{contents} 目次
---
local: true
---
```

$\newcommand{\ket}[1]{| #1 \rangle}$
$\newcommand{\expval}[3]{\langle #1 | #2 | #3 \rangle}$

+++

## はじめに <a id='introduction'></a>

近年、機械学習の分野において**深層学習**（**ディープラーニング**）が注目を浴びています。ディープラーニングは**ニューラルネットワーク**の隠れ層を多層にすることで、入力と出力の間の複雑な関係を学習することができます。その学習結果を使って、新しい入力データに対して出力を予測することが可能になります。ここで学習する量子機械学習アルゴリズムは、このニューラルネットワークの部分を変分量子回路に置き換えたものです。つまり、ニューラルネットワークでの各ニューロン層への重みを調節する代わりに、変分量子回路のパラメータ（例えば回転ゲートの回転角）を調整することで入力と出力の関係を学習しようという試みです。
量子力学の重ね合わせの原理から、**指数関数的に増える多数の計算基底**を使って状態を表現できることが量子コンピュータの強みです。この強みを生かすことで、データ間の複雑な相関を学習できる可能性が生まれます。そこに量子機械学習の最も大きな強みがあると考えられています。
In recent years, within the field of machine learning, particular attention has been turned to **deep learning**. Deep learning uses multiple hidden layers in **neural networks** to learn complex relationships between input and output. The results of this learning can be applied to new input data to predict output. The quantum machine learning algorithm you will study in this unit replaces the neural network portion with a variational quantum circuit. In other words, instead of adjusting the weighting of each neuron layer in the neural network, you will attempt to have the system learn the relationships between input and output by adjusting the parameters of the variational quantum circuit (such as the rotation angle of rotation gates). One of the strengths of quantum computers is that, due to the principles of quantum mechanical superposition, states can be expressed using exponentially increasing numbers of computational basis states. We can leverage this strength to learn complex mutual relationships between items of data. This is considered one of the greatest strengths of quantum machine learning.

多項式で与えられる数の量子ゲートを使って、指数関数的に増える関数を表現できる可能性があるところに量子機械学習の強みがありますが、誤り訂正機能を持たない中規模の量子コンピュータ (*Noisy Intermediate-Scale Quantum*デバイス, 略してNISQ）で、古典計算を上回る性能を発揮できるか確証はありません。しかしNISQデバイスでの動作に適したアルゴリズムであるため、2019年3月にはIBMの実験チームによる実機での実装がすでに行われ、結果も論文{cite}`quantum_svm`として出版されています。
We can express an exponential growing number of functions using a polynomially-determined number of quantum gates. This is one of the strengths of quantum machine learning. Nonetheless, there is no guarantee that medium-sized quantum computers without error correction functions, called *Noisy Intermediate-Scale Quantum* devices (NISQ devices), can exceed the performance of classical computers. However, an experimental team in IBM already implemented quantum machine learning on an actual QC in March 2019, because the algorithms used were more appropriate for a NISQ device. The results of this project have been published in an academic paper{cite}`quantum_svm`.

+++

## 機械学習と深層学習 <a id='ml'></a>

機械学習を一言で（大雑把に）説明すると、与えられたデータを元に、ある予測を返すような機械を実現する工程だと言えます。例えば、2種類の変数$\mathbf{x}$と$\mathbf{y}$からなるデータ（$(x_i, y_i)$を要素とするベクトル、$i$は要素の添字）があったとして、その変数間の関係を求める問題として機械学習を考えてみましょう。つまり、変数$x_i$を引数とする関数$f$を考え、その出力$\tilde{y_i}=f(x_i)$が$\tilde{y}_i\simeq y_i$となるような関数$f$をデータから近似的に求めることに対応します。
一般的に、この関数$f$は変数$x$以外のパラメータを持っているでしょう。なので、そのパラメータ$\mathbf{w}$をうまく調整して、$y_i\simeq\tilde{y}_i$となる関数$f=f(x,\mathbf{w}^*)$とパラメータ$\mathbf{w}^*$を求めることが機械学習の鍵になります。

Machine learning could be (very broadly) described as a series of processes that functions like a machine which returns predictions when given data. For example, imagine that you have data consisting of two variables, $x$ and $y$ (a vector made up of elements $(x_i, y_i)$ with $i$ as the subscript index). Let's look at machine learning for determining the relationship between these variables. Let us think of a function $f$ which uses variable $x_i$ as an argument. This function $f$ is approximated from the data such that output $\tilde{y_i}=f(x_i)$ is $\tilde{y}_i\simeq y_i$. Generally speaking, this function $f$ will have parameters other than variable $x$. For example, parameter $w$. The key, then, to machine learning is to properly adjust parameter $w$ and determine a parameter $\mathbf{w}^*$ and a function $f=f(x,\mathbf{w}^*)$ such that $y_i\simeq\tilde{y}_i$.

関数$f$を近似する方法の一つとして、現在主流になっているのが脳のニューロン構造を模式化したニューラルネットワークです。下図に示しているのは、ニューラルネットの基本的な構造です。丸で示しているのが構成ユニット（ニューロン）で、ニューロンを繋ぐ情報の流れを矢印で表しています。ニューラルネットには様々な構造が考えられますが、基本になるのは図に示したような層構造で、前層にあるニューロンの出力が次の層にあるニューロンへの入力になります。入力データ$x$を受ける入力層と出力$\tilde{y}$を出す出力層に加え、中間に複数の「隠れ層」を持つものを総称して深層ニューラルネットワークと呼びます。
The most common method currently being used to approximate $f$ is to use a neural network that simulates the neural structure of the brain. The figure below shows the basic structure of a neural network. The circles indicate structural units (neurons), and the arrows indicate the flow of information between connected neurons. Neural networks can have many different structures, but the basic structure is as shown below, with the output from neurons in one layer becoming the input for neurons in the next layer. In addition to the input layer, which accepts input data $x$, and the output layer, which outputs $\tilde{y}$, there are also so-called "hidden layers." Together, this structure is collectively referred to as a deep neural network.

```{image} figs/neural_net.png
:alt: var_circuit
:width: 500px
:align: center
```

では、もう少し数学的なモデルを見てみましょう。$l$層目にある$j$番目のユニット$u_j^l$に対して、前層（$l-1$番目）から$n$個の入力$o_k^{l-1}$ ($k=1,2,\cdots n$) がある場合、入力$o_k^{l-1}$への重みパラメータ$w_k^l$を使って
Let's look at the mathematical model in a bit more depth. Let us consider a unit, unit $u_j^l$, where $l$ is the layer number and $j$ is the number of the unit. If, in the preceding layer (the $l-1$ layer), there are $n$ inputs of $o_k^{l-1}$ ($k=1, 2, \cdots, n$), then let us consider output $o_j^l$, applying weighting parameter $w_k^l$ to input $o_k^{l-1}$, such that the following is true.

$$
o_j^l=g\left(\sum_{k=1}^n o_k^{l-1}w_k^l\right)
$$

となる出力$o_j^l$を考えます。図で示すと
In a figure, this would be represented as follows.

```{image} figs/neuron.png
:alt: var_circuit
:width: 350px
:align: center
```

になります。関数$g$は活性化関数と呼ばれ、入力に対して非線形な出力を与えます。活性化関数としては、一般的にはシグモイド関数やReLU（Rectified Linear Unit）等の関数が用いられることが多いです。
Function $g$ is called an activation function, and the output is nonlinear with respect to the input. Generally, functions such as sigmoid functions and ReLU (Rectified Linear Unit) functions are used as activation functions.

関数$f(x,\mathbf{w}^*)$を求めるために、最適なパラメータ$\mathbf{w}^*$を決定するプロセス（学習と呼ばれる）が必要です。そのために、出力$\tilde{y}$とターゲットとなる変数$y$の差を測定する関数$L(\mathbf{w})$を考えます（一般に損失関数やコスト関数と呼ばれます）。

To find function $f(x,\mathbf{w}^*)$, we need a process for determining the optimized parameter $\mathbf{w}^*$ (this is called "learning"). For this reason, we will consider a function $L(\mathbf{w})$ for measuring the difference between output $\tilde{y}$ and the target variable $y$ (this is generally called a "loss function" or a "cost function").

$$
L(\mathbf{w}) = \frac{1}{N}\sum_{i=1}^N L(f(x_i,\mathbf{w}),y_i)
$$

$N$は$(x_i, y_i)$データの数です。この損失関数$L(\mathbf{w})$を最小化するパラメータ$\mathbf{w}^*$を求めたいわけですが、それには誤差逆伝搬法と呼ばれる手法を使うことができることが知られています。この手法は、$L(\mathbf{w})$の各$w$に対する微分係数$\Delta_w L(\mathbf{w})$を求めて、

$N$ is the number of $(x_i,y_i)$ data. We want to find the parameter $\mathbf{w}^*$ that minimizes the loss function $L(\mathbf{w})$. One known way for doing so is the error back propagation method. This method seeks to determine the derivative $\Delta_w L(\mathbf{w})$ for each $w$ of $L(\mathbf{w})$ by updating $w$ as shown below.

$$
w'=w-\epsilon\Delta_w L(\mathbf{w})
$$

のように$w$を更新することで、$L(\mathbf{w})$を最小化するというものです（$w$と$w'$は更新前と更新後のパラメータ）。$\epsilon\:(>0)$は学習率と呼ばれるパラメータで、これは基本的には私たちが手で決めてやる必要があります。
Through this process, it minimizes $L(\mathbf{w})$ ($w$ and $w'$ represent the parameter before and after updating). $\epsilon\:(>0)$ is a parameter known as the "learning rate," and we must essentially decide this by hand.

+++

## 量子回路学習<a id='qml'></a>

変分量子回路を用いた量子回路学習アルゴリズムは、一般的には以下のような順番で量子回路に実装され、計算が行われます。
Generally speaking, with quantum machine learning algorithms that use variational quantum circuits, the quantum circuits are implemented using the procedure below and used to perform computation.

1. **学習データ**$\{(\mathbf{x}_i, y_i)\}$を用意する。$\mathbf{x}_i$は入力データのベクトル、$y_i$は入力データに対する真の値（教師データ）とする（$i$は学習データのサンプルを表す添字）。
2. 入力$\mathbf{x}$から何らかの規則で決まる回路$U_{\text{in}}(\mathbf{x})$（**特徴量マップ**と呼ぶ）を用意し、$\mathbf{x}_i$の情報を埋め込んだ入力状態$\ket{\psi_{\text{in}}(\mathbf{x}_i)} = U_{\text{in}}(\mathbf{x}_i)\ket{0}$を作る。
3. 入力状態にパラメータ$\boldsymbol{\theta}$に依存したゲート$U(\boldsymbol{\theta})$（**変分フォーム**）を掛けたものを出力状態$\ket{\psi_{\text{out}}(\mathbf{x}_i,\boldsymbol{\theta})} = U(\boldsymbol{\theta})\ket{\psi_{\text{in}}(\mathbf{x}_i)}$とする。
4. 出力状態のもとで何らかの**観測量**を測定し、測定値$O$を得る。例えば、最初の量子ビットで測定したパウリ$Z$演算子の期待値$\langle Z_1\rangle = \expval{\psi_{\text{out}}}{Z_1}{\psi_{\text{out}}}$などを考える。
5. $F$を適当な関数として、$F(O)$をモデルの出力$y(\mathbf{x}_i,\boldsymbol{\theta})$とする。
6. 真の値$y_i$と出力$y(\mathbf{x}_i,\boldsymbol{\theta})$の間の乖離を表す**コスト関数**$L(\boldsymbol{\theta})$を定義し、古典計算でコスト関数を計算する。
7. $L(\boldsymbol{\theta})$が小さくなるように$\boldsymbol{\theta}$を更新する。
8. 3-7のプロセスを繰り返すことで、コスト関数を最小化する$\boldsymbol{\theta}=\boldsymbol{\theta^*}$を求める。
9. $y(\mathbf{x},\boldsymbol{\theta^*})$が学習によって得られた**予測モデル**になる。

10. The learning data $\{(\mathbf{x}_i, y_i)\}$ is prepared. $\mathbf{x}_i$ represents the input data vector and $y_i$ represents the true value of the input data (teaching data) (the $i$ subscript indicates the learning data sample). 
11. Input $\mathbf{x}$ is used to prepare a circuit $U_{\text{in}}(\mathbf{x})$ (the feature map), which is decided by applying various rules. The input state $\ket{\psi_{\text{in}}(\mathbf{x}_i)} = U_{\text{in}}(\mathbf{x}_i)\ket{0}$, in which the $\mathbf{x}_i$ information is embedded, is created.
12. Gate U(\boldsymbol{\theta})$ (the variational form), which is dependent on parameter $\boldsymbol{\theta}$, is applied to the input state to produce the output state $\langle Z_1\rangle = \expval{\psi_{\text{out}}}{Z_1}{\psi_{\text{out}}}$. 
13. Some observable is measured based on the output state to obtain measurement value O. For example, consider the expectation value of a Pauli $Z$ operator measured for the first quantum bit, ⟨Z_1 ⟩=⟨ψ_out |Z_1 | ψ_out ⟩. 
14. Assuming some function $F$, $F(O)$ is set to the output $y(\mathbf{x}_i,\boldsymbol{\theta})$ of the model. 
15. A cost constant $L(\boldsymbol{\theta})$ is defined to express the gap between the true value $y_i$ and the output $y(\mathbf{x}_i,\boldsymbol{\theta})$. A classical computer is used to calculate the cost function. 
16. $L(\boldsymbol{\theta})$ is updated to reduce the size of $\boldsymbol{\theta}$. 
17. Steps 3 through 7 are repeated to minimize the cost constant such that $\boldsymbol{\theta}=\boldsymbol{\theta^*}$. 
18. $y(\mathbf{x},\boldsymbol{\theta^*})$ is the predictive model obtained through learning.

```{image} figs/var_circuit.png
:alt: var_circuit
:width: 700px
:align: center
```


この順に量子回路学習アルゴリズムを実装していきましょう。まず、必要なライブラリを最初にインポートします。
Let's implement the quantum machine learning algorithm using this process. First, let's import the required libraries.

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

## 初歩的な例<a id='example'></a>

ある入力$\{x_i\}$と、既知の関数$f$による出力$y_i=f(x_i)$が学習データとして与えられた時に、そのデータから関数$f$を近似的に求める問題を考えてみます。例として、$f(x)=x^3$としてみます。
Given an input, $\{x_i\}$ and learning data in the form of $y_i=f(x_i)$, output by known function $f$, think about how to approximate function $f$ using that data. In this example, let us define the function $f(x)=x^3$.

### 学習データの準備<a id='func_data'></a>

まず、学習データを準備します。$x_{\text{min}}$と$x_{\text{max}}$の範囲でデータを`num_x_train`個ランダムに取った後、正規分布に従うノイズを追加しておきます。`nqubit`が量子ビット数、`nlayer`が変分フォームのレイヤー数（後述）を表します。
First, prepare the learning data. After randomly selecting `num_x_train` items of data in the range of $x_{\text{min}}$ to $x_{\text{max}}$, add noise, using a normal distribution. `nqubit` is the number of quantum bits and `c_depth` is the depth of the variational form circuit (explained later).

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

# 学習用データセットの生成
x_train = rng.uniform(x_min, x_max, size=num_x_train)
y_train = func_to_learn(x_train)

# Apply noise with a normal distribution to the function
mag_noise = 0.05
y_train_noise = y_train + rng.normal(0., mag_noise, size=num_x_train)

# 検証用データセットの生成
x_validation = rng.uniform(x_min, x_max, size=num_x_validation)
y_validation = func_to_learn(x_validation) + rng.normal(0., mag_noise, size=num_x_validation)
```

### 量子状態の生成<a id='func_state_preparation'></a>

次に、入力$x_i$を初期状態$\ket{0}^{\otimes n}$に埋め込むための回路$U_{\text{in}}(x_i)$（特徴量マップ）を作成します。まず参考文献{cite}`quantum_circuit_learning`に従い、回転ゲート$R_j^Y(\theta)=e^{-i\theta Y_j/2}$と$R_j^Z(\theta)=e^{-i\theta Z_j/2}$を使って
Next, create a circuit $U_{\text{in}}(x_i)$ for embedding the initial state $\ket{0}^{\otimes n}$ in input $x_i$ (the feature map). First, as indicated in reference material{cite}`quantum_circuit_learning`, use rotation gates $R_j^Y(\theta)=e^{-i\theta Y_j/2}$ and $R_j^Z(\theta)=e^{-i\theta Z_j/2}$ to define the following.

$$
U_{\text{in}}(x_i) = \prod_j R_j^Z(\cos^{-1}(x^2))R_j^Y(\sin^{-1}(x))
$$

と定義します。この$U_{\text{in}}(x_i)$をゼロの標準状態に適用することで、入力$x_i$は$\ket{\psi_{\text{in}}(x_i)}=U_{\text{in}}(x_i)\ket{0}^{\otimes n}$という量子状態に変換されることになります。
Here, we will apply this $U_{\text{in}}(x_i)$ to the standard zero state to convert input $x_i$ to the quantum state $\ket{\psi_{\text{in}}(x_i)}=U_{\text{in}}(x_i)\ket{0}^{\otimes n}$.

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
    # parameter.arcsin()はparameterに値vが代入された時にarcsin(v)になるパラメータ表現
    u_in.ry(x.arcsin(), iq)
    # arccosも同様
    u_in.rz((x * x).arccos(), iq)

u_in.bind_parameters({x: x_train[0]}).draw('mpl')
```

### 変分フォームを使った状態変換<a id='func_variational_form'></a>

#### 変分量子回路$U(\boldsymbol{\theta})$の構成
次に、最適化すべき変分量子回路$U(\boldsymbol{\theta})$を作っていきます。これは以下の3つの手順で行います。
Next, we will create the variational quantum circuit $U(\boldsymbol{\theta})$ to be optimized. This is done using the following three steps.

1. 2量子ビットゲートの作成（$\to$ 量子ビットをエンタングルさせる）
2. 回転ゲートの作成
3. 1.と2.のゲートを交互に組み合わせ、1つの大きな変分量子回路$U(\boldsymbol{\theta})$を作る

4. Create a 2-quantum bit gate (→ entangle the quantum bits)
5. Creating the rotation gate 
6. Mutually combine the gates from steps 1 and 2 to create a single large variational quantum circuit $U(\boldsymbol{\theta})$.


#### 2量子ビットゲートの作成
ここではControlled-$Z$ゲート（$CZ$）を使ってエンタングルさせ、モデルの表現能力を上げることを目指します。
Here, we will use a Controlled-$Z$ gate ($CZ$) to entangle the quantum bits and aim to improve the descriptive capabilities of the model.

#### 回転ゲートと$U(\boldsymbol{\theta})$の作成
$CZ$ゲートを使ってエンタングルメントを生成する回路$U_{\text{ent}}$と、$j \:(=1,2,\cdots n)$番目の量子ビットに適用する回転ゲート
We will combine the circuit that causes entanglement through the use of $CZ$ gates, $U_{\text{ent}}$, with the product of multiple rotation gates applied to quantum bit $j \:(=1,2,\cdots n)$ to create $U(\boldsymbol{\theta})$, as follows.

$$
U_{\text{rot}}(\theta_j^l) = R_j^Y(\theta_{j3}^l)R_j^Z(\theta_{j2}^l)R_j^Y(\theta_{j1}^l)
$$

を掛けたものを組み合わせて、変分量子回路$U(\boldsymbol{\theta})$を構成します。ここで$l$は量子回路の層を表していて、$U_{\text{ent}}$と上記の回転ゲートを合計$d$層繰り返すことを意味しています。実際は、この演習では最初に回転ゲート$U_{\text{rot}}$を一度適用してから$d$層繰り返す構造を使うため、全体としては
Here, $l$ is the quantum circuit layer. This indicates that $U_{\text{ent}}$ and the above rotation gates will be repeatedly applied in a total of $d$ layers. In reality, in this exercise we will apply rotation gate $U_{\text{rot}}$ once, and then, to use the structure of $d$ levels of repetitions, we will use a variational quantum circuit with the following form.

$$
U\left(\{\theta_j^l\}\right) = \prod_{l=1}^d\left(\left(\prod_{j=1}^n U_{\text{rot}}(\theta_j^l)\right) \cdot U_{\text{ent}}\right)\cdot\prod_{j=1}^n U_{\text{rot}}(\theta_j^0)
$$

という形式の変分量子回路を用いることになります。つまり、変分量子回路は全体で$3n(d+1)$個のパラメータを含んでいます。$\boldsymbol{\theta}$の初期値ですが、$[0, 2\pi]$の範囲でランダムに設定するものとします。
In other words, the overall variational quantum circuit contains $3n(d+1)$ parameters. The initial value of $\boldsymbol{\theta}$ will be set randomly in the $[0, 2\pi]$ range.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
u_out = QuantumCircuit(nqubit, name='U_out')

# 長さ0のパラメータ配列
theta = ParameterVector('θ', 0)

# thetaに一つ要素を追加して最後のパラメータを返す関数
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

### 測定とモデル出力<a id='func_measurement'></a>

モデルの出力（予測値）として、状態$\ket{\psi_{\text{out}}(\mathbf{x},\boldsymbol{\theta})}=U(\boldsymbol{\theta})\ket{\psi_{\text{in}}(\mathbf{x})}$の元で最初の量子ビットを$Z$基底で測定した時の期待値を使うことにします。つまり$y(\mathbf{x},\boldsymbol{\theta}) = \langle Z_0(\mathbf{x},\boldsymbol{\theta}) \rangle = \expval{\psi_{\text{out}}(\mathbf{x},\boldsymbol{\theta})}{Z_0}{\psi_{\text{out}}(\mathbf{x},\boldsymbol{\theta})}$です。
For the model's output (predictions), let's use the expectation value when the first quantum bit is measured with the $Z$ basis state based on the state $\ket{\psi_{\text{out}}(\mathbf{x},\boldsymbol{\theta})}=U(\boldsymbol{\theta})\ket{\psi_{\text{in}}(\mathbf{x})}$. In other words, y(\mathbf{x},\boldsymbol{\theta}) = \langle Z_0(\mathbf{x},\boldsymbol{\theta}) \rangle = \expval{\psi_{\text{out}}(\mathbf{x},\boldsymbol{\theta})}{Z_0}{\psi_{\text{out}}(\mathbf{x},\boldsymbol{\theta})}$.

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
# 今回はバックエンドを利用しない（量子回路シミュレーションを簡略化した）Estimatorクラスを使う
estimator = Estimator()

# 与えられたパラメータの値とxの値に対してyの値を計算する
def yvals(param_vals, x_vals=x_train):
    circuits = list()
    for x_val in x_vals:
        # xだけ数値が代入された変分回路
        circuits.append(model.bind_parameters({x: x_val}))

    # 観測量はIIZ（右端が第0量子ビット）
    observable = SparsePauliOp('I' * (nqubit - 1) + 'Z')

    # shotsは関数の外で定義
    job = estimator.run(circuits, [observable] * len(circuits), [param_vals] * len(circuits), shots=shots)

    return np.array(job.result().values)

def objective_function(param_vals):
    return np.sum(np.square(y_train_noise - yvals(param_vals)))

def callback_function(param_vals):
    # lossesは関数の外で定義
    losses.append(objective_function(param_vals))

    if len(losses) % 10 == 0:
        print(f'COBYLA iteration {len(losses)}: cost={losses[-1]}')
```

コスト関数$L$として、モデルの予測値$y(x_i, \theta)$と真の値$y_i$の平均2乗誤差の総和を使っています。
For cost function $L$, we will use the sum of the mean squared error of the model's prediction values $y(x_i, \theta)$ and the true values, $y_i$.

では、最後にこの回路を実行して、結果を見てみましょう。
Last, let's implement this circuit and look at the results.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
# COBYLAの最大ステップ数
maxiter = 50
# COBYLAの収束条件（小さいほどよい近似を目指す）
tol = 0.05
# バックエンドでのショット数
shots = 1000

optimizer = COBYLA(maxiter=maxiter, tol=tol, callback=callback_function)
```

```{code-cell} ipython3
:tags: [remove-input]

# テキスト作成用のセル - わざと次のセルでエラーを起こさせる
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

# テキスト作成用のセルなので無視してよい

if os.getenv('JUPYTERBOOK_BUILD') == '1':
    import pickle

    with open('data/qc_machine_learning_xcube.pkl', 'rb') as source:
        min_result, losses = pickle.load(source)
```

コスト値の推移をプロットします。

```{code-cell} ipython3
plt.plot(losses)
```

+++ {"jupyter": {"outputs_hidden": false}, "pycharm": {"name": "#%%\n"}}

最適パラメータ値でのモデルの出力値をx_minからx_maxまで均一にとった100点で確認します。

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

# 結果を図示する
plt.plot(x_train, y_train_noise, "o", label='Training Data (w/ Noise)')
plt.plot(x_list, func_to_learn(x_list), label='Original Function')
plt.plot(x_list, np.array(y_pred), label='Predicted Function')
plt.legend();
```

生成された図を確認してください。ノイズを印加した学習データの分布から、元の関数$f(x)=x^3$をおおよそ導き出せていることが分かると思います。
Let's check the figure that was generated. As you can see, we have largely derived the original function, $f(x)=x^3$, from the learning data, to which noise had been added.

この実習では計算を早く収束させるために、COBYLAオプティマイザーをCallする回数の上限`maxiter`を50、計算をストップする精度の許容範囲`tol`を0.05とかなり粗くしています。`maxiter`を大きくするあるいは`tol`を小さくするなどして、関数を近似する精度がどう変わるか確かめてみてください（ただ同時に`maxiter`を大きくかつ`tol`を小さくしすぎると、計算に非常に時間がかかります）。
In order to rapidly produce calculation results in this exercise, we set the maximum number of times the COBYLA optimizer would be called (`maxiter`) to 50, and set the accuracy tolerance range at which calculation would be stopped (`tol`) to 0.05. Check how the accuracy of the function approximation varies if maxiter is raised or tol is lowered (note that if you raise maxiter too much while lowering tol too much, calculation will take an extremely large amount of time).

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