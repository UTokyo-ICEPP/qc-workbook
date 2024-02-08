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

# 物理系を表現する

量子コンピュータの並列性を利用した計算の代表例として、量子系のダイナミクスシミュレーションについて学びます。
You will learn about creating a dynamic simulation of a quantum system as a typical example of calculation using the parallel calculation capabilities of a quantum computer.

```{contents} 目次
---
local: true
---
```

$\newcommand{\bra}[1]{\langle #1 |}$
$\newcommand{\ket}[1]{| #1 \rangle}$
$\newcommand{\upket}{\ket{\!\uparrow}}$
$\newcommand{\downket}{\ket{\!\downarrow}}$
$\newcommand{\rightket}{\ket{\!\rightarrow}}$
$\newcommand{\leftket}{\ket{\!\leftarrow}}$

+++

## 量子系のダイナミクスとは

量子力学について少しでも聞いたことのある方は、量子力学の根幹にシュレーディンガー方程式というものが存在することを知っているかと思います。この方程式は
If you have even the most passing familiarity with quantum mechanics, you'll know that the Schrödinger equation lies at the heart of quantum mechanics. This equation can be expressed as shown below.

$$
i \hbar \frac{\partial}{\partial t} \ket{\psi (t)} = H \ket{\psi (t)}
$$

などと表現され、時刻$t$のある系の状態$\ket{\psi (t)}$の時間微分（左辺）が$\ket{\psi (t)}$へのハミルトニアンという演算子の作用で定まる（右辺）ということを表しています。ただしこの「微分形」の方程式は我々の目的には少し使いづらいので、ここでは等価な「積分形」にして
This shows that the derivative with respect to time of the state $\ket{\psi (t)}$ of a system with time $t$ (the left side of the equation) is decided by the action of an operator called a Hamiltonian on $\ket{\psi (t)}$ (the right side of the equation). This differential form equation is somewhat difficult for us to use, so we'll change it into its equivalent integral form, as shown below.

$$
\ket{\psi (t_1)} = T \left[ \exp \left( -\frac{i}{\hbar} \int_{t_0}^{t_1} H dt \right) \right] \ket{\psi (t_0)}
$$

と書いておきます。$T[\cdot]$は「時間順序演算子」と呼ばれ重要な役割を持ちますが、説明を割愛し、以下ではハミルトニアン$H$が直接時間に依存しない場合の
If the Hamiltonian $H$ is not directly dependent on the time, this can be written as follows.

$$
\ket{\psi (t_1)} = \exp \left( -\frac{i}{\hbar} H (t_1 - t_0) \right) \ket{\psi (t_0)}
$$

のみを考えます。量子状態に対する演算子（線形演算子）の指数関数もまた演算子なので、積分形のシュレーディンガー方程式は「$e^{-i/\hbar H (t_1-t_0)}$という演算子が系を時刻$t_0$の初期状態$\ket{\psi(t_0)}$から時刻$t_1$の状態$\ket{\psi(t_1)}$に発展させる」と読めます。さらに、定義上ハミルトニアンは「エルミート演算子」であり、それに虚数単位をかけて指数の冪にした$e^{-i/\hbar H t}$（以下これを時間発展演算子$U_H(t)$と呼びます）は「ユニタリ演算子」です（このあたりの線形代数の用語にあまり馴染みがなくても、そういうものかと思ってもらえれば結構です）。
Below, we will consider only this case. The exponential functions of operators used on quantum states (linear operators) are also operators, so the integral Schrödinger equation can be read as "the operator $e^{-i/\hbar H (t_1-t_0)}$ expands the system from its initial state of $\ket{\psi(t_0)}$ at $t_0$ to its state of $\ket{\psi(t_1)}$ at time $t_1$. Furthermore, by its definition, the Hamiltonian is a Hermitian operator, and $e^{-i/\hbar H t}$, which multiplies it by an imaginary unit and raises it to a power is a unitary operator (hereafter, this is referred to as the time evolution operator $U_H(t)$) (if you're not that familiar with these linear algebra terms, just keep this somewhere in the back of your mind).

ユニタリ演算子は量子計算の言葉で言えばゲートにあたります。したがって、ある量子系に関して、その初期状態を量子レジスタで表現でき、時間発展演算子を量子コンピュータの基本ゲートの組み合わせで実装できれば、その系のダイナミクス（＝時間発展）シミュレーションを量子コンピュータで行うことができます。
A unitary operator, in quantum calculation terminology, is a gate. Therefore, for a quantum system, if we can express the initial state using a quantum register and implement the time evolution operator through a combination of quantum computer basic gates, we can perform a simulation of the system's dynamics (=time evolution) using a quantum computer.

+++

### 例：核磁気の歳差運動

シミュレーションの詳しい話をする前に、これまで量子力学と疎遠だった方のために、ハミルトニアンや時間発展とは具体的にどういうことか、簡単な例を使って説明します。
Before getting into a detailed discussion of the simulation, for the benefit of readers who may become a bit rusty when it comes to quantum mechanics, let's go over what, specifically, Hamiltonians and time evolution are, using some basic examples. 

空間中に固定されたスピン$\frac{1}{2}$原子核一つを考えます。ある方向（Z方向とします）のスピン$\pm \frac{1}{2}$の状態をそれぞれ$\upket, \downket$で表します。量子力学に馴染みのない方のための説明例で大いに量子力学的な概念を使っていますが、何の話かわからなければ「2つの基底ケットで表現される、量子ビットのような物理系がある」と考えてください。量子ビットのような物理系なので、系の状態は一般に$\upket$と$\downket$の重ね合わせになります。
onsider an atomic nucleus with a spin of $\frac{1}{2}$, suspended in a fixed point in space. Its spin $\pm \frac{1}{2}$ status in a given direction (let's call it the Z direction) can be expressed as $\upket$ and $\downket$で. This explanation example for people who are unfamiliar with quantum mechanics uses many quantum mechanics concepts, so if you don't follow, just think of this as being a "physical system like a quantum bit which can be expressed with two basis state kets." This physical system is like a quantum bit, so the system is normally a superposition of |↑⟩ and |↓⟩.

時刻$t_0$で系が$\ket{\psi(t_0)} = \upket$にあるとします。時刻$t_1$での系の状態を求めることは
Let us say that the state of the system at $t_0$ is $\ket{\psi(t_0)} = \upket$. At $t_1$, we can determine the state of the system using the following equation.

$$
\ket{\psi (t_1)} = \alpha (t_1) \upket + \beta (t_1) \downket
$$

の$\alpha (t_1)$と$\beta (t_1)$を求めることに相当します。ここで$\alpha (t_0) = 1, \beta (t_0) = 0$です。
Determining the state of the system is equivalent to determining the state of $\alpha (t_1)$ and $\beta (t_1)$. Here, $\alpha (t_0) = 1$, $beta (t_0) = 0$.

この原子核に$X$方向の一定磁場をかけます。非ゼロのスピンを持つ粒子はスピンベクトル$\vec{\sigma}$と平行な磁気モーメント$\vec{\mu}$を持ち、磁場$\vec{B}$のもとでエネルギー$-\vec{B}\cdot\vec{\mu}$を得ます。ハミルトニアンとは実は系のエネルギーを表す演算子なので、この一定磁場だけに注目した場合の系のハミルトニアンは、何かしらの定数$\omega$とスピンベクトルの$X$成分$\sigma^X$を用いて$H = \hbar \omega \sigma^X$と書けます。
Now let us apply a constant magnetic field to the atomic nucleus in the $X$ direction. Particles with non-zero spin will have a magnetic moment $\vec{\mu}$ parallel to the spin vector $\vec{\sigma}$, and in magnetic field $\vec{B}$ will gain $-\vec{B}\cdot\vec{\mu}$ energy. The Hamiltonian is an operator that expresses the energy of the system. The Hamiltonian for a system with only a constant magnetic field, like this example, can be written as $H = \hbar \omega \sigma^X$, using a constant $\omega$ and the spin vector's $X$ component $\sigma^X$.

量子力学では$\sigma^X$は演算子であり、$\upket$と$\downket$に対して
In quantum mechanics, $\sigma^X$ is an operator and acts on $\upket$ and $\downket$ as below.

$$
\sigma^X \upket = \downket \\
\sigma^X \downket = \upket
$$

と作用します。時間発展演算子$U_H(t)$は
$U_H (t)$ is a time evolution operator.

$$
U_H(t) = \exp (-i \omega t \sigma^X) = \sum_{n=0}^{\infty} \frac{1}{n!} (-i \omega t)^n (\sigma^X)^n = I + (-i \omega t) \sigma^X + \frac{1}{2} (-i \omega t)^2 (\sigma^X)^2 + \frac{1}{6} (-i \omega t)^3 (\sigma^X)^3 \cdots
$$

ですが（$I$は恒等演算子）、上の$\sigma^X$の定義からわかるように$(\sigma^X)^2 = I$なので
($I$ is an identity operator.) Given the definition for $\sigma^X$ above, $(\sigma^X)^2 = I$, so this can be written as follows.

```{math}
:label: exp_sigmax
\begin{align}
\exp (-i \omega t \sigma^X) & = \left[ 1 + \frac{1}{2} (-i \omega t)^2 + \cdots \right] I + \left[(-i \omega t) + \frac{1}{6} (-i \omega t)^3 + \cdots \right] \sigma^X \\
& = \cos(\omega t) I - i \sin(\omega t) \sigma^X
\end{align}
```

と書けます。したがって、
Therefore:

```{math}
:label: spin_exact
\begin{align}
\ket{\psi (t_1)} = U_H(t_1 - t_0) \ket{\psi (t_0)} & = \exp [-i \omega (t_1 - t_0) \sigma^X] \upket \\
& = \cos[\omega (t_1 - t_0)] \upket - i \sin[\omega (t_1 - t_0)] \downket
\end{align}
```

です。任意の時刻$t_1$のスピンの状態が基底$\upket$と$\downket$の重ね合わせとして表現されました。
At any given time, $t_1$, the spin state can be expressed as the superposition of basis states $\upket$ and $\downket$.

このように、系のエネルギーの表式からハミルトニアンが決まり、その指数関数を初期状態に作用させることで時間発展後の系の状態が求まります。
In this way, the energy formula for the system determines its Hamiltonian, and it can be applied to the initial state of the exponential function to determine the state of the system after time evolution.

ちなみに、$\ket{\psi (t_1)}$は$t_1 = t_0$で$\upket$、$t_1 = t_0 + \pi / (2 \omega)$で$(-i)\downket$となり、以降$\upket$と$\downket$を周期的に繰り返します。実は、その間の状態はスピンが$Y$-$Z$平面内を向いている状態に相当します。スピンが0でない原子核に磁場をかけると、スピンと磁場の方向が揃っていなければ磁場の方向を軸にスピンが歳差運動（すりこぎ運動）をします。これはコマが重力中で起こす運動と同じで、核磁気共鳴（NMR、さらに医学応用のMRI）の原理に深く関わります。
When $t_1=t_0$, $\ket{\psi (t_1)}$ is $\upket$, while when $t_1 = t_0 + \pi / (2 \omega)$, it is $(-i)\downket$. \upket$ and \downket$ then repeat cyclically. In reality, the states in between correspond to spins on the $Y$ and $Z$ planes. When a magnetic field is applied to an atomic nucleus with non-zero spin, if the direction of the spin and the atomic field do not match, the spin will undergo precessional movement along the axis of the direction of the magnetic field. This is the same as the motion of a spinning top within a gravitational field, and is deeply tied to the principles of nuclear magnetic resonance, or NMR (a form of MRI used in medical applications).

+++

### 量子コンピュータ上での表現

すでに触れましたが、上の例で核のスピンは量子ビットのように2つの基底ケットを持ちます（2次元量子系です）。さらに、お気づきの方も多いと思いますが、$\sigma^X$の$\upket$と$\downket$への作用は$X$ゲートの$\ket{0}$と$\ket{1}$への作用そのものです。このことから、核磁気の歳差運動が極めて自然に量子コンピュータでシミュレートできることがわかるかと思います。
As we've already briefly touched on, in the example above the spin of the nucleus has two basis state kets, like a quantum bit (it is a two-dimensional quantum system). Furthermore, many readers have likely also noticed that $\sigma^X$ acts on $\upket$ and $\downket$ like the $X$ gate acts on $\ket{0}$ and $\ket{1}$. You can therefore see that the precessional movement of nuclear magnetism can be simulated in a quantum computer in an extremely natural manner.

実際には、時間発展演算子は$\sigma^X$そのものではなくその指数関数なので、量子コンピュータでも$\exp (-i \frac{\theta}{2} X)$に対応する$R_{x} (\theta)$ゲートを利用します。これまで紹介されませんでしたが、$R_{x}$ゲートはパラメータ$\theta$をとり、
In reality, the time evolution operator is not $\sigma^X$ itself, but an exponential function of it, so quantum computers also use the $R_{x} (\theta)$ gate, which corresponds to $\exp (-i \frac{\theta}{2} X)$. We have not previously introduced this, but the $R_x$ gate uses parameter $\theta$ and performs the following transformation.

$$
R_{x}(\theta)\ket{0} = \cos\frac{\theta}{2}\ket{0} - i\sin\frac{\theta}{2}\ket{1} \\
R_{x}(\theta)\ket{1} = -i\sin\frac{\theta}{2}\ket{0} + \cos\frac{\theta}{2}\ket{1}
$$

という変換を行います。上の核スピン系を量子コンピュータでシミュレートするには、1量子ビットで$R_{x} (2 \omega (t_1 - t_0)) \ket{0}$を計算する以下の回路を書けばいいだけです。
To simulate the above nuclear spin system with a quantum computer, you could calculate $R_{x} (2 \omega (t_1 - t_0)) \ket{0}$ with 1 quantum bit using the following circuit.

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
```

```{code-cell} ipython3
:tags: [remove-input]

circuit = QuantumCircuit(QuantumRegister(1, 'q'), ClassicalRegister(1, 'c'))
circuit.rx(Parameter(r'$2 \omega (t_1 - t_0)$'), 0)
circuit.measure(0, 0)
circuit.draw('mpl', initial_state=True)
```

### ハミルトニアンの対角化

再び量子コンピュータを離れて、量子・古典に関わらずデジタル計算機で量子ダイナミクスのシミュレーションをする際の一般論をします。
Let us step away from quantum computers again and talk in general about how quantum mechanics are simulated by digital computers, whether quantum or classical.

上の核スピンの例ではハミルトニアンが単純だったので、式{eq}`spin_exact`のように厳密解が求まりました。特に、導出において$(\sigma^X)^2 = I$という恒等式が非常に重要でした。しかし、一般のハミルトニアンでは、何乗しても恒等演算子の定数倍にたどり着く保証がありません。
In the nuclear spin example above, the Hamiltonian was simple, so we were able to determine the exact solution in Formula {eq}`spin_exact`. In deriving it, the identity formula of $(\sigma^X)^2 = I$ was extremely important. However, there is no guarantee that with ordinary Hamiltonians, raising to a certain power will produce a multiple of the identity operator.

累乗して恒等演算子にならないようなハミルトニアンであっても、系の次元が小さい場合は「対角化」という作業で厳密解を得られます。ハミルトニアンの対角化とは、ハミルトニアンの作用が実数をかけることと等しくなるようなケットを探してくること、つまり
Even for Hamiltonians which do not become identity operators when raised to other powers, if the system's dimensions are small, exact solutions can be determined through diagonalization. Hamiltonian diagonalization consists of searching for kets for which applying the Hamiltonian produces the same results as multiplying by a real number. In other words, it refers to searching for a $\ket{\phi_j}$ that would make the following true.

$$
H\ket{\phi_j} = \hbar \omega_j \ket{\phi_j}, \quad \omega_j \in \mathbb{R}
$$

が成り立つような$\ket{\phi_j}$を見つけることを指します。このような$\ket{\phi_j}$を「固有値$\hbar \omega_j$を持つ$H$の固有ベクトル」と呼びます。「エネルギー固有状態」と呼ぶこともあります。系の次元が$N$であれば、独立な固有ベクトルが$N$個存在します。
In this case, $\ket{\phi_j}$ is called an $H$ eigenvector with eigenvalue $\hbar \omega_j$. It is also called an energy eigenstate. If the system has $N$ dimensions, there are $N$ independent eigenvectors.

例えば上の例では$H = \hbar \omega \sigma^X$ですが、
For example, in the above example, $H = \hbar \omega \sigma^X$. However, consider the following two states.

```{math}
:label: left_right_kets
\rightket := \frac{1}{\sqrt{2}}(\upket + \downket) \\
\leftket := \frac{1}{\sqrt{2}}(\upket - \downket)
```

という2つの状態を考えると
This results in the following.

$$
\sigma^X \rightket = \rightket \\
\sigma^X \leftket = -\leftket
$$

なので、これらが固有値$\pm \hbar \omega$の$H$の固有ベクトルとなっていることがわかります。
As you can see, these are the $H$ eigenvectors for eigenvalues $\pm \hbar \omega$.

固有値$\hbar \omega_j$のハミルトニアン$H$の固有ベクトル$\ket{\phi_j}$は自動的に時間発展演算子$U_H(t)$の固有値$e^{-i\omega_j t}$の固有ベクトルでもあります。
The eigenvector $\ket{\phi_j}$ of Hamiltonian $H$ for eigenvalue $\hbar \omega_j$ is, automatically, also an eigenvector for eigenvalue $e^{-i\omega_j t}$ of time evolution operator $U_H (t)$.

$$
U_H(t) \ket{\phi_j} = \exp \left( -\frac{i}{\hbar} H t \right) \ket{\phi_j} = \exp (-i \omega_j t) \ket{\phi_j}.
$$

したがって、系の初期状態$\ket{\psi (t_0)}$が
Therefore, if the initial state of the system, $\ket{\psi (t_0)}$, is as follows:

$$
\ket{\psi (t_0)} = \sum_{j=0}^{N} c_j \ket{\phi_j}
$$

であれば、時刻$t_1$での状態は
Then the state at time $t_1$ is:

$$
\ket{\psi (t_1)} = \sum_{j=0}^{N} c_j U_H(t_1 - t_0) \ket{\phi_j} = \sum_{j=0}^{N} e^{-i \omega_j (t_1 - t_0)} c_j \ket{\phi_j},
$$

つまり、各固有ベクトルの振幅に、対応する位相因子をかけるだけで求まります。
In other words, the amplitude of each eigenvector can be determined simply by multiplying the corresponding phase factors.

再び核スピンの例を見ると、初期状態$\ket{\psi(t_0)} = \upket = 1/\sqrt{2} (\rightket + \leftket)$なので、
Looking at the example of the nuclear spin, the initial state is $\ket{\psi(t_0)} = \upket = 1/\sqrt{2} (\rightket + \leftket)$.

$$
\begin{align}
\ket{\psi(t_1)} & = \frac{1}{\sqrt{2}} \left( e^{-i\omega (t_1 - t_0)} \rightket + e^{i\omega (t_1 - t_0)} \leftket \right) \\
& = \frac{1}{2} \left[ \left( e^{-i\omega (t_1 - t_0)} + e^{i\omega (t_1 - t_0)} \right) \upket + \left( e^{-i\omega (t_1 - t_0)} - e^{i\omega (t_1 - t_0)} \right) \downket \right] \\
& = \cos [\omega (t_1-t_0)] \upket - i \sin [\omega (t_1-t_0)] \downket
\end{align}
$$

となり、式{eq}`spin_exact`が再導出できます。
We can therefore rederive Formula {eq}`spin_exact`.

このように、ハミルトニアンの対角化さえできれば、量子ダイナミクスのシミュレーションは位相をかけて足し算をするだけの問題に帰着します。しかし、上で言及したように、計算量の問題から、ハミルトニアンが対角化できるのは主に系の次元が小さいときに限ります。「対角化」という言葉が示唆するように、この操作は行列演算（対角化）を伴い、その際の行列の大きさは$N \times N$です。上の核スピンの例では$N=2$でしたが、もっと実用的なシミュレーションの場合、系の量子力学的次元は一般的に関係する自由度の数（粒子数など）の指数関数的に増加します。比較的小規模な系でもすぐに対角化にスーパーコンピュータが必要なスケールになってしまいます。
In this way, if we can simply perform Hamiltonian diagonalization, the simulation of the quantum dynamics becomes simply a matter of multiplying phases and then adding. However, as mentioned above, due to the issue of computational volume, Hamiltonian diagonalization can usually only be performed for systems with small dimensions. As the word "diagonalization" suggests, this operation involves matrix operations (diagonalization), so the size of the array will be $N \times N$. In the above nuclear spin example, $N=2$, but in a more practical simulation, the number of quantum mechanical dimensions in a system generally increases exponentially with respect to the related degree of freedom (number of particles, etc.). Even a relatively small system would require the use of a supercomputer to perform diagonalization.
+++

### 鈴木・トロッター分解

ハミルトニアンが対角化できない場合、ダイナミクスシミュレーションをするには、結局式{eq}`spin_exact`のように初期状態に時間発展演算子を愚直にかけていくことになります。これは、式{eq}`exp_sigmax`のように$U_H(t)$を閉じた形式で厳密に書けるなら簡単な問題ですが、そうでない場合は数値的に近似していく必要があります。その場合の常套手段は、行いたい時間発展$(t_1 - t_0)$を短い時間
When Hamiltonian diagonalization is not possible, to perform a dynamics simulation, we would ultimately have to simply multiply the initial state by a time evolution operator, as in Formula {eq}`spin_exact`. If $U_H (t)$ could strictly expressed in closed form, as in Formula {eq}`exp_sigmax`, this would be a simple issue, but when it can't, we must numerically approximate it. The standard method for doing this is to break down the time evolution $(t_1  - t_0)$ finely, as follows:

$$
\Delta t = \frac{t_1 - t_0}{M}, \quad M \gg 1
$$

に分割し、$\Delta t$だけの時間発展$U_H(\Delta t)$を考えることです。もちろん、$U_H(t)$が閉じた形式で書けないのなら当然$U_H(\Delta t)$も書けないので、時間を分割しただけでは状況は変わりません。しかし、$\Delta t$が十分短いとき、$U_H(\Delta t)$に対応する計算可能な近似演算子$\tilde{U}_{H;\Delta t}$を見つけることができる場合があり、この$\tilde{U}_{H;\Delta t}$での状態の遷移の様子がわかるのであれば、それを$M$回繰り返すことで、求める終状態が近似できることになります。

例えば、通常$H$はわかっており、任意の状態$\ket{\psi}$に対して$H\ket{\psi}$が計算できるので、$\mathcal{O}((\Delta t)^2)$を無視する近似で
Normally, we know $H$, so we can calculate $H\ket{\psi}$ for any state $\ket{\psi}$. For example, if we approximate, ignoring $\mathcal{O}((\Delta t)^2)$, we get the following:

$$
\tilde{U}_{H;\Delta t} = I - \frac{i \Delta t}{\hbar} H
$$

とすれば、まず$H\ket{\psi(t_0)}$を計算し、それを$i\Delta t/\hbar$倍して$\ket{\psi(t_0)}$から引き、その結果にまた$H$をかけて、…という具合に$\ket{\psi(t_1)}$が近似計算できます[^exact_at_limit]。
First, we calculate $H\ket{\psi(t_0)}$ and then multiply that by $i\Delta t/\hbar$ and subtract from $\ket{\psi(t_0)}$. We multiply the result by $H$...and in this fashion we can approximately calculate $\ket{\psi(t_1)}$[^exact_at_limit].

しかし、このスキームは量子コンピュータでの実装に向いていません。上で述べたように量子コンピュータのゲートはユニタリ演算子に対応するのに対して、$I - i\Delta t / \hbar H$はユニタリでないからです。代わりに、量子コンピュータでのダイナミクスシミュレーションでよく用いられるのが鈴木・トロッター分解という近似法です{cite}`nielsen_chuang_dynamics`。
However, this approach is not well-suited to implementation in a quantum computer. As discussed above, while quantum computer gates correspond to unitary operators, $I - i\Delta t / \hbar H$ is not unitary. Instead, what is often used in dynamics simulations on quantum computers is the Suzuki-Trotter transformation approximation method{cite}`nielsen_chuang_dynamics`.

鈴木・トロッター分解が使えるケースとは、

- $U_H(t)$は量子回路として実装が難しい。
- ハミルトニアンが$H = \sum_{k=1}^{L} H_k$のように複数の部分ハミルトニアン$\{H_k\}_k$の和に分解できる。
- 個々の$H_k$に対しては$U_{H_k}(t) = \exp(-\frac{i t}{\hbar} H_k)$が簡単に実装できる。

のような場合です。もしも$H$や$H_k$が演算子ではなく単なる実数であれば、$\exp\left(\sum_k A_k\right) = \prod_k e^{A_k}$なので、$U_H(t) = \prod_k U_{H_k}(t)$となります。ところが、一般に線形演算子$A, B$に対して、特殊な条件が満たされる（$A$と$B$が「可換」である）場合を除いて

$$
\exp(A + B) \neq \exp(A)\exp(B)
$$

なので、そのような簡単な関係は成り立ちません。しかし、

$$
\exp \left(- \frac{i \Delta t}{\hbar} H \right) = \prod_{k=1}^{L} \exp \left(-\frac{i \Delta t}{\hbar} H_k \right) + \mathcal{O}((\Delta t)^2)
$$

という、Baker-Campbell-Hausdorfの公式の応用式は成り立ちます。これによると、時間分割の極限では、

$$
\lim_{\substack{M \rightarrow \infty \\ \Delta t \rightarrow 0}} \left[ \prod_{k=1}^{L} \exp \left(-\frac{i \Delta t}{\hbar} H_k \right) \right]^M = \exp \left(-\frac{i}{\hbar} H (t_1 - t_0) \right).
$$

つまり、$U_H(\Delta t)$を

$$
\tilde{U}_{H;\Delta t} = \prod_k U_{H_k}(\Delta t)
$$

で近似すると、$[\tilde{U}_{H;\Delta t}]^M$と$U_H(t_1 - t_0)$の間の誤差は$\Delta t$を短くすることで[^sufficiently_small]いくらでも小さくできます。

鈴木・トロッター分解とは、このように全体の時間発展$U_H(t_1 - t_0)$を短い時間発展$U_H(\Delta t)$の繰り返しにし、さらに$U_H(\Delta t)$をゲート実装できる部分ユニタリの積$\prod_k U_{H_k}(\Delta t)$で近似する手法のことを言います。

[^exact_at_limit]: 実際、この手続きは$M \rightarrow \infty$の極限で厳密に$U(t_1 - t_0)$による時間発展となります。
[^sufficiently_small]: 具体的には、$\Omega = H/\hbar, \Omega_k = H_k/\hbar$として$\exp(-i\Delta t \Omega) - \prod_{k} \exp(-i\Delta t \Omega_k) = (\Delta t)^2/2 \sum_{k \neq l} [\Omega_k, \Omega_l] + \mathcal{O}((\Delta t)^3)$なので、任意の状態$\ket{\psi}$について$(\Delta t)^2 \sum_{k \neq l} \bra{\psi} [\Omega_k, \Omega_l] \ket{\psi} \ll 1$が成り立つとき、$\Delta t$が十分小さいということになります。
[^exact_at_limit]: This procedure produces the exact time evolution using $U(t_1 - t_0)$ at the limit of $M \rightarrow \infty$.
[^sufficiently_small]: Specifically, as $\Omega = H/\hbar$ and $\Omega_k = H_k/\hbar$, \exp(-i\Delta t \Omega) - \prod_{k} \exp(-i\Delta t \Omega_k) = (\Delta t)^2/2 \sum_{k \neq l} [\Omega_k, \Omega_l] + \mathcal{O}((\Delta t)^3)$, so when $(\Delta t)^2 \sum_{k \neq l} \bra{\psi} [\Omega_k, \Omega_l] \ket{\psi} \ll 1$ for a given state $\ket{\psi}$, $\Delta t$ will be sufficiently small.

+++

### なぜ量子コンピュータが量子ダイナミクスシミュレーションに向いているか

鈴木・トロッター分解がダイナミクスシミュレーションに適用できるには、ハミルトニアンが都合よくゲートで実装できる$H_k$に分解できる必要があります。これが常に成り立つかというと、答えはyes and noです。
For the Suzuki-Trotter transformation to be applied to a dynamics simulation, it must be possible to conveniently break the Hamiltonian down into $H_k$ that can be implemented using gates. Is this normally the case? Well, yes and no.

まず、$2^n$次元線形空間に作用するエルミート演算子は、$n$個の2次元部分系に独立に作用する基底演算子$\{I, \sigma^X, \sigma^Y, \sigma^Z\}$の積の線形和に分解できます。$\sigma^X$以外のパウリ演算子$\sigma^Y$と$\sigma^Z$はここまで登場しませんでしたが、重要なのは、2次元量子系に作用する$\sigma^X, \sigma^Y, \sigma^Z$がそれぞれ量子ビットに作用する$X, Y, Z$ゲート[^ygate]に、パウリ演算子の指数関数がそれぞれ$R_x, R_y, R_z$ゲート（総じて回転ゲートと呼びます）に対応するということです。つまり、対象の物理系の量子レジスタへの対応付けさえできれば、そのハミルトニアンは必ず基本的なゲートの組み合わせで表現できます。
First, the Hermitian operators that can be used in a $2^n$-dimension linear space can be broken down into the linear sum of basis state operators $\{I, \sigma^X, \sigma^Y, \sigma^Z\}$ that act independently in n number of two dimensional systems. While we have seen Pauli operator $\sigma^X$, we have not yet seen Pauli operators $\sigma^Y$ and $\sigma^Z$. However, what is important is that $\sigma^X$, $\sigma^Y$, and $\sigma^Z$, which act on two-dimensional quantum systems, correspond to the $X$, $Y$, and $Z$ gates used on their respective quantum bits[^ygate], and that the Pauli operator exponential functions correspond, respectively, to gates $R_x$, $R_y$, and $R_Z$ (collectively, this is referred to as a rotation gate). In other words, if we can associate them with the quantum registers of the physical system, the Hamiltonian can always be expressed using a combination of basic gates.

しかし、$n$ビットレジスタに作用する基底演算子の組み合わせは$4^n$通りあり、最も一般のハミルトニアンではその全ての組み合わせが寄与することも有りえます。その場合、指数関数的に多くのゲートを用いてしか時間発展演算子が実装できないことになります。それでは「都合よく分解できる」とは言えません。
However, there are $4^n$ combinations of basis state operators in an $n$-bit register, and the most common Hamiltonian could apply to all of those combinations. In that case, the only way to implement the time evolution operator would be to use an exponentially large number of gates. This could not even charitably be described as "conveniently breaking it down."

そもそも量子コンピュータで量子ダイナミクスシミュレーションを行う利点は、その計算効率にあります。
The whole advantage of performing a quantum dynamics simulation on a quantum computer is supposed to be its computation efficiency.

シミュレートする量子系の次元を$2^n$としたとき、古典計算機では、仮にハミルトニアンが対角化できても$2^n$回の位相因子の掛け算と同じ回数だけの足し算を行う必要があります。ハミルトニアンが対角化できず、時間を$M$ステップに区切って近似解を求めるとなると、必要な計算回数は$\mathcal{O}(2^nM)$となります。
When the number of dimensions in a quantum system to be simulated is $2^n$, with a classical computer, even if it were possible to diagonalize the Hamiltonian, you would need to perform the same number of additions as the product of $2^n$ phase factors. If you were unable to diagonalize the Hamiltonian, and instead tried to approximate by slicing the time up into $M$ steps, you would need to perform $\mathcal{O}(2^nM)$ calculations.

一方、同じ計算に$n$ビットの量子コンピュータを使うと、対角化できない場合のステップ数$M$は共通ですが、各ステップで必要な計算回数（＝ゲート数）はハミルトニアン$H$の基底演算子への分解$H_k$の項数$L$で決まります。個々の$H_k$は一般に$\mathcal{O}(n)$ゲート要するので、計算回数は$\mathcal{O}(nLM)$です。したがって、$L$が$\mathcal{O}(1)$であったり$\mathcal{O}(\mathrm{poly}(n))$（$n$の多項式）であったりすれば、量子コンピュータでの計算が古典のケースよりも指数関数的に早いということになります。
However, if you used an $n$-bit quantum computer to perform the same calculations, if you were unable to perform diagonalization, the number of steps, $M$, would be the same, but the number of calculations during each step (=the number of gates) would be decided by the number of terms, $L$, when breaking Hamiltonian $H$ into basis state operators $H_k$. Each individual $H_k$ would generally require an $\mathcal{O}(n)$ gate, so the number of computations would be $\mathcal{O}(nLM)$. Therefore, if $L$ were $\mathcal{O}(1)$ or $\mathcal{O}(\mathrm{poly}(n))$ (an $n$th polynomial), the calculation by the quantum computer would be exponentially faster than calculation with a classical computer.

したがって、逆に、ハミルトニアンが$4^n$通りの基底演算子に分解されてしまっては（$L=4^n$）、量子コンピュータの利点が活かせません[^exponential_memory]。
Therefore, conversely, if a Hamiltonian were broken down into $4^n$ basis state operators, then $L = 4^n$, and thus the advantages of the quantum computer could not be leveraged[^exponential_memory].

幸いなことに、通常我々がシミュレートしたいと思うような物理系では、$L$はせいぜい$\mathcal{O}(n^2)$で、$\mathcal{O}(n)$ということもしばしばあります。2体相互作用のある量子多体系などが前者にあたり、さらに相互作用が隣接した物体間のみである場合、後者が当てはまります。
Fortunately, for the types of physical systems we usually wish to emulate, $L$ is often at most $\mathcal{O}(n^2)$, and often $\mathcal{O}(n)$. The former would correspond to a quantum many-body system with two-body interaction, and the latter would correspond to a system in which there is only mutual interaction with an adjacent body.

[^ygate]: $Y$ゲートは変換$Y\ket{0} = i\ket{1}$、$Y\ket{1} = -i\ket{0}$を引き起こします。
[^exponential_memory]: 古典計算機でのシミュレーションでは、一般的には全ての固有ベクトルの振幅を記録しておくためのメモリ（$\mathcal{O}(2^n)$）も必要です。一方量子コンピュータでは（測定時に限られた情報しか取り出せないという問題はありますが）そのような制約がないので、指数関数的に多くのゲートを用いるハミルトニアンでも、一応後者に利点があると言えるかもしれません。
[^ygate]: The $Y$ gate causes transformations $Y\ket{0} = i\ket{1}$ and $Y\ket{1} = -i\ket{0}$.
[^exponential_memory]:  In a simulation using a classical computer, generally speaking, you would need ($\mathcal{O}(2^n)$) of memory to record the amplitudes of all of the eigenvectors. With a quantum computer, on the other hand, (although there is the issue of not being able to extract information except when performing measurement), this restriction does not apply, so even for Hamiltonians with an exponentially large number of gates, the quantum computer's benefits would outweigh those of the classical computer.

+++

## 実習：ハイゼンベルグモデルの時間発展

### モデルのハミルトニアン

ハミルトニアンの分解と言われてもピンと来ない方もいるかもしれませんし、ここからはダイナミクスシミュレーションの具体例をQiskitで実装してみましょう。
The concept of "breaking down the Hamiltonian" may not click with some readers, so from here on, let's implement a specific dynamics simulation in Qiskit.

ハイゼンベルグモデルという、磁性体のトイモデルを考えます。空間中一列に固定された多数のスピンを持つ粒子（電子）の系で、隣接スピンの向きによってエネルギーが決まるような問題です。
Let us consider a toy model of a magnetic body, the Heisenberg model. It is a system in which particles (electrons) with various spins are fixed in a line in space, and the amount of energy is decided by the direction of the spin of adjacent particles.

例えば、$n$スピン系で簡単な形式のハミルトニアンは
For example, a simplified Hamiltonian for the $n$-spin system would be as follows.

```{math}
:label: heisenberg
H = -J \sum_{j=0}^{n-2} (\sigma^X_{j+1}\sigma^X_{j} + \sigma^Y_{j+1}\sigma^Y_{j} + \sigma^Z_{j+1} \sigma^Z_{j})
```

です。ここで、$\sigma^{[X,Y,Z]}_j$は第$j$スピンに作用するパウリ演算子です。
Here, $\sigma^{[X,Y,Z]}_j$is a Pauli operator that acts on the $j$-th spin.

ただし、式{eq}`heisenberg`の和の記法には実は若干の省略があります。例えば第$j$項をより正確に書くと、
There is actually a slight omission in the addition notation in Formula {eq}`heisenberg`. For example, if we were to properly write out the $j$-th element, it would be as follows.

$$
I_{n-1} \otimes \dots \otimes I_{j+2} \otimes \sigma^X_{j+1} \otimes \sigma^X_{j} \otimes I_{j-1} \otimes \dots I_{0}
$$

です。ここで$\otimes$は線形演算子間の「テンソル積」を表しますが、聞き慣れない方は掛け算だと思っておいてください。重要なのは、式{eq}`heisenberg`の各項が、上で触れたように$n$個の基底演算子の積になっているということです。さらに、この系では隣接スピン間の相互作用しか存在しないため、ハミルトニアンが$n-1$個の項に分解できています。
Here, $\otimes$ represents the tensor product between the linear operators. If you are unfamiliar with the tensor product, think of it as multiplication. What is important is that each element in Formula {eq}`heisenberg` is the product of $n$ basis state operators, as mentioned above. Furthermore, in this system there is only mutual interaction between adjacent spins, so the Hamiltonian is broken down into $n-1$ elements.

この系では、隣接スピン間の向きが揃っている（内積が正）のときにエネルギーが低くなります[^quantum_inner_product]。少し考えるとわかりますが、すべてのスピンが完全に同じ方向を向いている状態が最もエネルギーの低いエネルギー固有状態です。そこで、最低エネルギー状態から少しだけずらして、スピンが一つだけ直角方向を向いている状態を始状態としたときのダイナミクスをシミュレートしてみましょう。
In this system, the amount of energy is low when the adjacent spin directions are aligned (the inner product is positive)[^quantum_inner_product]. If you think about it, this makes sense. The state when all of the spins are perfectly aligned in the same direction has the lowest energy eigenstate.  Let's simulate the dynamics using a starting state that is slightly different from the lowest energy state -- the state when the spin of just one of the particles is perpendicular to the rest.

核スピンのケースと同様に、それぞれのスピンについて+$Z$方向を向いた状態$\upket$を量子ビットの状態$\ket{0}$に、-$Z$方向の状態$\downket$を$\ket{1}$に対応づけます。このとき、上で見たように、パウリ演算子$\sigma^X, \sigma^Y, \sigma^Z$と$X, Y, Z$ゲートとが対応します。また、$J=\hbar\omega/2$とおきます。
As with the nuclear spin example, let us assign states in which the spin is in the +$Z$ direction, $\upket$, to the quantum bit state $\ket{0}$, and states in which the spin is in the -$Z$ direction, $\downket$, to the quantum bit state $\ket{1}$. Here, as we saw above, the Pauli operators $\sigma^X$, $\sigma^Y$, and $\sigma^Z$ correspond to gates $X$, $Y$, and $Z$. Let us also state that $J=\hbar\omega/2$.

時間発展演算子は
The time evolution operator is as follows.

$$
U_H(t) = \exp \left[ \frac{i\omega t}{2} \sum_{j=0}^{n-2} (\sigma^X_{j+1}\sigma^X_{j} + \sigma^Y_{j+1}\sigma^Y_{j} + \sigma^Z_{j+1} \sigma^Z_{j}) \right]
$$

ですが、ハミルトニアンの各項が互いに可換でないので、シミュレーションでは鈴木・トロッター分解を用いて近似します。各時間ステップ$\Delta t$での近似時間発展は
The Hamiltonian elements are not mutually variable, so we will use Suzuki-Trotter transformation to perform approximation in this simulation. The approximate time evolution for each time step $\Delta t$ is as follows.

$$
\tilde{U}_{H;\Delta t} = \prod_{j=0}^{n-2} \exp\left( \frac{i \omega \Delta t}{2} \sigma^X_{j+1}\sigma^X_{j} \right) \exp\left( \frac{i \omega \Delta t}{2} \sigma^Y_{j+1}\sigma^Y_{j} \right) \exp\left( \frac{i \omega \Delta t}{2} \sigma^Z_{j+1}\sigma^Z_{j} \right)
$$

です。

### 量子ゲートでの表現

これを回転ゲートと制御ゲートで表します。まず$\exp(\frac{i \omega \Delta t}{2} \sigma^Z_{j+1}\sigma^Z_{j})$について考えてみましょう。この演算子の$j$-$(j+1)$スピン系の4つの基底状態への作用は
Let us express this with a rotation gate and a controlled gate. First, let's consider $\exp(\frac{i \omega \Delta t}{2} \sigma^Z_{j+1}\sigma^Z_{j})$. This operator acts on the four basis states of a $j$-$(j+1)$ spin system as follows.

$$
\begin{align}
\upket_{j+1} \upket_{j} \rightarrow e^{i \omega \Delta t / 2} \upket_{j+1} \upket_{j} \\
\upket_{j+1} \downket_{j} \rightarrow e^{-i \omega \Delta t / 2} \upket_{j+1} \downket_{j} \\
\downket_{j+1} \upket_{j} \rightarrow e^{-i \omega \Delta t / 2} \downket_{j+1} \upket_{j} \\
\downket_{j+1} \downket_{j} \rightarrow e^{i \omega \Delta t / 2} \downket_{j+1} \downket_{j}
\end{align}
$$

です。つまり、2つのスピンの「パリティ」（同一かどうか）に応じて、かかる位相の符号が違います。
In other words, the sign of the phase is dependent on the parity of the two spins. 

パリティに関する演算をするにはCNOTを使います。例えば以下の回路
The CNOT gate is used to perform operations related to parity. For example, consider the following circuit.

[^quantum_inner_product]: これは量子力学的な系なので、もっと正確な表現は「隣接スピン間の内積が正であるようなハミルトニアンの固有状態の固有値が、そうでない固有状態の固有値より小さい」です。
[^quantum_inner_product]: This is a quantum mechanics system, so the more accurate way to express this would be to say that "the eigenvalues of the inherent states of the Hamiltonian when the inner products of adjacent spins are positive are greater than the eigenvalues of inherent states when that is not the case."

```{code-cell} ipython3
:tags: [remove-input]

circuit = QuantumCircuit(QuantumRegister(2, 'q'))
circuit.cx(0, 1)
circuit.rz(Parameter(r'-$\omega \Delta t$'), 1)
circuit.cx(0, 1)
circuit.draw('mpl')
```

によって、計算基底$\ket{00}, \ket{01}, \ket{10}, \ket{11}$はそれぞれ
This circuit transforms the computational basis states $\ket{00}$, $\ket{01}$, $\ket{10}$, and $\ket{11}$ as follows (double-check yourself).

$$
\begin{align}
\ket{00} \rightarrow e^{i \omega \Delta t / 2} \ket{00} \\
\ket{01} \rightarrow e^{-i \omega \Delta t / 2} \ket{01} \\
\ket{10} \rightarrow e^{-i \omega \Delta t / 2} \ket{10} \\
\ket{11} \rightarrow e^{i \omega \Delta t / 2} \ket{11}
\end{align}
$$

と変換するので（確認してください）、まさに$\exp(\frac{i \omega \Delta t}{2} \sigma^Z_{j+1}\sigma^Z_{j})$の表現になっています。
This is exactly as expressed with $\exp(\frac{i \omega \Delta t}{2} \sigma^Z_{j+1}\sigma^Z_{j})$.

残りの2つの演算子も同様にパリティに対する回転で表せますが、CNOTで表現できるのは$Z$方向のパリティだけなので、先にスピンを回転させる必要があります。$\exp(\frac{i \omega \Delta t}{2} \sigma^X_{j+1}\sigma^X_{j})$による変換は
The remaining two operators can also be expressed as rotation with respect to parity, but CNOT can only be used to express parity in the $Z$ direction, so the spin must first be rotated. If we perform transformation using $\exp(\frac{i \omega \Delta t}{2} \sigma^X_{j+1}\sigma^X_{j})$, we get the following.

$$
\begin{align}
\rightket_{j+1} \rightket_{j} \rightarrow e^{i \omega \Delta t / 2} \rightket_{j+1} \rightket_{j} \\
\rightket_{j+1} \leftket_{j} \rightarrow e^{-i \omega \Delta t / 2} \rightket_{j+1} \leftket_{j} \\
\leftket_{j+1} \rightket_{j} \rightarrow e^{-i \omega \Delta t / 2} \leftket_{j+1} \rightket_{j} \\
\leftket_{j+1} \leftket_{j} \rightarrow e^{i \omega \Delta t / 2} \leftket_{j+1} \leftket_{j}
\end{align}
$$

で、式{eq}`left_right_kets`から、次の回路が対応する変換を引き起こすことがわかります（これも確認してください）。
Then, given Formula {eq}`left_right_kets`, we know that we must perform the transformation that corresponds to the next circuit (double-check this yourself, as well).

```{code-cell} ipython3
:tags: [remove-input]

circuit = QuantumCircuit(QuantumRegister(2, 'q'))
circuit.h(0)
circuit.h(1)
circuit.cx(0, 1)
circuit.rz(Parameter(r'-$\omega \Delta t$'), 1)
circuit.cx(0, 1)
circuit.h(0)
circuit.h(1)
circuit.draw('mpl')
```

最後に、$\exp(\frac{i \omega \Delta t}{2} \sigma^Y_{j+1}\sigma^Y_{j})$に対応する回路は
Ultimately, the circuit that corresponds to $\exp(\frac{i \omega \Delta t}{2} \sigma^Y_{j+1}\sigma^Y_{j})$ is as follows[^sgate].

```{code-cell} ipython3
:tags: [remove-input]

circuit = QuantumCircuit(QuantumRegister(2, 'q'))
circuit.p(-np.pi / 2., 0)
circuit.p(-np.pi / 2., 1)
circuit.h(0)
circuit.h(1)
circuit.cx(0, 1)
circuit.rz(Parameter(r'-$\omega \Delta t$'), 1)
circuit.cx(0, 1)
circuit.h(0)
circuit.h(1)
circuit.p(np.pi / 2., 0)
circuit.p(np.pi / 2., 1)
circuit.draw('mpl')
```

です[^sgate]。

### 回路実装

やっと準備が整ったので、シミュレーションを実装しましょう。実機で走らせられるように、$n=5$, $M=10$, $\omega \Delta t = 0.1$とします。上で決めたように、ビット0以外が$\upket$、ビット0が$\rightket$という初期状態から始めます。各$\Delta t$ステップごとに回路のコピーをとり、それぞれのコピーで測定を行うことで、時間発展の様子を観察します。
We have finally completed our preparations, so let's implement the simulation. Let us set $n=5$, $M=10$, and $\omega \Delta t = 0.1$ so that the circuit can be run on the actual QC.  As we decided above, let us begin with all bits other than bit 0 in the $\upket$ state and bit 0 in the $\rightket$ state. We create a copy of the circuit for each $\Delta t$ step and perform measurement with each copy to observe the time evolution.

[^sgate]: $P(\pi/2)$ゲートは$S$ゲートとも呼ばれます。$P(-\pi/2)$は$S^{\dagger}$です。
[^sgate]: The $P(\pi/2)$ gate is also called an $S$ gate. $P(-\pi/2)$ is $S^{\dagger}$.

```{code-cell} ipython3
# まずは全てインポート
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.tools.monitor import job_monitor
from qiskit_aer import AerSimulator
from qiskit_ibm_provider import IBMProvider, least_busy
from qiskit_ibm_provider.accounts import AccountNotFoundError
# このワークブック独自のモジュール
from qc_workbook.dynamics import plot_heisenberg_spins
from qc_workbook.utils import operational_backend
```

```{code-cell} ipython3
n_spins = 5
M = 10
omegadt = 0.1

circuits = []

circuit = QuantumCircuit(n_spins)

# 第0ビットを 1/√2 (|0> + |1>) にする
circuit.h(0)

# Δtでの時間発展をM回繰り返すループ
for istep in range(M):
    # ハミルトニアンのn-1個の項への分解に関するループ
    for jspin in range(n_spins - 1):
        # ZZ
        circuit.cx(jspin, jspin + 1)
        circuit.rz(-omegadt, jspin + 1)
        circuit.cx(jspin, jspin + 1)

        # XX
        circuit.h(jspin)
        circuit.h(jspin + 1)
        circuit.cx(jspin, jspin + 1)
        circuit.rz(-omegadt, jspin + 1)
        circuit.cx(jspin, jspin + 1)
        circuit.h(jspin)
        circuit.h(jspin + 1)

        # YY
        circuit.p(-np.pi / 2., jspin)
        circuit.p(-np.pi / 2., jspin + 1)
        circuit.h(jspin)
        circuit.h(jspin + 1)
        circuit.cx(jspin, jspin + 1)
        circuit.rz(-omegadt, jspin + 1)
        circuit.cx(jspin, jspin + 1)
        circuit.h(jspin)
        circuit.h(jspin + 1)
        circuit.p(np.pi / 2., jspin)
        circuit.p(np.pi / 2., jspin + 1)

    # この時点での回路のコピーをリストに保存
    # measure_all(inplace=False) はここまでの回路のコピーに測定を足したものを返す
    circuits.append(circuit.measure_all(inplace=False))

print(f'{len(circuits)} circuits created')
```

量子回路シミュレーターで実行し、各ビットにおける$Z$方向スピンの期待値をプロットしましょう。プロット用の関数は比較的長くなってしまいますが実習の本質とそこまで関係しないので、[別ファイル](https://github.com/UTokyo-ICEPP/qc-workbook/blob/master/source/utils/dynamics.py)に定義してあります。関数はジョブの実行結果、系のスピンの数、初期状態、ステップ間隔を引数にとります。
Let's run this on a quantum circuit simulator and plot the expectation values of the $Z$ direction spins of each bit. The functions to be plotted are relatively long, but they aren't that important to the key points of this exercise, so they are defined in a [separate file](https://github.com/UTokyo-ICEPP/qc-workbook/blob/master/source/utils/dynamics.py). The functions use as their arguments the job execution results, the spin numbers of the system, the initial state, and the step spacing.

```{code-cell} ipython3
# 初期状態 |0> x |0> x |0> x |0> x 1/√2(|0>+|1>) は配列では [1/√2 1/√2 0 0 ...]
initial_state = np.zeros(2 ** n_spins, dtype=np.complex128)
initial_state[0:2] = np.sqrt(0.5)

shots = 100000

simulator = AerSimulator()

circuits_sim = transpile(circuits, backend=simulator)
sim_job = simulator.run(circuits_sim, shots=shots)
sim_counts_list = sim_job.result().get_counts()

plot_heisenberg_spins(sim_counts_list, n_spins, initial_state, omegadt, add_theory_curve=True)
```

ビット0でのスピンの不整合が徐々に他のビットに伝搬していく様子が観察できました。
As you can see, the mismatch of the spin of bit 0 gradually propagates to the other bits.

また、上のように関数`plot_heisenberg_spins`に`add_theory_curve=True`という引数を渡すと、ハミルトニアンを対角化して計算した厳密解のカーブも同時にプロットします。トロッター分解による解が、厳密解から少しずつずれていっている様子も観察できます。興味があれば$\Delta t$を小さく（$M$を大きく）して、ずれがどう変わるか確認してみてください。
If the `add_theory_curve=True` argument is passed to the `plot_heisenberg_spins` function, as shown above, the curves of the exact solution, calculated after diagonalizing the Hamiltonian, can also be plotted. As you can see, the solution produced by the Suzuki-Trotter transformation is slightly off from the exact solution at each point. If you are interested, try making the $\Delta t$ smaller (making the $M$ larger) and check how this changes the amount of deviation from the exact solution.

実機でも同様の結果が得られるか確認してみましょう。
Let's confirm whether you get the same results when using the actual QC.

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

# よりアクセス権の広いプロバイダを使える場合は、下を書き換える
instance = 'ibm-q/open/main'

try:
    provider = IBMProvider(instance=instance)
except IBMQAccountCredentialsNotFound:
    provider = IBMProvider(token='__paste_your_token_here__', instance=instance)

backend_list = provider.backends(filters=operational_backend(min_qubits=n_spins, min_qv=32))
backend = least_busy(backend_list)

print(f'Job will run on {backend.name()}')
```

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

circuits_ibmq = transpile(circuits, backend=backend)

job = backend.run(circuits_ibmq, shots=8192)

job_monitor(job, interval=2)

counts_list = job.result().get_counts()
```

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

plot_heisenberg_spins(counts_list, n_spins, initial_state, omegadt)
```
