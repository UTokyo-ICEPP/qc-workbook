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

# 【課題】位相推定によるスペクトル分解

```{contents} 目次
---
local: true
---
```

$\newcommand{\ket}[1]{|#1\rangle}$

+++

## エネルギースペクトルを求める

{doc}`後の実習 <vqe>`にも現れるように、物理学や化学において、系のエネルギー固有値（スペクトル）と対応する固有状態を求めるということは非常に重要な問題です。系のエネルギースペクトルを求めることとは、系を表すハミルトニアンを決め、それを対角化することに対応します。
In physics and chemistry, determining the eigenstates that correspond to the energy eigenvalues (spectra) of systems is an extremely important task, an example of which appears in a {doc}`later exercise<vqe>`. To determine the energy spectrum of a system, you decide on the Hamiltonian that expresses the system and then diagonalize it.

しかし、{doc}`前回の実習 <dynamics_simulation>`でも出てきたように、一般的に量子系の次元数は莫大で、ハミルトニアンの対角化の肝となる逆行列の計算をまともに行うことはできません。同実習では、そのような場合でも、ハミルトニアンが効率的な鈴木・トロッター分解を許す形であれば、量子コンピュータを使って時間発展シミュレーションを行えるということを示しました。ただし、このシミュレーションでは、系のエネルギー固有値や固有状態を明示的に利用しませんでした。
However, as we saw in the {doc}`previous exercise<dynamics_simulation>`, the number of dimensions in a typical quantum system is extremely large, so we cannot properly perform the inverse matrix calculations that are the key to diagonalizing the Hamiltonian. In that exercise, we found that even in that case, if we could perform efficient Suzuki-trotter transformation of the Hamiltonian, we could use a quantum computer to perform a time evolution simulation. However, in this simulation, the system's energy eigenvalues and eigenstates cannot be explicitly used.

実は、同じ時間発展シミュレーションの手法と、{doc}`shor`で登場した位相推定の手法を組み合わせると、エネルギー固有値も数値的に求めることができます{cite}`Aspuru-Guzik1704`。さらに、工夫によっては対応する固有状態も調べられます。ここでは外部磁場のかかったハイゼンベルグモデルを題材に、位相推定によるエネルギースペクトル分解をしてみましょう。
We can, though, combine the same time evolution simulation methods with the phase estimation method that appeared in {doc}`"Learning about the prime factorization algorithm"` to quantitatively determine the energy eigenvalues{cite}`Aspuru-Guzik1704`. We can also apply a little ingenuity to investigate the eigenstates. In this assignment, we will use phase estimation to perform energy spectrum factorization on a Heisenberg model with an external magnetic field.

+++

## ハイゼンベルグモデル再考

前回登場したハイゼンベルグモデルのハミルトニアンは
The Hamiltonian of the Heisenberg model, introduced in the previous section, is as follows.

$$
H = -J \sum_{j=0}^{n-2} (\sigma^X_{j+1}\sigma^X_{j} + \sigma^Y_{j+1}\sigma^Y_{j} + \sigma^Z_{j+1} \sigma^Z_{j}) \quad (J > 0)
$$

というものでした。このハミルトニアンが表しているのは、空間中で一列に並んだスピンを持つ粒子が、隣接粒子間で相互作用を及ぼしているような系でした。ここで相互作用は、スピンの向きが揃っているときにエネルギーが低くなるようなものでした。したがって、全てののスピンが同じ方向を向いているときにエネルギーが最も低くなることが予想されました。
This Hamiltonian represents a system in which there is a line of particles with spins, suspended in space, and adjacent particles mutually interact. In this mutual interaction, when spins are aligned, the amount of energy is low. Therefore, if all of the spins were aligned, we would expect the energy to be at its minimum.

今回は、このハミルトニアンに外部からの磁場の影響を入れます。外部磁場がある時は、スピンが磁場の方向を向いているときにエネルギーが低くなります。したがって、外部磁場を$+Z$方向にかけるとすれば、ハミルトニアンは
In this assignment, we will apply an external magnetic field to the Hamiltonian. When there is an external magnetic field, the level of energy is low when the spin is aligned with the magnetic field. Therefore, if we apply the external magnetic field in the +$Z$ direction, the Hamiltonian is as follows.

$$
H = -J \sum_{j=0}^{n-1} (\sigma^X_{j+1}\sigma^X_{j} + \sigma^Y_{j+1}\sigma^Y_{j} + \sigma^Z_{j+1} \sigma^Z_{j} + g \sigma^Z_j)
$$

となります。このハミルトニアンにはもう一点前回と異なる部分があります。前回はスピンに関する和を$j=0$から$n-2$まで取ることで、両端のスピンは「内側」のスピンとしか相互作用をしないような境界条件を採用していました。今回は和を$n-1$まで取っています。$\sigma^{X,Y,Z}_n$を$\sigma^{X,Y,Z}_0$と同一視することで、これは「周期境界条件」（一列ではなく環状に並んだスピン）を表します。
There is another difference between this Hamiltonian and the Hamiltonian from the previous section. Previously, we took the sum of the spins from $j=0$ to $n-2$, applying border conditions such that the spins of each particle on an end of the string of particles only interacted with the one particle next to it on the "inner side" of the string. This time, we will take the sum all the way to $n-1$. If we treat $\sigma^{X,Y,Z}_n$ the same as $\sigma^{X,Y,Z}_0$, this is a "periodic boundary condition" (the spin in a ring, instead of a string).

このようなハミルトニアンの固有値や固有状態がどのような値を取るのか、具体例で確認しましょう。最も簡単な$n=2, g=0$というケースを考え、直接対角化して厳密解を求めます。
Let's look at a specific example at possible values of the Hamiltonian's eigenvalues and eigenstates. Let's look at the simplest example, $n=2$, $g=0$, and directly diagonalize it to determine the exact solution.

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

最後の部分で、[`show_state`関数](https://github.com/UTokyo-ICEPP/qc-workbook/tree/master/source/qc_workbook/show_state.py)を利用して固有値と固有ベクトルを表示しました。最低エネルギー状態（固有値$-2J$）に対応する独立な固有ベクトルが3つあることがわかります。したがって、これらの固有ベクトルの任意の線形和もまた最低エネルギー状態です。励起状態（固有値$6J$）は$1/\sqrt{2} (-\ket{01} + \ket{10})$です。
In the last section, the [`show_state` function](https://github.com/UTokyo-ICEPP/qc-workbook/tree/master/source/qc_workbook/show_state.py) was used to show the eigenvalues and eigenvectors. We can see that there are three independent eigenvectors that correspond to the lowest energy state (eigenvalue $-2J$). Therefore, an arbitrary linear sum of these eigenvectors is also the lowest energy state. The excited state (eigenvalue $6J$) is $1/\sqrt{2} (-\ket{01} + \ket{10})$.

+++

## 位相推定によるスペクトル分解

それでは本題に入りましょう。{doc}`shor`で登場した下の図において、$U$が何らかのハミルトニアン$H$による適当な時間$\tau$の時間発展演算子$U_H(-\tau)$である場合、何が言えるでしょうか。
Let's now begin the main part of this exercise. In the figure below, which you saw in "Learning about the prime factorization algorithm," what can you conclude when $U$ is the time evolution operator $U_H(-\tau)$ for a time $\tau$ for Hamiltonian $H$?

```{image} figs/qpe_wo_iqft.png
:alt: qpe_wo_iqft
:width: 500px
:align: center
```

以下では、図中上のレジスタ（初期状態$\ket{0}$）を読み出し(readout)レジスタR、下のレジスタ（初期状態$\ket{\psi}$）を状態(state)レジスタSと呼びます。RとSのビット数をそれぞれ$n_R$, $n_S$とします。また、図では読み出しレジスタの最低位（1の位）に対応するビットが一番下に描かれており、Qiskitでの描画法と逆になっていることに注意してください。
Below, we will refer to the upper register in the figure above (initial state $\ket{0}$) as register R (for "readout"), and the lower register (initial state $\ket{\psi}$) as register S (for "state"). The number of bits for registers R and S are, respectively, $n_R$ and $n_S$. Furthermore, note that the bit that corresponds to the lowest place in the readout register in the figure (the 1s place) is written at bottom, the opposite of Qiskit's notation approach.

さて、ハミルトニアン$H$を、エネルギーの次元を持つ定数$\hbar \omega$と無次元のエルミート演算子$\Theta$とに分解します。
Now, let's break the Hamiltonian $H$ into the dimensional energy constant $\hbar \omega$ and the dimensionless Hermitian operator $\Theta$.

$$
H = \hbar \omega \Theta.
$$

ここで$\omega$はどう選んでも構いません。$\omega$を$x$倍大きく取れば、その分$\Theta$に$1/x$の係数をかけるだけです。実際には、後述するように$\Theta$の固有値の絶対値が1より少しだけ小さくなるように$\omega$を決めます。この書き換えにより
Here, we can select anything for $\omega$. If we increase $\omega$ $x$-fold, we can simply multiply $\Theta$ by $1/x$. In reality, as we see later, we will choose a $\omega$ such that the absolute value of $\Theta$'s eigenvalue is slightly greater than 1. The formula can thus be rewritten as follows.

$$
U_H(-\tau) \ket{\psi} = \exp\left(i\omega \tau \Theta\right) \ket{\psi}
$$

なので、図の回路に対応する演算子を$\Gamma$とおくと、
If we indicate the operator for the circuit in the figure as $\Gamma$, we arrive at the following.

$$
\Gamma \ket{0}_R \ket{\psi}_S = \frac{1}{\sqrt{2^{n_R}}} \sum_{j=0}^{2^{n_R} - 1} \exp\left(i j \omega \tau \Theta\right) \ket{j}_R \ket{\psi}_S
$$

です。この状態に対して、実習と同じように逆フーリエ変換を施します。
Then, as in the exercise, we apply an inverse Fourier transform.

$$
\text{QFT}^{\dagger}_R \Gamma \ket{0}_R \ket{\psi}_S = \frac{1}{2^{n_R}} \sum_{k=0}^{2^{n_R} - 1} \sum_{j=0}^{2^{n_R} - 1} \exp(i j \omega \tau \Theta) \exp\left(-\frac{2 \pi i j k}{2^{n_R}}\right) \ket{k}_R \ket{\psi}_S.
$$

さて、ここまで$\tau$は「適当な時間」としか決めて来ませんでしたが、実際にどんな値をとってもいいので、$\omega \tau = 2 \pi$となるように決めてしまいましょう。すると
So far, we have simply stated that $\tau$ is "a time," but it can actually have any value, so let us select a value for $\tau$ such that $\omega \tau = 2 \pi$. This produces the following.

$$
\text{QFT}^{\dagger}_R \Gamma \ket{0}_R \ket{\psi}_S = \frac{1}{2^{n_R}} \sum_{k=0}^{2^{n_R} - 1} \sum_{j=0}^{2^{n_R} - 1} \exp\left[\frac{2 \pi i j}{2^{n_R}} \left(2^{n_R} \Theta - k\right)\right] \ket{k}_R \ket{\psi}_S
$$

となります。したがって、$\Theta$の固有ベクトル$\{\ket{\phi_m}\}$で$\ket{\psi}$が
Therefore, using $\Theta$'s eigenvector $\{\ket{\phi_m}\}$, $\ket{\psi}$ can be written as shown below.

```{math}
:label: spectral_decomposition
\ket{\psi} = \sum_{m=0}^{2^{n_S} - 1} \psi_m \ket{\phi_m}
```

と書けるなら、対応する固有値を$\{\theta_m\}$とし、$\kappa_m = 2^{n_R} \theta_m$を用いて
Thus, using $\{\theta_m\}$ as the corresponding eigenvalue, we can use $\kappa_m = 2^{n_R} \theta_m$ to arrive at the following.

```{math}
:label: spectrum_estimation_final
\begin{align}
\text{QFT}^{\dagger}_R \Gamma \ket{0}_R \ket{\psi}_S & = \frac{1}{2^{n_R}} \sum_{k=0}^{2^{n_R} - 1} \sum_{j=0}^{2^{n_R} - 1} \sum_{m=0}^{2^{n_S} - 1} \psi_m \exp\left[\frac{2 \pi i j}{2^{n_R}} (\kappa_m - k)\right] \ket{k}_R \ket{\phi_m}_S \\
& = \sum_{k=0}^{2^{n_R} - 1} \sum_{m=0}^{2^{n_S} - 1} \psi_m f(\kappa_m - k) \ket{k}_R \ket{\phi_m}_S.
\end{align}
```

ただし、二つめの等号で、$f(\kappa_m - k) := \frac{1}{2^{n_R}} \sum_{j} \exp \left[2 \pi i j (\kappa_m - k) / 2^{n_R}\right]$と定義しました。
Here, the second equation operator is defined as $f(\kappa_m - k) := \frac{1}{2^{n_R}} \sum_{j} \exp \left[2 \pi i j (\kappa_m - k) / 2^{n_R}\right]$.

最後にこの状態を測定し、Rに現れるビット列から推定される$\theta_m = 2^{-n_R} \kappa_m$に$\hbar \omega$をかければ、$H$のエネルギー固有値がわかります。
Finally, we measure the state and multiply $\theta_m = 2^{-n_R} \kappa_m$, estimated from the bit sequence that appears in R, by $\hbar \omega$ to determine the energy eigenvalue of $H$.

なんだか掴みどころのない$\omega$や$\tau$といったパラメータが登場して、結局何をしたのか分かりにくいという方のために、別の見方で問題を整理しましょう。結局上で行ったことは、ハミルトニアン$H$を与えられたときに、
These $\omega$ and $\tau$ parameters may seem a bit slippery, and it may be hard to figure out what exactly we did, so let's look at the problem again from another perspective. Ultimately, what we did above was perform the following when given Hamiltonian $H$.

1. 固有値が$\lesssim 1$（負の値があり得る場合は、その絶対値が$\lesssim \frac{1}{2}$）になるように$H$を規格化する（規格化の定数は記録しておく）
2. 規格化した結果の演算子を$\Theta$として、$U = \exp(-2 \pi i \Theta)$の位相推定をする
3. 位相推定から得られた$\Theta$の固有値に、1.の規格化定数をかけてエネルギー固有値を得る

4. Standardize $H$ such that the eigenvalue is $\lesssim 1$ (when the value could be negative, such that the absolute value is $\lesssim \frac{1}{2}$) (record the constant when normalizing $H$).
5. Given $\Theta$ as the operator of the normalized results, perform phase estimation of $U = \exp(-2 \pi i \Theta)$. 
6. Multiply the eigenvalue of $\Theta$ determined through phase estimation by the normalization constant from step 1 to arrive at the energy eigenvalue.

という操作でした。ここで、$\Theta$の固有値を上のように決めるのは、読み出しレジスタに現れる固有値の数値が{ref}`オーバーフロー <signed_binary>`を起こさないようにするためです。
Here, we define the eigenvalue of $\Theta$ as above to prevent the value of the eigenvalue that appears in the readout register from {ref}`overflowing<signed_binary>`.

固有値を求める問題なのに固有値が具体的な値になるように規格化定数を選ぶというのは一見矛盾しているように聞こえます。しかし、{doc}`dynamics_simulation`で触れたように、量子コンピューティングで表現されるハミルトニアンは全て基底演算子${I, \sigma^X, \sigma^Y, \sigma^Z}$の積の線形和に分解できます。個々の基底演算子の積の固有値は$\pm 1$なので、ハミルトニアン$H$が基底演算子の積$\sigma_k$とエネルギー係数$h_k$で
Because we are trying to determine the eigenvalue in this problem, it might appear contradictory to select a normalization constant such that the eigenvalue is a specific value. However, as we touched on in {doc}`dynamics_simulation`, the Hamiltonians described by quantum computing can all be broken down into the linear sum of the products of basis state operators ${I, \sigma^X, \sigma^Y, \sigma^Z}$. The eigenvalue of the products of the individual basis state operators is $\pm 1$, so if we can break down Hamiltonian $H$ into the product of the basis state operators, $\sigma_k$, and the energy coefficient $h_k$, as follows:

$$
H = \sum_{k} h_k \sigma_k
$$

と分解できるとすれば、$H$の固有値の絶対値は最大で$\sum_{k} |h_k|$です。したがって、全く未知のハミルトニアンに対しても、まず規格化定数を$\hbar \omega = 2 \sum_{k} |h_k|$と取ればいいことがわかります。スペクトル推定の結果、最大固有値がもっと小さい値であれば、規格化定数を調節して再度計算を行えばいいだけです。
...then the absolute value of the eigenvalue of $H$ has a maximum value of $\sum_{k} |h_k|$. Therefore, as we can see, even for a completely unknown Hamiltonian, we can set $\hbar \omega = 2 \sum_{k} |h_k|$ as the normalization constant. If, as the result of spectral estimation, the maximum eigenvalue is a smaller value, we can simply adjust the normalization constant and perform the calculation again.

同じ理屈で読み出しレジスタRのビット数も決めることができます。今度は非ゼロで最も小さい固有値の絶対値を考えます。最小値は
The same logic is used in deciding the number of bits of readout register R. Now let's think about the smallest non-zero absolute eigenvalue. The smallest value is equal to or greater than the following.

$$
\mu = \min_{s_k = \pm 1} \left| \sum_{k} s_k h_k \right|
$$

以上です。この値を求めるには原理的には$2^{L}$（$L$はハミルトニアンの項数）通りの組み合わせを調べる必要がありますが、実際のハミルトニアンは通常そこまで複雑な形をしていないので、これも難しくないと考えられます。レジスタRのビット数$n_R$は$2^{n_R}\mu/(\hbar \omega)$が読みだせるように設定します。
Theoretically, we would have to investigate $2^{L}$ ($L$ here is the number of Hamiltonian terms) combinations to determine this value. However, actual Hamiltonians aren't normally that complex, so this is unlikely to be difficult. Let us set $n_R$, the number of bits in register R, so that $2^{n_R}\mu/(\hbar \omega)$ can be read from it.

これで規格化定数と読み出しレジスタの大きさが簡単に決められることがわかりましたが、問題は演算子$U$を作用させる対象の状態$\ket{\psi}$です。第$m$準位のエネルギー固有値を知りたい場合、式{eq}`spectral_decomposition`で$\psi_m \neq 0$でなければいけません。特殊なケースではスペクトル推定の前から固有ベクトルの「当たり」がついていることがあるかもしれませんが、当然、一般のハミルトニアンに対して、任意の$m$についてそのような状態を準備することはできません。
This has made it easy to decide the normalization constant and the size of the readout register. The problem is the state $\ket{\psi}$ on which to use operator $U$. If we wish to know the $m$-th level energy eigenvalue, in Formula{eq}`spectral_decomposition`, $\psi_m \neq 0$ must be true. In special cases, we may have an idea of the eigenvector before even performing spectral estimation, but, needless to say, for typical Hamiltonians, we cannot prepare those states for arbitrary values of $m$.

一方、系の最低エネルギー$\hbar \omega \theta_0$なら、{doc}`vqe`の手法などを用いて最低エネルギー状態の近似を実現し、それをSの初期状態とすることで、比較的に安定して求めることは可能です。そのため、上の手法は原理的にはスペクトルの完全な分解に使えますが、実際には最低エネルギーとその固有ベクトルを正確に求めるために利用することが多いと考えられます。
On the other hand, for the minimum system energy, $\hbar \omega \theta_0$, methods such as those presented in {doc}`vqe` can be used to approximate the minimum energy state. If this were set as the initial state of S, it would be possible to determine the value with a relatively high level of stability. Therefore, the above method can, in principle, be used to completely break down spectra, but in reality it is most commonly used to accurately determine the minimum energy and its eigenvector.

+++

## 問題1: スペクトル推定を実装し、厳密解と照らし合わせる

それでは、位相推定を利用してハイゼンベルグモデルのハミルトニアンのスペクトルを求めてみましょう。
Let us now use phase estimation to determine the spectrum of the Hamiltonian of the Heisenberg model.

$U_H(-\tau)$を量子コンピュータ上で計算するために、鈴木・トロッター分解を使います。$ZZ, XX, YY$回転ゲートの実装法は{doc}`dynamics_simulation`を参照してください。
We will use Suzuki-trotter transformation to calculate $U_H(-\tau)$ on a quantum computer. Refer to {doc}`dynamics_simulation` for information on how to implement rotation gates $ZZ$, $XX$, and $YY$.

次のセルでは$U_H(-2\pi/\omega)$を鈴木・トロッター分解した量子回路を返す関数を定義しています。引数`num_steps`で分解の細かさを指定できます。
The next cell defines a function that returns a function that corresponds to one step of the Suzuki-trotter transformation (called a "Trotter step"). The structure is a bit complex, but by using this approach, each returned function can take the phase for one step as an argument, which makes it possible to think of the physics of the Heisenberg model separately from the spectral estimation algorithm. The arguments of the outer function, n, g, and hbar_omega can also be used in the inner function.

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

次のセルでスペクトル推定のアルゴリズムを実装しています。この関数は状態レジスタ、読み出しレジスタ、時間発展回路を引数に取り、スペクトル推定の量子回路を返します。
In the next cell, we implement the spectral estimation algorithm. We define the function propagator, which returns the gate that corresponds to a power of U and uses as its arguments the function that implements the Trotter step (the function returned by the function above) and two integers. The spectrum_estimation function uses propagator to perform spectral estimation, taking as its arguments the circuit object and the Trotter step function.

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

この問題では、上で厳密解を求めた$n=2, g=0$のケースを調べます。今回はすでにエネルギー固有値を知っているので、ハミルトニアンの規格化定数を$\hbar \omega = 16J$として、読み出しレジスタの終状態が単純になるようにします。このとき読み出しは符号付きで、最大絶対値が$2^{n_R} (6/16)$なので、$n_R = 1 + 3$とすればオーバーフローを回避できます。
First, let's look at the $n=2$, $g=0$ case, for which we determined the exact solution above. 

次のセルでシミュレーションとスペクトル推定のパラメータを設定します。

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

上で正しく関数をかけているか確認しておきましょう。

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

状態レジスタの初期状態を
Let us begin with the following initial state for the state register.

```{math}
:label: two_qubit_init
\frac{1}{2}\ket{00} - \frac{1}{\sqrt{2}}\ket{01} + \frac{1}{2} \ket{11} = \frac{1}{2} \ket{\phi_0} + \frac{1}{2} \ket{\phi_1} + \frac{1}{2} \ket{\phi_2} + \frac{1}{2} \ket{\phi_3}
```

とする関数を書きます。ここで$\ket{\phi_i}$は最初に求めた固有ベクトルの4つの厳密解です。
Here, $\ket{\phi_i}$ are the four exact solutions for the eigenvectors we initially determined.

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

最後に全てを組み合わせます。

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

シミュレータで実行してヒストグラムを得ます。

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

状態レジスタの初期状態が式{eq}`two_qubit_init`なので、回路の終状態は
The initial state of the state register is shown in Formula{eq}`two_qubit_init`, so the final state of the circuit should be as follows.

$$
\frac{1}{2} \ket{-2}_{R} \ket{00}_{S} - \frac{1}{2\sqrt{2}} \ket{-2}_{R} \left( \ket{01}_{S} + \ket{10}_{S} \right) + \frac{1}{2} \ket{-2}_{R} \ket{11}_{S} - \frac{1}{2\sqrt{2}} \ket{6}_{R} \left( \ket{01}_{S} - \ket{10}_{S} \right)
$$

となるはずです。得られたヒストグラムはこの状態と無矛盾でしょうか？
Does the histogram that is produced conflict with this?

**提出するもの**

- 完成した`make_trotter_step_heisenberg`関数
- 完成した状態レジスタの初期化回路
- スペクトル推定の結果のヒストグラムと、その解釈

**Items to submit**:

- Completed make_trotter_step_heisenberg function 
- Completed state register initialization circuit 
- Histogram and explanation of spectral estimation results

+++

## 問題2: 非自明な系の振る舞いを調べる

今度は$n=4$のハイゼンベルグモデルで、全エネルギースペクトルを$g$の関数として求めてみましょう。$n=4$なので最初にやったような対角化もまだ簡単にできますが、ここではあえて量子計算のみに頼ってみます。
Next, let's determine all of the energy spectra of an $n=4$ Heisenberg model as a function of $g$. $n=4$, so as before, diagonalization is still easy, but this time, let's try relying on quantum calculation alone.

全てのエネルギー固有値を知るためにはSの初期状態を工夫しなければいけませんが、今回は事前の情報がないので、しらみつぶし戦略をとります。つまり、計算基底$\ket{0}$から$\ket{15}$までをそれぞれ初期状態としてスペクトル推定を繰り返し、得られる情報をつなぎ合わせて全体のスペクトルを求めます。
To discover all of the energy eigenvalues, we need to apply some ingenuity to the initial state of $S$. This time, we don't have any preliminary information, so we'll use an exhaustive search strategy. That is, we will use computational basis states $\ket{0}$ to $\ket{15}$ as initial states and perform spectral estimation for each. We will then link the resulting information to determine the overall spectrum.

全ての計算基底についてスペクトル推定をすると、どんな情報が得られるでしょうか。式{eq}`spectrum_estimation_final`で$\ket{\psi} = \ket{l}$ $(l=0,\dots,2^{n_S} - 1)$とおき、
If we perform spectral estimation for every computational basis, what kind of information will we produce? In Formula{eq}`spectrum_estimation_final`, let us set $\ket{\psi} = \ket{l}$ $(l=0,\dots,2^{n_S} - 1)$.

```{math}
:label: phim_decomposition
\ket{l} = \sum_{m=0}^{2^{n_S} - 1} c^l_m \ket{\phi_m}
```

とします。すると回路の終状態は
The final state of this circuit will then be as follows.

$$
\sum_{k=0}^{2^{n_R} - 1} \sum_{m=0}^{2^{n_S} - 1} c^l_m f(\kappa_m - k) \ket{k}_R \ket{\phi_m}_S
$$

となります。実は式{eq}`phim_decomposition`が成り立つとき
When Formula{eq}`phim_decomposition` is true, the following will also be true[^unitarity].

```{math}
:label: l_decomposition
\ket{\phi_m} = \sum_{l=0}^{2^{n_S} - 1} c^{l*}_m \ket{l}
```

も成り立つ[^unitarity]ので、終状態の回路をRとSの計算基底で測定し、$k, h$を得る確率を$P_l(k, h)$とすると
If we measure the computational basis of R and S for the circuit in its final state, if we define $P_l(k, h)$ as the probabilities of $k$ and $h$, we arrive at the following.

$$
P_l(k, h) = \left| \sum_{m=0}^{2^{n_S} - 1} c^l_m c^{h*}_m f(\kappa_m - k) \right|^2
$$

となります。$c^l_m$の値がわからなくても、これらの分布から
Even if we do not know the value of $c_m^l$, from this distribution, let us think of a method for producing the following.

$$
P(k) = \frac{1}{2^{n_S}} \sum_{m=0}^{2^{n_S} - 1} |f(\kappa_m - k)|^2
$$

を得る方法を考えてみてください。

$|f(\kappa_m - k)|$は$\kappa_m$近傍で鋭いピークを持つ分布なので、$P(k)$を$k$に対してプロットすれば、$m$個の（部分的に重なり合っている可能性のある）ピークが見られ、そこからエネルギー固有値が算出できます。
The distribution of $|f(\kappa_m - k)|$ has a sharp peak near $\kappa_m$, so if $P(k)$ is plotted with respect to $k$, we will observe $m$ peaks (with the possibility of partial overlaps). We can then calculate the energy eigenvalues from this.

試しに$n=2, g=0, \hbar \omega = 20J, n_R=4$で$P(k)$をプロットすると以下のようになります（問題1と異なり$\hbar \omega = 20J$なので、$\kappa_m$が整数ではありません）。ただし、負の固有値がわかりやすいように、$P(k - 2^{n_R}) = P(k)$として、$-2^{n_R - 1} \leq k < 2^{n_R - 1}$の範囲を表示しています。
Plotting $P(k)$ for $n=2$, $g=0$, $\hbar \omega = 20J$,$n_R=4$ would produce the following (unlike task 1, $\hbar \omega = 20J$な, so $\kappa_m$ is not an integer. To make it easy to determine negative eigenvalues, we set $P(k - 2^{n_R}) = P(k)$ and show the range of $-2^{n_R - 1} \leq k < 2^{n_R - 1}$.

```{image} figs/spectrum_estimation_example.png
:alt: spectrum_estimation_example
:width: 500px
:align: center
```

このようなプロットを$n=4$で$g$の値を0から0.5まで0.1刻みに変えながら作ってみましょう。
Let's create plots like this with $n=4$ and $g$ values in 0.1 increments, from 0 to 0.5.

まずは計算基底と$g$の値を引数に取り、終状態の確率分布を返す関数を定義します。通常のショットベースのシミュレータでは統計誤差の影響が乗るので、デフォルトでは状態ベクトルシミュレーションを使うことにします。
First, we define a function for returning the probability distribution of the final state, using the computational basis and value of $g$ as its arguments. Using the qasm_simulator would be a closer approximation to actual usage, but due to problems with running time and statistical error, we'll use the default statevector_simulator.

[^unitarity]: これは$\{\ket{l}\}$と$\{\ket{\phi_m}\}$がともに状態レジスタの正規直交基底を張る（変換行列がユニタリである）ことに起因します。
[^unitarity]: This is because both $\{\ket{l}\}$ and $\{\ket{\phi_m}\}$ have orthonormal state register basis states (unitary transformation matrices).

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

読み出しレジスタのビット数を決めます。スピンの数が4なので、$\hbar \omega = 8(3 + |g|)J$と取ります。すると、$g=0$のとき、上の議論によると$\Theta$の固有値の予想される最小絶対値は$1/24$ですが、実は系の対称性からその$n=4$倍の$1/6$が最小値になると予測できます。$|g| \ll 1$しか考えないので、外部磁場を摂動として考えて、結局$2^{n_R} / 6$が1より十分大きくなるように$n_R=5$とします。
Let's decide the number of bits in the readout register. The number of spins is 4, so we will use $\hbar \omega = 8(3 + |g|)J$. When $g=0$, based on the above theory, the smallest absolute value that we can expect the $\Theta$ eigenvalue to have is $1/24$, but in reality, due to the symmetry of the system, we predict the smallest value to be four times that, so $n=1/6$. $|g| \ll 1$ must be true, so if we think of the external magnetic field as a perturbation, ultimately $n_R=5$ so that $2^{n_R} / 6$ is sufficiently greater than 1.

回路のパラメータが決まったので、$g$を引数として$2^{n}$通りの計算基底に対して`get_spectrum_for_comp_basis`関数を呼ぶ関数を定義し、$g=0$について実行します（時間がかかります）。
Now that we have decided on the circuit's parameters, we define a function that calls the function `get_spectrum_for_comp_basis` for $2^n$ computational basis states, using $g$ as an argument, and we execute the function for $g=0$ (this will take some time).

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

得られた$P(k)$分布を、$k$をエネルギーに換算してプロットしてみましょう。
Now let's convert $k$ into energy and plot the resulting $P(k)$ distribution.

```{code-cell} ipython3
:tags: [remove-output]

plt.plot(np.linspace(-0.5 / energy_norm, 0.5 / energy_norm, 2 ** n_readout), spectra[0], 'o')
plt.xlabel('E/J')
plt.ylabel('P(E)')
```

続いて、同じ関数を$g=0.1, 0.2, 0.3, 0.4, 0.5$について実行して、それぞれのスペクトルから系のエネルギー固有値と$g$の関係をプロットしてください。
Next, let's execute the same function for $g=0.1$, 0.2, 0.3, 0.4, and 0.5, and plot the relationships between the system's energy eigenvalues and $g$ based on each spectrum.

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

**提出するもの**

- $P_l(k, h)$から$P(k)$を導出する方法の説明と、`get_full_spectrum`関数への実装
- $P(k)$からエネルギー固有値を抽出する方法を考案し、実装したコード（エネルギー固有値を求める全く別の方法を考えつけば、それでもよい）
- 16個のエネルギー固有値と$g$の関係のプロット

**Items to submit**:

- An explanation of how to derive $P(k)$ from $P_l(k, h)$ and its implementation in the `get_full_spectrum` function
- Implementation code for extracting energy eigenvalues from $P(k)$ (if you can think of a totally different method for determining energy eigenvalues, you may submit that instead)
- A plot of the relationship between the 16 energy eigenvalues and $g$


**HInt**:

（$P(k)$の導出に関して）
式{eq}`phim_decomposition`と{eq}`l_decomposition`を眺めると、
(With regard to deriving $P(k)$) If you look at Formulas{eq}`phim_decomposition` and {eq}`l_decomposition`, you will see that the following is true.

$$
\sum_{l=0}^{2^{n_S} - 1} c^l_m c^{l*}_n = \delta_{mn}
$$

が成り立つことがわかります。ここで$\delta_{mn}$はクロネッカーの$\delta$記号で、$m=n$のとき1、それ以外では0の値を持つ因子です。
Here, $\delta_{mn}$ is a factor using Kronecker's $\delta$ symbol which is 1 when $m=n$ is true and 0 when it is not.