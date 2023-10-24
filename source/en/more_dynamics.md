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

# 【課題】量子ダイナミクスシミュレーション・続

第三回の実習では量子計算の並列性と、その顕著な利用法としての量子ダイナミクスシミュレーションを取り上げました。また、実機で計算を行う際の実用的な問題として、回路の最適化や測定エラーの緩和についても議論しました。この課題はその直接の延長です。
In the third exercise, we looked at the parallelism of quantum calculations and, as a prominent example of how that can be used, at simulations of quantum dynamics. We also discussed how to optimize circuits and mitigate measurement error, practical issues involved in performing computations on actual QC. This assignment is a direct extension of what we have been discussing. 

```{contents} 目次
---
local: true
---
```
$\newcommand{\ket}[1]{|#1\rangle}$
$\newcommand{\plusket}{\ket{+}}$
$\newcommand{\minusket}{\ket{-}}$

+++

## 問題1: ハイゼンベルグモデル、X方向のスピン

+++

### 問題

実習ではハイゼンベルグモデルのシミュレーションをし、各スピンの$Z$方向の期待値の時間発展を追いました。しかし、シミュレーションそのものは最終的なオブザーバブル（観測量）によらず成立するので、（ほぼ）同じ回路を用いて系の他の性質を調べることもできます。そこで、各スピンの$X$方向の期待値の時間発展を測定する回路を書き、実習時と同様に時間に対してプロットしてください。
In the exercise, we simulated a Heisenberg model and tracked the time evolution of expectation values for each spin in the Z direction. However, simulations stand without ultimate observables, so we can use a (mostly) identical circuit to explore other properties of the system. In this task, create a circuit for measuring the time evolution of expectation values for each spin in the $X$ direction and, as in the exercise, plot them over time. 

**Hint**:

[プロット用関数`plot_heisenberg_spins`](https://github.com/UTokyo-ICEPP/qc-workbook/blob/master/source/utils/dynamics.py)で厳密解のカーブを書くとき、追加の引数`spin_component='x'`を渡すと$X$方向のスピンのプロットに切り替わります。ただし、実験結果の`counts_list`は相応する測定の結果となっていなければいけません。具体的には、各スピンについて「0が測定される＝スピンが+$X$を向いている、1が測定される＝スピンが-$X$を向いている」という対応付けが必要です。）
When we draw the curve of the exact solution using the [plotting function `plot_heisenberg_spins`](https://github.com/UTokyo-ICEPP/qc-workbook/blob/master/source/utils/dynamics.py), we can add the `spin_component='x'` argument to switch to plotting the spin in the $X$ direction. However, the counts_list of the experimental results must produce corresponding measurement results. Specifically, a measurement of $\theta$ must correspond to the spin being in the +$X$ direction, and a measurement of 1 must correspond to the spin being in the -$X$ direction. 

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

# 必要なモジュールを先にインポート
# First, import all required modules
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
# このワークブック独自のモジュール
# Modules unique to this workbook
from qc_workbook.dynamics import plot_heisenberg_spins, bit_expectations_sv, bit_expectations_counts, diagonalized_evolution
```

```{code-cell} ipython3
:tags: [remove-output]

n = 5
M = 10
omegadt = 0.1

shots = 100000

# Define the circuits
circuits = []

circuit = QuantumCircuit(n)

# Bit 0 in state 1/sqrt(2)(|0> + |1>)
circuit.h(0)

for istep in range(M):
    for j in range(n - 1):
        # ZZ
        circuit.cx(j, j + 1)
        circuit.rz(-omegadt, j + 1)
        circuit.cx(j, j + 1)

        # XX
        circuit.h(j)
        circuit.h(j + 1)
        circuit.cx(j, j + 1)
        circuit.rz(-omegadt, j + 1)
        circuit.cx(j, j + 1)
        circuit.h(j)
        circuit.h(j + 1)

        # YY
        circuit.p(-np.pi / 2., j)
        circuit.p(-np.pi / 2., j + 1)
        circuit.h(j)
        circuit.h(j + 1)
        circuit.cx(j, j + 1)
        circuit.rz(-omegadt, j + 1)
        circuit.cx(j, j + 1)
        circuit.h(j)
        circuit.h(j + 1)
        circuit.p(np.pi / 2., j)
        circuit.p(np.pi / 2., j + 1)

    # Copy of the circuit up to this point
    snapshot = circuit.copy()

    ##################
    ### EDIT BELOW ###
    ##################

    # Set up the observable for this snapshot
    #snapshot.?

    ##################
    ### EDIT ABOVE ###
    ##################

    snapshot.measure_all()
    circuits.append(snapshot)

simulator = AerSimulator()

circuits = transpile(circuits, backend=simulator)
sim_job = simulator.run(circuits, shots=shots)
sim_counts_list = sim_job.result().get_counts()

# Initial state as a statevector
initial_state = np.zeros(2 ** n, dtype=np.complex128)
initial_state[0:2] = np.sqrt(0.5)

plot_heisenberg_spins(sim_counts_list, n, initial_state, omegadt, add_theory_curve=True, spin_component='x')
```

**提出するもの**

- 完成した回路のコードとシミュレーション結果によるプロット
- 一般の方向のスピンの期待値を測定するためにはどうすればいいかの説明

**Items to submit**e: 
- The code for the completed circuit and the plotted simulation results 
- An explanation of how to measure expectation values for spin in any direction

+++

### おまけ: スピン総和

注：これは量子コンピューティングというより物理の問題なので、興味のある方だけ考えてみてください。
Note: This is more of a physics problem than a quantum computing problem, so feel free to skip it if you are not interested. 

上のハイゼンベルグモデルのシミュレーションで、初期状態の$X$, $Y$, $Z$方向のスピン期待値の全系での平均値$m_x$, $m_y$, $m_z$はそれぞれ
In the above Heisenberg model simulation, the system-wide average values $m_x$, $m_y$, and $m_z$ for the expectation values of spin in the $X$, $Y$, and $Z$ directions in the initial state are as indicated below. 

$$
m_x = \frac{1}{n} \sum_{j=0}^{n} \langle \sigma^{X}_j \rangle = \frac{1}{n} \\
m_y = \frac{1}{n} \sum_{j=0}^{n} \langle \sigma^{Y}_j \rangle = 0 \\
m_z = \frac{1}{n} \sum_{j=0}^{n} \langle \sigma^{Z}_j \rangle = \frac{n-1}{n}
$$

です。これらの平均値はどう時間発展するでしょうか。理論的議論をし、シミュレーションで数値的に確かめてください。
How do these average values evolve with time? Consider this from a theoretical standpoint and then quantitatively confirm your conclusions through simulation. 

+++

## 問題2: シュウィンガーモデル

これまで扱ったような、スピンに関連する現象とは異なる物理モデルのシミュレーションをしましょう。空間1次元、時間1次元の時空における量子電磁力学の模型「シュウィンガーモデル」を考えます。
Instead of simulating phenomena related to spin, as we have so far, let's now perform a simulation of a different physical model. Let's consider the Schwinger model, a quantum electrodynamics model with one spatial dimension and one time dimension. 

+++

### シュウィンガーモデルの物理

簡単に物理の解説をします（ここは読み飛ばしても差し支えありません）。といっても、まともにゼロから解説をしたらあまりにも長くなってしまうので、かなり前提知識を仮定します。興味のある方は参考文献{cite}`shifman_schwinger,Martinez_2016`などを参照してください。特に{cite}`Martinez_2016`は実際にこれから実装する回路をイオントラップ型量子コンピュータで実行した論文です。
First, let's briefly go over the physics (you may skip this section if you wish). Explaining everything from the basics would take an extremely long time, so we'll assume you have a great deal of prior knowledge. If you're interested, check reference materials{cite}`shifman_schwinger,Martinez_2016` in the Bibliography. {cite}`Martinez_2016`, in particular, is an essay about running the circuit we are about to create on a trapped-ion quantum computer. 

量子電磁力学とは量子場の理論の一種です。量子場の理論とは物質やその相互作用（力）をすべて量子力学的な「場」（時空中の各点に応じた値を持つ存在）で記述した理論で、素粒子論などで物質の根源的な性質を記述する際の基本言語です。量子場の理論において、一部の場を「物質場」とし、それに特定の対称性（$U(1)$ゲージ対称性）を持たせると、「電荷」が生み出され、電荷を持った場の間の相互作用を媒介する「光子場」が生じます。電荷を持った場と光子場の振る舞いを記述するのが量子電磁力学です。
Quantum electrodynamics are a type of quantum field theory. Quantum field theory describes all matter and mutual interaction (forces) as quantum mechanics fields (assigning numerical values to every point in space-time). It is the basic language used to indicate the fundamental properties of matter in elementary particle theory and the like. In quantum field theory, some fields are considered "matter fields." When they have specific symmetry ($U(1)$ gauge symmetry), "electric charges" are produced, and "photon fields," which act as the media through which fields with charges mutually interact, are created. Quantum electrodynamics express the behavior of fields with charges and photon fields. 

量子場の理論は「ラグランジアン」[^lagrangian]を指定すれば定まります。シュウィンガーモデルのラグランジアンは物質場（電子）$\psi$とゲージ場（光子）$A$からなり、
Quantum field theory is defined by specifying the Lagrangian[^lagrangian]. The Lagrangian of the Schwinger model is made up of the matter field (electron) $\psi$ and the gauge field (photon) $A$, as follows. 

```{math}
:label: schwinger_lagrangian
\mathcal{L} = -\frac{1}{4g^2} F^{\mu\nu}F_{\mu\nu} + \bar{\psi} (i\gamma^{\mu}D_{\mu} - m) \psi
```

です。ただし、これまでの物理系を扱った話と異なり、ここでは場の量子論の一般慣習に従って、光速$c$とプランク定数$\hbar$がともに1である単位系を使っています。
However, unlike the physical systems we've been dealing with so far, as is customary with quantum field theory, we will use a unit system in which both the speed of light, $c$, and the Planck constant, $\hbar$, are 1. 

式{eq}`schwinger_lagrangian`の指数$\mu, \nu$は0（時間次元）か1（空間次元）の値を取ります。$\frac{1}{2g} F_{\mu\nu}$は$A$の強度テンソル（電場）で
Exponents $\mu$ and $\nu$ in Formula{eq}`schwinger_lagrangian` are either 0 (time dimension) or 1 (spatial dimension). $\frac{1}{2g} F_{\mu\nu}$ is $A$'s strength tensor (electrical field). 

$$
F_{\mu\nu} = \partial_{\mu} A_{\nu} - \partial_{\nu} A_{\mu}
$$

です。$\psi$は物質と反物質を表す2元スピノルで、$m$がその質量となります。$\{\gamma^0, \gamma^1\}$は2次元のクリフォード代数の表現です。
$\psi$ is a two component spinor that represents matter and antimatter, and $m$ is the mass. $\{\gamma^0, \gamma^1\}$ is a two-dimensional Clifford algebra expression. 

このラグランジアンを元に、Kogut-Susskindの手法{cite}`PhysRevD.10.732`でモデルを空間格子（格子間隔$a$）上の場の理論に移すと、そのハミルトニアンは
Based on this Lagrangian, if we use the Kogut-Susskind method{cite}`PhysRevD.10.732` to move the model to field theory on a spatial lattice (with lattice spacing $a$), the Hamiltonian becomes as follows. 

```{math}
:label: kogut_susskind_hamiltonian
H = \frac{1}{2a} \bigg\{ -i \sum_{j=0}^{n-2} \left[ \Phi^{\dagger}_{j} e^{i\theta_{j}} \Phi_{j+1} + \Phi_{j} e^{-i\theta_{j}} \Phi^{\dagger}_{j+1} \right] + 2 J \sum_{j=0}^{n-2} L_{j}^2 + 2 \mu \sum_{j=0}^{n-1} (-1)^{j+1} \Phi^{\dagger}_{j} \Phi_{j} \bigg\}
```

となります。ここで$J = g^2 a^2 / 2$, $\mu = m a$, また$\Phi_j$はサイト$j$上の（1元）物質場、$\theta_j$は$j$上のゲージ場、$L_j$は格子$j$と$j+1$間の接続上の電場です。
Here, $J = g^2 a^2 / 2$, $\mu = m a$, $\Phi_j$ is a (one-dimensional) matter field on site $j$, $\theta_j$ is a gauge field on $j$, and $L_j$ is an electrical field on the connection between lattice $j$ and $j+1$. 

Kogut-Susskindハミルトニアンにおける物質場はstaggered fermionsと呼ばれ、隣接サイトのうち片方が物質を、もう一方が反物質を表します。約束として、ここでは$j$が偶数のサイトを物質（電荷-1）に、奇数のサイトを反物質（電荷1）に対応付けます。一般に各サイトにおける物質の状態は、フェルミ統計に従って粒子が存在する・しないという2つの状態の重ね合わせです。サイト$j$の基底$\plusket_j$と$\minusket_j$を、$\Phi_j$と$\Phi^{\dagger}_j$が
The matter field of the Kogut-Susskind Hamiltonian are called staggered fermions, and on adjacent sites one side is matter while the other side is antimatter. Here, for $j$, even-numbered sites correspond to matter (electric charge -1) and odd-numbered sites correspond to antimatter (electric charge 1). Generally speaking, the state of matter at each site is a superposition of the two states of particles existing or not existing, in accordance with Fermi statistics. $\Phi_j$ and $\Phi^{\dagger}_j$ are defined as states that act on the basis states $\plusket_j$ and $\minusket_j$ of site $j$ as follows. 

```{math}
:label: creation_annihilation
\Phi_j \plusket_j = \minusket_j \\
\Phi_j \minusket_j = 0 \\
\Phi^{\dagger}_j \plusket_j = 0 \\
\Phi^{\dagger}_j \minusket_j = \plusket_j
```

と作用する状態と定めます。質量項の符号から、偶数サイトでは$\minusket$が粒子が存在する状態、$\plusket$が存在しない状態を表現し、奇数サイトでは逆に$\plusket$が粒子あり、$\minusket$が粒子なしを表すことがわかります。つまり、$\Phi^{\dagger}_j$と$\Phi_j$はサイト$j$における電荷の上昇と下降を引き起こす演算子です。
Based on the signs of the mass terms, for even-numbered sites, $\minusket$ indicates that a particle is present and $\plusket$ indicates that it is not. Conversely, for odd-numbered sites, $\plusket$ indicates that a particle is present, and $\minusket$ indicates that it is not. In other words, $\Phi^{\dagger}_j$ and $\Phi_j$ are operators that raise and lower the electric charge at site $j$. 

+++

### ハミルトニアンを物質場のみで記述する

$\newcommand{\flhalf}[1]{\left\lfloor \frac{#1}{2} \right\rfloor}$

このままのハミルトニアンではまだデジタルモデルが構築しにくいので、ゲージを固定して$\theta$と$L$を除いてしまいます[^another_approach]。まず$\Phi_j$を以下のように再定義します。
It would be difficult to create a digital model using this Hamiltonian as-is, so let's set the gauge as a constant and eliminate $\theta$ and $L$[^another_approach]. First, let's redefine $\Phi_j$ as follows. 

$$
\Phi_j \rightarrow \prod_{k=0}^{j-1} e^{-i\theta_{k}} \Phi_j.
$$

また、ガウスの法則から、サイト$j$の電荷$\rho_j$が同じサイトの電場の発散と等しいので、
Also, because of Gauss' law, the electric charge $\rho_j$ at site $j$ is equal to the divergence of the site's electric field, so the following is true. 

$$
L_j - L_{j-1} = \rho_j \\
\therefore L_j = \sum_{k=0}^{j} \rho_k
$$

となります。ただし、サイト0に系の境界の外から作用する電場はないもの（$L_{-1} = 0$）としました。
However, site 0 is defined as not having any electrical charges from outside the system's borders that act on it, ($L_{-1} = 0$). 

質量項と同様にサイトの偶奇を考慮した電荷は
Like with mass terms, each site's even or odd numbering is taken into consideration in the electrical charge, which is as follows. 

$$
\rho_k = \Phi_{k}^{\dagger} \Phi_{k} - (k+1 \bmod 2)
$$

なので、
And thus:

$$
L_j = \sum_{k=0}^{j} \Phi_{k}^{\dagger} \Phi_{k} - \flhalf{j} - 1
$$

となります。ここで$\flhalf{j}$は切り捨ての割り算$[j - (j \bmod 2)]/2$（Pythonでの`j // 2`と同等）です。この電場を式{eq}`kogut_susskind_hamiltonian`に代入して
Here, the thick lined-fraction $\flhalf{j}$ is rounded down division $[j - (j \bmod 2)]/2$ (equivalent to `j // 2` in Python). Substituting this electrical field into Formula{eq}`kogut_susskind_hamiltonian` give us the following. 

$$
H = \frac{1}{2a} \left\{ -i \sum_{j=0}^{n-2} \left[ \Phi^{\dagger}_{j} \Phi_{j+1} + \Phi_j \Phi^{\dagger}_{j+1} \right] + 2J \sum_{j=0}^{n-2} \left[\sum_{k=0}^{j} \Phi_{k}^{\dagger} \Phi_{k} - \flhalf{j} - 1 \right]^2 + 2\mu \sum_{j=0}^{n-1} (-1)^{j+1} \Phi^{\dagger}_{j} \Phi_{j} \right\}
$$

が得られます。

+++

### ハミルトニアンをパウリ行列で表現する

最後に、$\plusket$と$\minusket$をスピン$\pm Z$の状態のようにみなして、$\Phi^{\dagger}_j\Phi_j$と$\Phi^{\dagger}_j\Phi_{j+1}$をパウリ行列で表現します。式{eq}`creation_annihilation`から
前者は
Lastly, we can consider $\plusket$ and $\minusket$ to be spin states $\pm Z$, and express \Phi^{\dagger}_j\Phi_j$と and $\Phi^{\dagger}_j\Phi_{j+1}$ as Pauli matrices. Based on Formula{eq}`creation_annihilation`, the former can be expressed as follows. 

$$
\Phi^{\dagger}_j\Phi_j \rightarrow \frac{1}{2} (\sigma^Z_j + 1)
$$

と表現できることがわかります。一方、$\Phi^{\dagger}_j\Phi_{j+1}$に関しては、やや込み入った議論{cite}`PhysRevD.13.1043`の末、
With regard to $\Phi^{\dagger}_j\Phi_{j+1}$, as the result of some somewhat complex discussion{cite}`PhysRevD.13.1043`, we find that the following expression is correct. 

$$
\Phi^{\dagger}_j\Phi_{j+1} \rightarrow i \sigma^-_{j+1} \sigma^+_{j}
$$

が正しい表現であることがわかっています。ここで、
Here, 

$$
\sigma^{\pm} = \frac{1}{2}(\sigma^X \pm i \sigma^Y)
$$

です。また、このワークブックでの約束に従って、右辺の$j$と$j+1$の順番をひっくり返してあります。

ハミルトニアンには$\Phi_j\Phi^{\dagger}_{j+1} \rightarrow i \sigma^+_{j+1}\sigma^-_{j}$も登場するので、二つの項を合わせると
$\Phi_j\Phi^{\dagger}_{j+1} \rightarrow i \sigma^+_{j+1}\sigma^-_{j}$ also appears, so combining the two terms produces the following. 

$$
\Phi^{\dagger}_{j} \Phi_{j+1} + \Phi_j \Phi^{\dagger}_{j+1} \rightarrow \frac{i}{2} (\sigma^X_{j+1}\sigma^X_j + \sigma^Y_{j+1}\sigma^Y_j)
$$

となります。まとめると、
In summary, we arrive at the following.

$$
H \rightarrow \frac{1}{4a} \left\{ \sum_{j=0}^{n-2} (\sigma^X_{j+1}\sigma^X_j + \sigma^Y_{j+1}\sigma^Y_j) + J \sum_{j=1}^{n-2} (n - j - 1) \sum_{k=0}^{j-1} \sigma^Z_j \sigma^Z_k + \sum_{j=0}^{n-1} \left[ (-1)^{j+1} \mu - J \flhalf{n-j} \right] \sigma^Z_j \right\}
$$

です。ただし、計算過程で現れる定数項（恒等演算子に比例する項）は時間発展において系の状態に全体位相をかける作用しか持たないため、無視しました。
However, the constant terms that appear during the course of this calculation (terms that are proportional to the identity operator) only act on the overall phase of the system's state as it evolves over time, so we have disregarded them. 

+++

### 問題

上のシュウィンガーモデルのハミルトニアンによる時間発展シミュレーションを、$\plusket$と$\minusket$をそれぞれ$\ket{0}$と$\ket{1}$に対応させて、8ビット量子レジスタに対して実装してください。初期状態は真空、つまりどのサイトにも粒子・反粒子が存在しない状態$\ket{+-+-+-+-}$とし、系全体の粒子数密度の期待値
Implement the following time evolution simulation of the Schwinger model using the Hamiltonian, associating $\plusket$ and $\minusket$ with $\ket{0}$ and $\ket{1}$, respectively, using an 8-bit quantum register. Set the initial state to a vacuum -- that is, $\ket{+-+-+-+-}$, and plot the system's overall particle density expectation value, shown in the formula below, as a function of time. 

$$
\nu = \left\langle \frac{1}{n} \sum_{j=0}^{n-1} \frac{1}{2} \left[(-1)^{j+1} \sigma^Z_j + 1\right] \right\rangle
$$

を時間の関数としてプロットしてください。余裕があれば、各サイトにおける粒子数、電荷、サイト間の電場などの期待値の時間変化も観察してみましょう。
If this is relatively easy for you, also try observing how expectation values such as the number of particles, electric charge, and electrical field between sites change over time at each site. 

ハミルトニアンのパラメターは、$J = 1$, $\mu = 0.5$とします（他の$J$や$\mu$の値もぜひ試してみてください）。$\omega = 1/(2a)$とおき、鈴木・トロッター分解における時間ステップ$\Delta t$の大きさ$\omega \Delta t = 0.2$として、時間$\omega t = 2$までシミュレーションをします。
Use the following for Hamiltonian parameters: $J = 1$, $\mu = 0.5$ (also try it out using different $J$ and $\mu$ values). Set $\omega = 1/(2a)$ and the size of the Suzuki-trotter transformation time step $\Delta t$ to $\omega \Delta t = 0.2$, and then simulate time $\omega t = 2$. 

**解説**:
**Explanation**: 

偶数サイトでは$\plusket$が物質粒子の存在しない状態、奇数サイトでは$\minusket$が反物質粒子の存在しない状態を表すので、初期状態は粒子数密度0となります。しかし、場の量子論においては場の相互作用によって物質と反物質が対生成・対消滅を起こし、一般に系の粒子数の期待値は時間とともに変化します。
At even-numbered sites, $\plusket$ represents the state when there is no matter particle present, while at odd-numbered sites, $\minusket$ represents the state when there is no matter particle present. Therefore, the particle density in the system's initial state is 0. However, in quantum field theory, the fields act on each other, generating pairs of matter and antimatter particles which annihilate each other, so in general the expectation values of the number of particles changes with time. 

**ヒント**:
**Hint**:

上のハミルトニアンのパラメターの値は参考文献{cite}`Martinez_2016`と同一です（ただし$\sigma$積の順番は逆です）。したがって、$n=4$, $\omega \Delta t = \pi/8$とすれば、論文中の図3aを再現できるはずです。答え合わせに使ってください。
The above Hamiltonian parameter values are identical to those indicated in reference material{cite}`Martinez_2016`. Therefore, if $n=4$ and $\omega \Delta t = \pi/8$, you should be able to replicate what is shown in Figure 3a of the paper. Use it to check your answer. 

また、問題を解くためのヒントではありませんが、ハイゼンベルグモデルと同様にこのモデルでも対角化による厳密解を比較的簡単にプロットできるように道具立てがしてあります。下のコードのテンプレートでは、シミュレーション回路と厳密解を計算するためのハミルトニアンのパウリ行列分解だけ指定すれば、`plot_heisenberg_spins`と同様のプロットが作成されるようになっています。パウリ行列分解を指定するには、`paulis`と`coeffs`という二つのリストを作り、Qiskitの`SparsePauliOp`というクラスに渡します。これらのリストの長さはハミルトニアンの項数で、`paulis`の各要素は対応する項のパウリ行列のリスト、`coeffs`の各要素はその項にかかる係数にします。例えば
Furthermore, although this is not a hint for solving the task, the groundwork has been laid for plotting the exact solution with relative ease by applying the same diagonalize to this model as you did to the Heisenberg model. With the code template below, you can perform plotting in the same way as with `plot_heisenberg_spins` simply by specifying the simulation circuit and the Pauli matrix decomposition of the Hamiltonian used to calculate the exact solution. To specify the Pauli matrix decomposition, create two lists, `paulis` and `coeffs`. The length of each of these lists is the number of Hamiltonian terms. In the `paulis` list, each of the elements corresponds to a term's Pauli matrix. In the `coeffs` list, each element is the coefficient of that term. For example, for the following Hamiltonian, the lists' contents would be as shown below. 

$$
H = 0.5 \sigma^X_2 \sigma^Y_1 I_0 + I_2 \sigma^Z_1 \sigma^X_0
$$

というハミルトニアンに対しては、

```{code-block} python
paulis = ['XYI', 'IZX']
coeffs = [0.5, 1.]
```

です。

[^lagrangian]: ここで「ラグランジアン」と呼ばれているのは本来「ラグランジアン密度」で、正しくはこれを空間積分したものがラグランジアンですが、素粒子論の文脈で「ラグランジアン」と言った場合はほぼ100%積分する前のものを指します。
[^another_approach]: 参考文献{cite}`Shaw2020quantumalgorithms`では、別のアプローチで同じハミルトニアンの量子回路実装をしています。
[^lagrangian]: Here, what we call the "Lagrangian" is normally called the "Lagrangian density." Properly speaking, the Lagrangian is this integrated over space and time, but in the context of elementary particle theory, almost 100% of the time, the term "Lagrangian" refers the Lagrangian density before integration. 
[^another_approach]: In reference material{cite}`Shaw2020quantumalgorithms`, a different approach is made to implement a quantum circuit with the same Hamiltonian.  


```{code-cell} ipython3
:tags: [raises-exception, remove-output]

def number_density(bit_exp):
    particle_number = np.array(bit_exp) # shape (T, n)
    # Particle number is 1 - (bit expectation) on odd sites
    particle_number[:, 1::2] = 1. - particle_number[:, 1::2]

    return np.mean(particle_number, axis=1)


n = 8 # number of sites
J = 1. # Hamiltonian J parameter
mu = 0.5 # Hamiltonian mu parameter

## Quantum circuit experiment

M = 10 # number of Trotter steps
omegadt = 0.2 # Trotter step size

shots = 100000

# Define the circuits
circuits = []

circuit = QuantumCircuit(n)

# Initial state = vacuum
circuit.x(range(1, n, 2))

for istep in range(M):
    ##################
    ### EDIT BELOW ###
    ##################

    #circuit.?

    ##################
    ### EDIT ABOVE ###
    ##################

    circuits.append(circuit.measure_all(inplace=False))

# Run the circuits in the simulator
simulator = AerSimulator()

circuits = transpile(circuits, backend=simulator)
sim_job = simulator.run(circuits, shots=shots)
sim_counts_list = sim_job.result().get_counts()
```

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

## Numerical solution through diagonalization

##################
### EDIT BELOW ###
##################

paulis = ['I' * n]
coeffs = [1.]

##################
### EDIT ABOVE ###
##################

hamiltonian = SparsePauliOp(paulis, coeffs).to_matrix()

# Initial state as a statevector
initial_state = np.zeros(2 ** n, dtype=np.complex128)
vacuum_state_index = 0
for j in range(1, n, 2):
    vacuum_state_index += (1 << j)
initial_state[vacuum_state_index] = 1.

## Plotting

# Plot the exact solution
time_points, statevectors = diagonalized_evolution(hamiltonian, initial_state, omegadt * M)
_, bit_exp = bit_expectations_sv(time_points, statevectors)

plt.plot(time_points, number_density(bit_exp))

# Prepend the "counts" (=probability distribution) for the initial state to counts_list
initial_probs = np.square(np.abs(initial_state))
fmt = f'{{:0{n}b}}'
initial_counts = dict((fmt.format(idx), prob) for idx, prob in enumerate(initial_probs) if prob != 0.)
sim_counts_list_with_init = [initial_counts] + sim_counts_list

# Plot the simulation results
time_points = np.linspace(0., omegadt * M, M + 1, endpoint=True)
_, bit_exp = bit_expectations_counts(time_points, sim_counts_list_with_init, n)

plt.plot(time_points, number_density(bit_exp), 'o')
```

+++ {"tags": ["remove-output"]}

**提出するもの**
- 完成した回路のコードとシミュレーション結果によるプロット

**Items to submit**: 
- The code for the completed circuit and the plotted simulation results
