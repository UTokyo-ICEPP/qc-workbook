---
jupytext:
  notebook_metadata_filter: all
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
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
  version: 3.8.10
---

# 【課題】量子相関を調べる

第一回の実習ではCHSH不等式の破れを調べるために、2つの量子ビットの相関関数$C^{i} \, (i=0,1,2,3)$という量を量子コンピュータを使って計算しました。この課題では、この量をもう少し細かく調べてみましょう。

```{contents} 目次
---
local: true
---
```

$\newcommand{\ket}[1]{|#1\rangle}$
$\newcommand{\bra}[1]{\langle#1|}$

+++

## QCシミュレータの使い方

実習で見たように、QCで実現される量子状態は、量子力学の公理に基づいて理論的に計算・予測できます。そこで用いられる数学的操作も単なる足し算や掛け算（線形演算）なので、実はQCの量子状態は（古典）計算機で比較的簡単に計算できます。当然のことですが、QCは何も魔法のブラックボックスというわけではありません。

ただし、古典計算機で量子状態を再現するためには、特殊な場合を除いて、量子ビット数の指数関数的な量のメモリが必要になります。これも前半で見たように、$n$量子ビットあれば、系の自由度（degrees of freedom / dof: 実数自由パラメータの数）は$2^{n+1} - 2$ですので、例えば各自由度を64ビット（＝8バイト）の浮動小数点で表現するとしたら、必要なメモリは(-2を無視して)

$$
2^3\, \mathrm{(bytes / dof)} \times 2^{n+1}\, \mathrm{(dof)} = 2^{n+4}\, \mathrm{(bytes)}
$$

なので、$n=16$で1 MiB、$n=26$で1 GiB、$n=36$で1 TiBです。現在の計算機では、ハイエンドワークステーションでRAMが$\mathcal{O}(1)$ TiB、スパコン「富岳」で5 PB (~2<sup>52</sup> bytes)なのに対し、QCではすでに$n=127$のものが存在するので、既に古典計算機でまともにシミュレートできない機械が存在していることになります。

しかし、逆に言うと、$n \sim 30$程度までの回路であれば、ある程度のスペックを持った計算機で厳密にシミュレートできるということが言えます。じっさい世の中には[数多くの](https://quantiki.org/wiki/list-qc-simulators)シミュレータが存在します。Qiskitにも様々な高機能シミュレータが同梱されています。

シミュレーションはローカル（手元のPythonを動かしているコンピュータ）で実行できるので、ジョブを投げて結果を待つ時間が省けます。この課題ではたくさんの細かい量子計算をするので、実機を使わず、`qasm_simulator`というQiskitに含まれるシミュレータを利用します。

Qiskitのシミュレータには`Aer`というオブジェクトからアクセスします。`Aer`は実習で登場した`IBMQ`と同様の構造をしており、複数のシミュレータをバックエンドとして管理しています。そのうちの`qasm_simulator`を取り出します。

```{code-cell} ipython3
# まずは全てインポート
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.visualization import plot_histogram

print('notebook ready')
```

```{code-cell} ipython3
# シミュレータをバックエンドとして使うときは、IBMQのプロバイダではなくAerのget_backend()を呼ぶ
simulator = Aer.get_backend('qasm_simulator')
print(simulator.name())
```

実習の内容を再現してみましょう。

```{code-cell} ipython3
:tags: [remove-output]

circuits = []

circuit = QuantumCircuit(2, name='circuit0')
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(-np.pi / 4., 1)
circuits.append(circuit)

circuit = QuantumCircuit(2, name='circuit1')
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(-3. * np.pi / 4., 1)
circuits.append(circuit)

circuit = QuantumCircuit(2, name='circuit2')
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(-np.pi / 4., 1)
circuit.ry(-np.pi / 2., 0)
circuits.append(circuit)

circuit = QuantumCircuit(2, name='circuit3')
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(-3. * np.pi / 4., 1)
circuit.ry(-np.pi / 2., 0)
circuits.append(circuit)

for circuit in circuits:
    circuit.measure_all()
```

```{code-cell} ipython3
:tags: [remove-output]

# シミュレータにはショット数の制限がないので、時間の許す限りいくらでも大きい値を使っていい
shots = 10000

# 実習と同じく transpile() - 今は「おまじない」と思ってよい
circuits = transpile(circuits, backend=simulator)
# シミュレータもバックエンドと同じように振る舞うので、runメソッドで回路とショット数を受け取り、ジョブオブジェクトを返す
sim_job = simulator.run(circuits, shots=shots)

# シミュレータから渡されたジョブオブジェクトは実機のジョブと全く同じように扱える
sim_result = sim_job.result()

C = np.zeros(4, dtype=float)
for idx in range(4):
    counts = sim_result.get_counts(idx)
    ax = plt.figure().add_subplot()
    plot_histogram(counts, ax=ax)
    
    C[idx] = counts.get('00', 0) + counts.get('11', 0) - counts.get('01', 0) - counts.get('10', 0)
    
C /= shots
    
S = C[0] - C[1] + C[2] + C[3]
print('S =', S)
```

上のように、`qasm_simulator`は実機と同様に`run`関数を実行でき、ヒストグラムデータを返します。実機ではショット数に制限がありますが、シミュレータにはありません。ただしショット数が多いほど、当然実行に時間がかかります。といってもこの程度の回路であれば常識的なショット数ならほぼ瞬間的にジョブの実行が終わるので、上の例では実習で使った`job_monitor()`関数を使用していません。また、シミュレータにはノイズがない[^simulator_noise]ので、$S$の計算結果が統計誤差の範囲内で理論値と一致していることが見て取れます。

[^simulator_noise]: 標準設定において。実機の振る舞いに従うよう、あえてノイズを加えるような設定も存在します。

+++

## 測定基底の変換

さて、おさらいをすると、上の$C^{0,1,2,3}$を計算する4つの回路は以下のようなものでした。

```{code-cell} ipython3
:tags: [remove-input]

for circuit in circuits:
    ax = plt.figure().add_subplot()
    circuit.draw('mpl', ax=ax)
```

ベル状態を作るところまですべての回路で共通で、その後それぞれ異なる角度の$R_y$ゲートをかけています。実習では深く立ち入りませんでしたが、この$R_y$ゲートはベル状態を$\{\ket{0}, \ket{1}\}$とは異なる基底で測定するために置かれています。どういうことか、以下で説明します。

ここまで「測定」とはレジスタの量子状態$\sum_{j=0}^{2^n-1} c_j \ket{j}$からビット列$j$を確率$|c_j|^2$で得る行為である、と説明してきました。しかし、本来「測定」はもっと一般的な概念です。それを理解するために、まず、量子力学的には計算基底状態$\ket{j}$は何ら特別な状態ではないということを理解しておきましょう。

例えば1量子ビットにおいて、計算基底状態は$\ket{0}$と$\ket{1}$ですが、以下のような状態$\ket{\theta}$と$\ket{\theta + \pi}$を考えます。

```{math}
:label: theta_ket_def
\ket{\theta} := R_y(\theta)\ket{0} = \cos\frac{\theta}{2}\ket{0} + \sin\frac{\theta}{2}\ket{1} \\
\ket{\theta + \pi} := R_y(\theta)\ket{1} = -\sin\frac{\theta}{2}\ket{0} + \cos\frac{\theta}{2}\ket{1}
```

すると、

$$
\ket{0} = \cos\frac{\theta}{2}\ket{\theta} - \sin\frac{\theta}{2}\ket{\theta + \pi} \\
\ket{1} = \sin\frac{\theta}{2}\ket{\theta} + \cos\frac{\theta}{2}\ket{\theta + \pi},
$$

つまり、計算基底状態が$\ket{\theta}$と$\ket{\theta + \pi}$の重ね合わせとして表現できます。量子ビットの任意の状態は$\ket{0}$と$\ket{1}$の重ね合わせで表現できるので、$\ket{\theta}$と$\ket{\theta + \pi}$の重ね合わせでも表現できるということになります。そのようなときは$\ket{\theta}$と$\ket{\theta + \pi}$を基底として状態を表しているわけです。

一般に、量子力学的には、2つの異なる状態$\ket{a} \nsim \ket{b}$を考えれば、それらの重ね合わせで量子ビットの任意の状態が表現できます。そして、$\ket{a}$と$\ket{b}$が直交する[^orthogonal]ときに状態$\ket{\psi}$が

$$
\ket{\psi} = \alpha \ket{a} + \beta \ket{b}
$$

と表現されるならば、「基底$\ket{a}$と$\ket{b}$についての測定」という操作を考えることができます。$\ket{\psi}$に対してそのような測定をすると、状態$\ket{a}$が確率$|\alpha|^2$で、状態$\ket{b}$が確率$|\beta|^2$で得られます。

量子計算においても、アルゴリズムの一部として、計算の結果実現した状態を特定の基底で測定するということが多々あります。ところが、ここで若干問題があります。量子コンピュータは実装上、計算基底でしか測定ができないのです。量子力学の理論的には特別でない$\ket{0}$と$\ket{1}$ですが、量子コンピュータという実態にとっては具体的な対応物があるのです。

そこで、量子計算では、状態を任意の基底で測定することを諦め、反対に状態を変化させてしまいます。例えば、本当は上の$\ket{\theta}$と$\ket{\theta + \pi}$という基底で量子ビットの状態$\ket{\psi} = \alpha \ket{\theta} + \beta \ket{\theta + \pi}$を測定したいとします。しかし計算基底でしか測定ができないので、代わりに$R_y(-\theta)$を$\ket{\psi}$にかけます。すると式{eq}`theta_ket_def`から

$$
R_y(-\theta)\ket{\theta} = \ket{0} \\
R_y(-\theta)\ket{\theta + \pi} = \ket{1}
$$

なので、

$$
R_y(-\theta)\ket{\psi} = \alpha \ket{0} + \beta \ket{1}
$$

が得られます。この$R_y(-\theta)\ket{\psi}$を計算基底で測定した結果は、$\ket{\psi}$を$\ket{\theta}, \ket{\theta + \pi}$基底で測定した結果と等価です。

このように、測定を行いたい基底（ここでは$\ket{\theta}, \ket{\theta + \pi}$）を$\ket{0}, \ket{1}$から得るための変換ゲート（$R_y(\theta)$）の逆変換を測定したい状態にかけることで、計算基底での測定で求める結果を得ることができます。

+++

## 観測量の期待値とその計算法

課題の説明に入る前に、さらに話が込み入ってきますが、量子計算でも多出する（ワークブックでは特に{doc}`vqe`以降）概念である「観測量の期待値」について説明します。

観測量とはそのまま「観測できる量」のことで、量子状態から取り出せる（古典的）情報のこととも言えます。例えば、何かしらの粒子の運動を量子力学的に記述した場合、その粒子の位置や運動量などが観測量です。

ケットで表される量子状態に対して、観測量はケットに作用する「エルミート演算子」で表現されます。細かい定義は参考文献に譲り、ここで必要な最小限のことだけ述べると、エルミート演算子は**対角化可能で、実数の固有値を持つ**という性質を持っています。つまり、$A$が$N$次元の量子状態の空間のエルミート演算子であれば、

$$
A \ket{\phi_j} = a_j \ket{\phi_j}
$$

が成り立つような状態$\ket{\phi_j}$と実数$a_j$の組が$N$個存在し、各$\ket{\phi_j} \, (j=0,\dots,N-1)$は互いに直交します。このとき、$\ket{\phi_j}$を固有ベクトル、$a_j$を固有値と呼びます。そして、この固有値$a_j$が、演算子$A$で表される観測量の値に対応します[^continuous_observables]。

さて、$A$が$n$ビット量子レジスタに対する演算子であるとします。レジスタの状態$\ket{\psi}$が$A$の固有ベクトルで

$$
\ket{\psi} = \sum_{j=0}^{2^n-1} \gamma_j \ket{\phi_j}
$$

と分解されるとき、この状態を固有ベクトル基底$\{\ket{\phi_j}\}$で測定すると、状態$\ket{\phi_j}$が確率$|\gamma_j|^2$で得られます。そして、状態が$\ket{\phi_j}$であれば観測量$A$の値は$a_j$です。

そのような測定を多数回繰り返して$A$の値の期待値を求めることを考えます。期待値の定義は「確率変数のすべての値に確率の重みを付けた加重平均」です。$A$が確率分布$\{|\gamma|^2_j\}$に従って値$\{a_j\}$を取るので、

$$
\bra{\psi} A \ket{\psi} = \sum_{j=0}^{2^n-1} a_j |\gamma_j|^2
$$

となります。ここから、量子コンピュータにおいて観測量の期待値を計算する方法を見出すことができます。具体的には、

1. 観測量をエルミート演算子で表現する
1. 演算子を対角化し、固有値と対応する固有ベクトルを求める
1. 固有ベクトルを基底として、レジスタの状態を測定する
1. 測定から得られた確率分布を重みとして固有値の平均値を取る

です。3の測定の際には、上のセクションで説明した測定基底の変換を利用します。

+++

## CHSH不等式を見直す

実習の中で、

$$
C = P_{00} - P_{01} - P_{10} + P_{11}
$$

という量を計算しました。ここで、$P_{lm}$は2つの量子ビットでそれぞれ$l, m \, (=0,1)$が得られる確率でした（コードで言えば`counts.get('lm') / shots`に対応）。実はこの量$C$は、2ビットレジスタにおけるある観測量の期待値として捉えることができます。

まず、1つの量子ビットに対して、固有値が$\pm 1$であるような観測量$\sigma^{\theta}$を考えます。そのような$\sigma^{\theta}$は無数に存在し、固有ベクトルで区別できます。$\theta$は固有ベクトルを決める何らかのパラメータです[^specifying_eigenvectors]。例えば$\sigma^0$という観測量を、計算基底を固有ベクトルとして、

$$
\sigma^0 \ket{0} = \ket{0} \\
\sigma^0 \ket{1} = -\ket{1}
$$

で定義できる、という具合です。

次に、2つの量子ビットA, Bからなるレジスタを考え、$\sigma^{\kappa}$をAの、$\sigma^{\lambda}$をBの観測量とします。また、それぞれの演算子の固有ベクトルを

$$
\sigma^{\kappa} \ket{\kappa_{\pm}} = \pm \ket{\kappa_{\pm}} \\
\sigma^{\lambda} \ket{\lambda_{\pm}} = \pm \ket{\lambda_{\pm}}
$$

で定義します。これらの固有ベクトルを使って、レジスタの状態$\ket{\psi}$を

$$
\ket{\psi} = c_{++} \ket{\lambda_+}_B \ket{\kappa_+}_A + c_{+-} \ket{\lambda_+}_B \ket{\kappa_-}_A + c_{-+} \ket{\lambda_-}_B \ket{\kappa_+}_A + c_{--} \ket{\lambda_-}_B \ket{\kappa_-}_A
$$

と分解します。ケットはAが右、Bが左になるよう並べました。すると、積$\sigma^{\lambda}_B \sigma^{\kappa}_A$の$\ket{\psi}$に関する期待値は、

$$
\bra{\psi} \sigma^{\lambda}_B \sigma^{\kappa}_A \ket{\psi} = |c_{++}|^2 - |c_{+-}|^2 - |c_{-+}|^2 + |c_{--}|^2
$$

です。

最後に、同じ結果を計算基底での測定で表すために、$\ket{\psi}$に対して基底変換を施します。$\{\ket{\kappa_{\pm}}, \ket{\lambda_{\pm}}\}$が何らかのパラメータ付きゲート$R(\theta)$を通して計算基底と

$$
\ket{\kappa_+} = R(\kappa) \ket{0} \\
\ket{\kappa_-} = R(\kappa) \ket{1} \\
\ket{\lambda_+} = R(\lambda) \ket{0} \\
\ket{\lambda_-} = R(\lambda) \ket{1}
$$

で結びついているなら、状態$\ket{\psi'} = R^{-1}_B(\lambda) R^{-1}_A(\kappa) \ket{\psi}$を計算基底で測定したとき、

$$
P_{00} = |c_{++}|^2 \\
P_{01} = |c_{+-}|^2 \\
P_{10} = |c_{-+}|^2 \\
P_{11} = |c_{--}|^2
$$

が成り立ちます。確認のためはっきりさせておくと、左辺は$\ket{\psi'}$を計算基底で測定し、ビット列00, 01, 10, 11を得る確率です。つまり、最初の$C$は

$$
C = \bra{\psi'} \sigma^0_B \sigma^0_A \ket{\psi'} = \bra{\psi} \sigma^{\lambda}_B \sigma^{\kappa}_A \ket{\psi}
$$

を表していたのでした。

これを踏まえて、CHSH不等式の左辺は結局何を計算していたのか、見直してみましょう。ここでベル状態を

$$
\ket{\Psi} = \frac{1}{\sqrt{2}} \left(\ket{00} + \ket{11}\right)
$$

とおき、第0ビットをA、第1ビットをBとします。$R_y(\pi/4)\ket{0}, R_y(\pi/2)\ket{0}, R_y(3\pi/4)\ket{0}$が固有値$+1$の固有ベクトルとなるような演算子をそれぞれ$\sigma^{\pi/4}, \sigma^{\pi/2}, \sigma^{3\pi/4}$とすると、

$$
S = C^0 - C^1 + C^2 + C^3 = \bra{\Psi} \sigma^{\pi/4}_B \sigma^0_A \ket{\Psi} - \bra{\Psi} \sigma^{3\pi/4}_B \sigma^0_A \ket{\Psi} + \bra{\Psi} \sigma^{\pi/4}_B \sigma^{\pi/2}_A \ket{\Psi} + \bra{\Psi} \sigma^{3\pi/4}_B \sigma^{\pi/2}_A \ket{\Psi}
$$

がわかります。

観測量$\sigma^{\theta}$を用いてCHSH不等式をより正確に表現すると、

> 4つのパラメータ$\kappa, \lambda, \mu, \nu$を用いて  
> $S(\kappa, \lambda, \mu, \nu) = \langle \sigma^{\kappa}\sigma^{\lambda} \rangle - \langle \sigma^{\kappa}\sigma^{\nu} \rangle + \langle \sigma^{\mu}\sigma^{\lambda} \rangle + \langle \sigma^{\mu}\sigma^{\nu} \rangle$  
> という量を定義すると、エンタングルメントのない古典力学において$|S| \leq 2$である

となります。

実習で用いた$\sigma^{\theta}$のパラメータは、$|S|$の値を最大化するものでした。次のセルで数値的にこのことを確かめています。

```python tags=["remove-output"]
def quantum_S(x):
    """
    Three-parameter function to be minimized. Returns -|S| = -|<sigma^k sigma^l> - <sigma^k sigma^n> + <sigma^m sigma^l> + <sigma^m sigma^n>| with k fixed to 0.
    """
    
    return -np.abs(np.cos(-x[0]) - np.cos(-x[2]) + np.cos(x[1] - x[0]) + np.cos(x[1] - x[2]))

# Initial values
x0 = np.array([np.pi / 2., 0., np.pi / 2.])
# Bounds on all parameters ([0, pi])
bounds = Bounds([0.] * 3, [np.pi] * 3)
# Minimize using scipy.optimize.minimize
res = minimize(quantum_S, x0, method='trust-constr', bounds=bounds)
xopt = res.x / np.pi
print(f'argmax |S|: kappa=0, lambda={xopt[0]:.2f}pi, mu={xopt[1]:.2f}pi, nu={xopt[2]:.2f}pi')
```

[^orthogonal]: ここまで状態ケットの内積に関する説明をしていませんが、線形代数に慣れている方は「量子力学の状態ケットはヒルベルト空間の要素である」と理解してください。慣れていない方は、量子ビットの直行する状態とは$\ket{0}$と$\ket{1}$に同じゲートをかけて得られる状態のことを言うと覚えておいてください。
[^continuous_observables]: 上で粒子の位置や運動量という例を挙げたので、$a_j \, (j=0,\dots,N-1)$と離散的な固有値が観測量の値だというと混乱するかもしれません。位置や運動量といった連続的な値を持つ観測量は、無数に固有値を持つエルミート演算子で表されます。そのような演算子のかかる状態空間は無限次元です。
[^specifying_eigenvectors]: 量子ビットの一般の状態は実パラメータ2つで決まるので、$\sigma^{\theta, \phi}$などと書いたほうがより明示的ですが、ここでの議論では結局1パラメータしか使わないので、「何か一般の（次元を指定しない）パラメータ」として$\theta$と置いています。

+++

## 問題: ベル状態と可分状態の違い

### 一般の$\sigma$演算子の期待値

上のように$R_y(\theta)\ket{0}$が固有値$+1$の固有ベクトルとなるような演算子を$\sigma^{\theta}$として、$\bra{\Psi} \sigma^{\phi}_B \sigma^{\theta}_A \ket{\Psi}$を計算してみましょう。まず基底変換を具体的に書き下します。

$$
\begin{align}
\ket{\Psi'} & = R_{y,B}(-\phi) R_{y,A}(-\theta) \frac{1}{\sqrt{2}} \left( \ket{00} + \ket{11} \right) \\
& = \frac{1}{\sqrt{2}} \left\{ \left[ \cos\left(\frac{\phi}{2}\right)\ket{0} - \sin\left(\frac{\phi}{2}\right)\ket{1} \right] \left[ \cos\left(\frac{\theta}{2}\right)\ket{0} - \sin\left(\frac{\theta}{2}\right)\ket{1} \right] + \left[ \sin\left(\frac{\phi}{2}\right)\ket{0} + \cos\left(\frac{\phi}{2}\right)\ket{1} \right] \left[ \sin\left(\frac{\theta}{2}\right)\ket{0} + \cos\left(\frac{\theta}{2}\right)\ket{1} \right] \right\} \\
& = \frac{1}{\sqrt{2}} \left[ \cos\left(\frac{\theta - \phi}{2}\right)\ket{00} - \sin\left(\frac{\theta - \phi}{2}\right)\ket{01} + \sin\left(\frac{\theta - \phi}{2}\right)\ket{10} + \cos\left(\frac{\theta - \phi}{2}\right)\ket{11} \right].
\end{align}
$$

したがって、

```{math}
:label: quantum_correlation
\begin{align}
\bra{\Psi} \sigma^{\phi}_B \sigma^{\theta}_A \ket{\Psi} & = \bra{\Psi'} \sigma^{0}_B \sigma^{0}_A \ket{\Psi'} \\
& = \cos^2\left(\frac{\theta - \phi}{2}\right) - \sin^2\left(\frac{\theta - \phi}{2}\right) \\
& = \cos(\theta - \phi)
\end{align}
```

となります。

### 問題

上の計算結果を量子回路でも確認してみましょう。実習のように2ビット量子レジスタをベル状態にし、2つの量子ビットに適当な$R_y$ゲートをかけ、期待値$C$を$R_y$ゲートのパラメータの値の差の関数としてプロットします。

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

# Define theta and phi values
thetas = np.array([0., np.pi / 4., np.pi / 2., 3. * np.pi / 4., np.pi])
theta_labels = ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
phis = np.linspace(0., np.pi, 16, endpoint=True)

# Construct a circuit for each (theta, phi) pair
circuits = []
for itheta, theta in enumerate(thetas):
    for iphi, phi in enumerate(phis):
        circuit = QuantumCircuit(2, name=f'circuit_{itheta}_{iphi}')

        ##################
        ### EDIT BELOW ###
        ##################

        #circuit.?
        
        ##################
        ### EDIT ABOVE ###
        ##################
        
        circuit.measure_all()

        circuits.append(circuit)

# Execute the circuit in qasm_simulator and retrieve the results
simulator = Aer.get_backend('qasm_simulator')
shots = 10000
circuits = transpile(circuits, backend=simulator)
sim_job = simulator.run(circuits, shots=shots)
result = sim_job.result()

# Plot C versus (theta - phi) for each theta
icirc = 0
for itheta, theta in enumerate(thetas):
    x = theta - phis
    y = np.zeros_like(x)

    for iphi, phi in enumerate(phis):
        counts = result.get_counts(circuits[icirc])

        ##################
        ### EDIT BELOW ###
        ##################

        #y[iphi] = ?
        
        ##################
        ### EDIT ABOVE ###
        ##################

        icirc += 1

    plt.plot(x, y, 'o', label=theta_labels[itheta])

plt.legend()
plt.xlabel(r'$\theta - \phi$')
plt.ylabel(r'$\langle \Psi | \sigma^{\phi}_B \sigma^{\theta}_A | \Psi \rangle$')
```

式{eq}`quantum_correlation`では、2体系の期待値がパラメータ$\theta$と$\phi$だけを含む関数の積として記述できないことがはっきり示されています。これはベル状態の持つエンタングルメントの現れで、2体系をバラバラに考えることができないということを言っています。

それでは、$C$や$S$をエンタングルメントのない状態（可分状態）に対して計算してみたら、何が得られるでしょうか。例えば、ベル状態$1/\sqrt{2}(\ket{00} + \ket{11})$の代わりに、「確率1/2で$\ket{00}$、確率1/2で$\ket{11}$」という状態を考えます。

まずは$|S(\kappa, \lambda, \mu, \nu)|$の最大値を求めます。

```{code-cell} ipython3
:tags: [remove-output]

def classical_S(params):
    """
    Four-parameter function to be minimized. Returns -|S| = -|<sigma^k sigma^l> - <sigma^k sigma^n> + <sigma^m sigma^l> + <sigma^m sigma^n>|.
    
    Args:
        params (np.ndarray): Values of kappa, lambda, mu, nu
        
    Returns:
        float: -|S|
    """
    
    k, l, m, n = params
    
    S = 0.
    
    # S from |00>
    ##################
    ### EDIT BELOW ###
    ##################
    #sksl_00 = 
    #sksn_00 = 
    #smsl_00 = 
    #smsn_00 = 
    #S += (sksl_00 - sksn_00 + smsl_00 + smsn_00) * 0.5
    ##################
    ### EDIT ABOVE ###
    ##################
    
    # S from |11>
    ##################
    ### EDIT BELOW ###
    ##################
    #sksl_11 = 
    #sksn_11 = 
    #smsl_11 = 
    #smsn_11 = 
    #S += (sksl_11 - sksn_11 + smsl_11 + smsn_11) * 0.5
    ##################
    ### EDIT ABOVE ###
    ##################
    
    return -np.abs(S)

# Initial values
x0 = np.array([0., np.pi / 2., 0., np.pi / 2.])
# Bounds on all parameters ([0, pi])
bounds = Bounds([0.] * 4, [np.pi] * 4)
# Minimize using scipy.optimize.minimize
res = minimize(classical_S, x0, method='trust-constr', bounds=bounds)
xopt = res.x / np.pi
print(f'argmax |S|: kappa={xopt[0]:.3f}pi, lambda={xopt[1]:.3f}pi, mu={xopt[2]:.3f}pi, nu={xopt[3]:.3f}pi')
print('max |S|:', -res.fun)
```

+++ {"tags": ["remove-output"]}

この状態を量子回路で実装して、上の実験を繰り返してみましょう。ただし、実際に確率1/2で異なる状態が現れるような「混合状態」は量子回路では表現しにくいので、$\ket{00}$と$\ket{11}$を初期状態とする回路を一つずつ用意して、それらから得る$C$の平均値を求めることにします。

```{code-cell} ipython3
:tags: [remove-output]

# Construct a circuit for each (theta, phi) pair
circuits = []
for itheta, theta in enumerate(thetas):
    for iphi, phi in enumerate(phis):
        circuit_00 = QuantumCircuit(2, name=f'circuit_{itheta}_{iphi}_00')
        circuit_11 = QuantumCircuit(2, name=f'circuit_{itheta}_{iphi}_11')

        ##################
        ### EDIT BELOW ###
        ##################

        #circuit_00.?
        #circuit_11.?
        
        ##################
        ### EDIT ABOVE ###
        ##################
        
        circuit_00.measure_all()
        circuit_11.measure_all()

        circuits.append(circuit_00)
        circuits.append(circuit_11)

# Execute the circuit in qasm_simulator and retrieve the results
shots = 10000
circuits = transpile(circuits, backend=simulator)
sim_job = simulator.run(circuits, shots=shots)
result = sim_job.result()

# Plot C versus (theta - phi) for each theta
icirc = 0
for itheta, theta in enumerate(thetas):
    x = theta - phis
    y = np.zeros_like(x)

    for iphi, phi in enumerate(phis):
        c_00 = result.get_counts(circuits[icirc])
        c_11 = result.get_counts(circuits[icirc + 1])

        ##################
        ### EDIT BELOW ###
        ##################

        #y[iphi] = ?
        
        ##################
        ### EDIT ABOVE ###
        ##################

        icirc += 2

    plt.plot(x, y, 'o', label=theta_labels[itheta])

plt.legend()
plt.xlabel(r'$\theta - \phi$')
plt.ylabel(r'$\langle \Psi | \sigma^{\theta}_A \sigma^{\phi}_B | \Psi \rangle$')
```

+++ {"tags": ["raises-exception", "remove-output"]}

ベル状態からの結果と比べてみて、何が言えるでしょうか。

**提出するもの**

- 完成した回路のコード（EDIT BELOW / EDIT ABOVEの間を埋める）とシミュレーション結果によるプロット
- ベル状態と可分状態の混合状態とでの2体相関の違いに関する考察
