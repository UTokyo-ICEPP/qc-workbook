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

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# 【課題】グローバーアルゴリズムを使った量子振幅推定

{doc}`グローバーアルゴリズム <grover>`は求める答えの位相を反転させるという操作を行うことによって、構造を持たないデータベースから答えを見つけることができるというものでした。

グローバーアルゴリズムでは、 最初にアダマール演算子をかけて均等重ね合わせの状態$\ket{s} = H^{\otimes n}\ket{0}^{\otimes n} = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}\ket{x$を作りました。
ここでは、$\ket{s}$の代わりに、$\ket{0}^{\otimes n}$に既知のユニタリー$U$を適用して作った状態$\ket{\psi}=U\ket{0}^{\otimes n}$を持ってくるとしましょう。
この状態$\ket{\psi}$は求める答え$\ket{w}$の成分を含んでいますが、その振幅が分からないとします。

{ref}`ここ <grover_geometry>`で使った2次元平面での記述方法に従うと、$\ket{\psi}$も

$$
\ket{\psi} =: \cos\frac\theta2\ket{w^{\perp}}+\sin\frac\theta2\ket{w} 
$$

と書くことができます。$\sin\frac\theta2$が$\ket{w}$の振幅ですが、この$\theta$の値が分からないという状況です。

この書き方に従えば、オラクル$U_w$は前と同じく$U_w=I-2\ket{w}\bra{w}=\begin{bmatrix}1&0\\0&-1\end{bmatrix}$です。

$U_0=2\ket{0}\bra{0}^{\otimes n}-I$なので、均等重ね合わせ$\ket{s}$の場合はDiffuserは

$$
\begin{aligned} 
U_s &= H^{\otimes n}U_0H^{\otimes n}\\ 
&=2\ket{s}\bra{ s}-I\\ 
\end{aligned}
$$

でしたが、今は$\ket{\psi}$としているため

$$
\begin{aligned} 
U_s &= UU_0U^\dagger\\ 
&=2\ket{\psi}\bra{\psi}-I\\
&=\begin{bmatrix}\cos\theta&\sin\theta\\\sin\theta&-\cos\theta\end{bmatrix}
\end{aligned} 
$$

になります。$\theta$を使った行列表記は前と同じです。

つまり下図にある通り、$\ket{\psi}=\cos\frac\theta2\ket{w^{\perp}}+\sin\frac\theta2\ket{w}$と書けている場合、$G$は$\ket{\psi}$を$\ket{w}$に向かって角度$\theta$だけ回転するという訳で、グローバーアルゴリズムと同じ操作になっています。

```{image} figs/grover_qae1.png
:alt: grover_qae1
:width: 300px
:align: center
```

## 量子振幅推定

$\ket{\psi}$に対するグローバーのアルゴリズムと量子位相推定の方法を組み合わせて振幅の推定を行う方法が[1]で提案されました。

グローバーの反復$G=U_sU_w$は

$$
\begin{aligned} 
G &= U_sU_w\\ 
&= \begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix} \end{aligned} 
$$

なので、

$$
\begin{aligned} 
G\ket{w} &= -\sin\theta\ket{w^{\perp}}+\cos\theta\ket{w}\\ 
G\ket{w^{\perp}} &= \cos\theta\ket{w^{\perp}}+\sin\theta\ket{w}
\end{aligned} 
$$

と書けます。

ここで$\ket{\psi_{\pm}} := \frac{1}{\sqrt{2}}(\ket{w}\pm i\ket{w^{\perp}})$という状態を定義すると

$$
\begin{aligned} 
G\ket{\psi_{\pm}} &= \frac{1}{\sqrt{2}}(G\ket{w}\pm iG\ket{w^{\perp}}) \\
&= \frac{1}{\sqrt{2}}(\cos\theta\pm i\sin\theta)(\ket{w}\pm i\ket{w^{\perp}}) \\
&= e^{\pm i\theta}\ket{\psi_{\pm}}
\end{aligned} 
$$

となり、$\ket{\psi_{\pm}}$は$G$の固有ベクトル、$e^{\pm i\theta}$が固有値であることが分かります。

このことから、ある状態$\ket{\psi}$を準備して、その状態に量子位相推定の方法を使ってグローバーの反復$G$を作用させて位相推定を行えば、角度$\theta/(2\pi)$を求めることができます。この角度が分かれば、求める答え$\ket{w}$の振幅$\sin\frac\theta2$を推定することができます。これが量子振幅推定と呼ばれる方法です。



+++ {"pycharm": {"name": "#%% md\n"}}

## 問題設定

この課題では、振幅が分かっている状態を前もって準備しておくことにします。その状態に量子振幅推定のための量子回路を適用して、振幅が正しく評価できることを示してもらいます。


(qae_qiskit)=
### Qiskitでの実装

まず必要な環境をセットアップします。

```{code-cell} ipython3
---
editable: true
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
slideshow:
  slide_type: ''
---
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Math

# Qiskit関連のパッケージをインポート
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import AerSimulator

# ワークブック独自のモジュール
from qc_workbook.show_state import statevector_expr
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

状態準備として、 3量子ビットの回路でGHZ状態を作ることにします。求める答えの状態$\ket{w}$を$\ket{111}$として、この状態の振幅が$\sin\frac\theta2$となる状態

$$
\ket{\psi}=\cos\frac\theta2\ket{000}+\sin\frac\theta2\ket{111}
$$

を作る量子回路を作ります。


```{code-cell} ipython3
---
editable: true
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
slideshow:
  slide_type: ''
tags: [remove-output]
---
n = 3

state_prep = QuantumCircuit(n)

##################
### EDIT BELOW ###
##################

# state_prepの回路を書いてください

##################
### ABOVE BELOW ###
##################

state_prep.draw('mpl')
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}


次に、state_prepで状態を作って確認します。

```{code-cell} ipython3
---
editable: true
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
slideshow:
  slide_type: ''
---
simulator = AerSimulator(method='statevector')

def get_statevector_array(circuit):
    # 渡された回路のコピーを使う
    circuit = circuit.copy()
    # 量子回路の終状態の状態ベクトルを保存するインストラクション
    circuit.save_statevector()
    # 再び「おまじない」のtranspileをしてから、run()に渡す
    circuit = transpile(circuit, backend=simulator)
    job = simulator.run(circuit)
    result = job.result()
    qiskit_statevector = result.data()['statevector']

    # result.data()['statevector']は通常の配列オブジェクト（ndarray）ではなくqiskit独自のクラスのインスタンス
    # ただし np.asarray() で numpy の ndarray に変換可能
    return np.asarray(qiskit_statevector)

statevector = get_statevector_array(state_prep)
expr = statevector_expr(statevector)
Math(expr)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

次にオラクル $U_w$とDiffuser $U_s$を作る回路を書いて、それらからグローバーの反復$G=U_sU_w$の量子回路を作ってください。


```{code-cell} ipython3
---
editable: true
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
slideshow:
  slide_type: ''
tags: [remove-output]
---

oracle = QuantumCircuit(n)
diffuser = QuantumCircuit(n)

##################
### EDIT BELOW ###
##################

# oracleの回路を書いてください
# diffuserの回路を書いてください

##################
### ABOVE BELOW ###
##################

grover_iter = QuantumCircuit(n)

##################
### EDIT BELOW ###
##################

grover_iter.append(oracle.to_gate(), list(range(n)))
grover_iter.append(diffuser.to_gate(), list(range(n)))

##################
### ABOVE BELOW ###
##################

grover_iter.decompose().draw('mpl')
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}


量子位相推定の回路は以下のようにします。

```{code-cell} ipython3
---
editable: true
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
slideshow:
  slide_type: ''
---
# 測定用レジスタの量子ビット数
n_meas = 3

# 位相測定用のレジスタ
qreg_meas = QuantumRegister(n_meas, name='meas')
# 固有ベクトルを保持するレジスタ
qreg_aux = QuantumRegister(n, name='aux')
# 位相測定の結果が書き出される古典レジスタ
creg_meas = ClassicalRegister(n_meas, name='out')

# 2つの量子レジスタと1つの古典レジスタから量子回路を作る
qc = QuantumCircuit(qreg_meas, qreg_aux, creg_meas)

# それぞれのレジスタを初期化
qc.h(qreg_meas)
qc.barrier()

# 状態準備の回路state_prepを固有ベクトルを保持するレジスタに入れる
qc.append(state_prep, qargs=qreg_aux)

qc.barrier()

##################
### EDIT BELOW ###
##################

# 位相推定用のレジスタを制御ビットとして、制御Gゲートを適用する回路を書いてください。

##################
### ABOVE BELOW ###
##################

```

+++ {"editable": true, "slideshow": {"slide_type": ""}}


逆量子フーリエ変換の回路は以下のものを使います。

```{code-cell} ipython3
---
editable: true
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
slideshow:
  slide_type: ''
---
def qft_dagger(qreg):
    """逆量子フーリエ変換用の回路"""
    qc = QuantumCircuit(qreg)

    for j in range(qreg.size // 2):
        qc.swap(qreg[j], qreg[-1 - j])

    for itarg in range(qreg.size):
        for ictrl in range(itarg):
            power = ictrl - itarg - 1
            qc.cp(-2. * np.pi * (2 ** power), ictrl, itarg)

        qc.h(itarg)

    qc.name = "QFT^dagger"
    return qc

qc.barrier()
qc.append(qft_dagger(qreg_meas), qargs=qreg_meas)
qc.barrier()
qc.measure(qreg_meas, creg_meas)
qc.draw('mpl')
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}


シミュレータで実行して、結果を確かめましょう。

```{code-cell} ipython3
---
editable: true
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
slideshow:
  slide_type: ''
---
from qiskit.primitives import Sampler
sampler = Sampler()

# Now run the job and examine the results
sampler_job = sampler.run(qc)
result = sampler_job.result()

from qiskit.visualization import plot_distribution
plt.style.use('dark_background')
plot_distribution(result.quasi_dists[0])
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}


**提出するもの**
- この問題を解く量子回路
- 45を13に変換するビット押し下げパターンを高確率で見つけていることが分かる結果
