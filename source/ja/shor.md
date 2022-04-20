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

+++ {"pycharm": {"name": "#%% md\n"}}

# 素因数分解アルゴリズムを学習する

+++

この実習では**ショアのアルゴリズム**を学習します。名前を聞いたことがある人もいるかもしれませんが、ショアのアルゴリズム{cite}`shor,nielsen_chuang_qft_app`は最も有名な量子アルゴリズムと言っても良いでしょう。ショアのアルゴリズムの元になっている**量子位相推定**と呼ばれる手法を学んだ後、ショアのアルゴリズムの各ステップを実例とともに紹介します。最後に、Qiskitを使用してショアのアルゴリズムを実装し、実際に素因数分解を行ってみます。

```{contents} 目次
---
local: true
---
```

$\newcommand{\ket}[1]{|#1\rangle}$
$\newcommand{\modequiv}[3]{#1 \equiv #2 \pmod{#3}}$
$\newcommand{\modnequiv}[3]{#1 \not\equiv #2 \pmod{#3}}$

(shor_introduction)=
## はじめに

古典計算をはるかに上回る能力を持つ量子計算の一つの例として、最も広く知られているのがショアの量子計算アルゴリズムでしょう。このアルゴリズムが考える問題は、大きな正の数を二つの素数の積に分解するというもので、問題自体はシンプルです。しかし、古典計算では素因数分解の有効なアルゴリズムが知られておらず、数が大きくなると**指数関数的に計算量が増える**と予想されています。
ショアのアルゴリズムを用いれば、同じ問題を**多項式時間で解くことができる**と考えられています（一般的に、問題のサイズに対して多項式で増える計算時間で解くことができれば、それは有効なアルゴリズムだと見なされます）。

古典計算での素因数分解の難しさは、現在広く使われている鍵暗号技術の元になっています。なので、指数関数的に高速なショアのアルゴリズムが量子コンピュータで実現されれば、秘密の情報が暴かれる可能性があります。ショアのアルゴリズムが大きく注目される理由はそこにあります。

+++

(qpe)=
## 量子位相推定

まず、ショアのアルゴリズムの元になっている「**量子位相推定」**（*Quantum Phase Estimation*, QPE）と呼ばれる手法について学びましょう。ショアのアルゴリズムを理解していけば、ショアのアルゴリズムの核心部分は、実はほぼQPEそのものであることが見えてくると思います。QPEの理解には「**量子フーリエ変換**」（*Quantum Fourier Transform*, QFT）の理解が欠かせませんが、QFTについては、この[実習](circuit_from_scratch.ipynb)の問題7、もしくは参考文献[1]を参照してください。

QPEはとても大事な計算手法で、ショアのアルゴリズムだけでなく、いろいろな量子アルゴリズムのサブルーチンとしても使われています。
%（モンテカルロサンプラーの[実習](mc_sampler.ipynb)でも出てきます）。

QPEが考える問題は、「あるユニタリー演算$U$に対して$U\ket{\psi}=e^{2\pi i\theta}\ket{\psi}$となる固有ベクトル$\ket{\psi}$が与えられるとして、その固有値$e^{2\pi i\theta}$の位相$\theta$を求めることができるか？」という問題です。

+++

(qpe_1qubit)=
### 1量子ビットの位相推定
まず、下図にあるような量子回路を考えてみましょう。上側の量子ビットは$\ket{0}$、下側の量子ビットには$U$の固有ベクトル$\ket{\psi}$が初期状態として与えられているとします。

```{image} figs/qpe_1qubit.png
:alt: qpe_1qubit
:width: 300px
:align: center
```

この場合、量子回路の1-3の各ステップでの量子状態は、以下のように表現できます。

- ステップ 1 : $\frac{1}{\sqrt{2}}(\ket{0}\ket{\psi}+\ket{1}\ket{\psi})$
- ステップ 2 : $\frac{1}{\sqrt{2}}(\ket{0}\ket{\psi}+\ket{1} e^{2\pi i\theta}\ket{\psi})$
- ステップ 3 : $\frac{1}{2}\left[(1+e^{2\pi i\theta})\ket{0}+(1-e^{2\pi i\theta})\ket{1}\right]\ket{\psi}$

この状態で上側の量子ビットを測定すると、$|(1+e^{2\pi i\theta})/2|^2$の確率で0、$|(1-e^{2\pi i\theta})/2|^2$の確率で1を測定するでしょう。つまり、この確率の値から位相$\theta$を求めることができるわけです。
しかし、$\theta$の値が小さい場合（$\theta\ll1$）、ほぼ100%の確率で0を測定し、ほぼ0％の確率で1を測定することになるため、100%あるいは0%からのずれを精度良く決めるためには測定を何度も繰り返す必要が出てきます。これではあまり優位性を感じませんね。。

```{hint}
この制御$U$ゲートの部分がショアのアルゴリズムの「オラクル」に対応しています。その後ろには逆量子フーリエ変換が来ますが、1量子ビットの場合は$H$ゲート（$H=H^\dagger$）です。つまり、1量子ビットの量子フーリエ変換は$H$ゲートそのものということですね。
```

本題に戻って、では少ない測定回数でも精度良く位相を決める方法は、何か考えられるでしょうか。。

+++ {"pycharm": {"name": "#%% md\n"}}

(qpe_nqubit)=
### $n$量子ビットの位相推定
そこで、上側のレジスタを$n$量子ビットに拡張した量子回路（下図）を考えてみましょう。

```{image} figs/qpe_wo_iqft.png
:alt: qpe_wo_iqft
:width: 500px
:align: center
```

それに応じて、下側のレジスタにも$U$を繰り返し適用することになりますが、鍵になるのは$U$を2の羃乗回適用するところです。それがどういう意味を持つのかを理解するために、準備として$U^{2^x}\ket{\psi}$が以下のように書けることを見ておきます（まあ当然と言えば当然ですが）。

$$
\begin{aligned}
U^{2^x}\ket{\psi}&=U^{2^x-1}U\ket{\psi}\\
&=U^{2^x-1}e^{2\pi i\theta}\ket{\psi}\\
&=U^{2^x-2}e^{2\pi i\theta2}\ket{\psi}\\
&=\cdots\\
&=e^{2\pi i\theta2^x}\ket{\psi}
\end{aligned}
$$

この量子回路に対して、同様に1, 2, ... $n+1$の各ステップでの量子状態を追跡していくと、以下のようになることが分かります。

- ステップ 1 : $\frac{1}{\sqrt{2^n}}(\ket{0}+\ket{1})^{\otimes n}\ket{\psi}$
- ステップ 2 : $\frac{1}{\sqrt{2^n}}(\ket{0}+e^{2\pi i\theta2^{n-1}}\ket{1})(\ket{0}+\ket{1})^{\otimes n-1}\ket{\psi}$
- $\cdots$
- ステップ $n+1$ : $\frac{1}{\sqrt{2^n}}(\ket{0}+e^{2\pi i\theta2^{n-1}}\ket{1})(\ket{0}+e^{2\pi i\theta2^{n-2}}\ket{1})\cdots(\ket{0}+e^{2\pi i\theta2^0}\ket{1})\ket{\psi}$

ステップ $n+1$後の$n$ビットレジスタの状態をよく見ると、この状態はQFTで$j$を$2^n\theta$としたものと同等であることが分かります。つまり、この$n$ビットレジスタに逆フーリエ変換$\rm{QFT}^\dagger$を適用すれば、状態$\ket{2^n\theta}$が得られるわけです！この状態を測定すれば、$2^n\theta$つまり固有値の位相$\theta$（を$2^n$倍したもの）を求めることができるというのがQPEです（下図）。

(qpe_nqubit_fig)=
```{image} figs/qpe.png
:alt: qpe
:width: 700px
:align: center
```

ただし、一般に$2^n \theta$が整数であるとは限りません。非整数値に対する逆フーリエ変換については、{ref}`補足ページ <nonintegral_fourier>`を参照してください。

+++ {"pycharm": {"name": "#%% md\n"}}

(qpe_imp)=
## QPEの簡単な例を実装する

では実際に、簡単な回路を使って量子位相推定を実装してみましょう。

まず、あるユニタリー演算$U$に対して$U\ket{\psi}=e^{2\pi i\theta}\ket{\psi}$となる固有ベクトル$\ket{\psi}$が必要ですが、ここでは$U$として1量子ビットの$S$ゲート（位相$\sqrt{Z}$ゲート）を考えてみます。$\ket{1}=\begin{pmatrix}0\\1\end{pmatrix}$として、$S\ket{1}=e^{i\pi/2}\ket{1}$となるので$\ket{1}$は$S$の固有ベクトル、$e^{i\pi/2}$がその固有値です。QPEは固有値$e^{2\pi i\theta}$の位相$\theta$を求めるので、$S$の場合は$\theta=1/4$を求めることに相当します。これを回路を使って実際に確かめてみます。

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
# Tested with python 3.8.12, qiskit 0.34.2, numpy 1.22.2
import matplotlib.pyplot as plt
import numpy as np
import math

from qiskit import IBMQ, QuantumCircuit, Aer, transpile
from qiskit.providers.ibmq import least_busy, IBMQAccountCredentialsNotFound
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

# ワークブック独自のモジュール
from qc_workbook.utils import operational_backend
```

+++ {"pycharm": {"name": "#%% md\n"}}

固有ベクトル$\ket{1}$を保持する1量子ビットレジスタと、位相推定用のレジスタ（3量子ビット）を持つ回路を作ります。$\ket{1}$は$\ket{0}$にパウリ$X$を掛けて作り、位相推定用の制御$S$ゲートを$2^x$回適用します。

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
n_mq = 3

# n_mq+1ビット量子回路
qc = QuantumCircuit(n_mq+1, n_mq)

for qubit in range(n_mq):
    qc.h(qubit)
qc.x(n_mq)

# angle/(2*pi)がQPEで求めたい位相
angle = math.pi/2
#angle = 2*math.pi/3

nr = 1
for qubit in range(n_mq):
    for i in range(nr):
        qc.cp(angle, qubit, n_mq)
    nr *= 2
```

+++ {"pycharm": {"name": "#%% md\n"}}

位相推定用のレジスタに逆量子フーリエ変換を適用して、量子ビットを測定します。

この{doc}`実習 <circuit_from_scratch>`の問題7を参考にして、QFTの**逆回路**`qft_dagger(n)`を書いてみてください。引数の$n$は測定用ビットの数`n_count`が入ることに注意します。

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
def qft_dagger(n):
    """逆量子フーリエ変換用の回路"""
    qc = QuantumCircuit(n)

    ##################
    ### EDIT BELOW ###
    ##################

    #qc?

    ##################
    ### EDIT ABOVE ###
    ##################

    qc.name = "QFT^dagger"
    return qc

qc.barrier()
qc.append(qft_dagger(n_mq),list(range(n_mq)))
qc.barrier()
for n in range(n_mq):
    qc.measure(n,n)
qc.draw('mpl')
```

+++ {"pycharm": {"name": "#%% md\n"}}

(qpe_imp_simulator)=
### シミュレータでの実験

シミュレータで実行して、測定結果の確率分布を作ってみます。

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
---
simulator = Aer.get_backend('aer_simulator')
shots = 2048
qc_tr = transpile(qc, backend=simulator)
results = simulator.run(qc_tr, shots=shots).result()
answer = results.get_counts()

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

+++ {"pycharm": {"name": "#%% md\n"}}

答えは10進数で2になっていますね。ここで測定した答えは$\ket{2^n\theta}$だったことを思い出すと、$\theta=2/2^3=1/4$となって正しい$\theta$が得られたことが分かります。

ここで見た量子回路はシンプルですが、いろいろ拡張して振る舞いを理解するのに役立ちます。例えば、以下の問題を調べてみてください。
- （グローバル因子を除いて）$S=R_Z(\pi/2)$ゲートの例を見ましたが、角度を$0<\phi<\pi$の範囲で変えた$R_Z(\phi)$ゲートではどうなるでしょうか？
- 角度の選び方によっては、得られる位相の精度が悪くなります。その場合、どうすればより良い精度で測定できるでしょうか？
- $S$ゲートの場合固有ベクトルは$\ket{1}$でしたが、$\ket{1}$以外の状態を使うとどうなりますか？特に固有ベクトルと線形従属なベクトルを使った場合の振る舞いを見てみてください。

+++ {"pycharm": {"name": "#%% md\n"}}

(qpe_imp_real)=
### 量子コンピュータでの実験

最後に量子コンピュータで実行して、結果を確認してみましょう、以下のようにすることで、現時点でbusyでないマシンを優先的に選んで実行してくれます。

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
# 量子コンピュータで実行する場合
try:
    IBMQ.load_account()
except IBMQAccountCredentialsNotFound:
    IBMQ.enable_account('__paste_your_token_here__')

provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')

backend_list = provider.backends(filters=operational_backend(min_qubits=6))
backend = least_busy(backend_list)
print("least busy backend: ", backend)
```

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
# 最も空いているバックエンドで回路を実行します。キュー内のジョブの実行をモニターします。
qc_tr = transpile(qc, backend=backend, optimization_level=3)
job = backend.run(qc_tr, shots=shots)
job_monitor(job, interval=2)
```

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
# 計算結果
results = job.result()
answer = results.get_counts()
show_distribution(answer)
```

(shor_algo)=
## ショアのアルゴリズム

では、そろそろ本題であるショアのアルゴリズムに入っていきましょう。ショアのアルゴリズムが考えるのは「ある正の合成数$N$を、自明ではない素数の積$N=qp$に分解する」という問題です。

まず、整数の剰余についての表記法をおさらいしておきます。以下のような整数$x$の並びを考えたとき、例えば3で割った余りを$y$とすると

|x|0|1|2|3|4|5|6|7|8|9|
|-|-|-|-|-|-|-|-|-|-|-|
|y|0|1|2|0|1|2|0|1|2|0|

ですね。この時、$\modequiv{x}{y}{3}$と書くものとします（$k$を0以上の整数とすれば、$x=3k+y$と書くこともできます）。

ショアのアルゴリズムの流れを書くと、以下のようなフローチャートになります。黒字の部分は古典計算で実行し、青地の部分を量子コンピュータで実行することになります。「アルゴリズムの一部でしか量子計算を使わないのはなぜ？」と思うかもしれませんが、この青地の部分を古典計算で実行するのが難しいというのがその理由です。つまり、古典計算で十分なところは古典計算にまかして、古典計算では難しい部分を量子計算で実行するというのがその発想の大元にあります。なぜ青地の部分が古典計算では難しいのかは追々明らかになります。

(shor_algo_fig)=
```{image} figs/shor_flow.png
:alt: shor_flow
:width: 500px
:align: center
```

+++ {"pycharm": {"name": "#%% md\n"}}

(factoring_example)=
### 素因数分解の例
簡単な例として、$N=15$の素因数分解をこのアルゴリズムに沿って考えてみましょう。

例えば、15に素な数として仮に$a=7$を選んだとします。そこで$7^x$を15で割った余りを$y$とすると

|x|0|1|2|3|4|5|6|$\cdots$|
|-|-|-|-|-|-|-|-|-|
|y|1|7|4|13|1|7|4|$\cdots$|

のようになります。つまり、$\modequiv{7^r}{1}{15}$を満たす最小の非自明な$r$は4になることが分かります。
$r=4$は偶数なので、$\modequiv{x}{7^{4/2}}{15}$が定義でき、$x=4$です。$x+1 = \modnequiv{5}{0}{15}$なので、

$$
\{p,q\}=\{\gcd(5,15), \gcd(3,15)\}=\{5,3\}
$$

となって、$15=5\times3$が得られました!

+++

(shor_circuit)=
### 量子回路

では次に、$N=15$の素因数分解を実装する量子回路を考えていきましょう。いきなり答えを書いてしまうようですが、回路自体は以下のような構成をしています。

(shor_circuit_fig)=
```{image} figs/shor.png
:alt: shor
:width: 700px
:align: center
```

上段にある4個の量子ビットが測定用のレジスタ、下段の4個の量子ビットが作業用のレジスタに対応します。それぞれのレジスタが4つずつなのは、15が4ビット（$n=4$）で表現できるからです（15の2進数表記 = $1111_2$）。状態は全て$\ket{0}$に初期化されているものとして、測定用ビットの状態を$\ket{x}$、作業用ビットの状態を$\ket{w}$とします。
$U_f$は以下のようなオラクル

```{image} figs/shor_oracle2.png
:alt: shor_oracle2
:width: 300px
:align: center
```

で、作業用ビットの出力状態が$\ket{w\oplus f(x)}$になるものと理解しておきます（詳細は後で説明します）。関数$f(x)$は$f(x) = a^x \bmod N$とします。

では、同様に回路のステップ 1-5ごとに量子状態を見ていきましょう。まずステップ 1で測定用量子ビットの等しい重ね合わせ状態を生成します。各計算基底は0から15までの整数で書いておくことにします。

- ステップ 1 :$\frac{1}{\sqrt{2^4}}\left[\sum_{j=0}^{2^4-1}\ket{j}\right]\ket{0}^{\otimes 4} = \frac{1}{4}\left[\ket{0}+\ket{1}+\cdots+\ket{15}\right]\ket{0}^{\otimes 4}$

オラクル$U_f$を適用した後の状態は、オラクルの定義から以下のようになります。

- ステップ 2 :

$$
\begin{aligned}
&\frac{1}{4}\left[\ket{0}\ket{0 \oplus (7^0 \bmod 15)}+\ket{1}\ket{0 \oplus (7^1 \bmod 15)}+\cdots+\ket{15}\ket{0 \oplus (7^{15} \bmod 15)}\right]\\
=&\frac{1}{4}\left[\ket{0}\ket{1}+\ket{1}\ket{7}+\ket{2}\ket{4}+\ket{3}\ket{13}+\ket{4}\ket{1}+\cdots+\ket{15}\ket{13}\right]
\end{aligned}
$$

ステップ 2の後に作業用ビットを測定します。$\ket{w}$は$\ket{7^x \bmod 15}$、つまり$\ket{1}$, $\ket{7}$, $\ket{4}$, $\ket{13}$のどれかなので、例えば測定の結果13が得られたとします。その場合、測定用ビットの状態は

- ステップ 3 :$\frac{1}{2}\left[\ket{3}+\ket{7}+\ket{11}+\ket{15}\right]$

となります。次に、測定用ビットに逆量子フーリエ変換$\rm{QFT}^\dagger$を適用します。逆量子フーリエ変換はある状態$\ket{j}$を$\ket{j} \to \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1}e^{\frac{-2\pi ijk}{N}}\ket{k}$に変換するので、

- ステップ 4 :

$$
\begin{aligned}
&\frac{1}{2}QFT^\dagger\left[\ket{3}+\ket{7}+\ket{11}+\ket{15}\right]\\
=&\frac{1}{2}\frac1{\sqrt{2^4}}\sum_{k=0}^{2^4-1}\left[e^{\frac{-2\pi i\cdot3k}{2^4}}+e^{\frac{-2\pi i\cdot7k}{2^4}}+e^{\frac{-2\pi i\cdot11k}{2^4}}+e^{\frac{-2\pi i\cdot15k}{2^4}}\right]\ket{k}\\
=&\frac{1}{8}\left[4\ket{0}+4i\ket{4}-4\ket{8}-4i\ket{12}\right]
\end{aligned}
$$

となります。ここで、状態として$\ket{0}$, $\ket{4}$, $\ket{8}$, $\ket{12}$しか出てこないところが鍵で、量子状態の干渉を使って不要な答えの振幅を小さくしているわけです。

- ステップ 5 :最後に測定用ビットを測定すると、0, 4, 8, 12がそれぞれ1/4の確率で得られます。

ステップ 2で$7^x \bmod 15$を計算しているので想像がつきますが、すでに繰り返しの兆候が現れていますね。

(shor_measurement)=
### 測定結果の解析

この測定結果の意味を考察してみましょう。ショアのアルゴリズムの{ref}`回路 <shor_circuit_fig>`と$n$量子ビット位相推定の{ref}`回路 <qpe_nqubit_fig>`の類似性から、ここではこの2つが同一の働きをするものと仮定してみます（以下で補足説明します）。その場合、測定用レジスタは固有値$e^{2\pi i\theta}$の位相$\theta$を$2^4=16$倍したものになっているはずです。つまり、例えば測定用レジスタを測定した結果が仮に4の場合、位相$\theta$は$\theta=4/16=0.25$です。この位相の値は何を意味しているのでしょうか？

ショアのアルゴリズムの量子回路として、これまで$\ket{w}=\ket{0}^{\otimes n}$を初期状態として、$U_f\ket{x}\ket{w}=\ket{x}\ket{w\oplus f(x)}$ $(f(x) = a^x \bmod N)$ となるオラクル$U_f$を考えてきました。この$U_f$を実装するために、以下のようなユニタリー演算子$U$を考えてみます。

$$
U\ket{m} =
\begin{cases}
\ket{am \bmod N)} & 0 \leq m \leq N - 1 \\
\ket{m} & N \leq m \leq 2^n-1
\end{cases}
$$

このユニタリーは、

$$
U^{x}\ket{1} = U^{x-1} \ket{a \bmod N} = U^{x-2} \ket{a^2 \bmod N} = \cdots = \ket{a^x \bmod N}
$$

を満たすので、$w=0$とした$U_f\ket{x}\ket{0}$を$U$を使って実装することができます。

$$
\begin{aligned}
U_f\ket{x}\ket{0}&=\ket{x}\ket{0 \oplus (a^x \bmod N)}\\
&=\ket{x}\ket{a^x \bmod N}\\
&=U^x\ket{x}\ket{1}
\end{aligned}
$$

ここで、天下り的ではありますが

$$
\ket{\psi_s} \equiv \frac{1}{\sqrt{r}}\sum_{k=0}^{r-1}e^{-2\pi isk/r}\ket{a^k \bmod N}
$$

（$s$は$0<s<r-1$の整数）となるベクトル$\ket{\psi_s}$を定義すると、

$$
\frac{1}{\sqrt{r}}\sum_{s=0}^{r-1}\ket{\psi_s}=\ket{1}
$$

が導けると同時に、$\ket{\psi_s}$は$U$の固有ベクトルであり、固有値$e^{2\pi is/r}$を持つことが分かります。

$$
U\ket{\psi_s}=e^{2\pi is/r}\ket{\psi_s}
$$

つまり、ショアのアルゴリズムのオラクル$U_f$による操作は、固有値$e^{2\pi is/r}$を持つ固有ベクトル$\ket{\psi_s}$の重ね合わせ状態$\ket{1}$にユニタリー$U$を$x$回適用することと同等なわけです。量子位相推定の{ref}`回路 <qpe_nqubit_fig>`と比較すれば、やっていることはほぼ同じですね。その後に逆QFTを掛けるということまで考えれば、まさにQPEそのものの操作を行っていることに対応しています。

QPEで得られる位相は何だったかを思い出すと、それは$U\ket{\psi}=e^{2\pi i\theta}\ket{\psi}$なるユニタリー演算$U$と固有ベクトル$\ket{\psi}$に対する固有値$e^{2\pi i\theta}$に含まれる位相$\theta$でした。以上のことから、ショアのアルゴリズムから得られる位相$\theta$は、$s/r$（の整数倍）の意味を持つことも分かるでしょう。

+++

(continued_fractions)=
### 連分数展開
以上の考察から、測定の結果得られる位相は$\theta \approx s/r$であることが分かりました。この結果から位数$r$を求めるために**連分数展開**という手法を使いますが、詳細は他の文献に委ねます（{cite}`nielsen_chuang_qft_app`にもp.230に記述があります）。この手法を使うことで、$\theta$に最も近い分数として$s/r$を求めることができます。

例えば$\theta=0.25$の場合、$r=4$が得られます（頻度は小さいですが$r=8$が得られる可能性もあります）。ここまでできれば、あとは古典計算のみで求める素因数に分解することができますね（{ref}`ここ <factoring_example>`を参照）。

+++

(modular_exponentiation)=
### 剰余指数化
オラクル$U_f$による操作$U_f\ket{x}\ket{w}=\ket{x}\ket{w\oplus f(x)}$とはどういうものか、もう少し考えてみましょう。$f(x) = a^x \bmod N$は、$x$の2進数表記

$$
x=(x_{n-1}x_{n-2}\cdots x_0)_2 = 2^{n-1}x_{n-1}+2^{n-2}x_{n-2}+\cdots+2^0x_0
$$

を使って

$$
\begin{aligned}
f(x) & = a^x \bmod N \\
 & = a^{2^{n-1}x_{n-1}+2^{n-2}x_{n-2}+\cdots+2^0x_0} \bmod N \\
 & = a^{2^{n-1}x_{n-1}}a^{2^{n-2}x_{n-2}}\cdots a^{2^0x_0} \bmod N
\end{aligned}
$$

と書くことができます。つまり、この関数は以下のようなユニタリー演算を考えれば実装することができます。

```{image} figs/shor_oracle.png
:alt: shor_oracle
:width: 600px
:align: center
```

$n$量子ビットQPEの{ref}`回路 <qpe_nqubit_fig>`と比較すれば、このユニタリーはQPEの$U^{2^x}$演算を実装しているものだと分かるでしょう。このように、第2レジスタ（上図では一番下のワイヤに繋がるレジスタ）の内容に、第1レジスタの各ビットで制御された$a^x \bmod N$を適用してQPEの$U^{2^x}$演算を実現する手法を、**剰余指数化**と呼びます。

(shor_imp)=
## アルゴリズムの実装
ここから、ショアのアルゴリズムを実装していきます。

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
from numpy.random import randint
from fractions import Fraction
```

+++ {"pycharm": {"name": "#%% md\n"}}

(shor_imp_period)=
### 位数の発見

まず最初に、繰り返しの位数（周期）を発見するアルゴリズムを見てみます。

$N$を正の整数として、関数$f(x) = a^x \bmod N$の振る舞いを考えます。[ショアのアルゴリズム](#shor_algo_fig)に立ち返ってみると、
ここで$a$は$N$と互いに素な$N$未満の正の整数で、位数$r$は$\modequiv{a^r}{1}{N}$を満たす非ゼロの最小の整数でした。
以下のグラフにこの関数の例を示します。 ポイント間の線は周期性を確認するためのものです。

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
N = 35
a = 3

# プロットするデータを計算する
xvals = np.arange(35)
yvals = [np.mod(a**x, N) for x in xvals]

# matplotlibを使って描画
fig, ax = plt.subplots()
ax.plot(xvals, yvals, linewidth=1, linestyle='dotted', marker='x')
ax.set(xlabel='$x$', ylabel='$%i^x$ mod $%i$' % (a, N),
       title="Example of Periodic Function in Shor's Algorithm")
try: # グラフ上にrをプロット
    r = yvals[1:].index(1) + 1
    plt.annotate(text='', xy=(0,1), xytext=(r,1), arrowprops=dict(arrowstyle='<->'))
    plt.annotate(text='$r=%i$' % r, xy=(r/3,1.5))
except:
    print('Could not find period, check a < N and have no common factors.')
```

(shor_imp_oracle)=
### オラクルの実装
以下では、$N=15$を素因数に分解してみます。上で説明したように、$U\ket{m}=\ket{am \bmod N}$となるユニタリー$U$を$x$回繰り返すことで、オラクル$U_f$を実装します。

練習問題として、$U\ket{m}=\ket{am \bmod 15}$を実行する関数`U_amod15`を以下に実装してください（`U_amod15`は制御ゲートですが、標的ビットのユニタリー演算に対応する部分を書いてみてください）。

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
def c_amod15(a, power):
    """mod 15による制御ゲート"""
    if a not in [2,4,7,8,11,13,14]:
        raise ValueError("'a' must be 2,4,7,8,11,13 or 14")

    U = QuantumCircuit(4)

    ##################
    ### EDIT BELOW ###
    ##################

    #U.?

    ##################
    ### EDIT ABOVE ###
    ##################

    # 以下で制御ゲートに変換
    U = U.to_gate()
    U.name = f"{a}^{power} mod 15"
    c_U = U.control()
    return c_U
```

+++ {"pycharm": {"name": "#%% md\n"}}

`power`は繰り返しの回数を表します。

(shor_imp_circuit)=
### 回路全体の実装

測定用ビットとして、8量子ビットを使います。

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
# 測定用ビットの数
n_count = 8

a = 7
```


+++ {"pycharm": {"name": "#%% md\n"}}


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
# n_count個の測定用量子ビットと、Uを操作するための4つの作業用量子ビットで量子回路を作る
qc = QuantumCircuit(n_count+4, n_count)

# 測定用量子ビットにHゲートをかけて初期化
qc.h(list(range(n_count)))

# 作業用量子レジスタを|1>の状態にする
qc.x(n_count)

# 制御Uゲートを適用
for q in range(n_count):
    qc.append(c_amod15(a, 2**q), [q]+[i+n_count for i in range(4)])

# 逆QFTを適用
qc.append(qft_dagger(n_count), list(range(n_count)))

# 回路を測定
qc.measure(list(range(n_count)), list(range(n_count)))
qc.draw('mpl')
```

シミュレータで実行して、結果を確認してみます。

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
qc = transpile(qc, backend=simulator)
results = simulator.run(qc, shots=2048).result()
answer = results.get_counts()

show_distribution(answer)
```

+++ {"pycharm": {"name": "#%% md\n"}}

(shor_imp_ana)=
### 計算結果の解析
出力された結果から、位相を求めてみます。

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
rows, measured_phases = [], []
for output in answer:
    decimal = int(output, 2)  # 10進数に変換
    phase = decimal/(2**n_count)
    measured_phases.append(phase)
    # これらの値をテーブルの行に追加：
    rows.append([f"{decimal}",
                 f"{decimal}/{2 ** n_count} = {phase:.2f}"])
# 結果を表示
print('Register Output              Phase')
print('----------------------------------')

# 回路を実装できたら、以下のコードをアンコメントして結果を確認
#for i in range(len(rows)):
#    print(f'{rows[i][0]:15s} {rows[i][1]:18s}')
```

得られた位相の情報から、連分数アルゴリズムを使用して$s$と$r$を見つけることができます。Pythonの組み込みの`fractions`(分数)モジュールを使用して、小数を`Fraction`オブジェクトに変換できます。

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
---
rows = []
for phase in measured_phases:
    frac = Fraction(phase).limit_denominator(15)
    rows.append([phase, f"{frac.numerator}/{frac.denominator}", frac.denominator])

# 結果を表示
print('     Phase   Fraction     Guess for r')
print('-------------------------------------')

# 回路を実装できたら、以下のコードをアンコメントして結果を確認
#for i in range(len(rows)):
#    print(f'{rows[i][0]:10f} {rows[i][1]:10s} {rows[i][2]:15d}')
```

`limit_denominator`メソッドを使って、分母が特定の値（ここでは15）を下回る分数で、最も位相の値に近いものを得ています。

測定された結果のうち、2つ（64と192）が正しい答えである$r=4$を与えたことが分かります。

+++

## 参考文献

```{bibliography}
:filter: docname in docnames
```
