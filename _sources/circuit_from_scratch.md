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

# 単純な量子回路をゼロから書く

+++

{doc}`chsh_inequality`で、量子回路の基本的な書き方と実機での実行の仕方を学びました。そこで出てきたように、QCでは量子レジスタに特定の量子状態を実現することが計算を行うこととなります。そこで今回は、基本的なゲートを組み合わせて様々な量子状態を作る実習を行います。

```{contents} 目次
---
local: true
---
```

$\newcommand{\ket}[1]{|#1\rangle}$

+++

## 状態ベクトルシミュレータで量子状態を調べる

第一回の課題で利用したQCシミュレータ`qasm_simulator`は、実機と同様に量子回路の測定結果をヒストグラムとして返してくれるものでした。測定結果を返すということは、その出力からは量子状態の複素位相を含めた振幅は読み取れません。今回は作った量子状態をより詳細に調べるために、`statevector_simulator`（状態ベクトルシミュレータ）を利用します。状態ベクトルシミュレータは、回路の終状態におけるすべての計算基底の確率振幅、つまりその量子状態に関する最も完全な情報を返します。

シミュレータへのアクセスは`qasm_simulator`同様`Aer`からです。

```{code-cell} ipython3
:tags: [remove-output]

# まずは全てインポート
import numpy as np
from IPython.display import Math
from qiskit import QuantumCircuit, Aer, transpile
# qc_workbookはこのワークブック独自のモジュール（インポートエラーが出る場合はPYTHONPATHを設定するか、sys.pathをいじってください）
from qc_workbook.show_state import statevector_expr

print('notebook ready')
```

```{code-cell} ipython3
simulator = Aer.get_backend('statevector_simulator')
print(simulator.name())
```

シミュレートする回路を作ります。ここでは例として前半で登場した回路1を使います。測定以外のゲートをかけていきます。

```{code-cell} ipython3
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(-3. * np.pi / 4., 1)

circuit.draw('mpl')
```

```{caution}
回路に測定操作(`measure_all()`)を加えてしまうと、`statevector_simulator`を利用して測定前の回路の量子状態を確認することができなくなります。
```

状態ベクトルシミュレーションを実行する際も`transpile`とバックエンドの`run`関数を使い、ジョブオブジェクトから結果を得ます。ただし、今回は計算結果からカウントではなく状態ベクトル（＝量子振幅の配列）を得るので`get_counts`ではなく`result.data()['statevector']`を参照します。

```{code-cell} ipython3
# 再び「おまじない」のtranspileをしてから、run()に渡す
circuit = transpile(circuit, backend=simulator)
job = simulator.run(circuit)
result = job.result()
qiskit_statevector = result.data()['statevector']

# result.data()['statevector']は通常の配列オブジェクト（ndarray）ではなくqiskit独自のクラスのインスタンス
print(type(qiskit_statevector))

# ただし np.asarray() で numpy の ndarray に変換可能
statevector = np.asarray(qiskit_statevector)
print(type(statevector), statevector.dtype)
print(statevector)
```

状態ベクトルは[numpy](https://numpy.org/)の`ndarray`配列オブジェクトとして実装されています。データタイプは128ビットの複素数（実部と虚部それぞれ64ビット）です。配列のインデックスがそのまま二進数としてみなした計算基底の値に対応しています。

状態ベクトルデータから数式の文字列を作る[`statevector_expr`という関数](https://github.com/UTokyo-ICEPP/qc-workbook/tree/master/source/utils/show_state.py)はこのワークブックのレポジトリからインポートしています。

```{code-cell} ipython3
expr = statevector_expr(statevector)

# Math()はLaTeXをタイプセットする関数
Math(expr)
```

前回より、{eq}`回路1 <eqn-circuit1>`は$\frac{1}{\sqrt{2}} (s\ket{00} + c\ket{01} - c\ket{10} + s\ket{11})$（$s:=\sin(\pi/8)=0.271\sqrt{2}$, $c:=\cos(\pi/8)=0.653\sqrt{2}$）という状態を作るもののはずでした。実際に計算された終状態の状態ベクトルが一致しています。

+++

(other_gates)=
## その他のゲート

{doc}`chsh_inequality`で、よく使われる1量子ビットゲートと2量子ビット制御ゲートを紹介しましたが、他にも知っておくと便利なゲートがいくつかあります。

+++

### 1、2量子ビット

```{list-table}
:header-rows: 1
* - ゲート名
  - 対象量子ビット数
  - 説明
  - Qiskitコード
* - $\sqrt{X}$, SX
  - 1
  - 計算基底それぞれに対して、以下の変換をする。
    ```{math}
    \sqrt{X} \ket{0} = \frac{1}{2} \left[(1 + i) \ket{0} + (1 - i) \ket{1}\right] \\
    \sqrt{X} \ket{1} = \frac{1}{2} \left[(1 - i) \ket{0} + (1 + i) \ket{1}\right]
    ```
  - circuit.sx(i)
* - $U_1$, $P$
  - 1
  - パラメータ$\phi$を取り、$\ket{1}$に位相$e^{i\phi}$をかける。$P(\phi)$は$R_{z}(\phi)$と等価なので、1量子ビットゲートとして利用する分にはどちらでも同じ結果が得られる。
  - `circuit.p(phi, i)`
* - $U_3$
  - 1
  - パラメータ$\theta, \phi, \lambda$を取り、単一量子ビットの任意の状態を実現する。
    ```{math}
    U_3(\theta, \phi, \lambda)\ket{0} = \cos\left(\frac{\theta}{2}\right) \ket{0} + e^{i\phi}\sin\left(\frac{\theta}{2}\right) \ket{1} \\
    U_3(\theta, \phi, \lambda)\ket{1} = -e^{i\lambda}\sin\left(\frac{\theta}{2}\right) \ket{0} + e^{i(\phi+\lambda)}\cos\left(\frac{\theta}{2}\right) \ket{1}
    ```
  - `circuit.u3(theta, phi, lambda, i)`
* - $C^i_j[U_1]$, $C^i_j[P]$
  - 2
  - パラメータ$\phi$を取り、ビット$i, j$が1であるような計算基底の位相を$\phi$前進させる。
    ```{math}
    C^i_j[U_1(\phi)] \ket{00} = \ket{00} \\
    C^i_j[U_1(\phi)] \ket{01} = \ket{01} \\
    C^i_j[U_1(\phi)] \ket{10} = \ket{10} \\
    C^i_j[U_1(\phi)] \ket{11} = e^{i\phi} \ket{11}
    ```
  - `circuit.cp(phi, i, j)`
* - SWAP
  - 2
  - 二つの量子ビットの量子状態を交換する。CNOT三回で実装できる。
    ```{math}
    \begin{align}
    & C^0_1[X]C^1_0[X]C^0_1[X] (\alpha \ket{00} + \beta \ket{01} + \gamma \ket{10} + \delta \ket{11}) \\
    = & C^0_1[X]C^1_0[X] (\alpha \ket{00} + \beta \ket{01} + \gamma \ket{11} + \delta \ket{10}) \\
    = & C^0_1[X] (\alpha \ket{00} + \beta \ket{11} + \gamma \ket{01} + \delta \ket{10}) \\
    = & \alpha \ket{00} + \beta \ket{10} + \gamma \ket{01} + \delta \ket{11}
    \end{align}
    ```
  - `circuit.swap(i, j)`
```

+++

### 3量子ビット以上

量子レジスタに対する全ての操作が1量子ビットゲートと2量子ビット制御ゲート（実際にはCNOTのみ）の組み合わせで表現できることはすでに触れましたが、実際に量子回路を書くときには3つ以上の量子ビットにかかるゲートを使用したほうがアルゴリズムをコンパクトに表現できます（最終的には1量子ビットゲートと2量子ビット制御ゲートに分解して実行されます）。

3量子ビット以上にかかるゲートも基本は制御ゲートで、よく使われるのは制御ビットが複数あるような多重制御（multi-controlled）ゲート$C^{ij\dots}_{k}[U]$です。そのうち制御ビットが2つで標的ビット1つに$X$をかける$C^{ij}_{k}[X]$は特別多用され、Toffoliゲートとも呼ばれます。

多重制御ゲートでない3量子ビットゲートの例として、controlled SWAPというものも存在します。制御ビット1つで他の2つの量子ビットの間の量子状態交換を制御します。当然制御ビットを増やしたmulti-controlled SWAPというものも考えられます。

Qiskitで多重制御ゲートに対応するコードは一般に`circuit.mc?(parameters, [controls], target)`という形を取ります。例えばToffoliゲートは

```{code-block} python
circuit.mcx([i, j], k)
```

で、$C^{ij}_{k}[R_y]$は

```{code-block} python
circuit.mcry(theta, [i, j], k)
```

となります。

+++

## 量子状態生成

それでは、これまでに登場したゲートを使って量子状態生成のエクササイズをしましょう。

実は`statevector_expr`関数には回路オブジェクトを直接渡すこともできるので（内部でシミュレータを実行し状態ベクトルオブジェクトを取得する）、ここからはその機能を利用します。また、コード中登場する`amp_norm`や`phase_norm`は、表示される数式において振幅や位相の共通因子をくくりだすために設定されています。

+++

### 問題1: 1量子ビット、相対位相付き

**問題**

1量子ビットに対して状態

$$
\frac{1}{\sqrt{2}}\left(\ket{0} + i\ket{1}\right)
$$

を作りなさい。

```{code-cell} ipython3
:tags: [remove-output]

circuit = QuantumCircuit(1)
##################
### EDIT BELOW ###
##################
circuit.h(0)
# circuit.?
##################
### EDIT ABOVE ###
##################

expr = statevector_expr(circuit, amp_norm=(np.sqrt(0.5), r'\frac{1}{\sqrt{2}}'))
Math(expr)
```

**解答**

````{toggle}
正解は無限に存在しますが、上の回路ですでにアダマールゲートがかかっているので、最も手っ取り早いのは

```{code-block} python
circuit.h(0)
circuit.p(np.pi / 2., 0)
```

でしょう。自分で入力して確認してみてください。

また、上で触れたように、単一量子ビットの場合、$U_3$ゲートを使えば任意の状態を一つのゲート操作で生成できます。$U_3$を使った解答は

```{code-block} python
circuit.u3(np.pi / 2., np.pi / 2., 0., 0)
```

です。
````

+++

### 問題2: ベル状態、相対位相付き

**問題**

2量子ビットに対して状態

$$
\frac{1}{\sqrt{2}}\left(\ket{0} + i\ket{3}\right)
$$

を作りなさい。

```{code-cell} ipython3
:tags: [remove-output]

circuit = QuantumCircuit(2)

##################
### EDIT BELOW ###
##################
# circuit.?
circuit.cx(0, 1)
##################
### EDIT ABOVE ###
##################

expr = statevector_expr(circuit, amp_norm=(np.sqrt(0.5), r'\frac{1}{\sqrt{2}}'))
Math(expr)
```

**解答**

````{toggle}
問題1の結果を利用すると簡単です。初期状態は両ビットとも$\ket{0}$なので、2ビットレジスタに対して問題1の操作をすると状態$1/\sqrt{2}(\ket{0} + i\ket{1})\ket{0}$ができます。ここでビット0から1にCNOTをかけると、$\ket{1}\ket{0} \rightarrow \ket{1}\ket{1}$となるので、望む状態が実現します。

```{code-block} python
circuit.h(0)
circuit.p(np.pi / 2., 0)
circuit.cx(0, 1)
```
````

+++

### 問題3: GHZ状態

**問題**

3量子ビットに対して状態

$$
\frac{1}{\sqrt{2}} (\ket{0} + \ket{7})
$$

を作りなさい。

```{code-cell} ipython3
:tags: [remove-output]

circuit = QuantumCircuit(3)

##################
### EDIT BELOW ###
##################
# circuit.?
circuit.mcx([0, 1], 2)
##################
### EDIT ABOVE ###
##################

expr = statevector_expr(circuit, amp_norm=(np.sqrt(0.5), r'\frac{1}{\sqrt{2}}'))
Math(expr)
```

+++ {"tags": ["hide-cell"]}

**解答**

````{toggle}
Bell状態の生成と発想は同じですが、多重制御ゲート$C^{01}_2[X]$を利用します。

```{code-block} python
circuit.h(0)
circuit.cx(0, 1)
circuit.mcx([0, 1], 2)
```
````

+++

### 問題4: Equal superposition

**問題**

一般の$n$量子ビットに対して状態

$$
\frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n-1} \ket{k}
$$

を作る回路を考え、$n=4$のケースを実装しなさい。

```{code-cell} ipython3
:tags: [remove-output]

num_qubits = 4

circuit = QuantumCircuit(num_qubits)

##################
### EDIT BELOW ###
##################
# circuit.?
##################
### EDIT ABOVE ###
##################

sqrt_2_to_n = 2 ** (num_qubits // 2)
expr = statevector_expr(circuit, amp_norm=(1. / sqrt_2_to_n, r'\frac{1}{%d}' % sqrt_2_to_n))
Math(expr)
```

+++ {"tags": ["hide-cell"]}

**解答**

````{toggle}
Equal superpositionと呼ばれるこの状態は、量子計算でおそらく一番多様される状態です。一見複雑に思えますが、$n$桁の二進数で$0$から$2^n-1$までの数を表現すると、全ての0/1の組み合わせが登場するということを考えると、

$$
\frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n-1} \ket{k} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{1}) \otimes \frac{1}{\sqrt{2}}(\ket{0} + \ket{1}) \otimes \cdots \otimes \frac{1}{\sqrt{2}}(\ket{0} + \ket{1})
$$

と、$n$個の$1/\sqrt{2}(\ket{0} + \ket{1})$の積と等しいことがわかります。つまり各ビットに対してアダマールゲートをかけた状態にほかならず、

```{code-block} python
for i in range(4):
    circuit.h(i)
```
が正解です。
````

+++

### 問題5: 特定の基底の符号を反転させる

**問題**

問題4の4ビットequal superposition状態において、基底$\ket{5}$の符号を反転させなさい。

```{code-cell} ipython3
:tags: [remove-output]

num_qubits = 4

circuit = QuantumCircuit(num_qubits)

##################
### EDIT BELOW ###
##################
for i in range(num_qubits):
    circuit.h(i)
    
# circuit.?
##################
### EDIT ABOVE ###
##################

sqrt_2_to_n = 2 ** (num_qubits // 2)
expr = statevector_expr(circuit, amp_norm=(1. / sqrt_2_to_n, r'\frac{1}{%d}' % sqrt_2_to_n))
Math(expr)
```

+++ {"tags": ["hide-cell"]}

**解答**

````{toggle}
$\ket{5}=\ket{0101}$の符号を反転させるには、4つの量子ビットが0, 1, 0, 1という値を取っている状態のみに作用するゲートを使います。必然的に多重制御ゲートが用いられますが、残念ながら制御ゲートはビットの値が1であるときにしか働きません。そこで、まず$\ket{0101}$を$\ket{1111}$に変換し、その上で$C^{012}_{3}[P(\pi)]$を使います[^mcz]。符号を反転させたら、最初の操作の逆操作をして$\ket{1111}$を$\ket{0101}$に変換します。

```{code-block} python
for i in range(4):
    circuit.h(i)

circuit.x(1)
circuit.x(3)
circuit.mcp(np.pi, [0, 1, 2], 3)
circuit.x(1) # Xの逆操作はX
circuit.x(3)
```

ビット1と3にかかっている$X$ゲートは当然他の基底にも影響するので、例えば$\ket{0011}$は$\ket{0110}$となります。しかし、1と3に$X$をかけて$\ket{1111}$になるのは$\ket{0101}$だけなので、逆操作でちゃんと「後片付け」さえすれば、この一連の操作で$\ket{5}$だけをいじることができます。

[^mcz]: （多重）制御$P$ゲートはどのビットを制御と呼んでも標的と呼んでも同じ作用をするので、$C^{123}_0[P(\pi)]$でもなんでも構いません。
````

+++

### 問題6: Equal superpositionに位相を付ける

**問題**

一般の$n$量子ビットに対して状態

$$
\frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1} e^{2\pi i s k/2^n} \ket{k} \quad (s \in \mathbb{R})
$$

を作る回路を考え、$n=6, s=2.5$のケースを実装しなさい。

```{code-cell} ipython3
:tags: [output_scroll, remove-output]

num_qubits = 6

circuit = QuantumCircuit(num_qubits)

s = 2.5

##################
### EDIT BELOW ###
##################
#for i in range(num_qubits):
#    circuit.h(i)
#
# circuit.?
##################
### EDIT ABOVE ###
##################

sqrt_2_to_n = 2 ** (num_qubits // 2)
amp_norm = (1. / sqrt_2_to_n, r'\frac{1}{%d}' % sqrt_2_to_n)
phase_norm = (2 * np.pi / (2 ** num_qubits), r'\frac{2 \pi i}{%d}' % (2 ** num_qubits))
expr = statevector_expr(circuit, amp_norm, phase_norm=phase_norm)
Math(expr)
```

+++ {"tags": ["hide-cell"]}

**解答**

````{toggle}
この回路は次の問題の量子フーリエ変換と深く関係しますが、量子フーリエ変換では任意の状態$\ket{s}$（このとき$s$は整数）を上の状態に変換するのに対し、この問題では$s$は外から与えられているパラメータです。そのため回路もだいぶ簡単になり、実は制御ゲートなしで実装できます。

問題4と同じように、この状態も個々の量子ビットにおける$\ket{0}$と$\ket{1}$の重ね合わせの積として表現できます。そのためには$k$の二進数表現$k = \sum_{m=0}^{n-1} 2^m k_m \, (k_m=0,1)$を用いて、$\ket{k}$の位相を

$$
\exp\left(2\pi i \frac{sk}{2^n}\right) = \exp \left(2\pi i \frac{s}{2^n} \sum_{m=0}^{n-1} 2^m k_m\right) = \prod_{m=0}^{n-1} \exp \left(2\pi i \frac{s}{2^{n-m}} k_m\right)
$$

と分解します。一方

$$
\ket{k} = \ket{k_{n-1}} \otimes \ket{k_{n-2}} \otimes \cdots \ket{k_0}
$$

なので、

$$
e^{2\pi i sk/2^n} \ket{k} = e^{2\pi i sk_{n-1}/2} \ket{k_{n-1}} \otimes \cdots \otimes e^{2\pi i sk_{1}/2^{n-1}} \ket{k_1} \otimes e^{2\pi i sk_{0}/2^n} \ket{k_{0}}.
$$

最後に、$k_m = 0$ならば$e^{2\pi i sk_{m}/2^n} = 1$であることから、

$$
\frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1} e^{2\pi i sk/2^n} \ket{k} = \frac{1}{\sqrt{2}} \left(\ket{0} + e^{2\pi i s/2} \ket{1}\right) \otimes \cdots \otimes \frac{1}{\sqrt{2}} \left(\ket{0} + e^{2\pi i s/2^n} \ket{1}\right)
$$

がわかります。

```{code-block} python
for m in range(num_qubits):
    circuit.h(m)
    circuit.p(2. * np.pi * s / (2 ** (num_qubits - m)), m)
```

が正解となります。
````

+++

(QFT)=
### 問題7: 量子フーリエ変換

**問題**

$n$量子ビットレジスタの状態$\ket{j} \, (j \in \{0,1,\dots,2^n-1\})$を以下のように変換する回路を考え、$n=6, j=23$のケースを実装しなさい。

$$
\ket{j} \rightarrow \frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1} e^{2\pi i jk/2^n} \ket{k}
$$

```{code-cell} ipython3
:tags: [output_scroll, remove-output]

num_qubits = 6

circuit = QuantumCircuit(num_qubits)

j = 23

## jの２進数表現で値が1になっているビットに対してXを作用させる -> 状態|j>を作る

# まずjの２進数表現を得るために、unpackbitsを利用（他にもいろいろな方法がある）
# unpackbitsはuint8タイプのアレイを引数に取るので、jをその形に変換してから渡している
j_bits = np.unpackbits(np.asarray(j, dtype=np.uint8), bitorder='little')

# 次にj_bitsアレイのうち、ビットが立っているインデックスを得る
j_indices = np.nonzero(j_bits)[0]

# 最後にcircuit.x()
for idx in j_indices:
    circuit.x(idx)
    
## Alternative method
#for i in range(num_qubits):
#    if ((j >> i) & 1) == 1:
#        circuit.x(i)

##################
### EDIT BELOW ###
##################
# circuit.?
##################
### EDIT ABOVE ###
##################

sqrt_2_to_n = 2 ** (num_qubits // 2)
amp_norm = (1. / sqrt_2_to_n, r'\frac{1}{%d}' % sqrt_2_to_n)
phase_norm = (2 * np.pi / (2 ** num_qubits), r'\frac{2 \pi i}{%d}' % (2 ** num_qubits))
expr = statevector_expr(circuit, amp_norm=amp_norm, phase_norm=phase_norm)
Math(expr)
```

この操作は量子フーリエ変換（Quantum Fourier transform, QFT）と呼ばれ、{doc}`Shorの素因数分解 <shor>`を含め多くのアルゴリズムに応用されています{cite}`nielsen_chuang_qft`。現在知られている中で最も重要な量子サブルーチン（アルゴリズムの部品）と言っても過言ではないでしょう。

古典計算機で離散フーリエ変換$j \rightarrow \{e^{2 \pi i j k/2^n}\}_k$を計算するには$\mathcal{O}(n2^n)$回の演算が必要であることが知られています。一方、QFTは最も効率的な実装{cite}`qft_nlogn`で$\mathcal{O}(n \log n)$個のゲートしか利用しません（下の実装では$\mathcal{O}(n^2)$）。つまり、QCは古典計算機に比べて指数関数的に早くフーリエ変換を実行できます。

**解答**

````{toggle}
前の問題同様、基本は$H$ゲート$P$ゲートを使っていきますが、今回は$\ket{j}$という「入力」があるので、制御ゲートを利用する必要があります。

ビットn-1に着目しましょう。前回同様$j_m \, (m=0,\dots,n-1, \, j_m=0,1)$を使って$j$の二進数表現を$j=\sum_{m=0}^{n-1} 2^m j_m$としておきます。ビットn-1の初期状態は$\ket{j_{n-1}}_{n-1}$なので、アダマールゲートをかけると

$$
H\ket{j_{n-1}}_{n-1} = \frac{1}{\sqrt{2}} \left[\ket{0}_{n-1} + e^{2 \pi i \frac{j_{n-1}}{2}} \ket{1}_{n-1}\right]
$$

です。さらに、$j_{0}, \dots j_{n-2}$の情報をこのビットに盛り込むためにビット0からn-2まででそれぞれ制御した$C^m_{n-1}[P]$を使います。ただし、かける位相は制御ビットごとに異なります。

$$
\begin{align}
& C^{n-2}_{n-1}\left[P\left(\frac{2^{n-2} \cdot 2 \pi}{2^n}\right)\right] \cdots C^{0}_{n-1}\left[P\left(\frac{2 \pi}{2^n}\right)\right] (H\ket{j_{n-1}}_{n-1}) \ket{j_{n-2}}_{n-2} \cdots \ket{j_0}_0 \\
= & \frac{1}{\sqrt{2}} \left[\ket{0}_{n-1} + \exp \left(2 \pi i \frac{\sum_{m=0}^{n-1} 2^{m} j_m}{2^n}\right) \ket{1}_{n-1}\right] \ket{j_{n-2}}_{n-2} \cdots \ket{j_0}_0
\end{align}
$$

次に、ビットn-2にも同様の操作をします。ただし、制御はビットn-3以下からのみです。

$$
\begin{align}
& C^{n-3}_{n-2}\left[P\left(\frac{2^{n-2} \cdot 2 \pi}{2^n}\right)\right] \cdots C^{0}_{n-2}\left[P\left(\frac{2 \cdot 2 \pi}{2^n}\right)\right] (H\ket{j_{n-2}}_{n-2}) \cdots \ket{j_0}_0 \\
= & \frac{1}{\sqrt{2}} \left[\ket{0}_{n-2} + \exp \left(2 \pi i \frac{2 \sum_{m=0}^{n-2} 2^{m} j_m}{2^n}\right) \ket{1}_{n-2}\right] \ket{j_{n-3}}_{n-3} \cdots \ket{j_0}_0
\end{align}
$$

これをビット0まで繰り返します。制御は常に位数の小さいビットからです。

$$
\begin{align}
& C^{n-4}_{n-3}\left[P\left(\frac{2^{n-2} \cdot 2 \pi}{2^n}\right)\right] \cdots C^{0}_{n-3}\left[P\left(\frac{2^2 \cdot 2 \pi}{2^n}\right)\right] (H\ket{j_{n-3}}_{n-3}) \cdots \ket{j_0}_0 \\
= & \frac{1}{\sqrt{2}} \left[\ket{0}_{n-3} + \exp \left(2 \pi i \frac{2^2 \sum_{m=0}^{n-3} 2^{m} j_m}{2^n}\right) \ket{1}_{n-3}\right] \ket{j_{n-4}}_{n-4} \cdots \ket{j_0}_0
\end{align}
$$

$$
\dots
$$

$$
\begin{align}
& C^{0}_{1}\left[P\left(\frac{2^{n-2} \cdot 2 \pi}{2^n}\right)\right] (H\ket{j_{1}}_{1}) \ket{j_0}_0 \\
= & \frac{1}{\sqrt{2}} \left[\ket{0}_{1} + \exp \left(2 \pi i \frac{2^{n-2} \sum_{m=0}^{1} 2^{m} j_m}{2^n}\right) \ket{1}_{1}\right] \ket{j_0}_0
\end{align}
$$

$$
\begin{align}
& H\ket{j_0}_0 \\
= & \frac{1}{\sqrt{2}} \left[\ket{0}_0 + \exp \left(2 \pi i \frac{2^{n-1} \sum_{m=0}^{0} 2^{m} j_m}{2^n}\right) \ket{1}_{0}\right].
\end{align}
$$

ここまでの全ての操作を順に適用すると、

$$
(H_0) (C^{0}_{1}[P]H_1) \cdots (C^{n-2}_{n-1}[P] \cdots C^{0}_{n-1}[P]H_{n-1}) \, \ket{j_{n-1}}_{n-1} \ket{j_{n-2}}_{n-2} \cdots \ket{j_{0}}_0 \\
= \frac{1}{\sqrt{2^n}} \left[\ket{0}_{n-1} + \exp \left(2 \pi i \frac{\sum_{m=0}^{n-1} 2^{m} j_m}{2^n}\right) \ket{1}_{n-1}\right] \cdots \left[\ket{0}_0 + \exp \left(2 \pi i \frac{2^{n-1} \sum_{m=0}^{0} 2^{m} j_m}{2^n}\right) \ket{1}_0\right].
$$

$\newcommand{tk}{\tilde{k}}$

例によって全ての量子ビットに$\ket{0}$と$\ket{1}$が現れるので、右辺は振幅$\{c_k\}$を用いて$\sum_{k=0}^{2^n-1} c_k \ket{k}$の形で表せます。後の利便性のために$k$の代わりに整数$\tk$とその二進数表現$\tk_{l}$を使って、

$$
c_{\tk} = \exp \left(\frac{2 \pi i}{2^n} \sum_{l=0}^{n-1}\sum_{m=0}^{l} 2^{n-1-l+m} j_m \tk_l \right).
$$

二重和が厄介な形をしていますが、$j_m, \tk_l \in {0, 1}$なので、$n-1-l+m >= n$のとき和の中身が$2^n$の倍数となり、$\exp (2 \pi i / 2^n \cdot a 2^n) = 1 \, \forall a \in \mathbb{N}$であることを用いると、$m$についての和を$n-1$までに拡張できます。さらに$k_l = \tk_{n-1 - l}$と定義すると、

$$
c_{\tk} = \exp \left(\frac{2 \pi i}{2^n} \sum_{m=0}^{n-1} 2^m j_m \sum_{l=0}^{n-1} 2^l k_l \right).
$$

つまり、ここまでの操作で得られる状態は

$$
\frac{1}{\sqrt{2^n}} \sum_{k_l=0,1} \exp \left(\frac{2 \pi i}{2^n} j \sum_{l=0}^{n-1} 2^l k_l \right) \ket{k_0}_{n-1} \cdots \ket{k_{n-1}}_0
$$

であり、求める状態に対してビット順序が逆転したものになっていることがわかります。したがって、最後にSWAPを使ってビット順序を逆転させれば量子フーリエ変換が完成します。

正解は

```{code-block} python
for itarg in range(num_qubits - 1, -1, -1):
    circuit.h(itarg)
    for ictrl in range(itarg - 1, -1, -1):
        power = ictrl - itarg - 1 + num_qubits
        circuit.cp((2 ** power) * 2. * np.pi / (2 ** num_qubits), ictrl, itarg)
        
for i in range(num_qubits // 2):
    circuit.swap(i, num_qubits - 1 - i)
```

です。
````

+++

## 参考文献

```{bibliography}
:filter: docname in docnames
```
