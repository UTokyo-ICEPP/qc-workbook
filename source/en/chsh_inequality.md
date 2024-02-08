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

# CHSH不等式の破れを確認する

+++

In the first exercise, you will confirm that the quantum computer realizes quantum mechanical states -- entanglement, in particular. You will be introduced to the concepts of quantum mechanics and the fundamentals of quantum computing through this exercise.

```{contents} 目次
---
local: true
---
```

$\newcommand{\ket}[1]{|#1\rangle}$
$\newcommand{\rmI}{\mathrm{I}}$
$\newcommand{\rmII}{\mathrm{II}}$
$\newcommand{\rmIII}{\mathrm{III}}$
$\newcommand{\rmIV}{\mathrm{IV}}$

+++

## 本当に量子コンピュータなのか？

このワークブックの主旨が量子コンピュータ（QC）を使おう、ということですが、QCなんて数年前までSFの世界の存在でした。それが今やクラウドの計算リソースとして使えるというわけですが、ではそもそも私たちがこれから使おうとしている機械は本当にQCなのでしょうか。どうしたらそれが調べられるでしょうか。
The aim of this workbook is to use a quantum computer (QC), but until just a few years ago, QCs were something that only existed in science fiction. They can now be used as computational resources over the cloud -- but are the devices that we are going to use in these exercises really QCs? How can we check?

QCの基本的な仕組みは、「何らかの物理的な系（超電導共振器や冷却原子など）をうまく操作して、求める計算の結果がその系の量子状態に表現されるようにする」ということです。つまり、量子状態が長く保たれてかつ思うように操作できる対象と、「計算」という実体のなさそうなものを具体的な「量子操作」に対応させるアルゴリズムの両方があって初めてQCが成り立ちます。アルゴリズムの部分はこのワークブックを通じて少しずつ紹介していくので、今回は「量子状態が保たれ、それを操作できる」ということを確認してみましょう。
The fundamental way that QCs work is that they skillfully operate a physical system (such as superconducting resonators or cold atoms) such that the results of computations are expressed in the quantum state of the system. In other words, a computer can be called a quantum computer only when it can maintain its quantum states for long periods of time, be used to freely manipulate them, and offer algorithms that link immaterial "calculation" with physical "quantum manipulation." The algorithm portion will be introduced little by little through this workbook, so here, let's confirm that the QC can really maintain quantum states for long periods of time and make it possible to freely manipulate them.

+++

## CHSH不等式

量子力学的状態が実際に存在するかどうかを確かめる実験として、2022年のノーベル物理学賞でも取り上げられたCHSH不等式{cite}`chsh`の検証というものがあります。かいつまんで言うと、CHSH不等式とは「二体系の特定の観測量について、エンタングルメントなど量子力学固有の現象がなければ保たれる不等式」です。やや回りくどいロジックですが、つまりQC（だと考えられる機械）で測ったこの観測量の値がCHSH不等式を破っていれば、その機械は実際に量子現象を利用しているかもしれないということになります。
One way to experimentally verify if quantum mechanical states really exist in a QC is to verify the CHSH inequality [CHSH69,NC00]. To put it briefly, the CHSH inequality is an inequality of specific observables in a two-body system that is maintained unless there are quantum mechanics-specific phenomena such as entanglement.  The logic is somewhat circuitous, but if the values of these observables, when measured by a device that is (believed to be) a QC, violate the CHSH inequality, the device may actually be using quantum phenomena.

通常このような実験を行うには高度なセットアップ（レーザーと非線形結晶、冷却原子など）が必要ですが、クラウドQCではブラウザひとつしか要りません。このワークブックではJupyter NotebookでPythonのプログラムを書き、<a href="https://quantum-computing.ibm.com/" target="_blank">IBM Quantum</a>の量子コンピュータを利用します。
Normally, this type of experiment would require a complicated setup (involving a laser, non-linear crystal, cold atoms, etc.), but with a cloud-based QC, all that's needed is a simple browser. In this workbook, you will use Jupyter Notebook to write a Python program and then use quantum computer through <a href="https://quantum-computing.ibm.com/" target="_blank">IBM Quantum</a>.

+++

## Qiskitの基本構造

IBM QuantumのQCで量子計算を実行するには、IBMの提供する<a href="https://qiskit.org/" target="_blank">Qiskit</a>というPythonライブラリを利用します。Qiskitの基本的な使い方は

1. 使用する量子ビットの数を決め、量子計算の操作（ゲート）をかけて、量子回路を作る
1. 回路を実行して計算結果を得る。ここでは二通りのオプションがあり、
   - 回路をQCの実機に送り、実行させる。
   - 回路をシミュレートする。
1. 計算結果を解析する。

です。以下でこの流れを一通り、重要な概念の説明を混ぜながら実行してみましょう。ただし、今回は実機のみ利用します。回路のシミュレーションに関しては{doc}`第一回の課題 <nonlocal_correlations>`を参照してください。

<a href="https://qiskit.org/" target="_blank">Qiskit</a>, a Python library provided by IBM, is used when implementing quantum calculation on a IBM Q System One QC. The basic procedure for using Qiskit is as follows:
1. Decide on the number of quantum bits to use
2. Apply quantum calculation operations (gates) to the quantum bits to create a quantum circuit
3. Implement the circuit and produce calculation results. Here, there are two options:
   - Send the circuit to the actual QC device and implement it. 
   - Simulate the circuit.
4. Analyze the calculation results.

You will perform this process using the procedure below, which includes explanations of important concepts. In this exercise, you will only use the actual device. Please refer to the {doc}`first assignment <nonlocal_correlations>` for information on simulating circuits.

Qiskitの機能は上のような基本的な量子回路の設計・実行だけではなく、非常に多岐に渡ります。基本的な使い方に関しても多少複雑なところがあるので、わからないことがあれば<a href="https://qiskit.org/documentation/" target="_blank">Qiskitのドキュメンテーション</a>をあたってみましょう。

+++

### 量子ビット、量子レジスタ

**量子ビット**（qubit=キュビット）とは量子コンピュータの基本構成要素のことで、量子情報の入れ物の最小単位です。そして、量子ビットの集まりを量子レジスタと呼びます。
**Quantum bits**, or qubits, are the fundamental elements that make up quantum computers. They are the smallest possible unit of quantum information. A number of quantum bits gathered together is referred to as a quantum register.

量子レジスタは量子コンピュータ中で常に一つの「状態」にあります。量子レジスタの状態を物理学の習わしに従ってしばしば「ケット」という$\ket{\psi}$のような記号で表します[^mixed_state]。量子力学に不慣れな方はこの記法で怯んでしまうかもしれませんが、ケット自体はただの記号なのであまり気にしないでください。別に「枠」なしで$\psi$と書いても、絵文字を使って🔱と書いても、何でも構いません。
Quantum registers in quantum computers always have one "state." Following from a common practice used in physics, the states of quantum registers are referred to as "kets" and indicated with the form $\ket{\psi}$[^mixed_state]. If you are unfamiliar with quantum mechanics, this notation method may look intimidating, but the ket itself is merely a symbol, so there is no need to be overly concerned. You could also write $\psi$ by itself, without enclosing, or even use the 🔱 emoji. Anything would work.

重要なのは各量子ビットに対して2つの**基底状態**が定義できることで、量子計算の習わしではそれらを$\ket{0}$と$\ket{1}$で表し、「計算基底」とも呼びます[^basis]。そして、量子ビットの任意の状態は、2つの複素数$\alpha, \beta$を使って
What's important is that each two **basis states** can be defined for each quantum bit. By quantum calculation custom, these are represented as $\ket{0}$ and $\ket{1}$, and together are called the computational basis[^basis]. Any state of a quantum bit can then be expressed through a "superimposition" of the two basis states, using the complex numbers $\alpha$ and $\beta$, as shown below.

$$
\alpha \ket{0} + \beta \ket{1}
$$

と2つの基底の「重ね合わせ」で表せます。ここで$\alpha, \beta$を確率振幅、もしくは単に**振幅**（amplitude）と呼びます。繰り返しですが別に表記法自体に深い意味はなく、例えば同じ状態を$[\alpha, \beta]$と書いてもいいわけです[^complexarray]。
Here, $\alpha$ and $\beta$ are called probability amplitudes, or simply **amplitudes**. Again, the actual formatting used is not particularly significant. The states could just as well be written as $[\alpha, \beta]$[^complexarray].

量子ビットの任意の状態が2つの複素数で表せるということは、逆に言えば一つの量子ビットには2つの複素数に相当する情報を記録できるということになります。ただこれには少し注釈があって、量子力学の決まりごとから、$\alpha$と$\beta$は
Because any state of a quantum bit can be expressed using two complex numbers, the amount of information that can be contained by any single quantum bit is equivalent to two complex numbers. However, it is also important to note that, due to the rules of quantum mechanics, the relationship between $\alpha$ and $\beta$ must also satisfy the following requirement.

$$
|\alpha|^2 + |\beta|^2 = 1
$$

という関係を満たさなければならず、かつ全体の位相（global phase）は意味を持たない、つまり、任意の実数$\theta$に対して
Furthermore, the global phase is not significant. In other words, for an arbitrary real number $\theta$:

$$
\alpha \ket{0} + \beta \ket{1} \sim e^{i\theta} (\alpha \ket{0} + \beta \ket{1})
$$

（ここで $\sim$ は「同じ量子状態を表す」という意味）である、という制約があります。
(Here, $\sim$ means "expresses the same quantum state")

複素数1つは実数2つで書けるので、$\alpha$と$\beta$をあわせて実数4つ分の情報が入っているようですが、2つの拘束条件があるため、実際の自由度は 4-2=2 個です。自由度の数をあらわにして量子ビットの状態を記述するときは、
A single complex number can be written using two real numbers, so $\alpha$ and $\beta$ together would appear to have the same amount of information as four real numbers, but due to these two constraints, the actual degree of freedom is 4-2=2. The state of the quantum bit, indicating the degree of freedom, can be expressed as shown below.

$$
e^{-i\phi/2}\cos\frac{\theta}{2}\ket{0} + e^{i\phi/2}\sin\frac{\theta}{2}\ket{1}
$$

と書いたりもします。この表記法をブロッホ球表現と呼ぶこともあります。
This method of notation is called Bloch sphere notation.

面白くなるのは量子ビットが複数ある場合です。例えば量子ビット2つなら、それぞれに$\ket{0}, \ket{1}$の計算基底があるので、任意の状態は
Where things get interesting is when one is working with multiple quantum bits. For example, if there are two quantum bits, each has a $\ket{0}$,$\ket{1}$ computational basis, so any state is a superimposition produced using four complex numbers.

$$
\alpha \ket{0}\ket{0} + \beta \ket{0}\ket{1} + \gamma \ket{1}\ket{0} + \delta \ket{1}\ket{1}
$$

と4つの複素数を使った重ね合わせになります。2つの量子ビットの基底を並べた$\ket{0}\ket{0}$のような状態が、このレジスタの計算基底ということになります。$\ket{00}$と略したりもします。
The state in which the basis states of the two quantum bits is $\ket{0}\ket{0}$ is this register's computational basis. It can be abbreviated as $\ket{00}$.

上で登場した量子力学の決まりごとはこの場合
Because of the rules of quantum mechanics discussed above, the following are true.

$$
|\alpha|^2 + |\beta|^2 + |\gamma|^2 + |\delta|^2 = 1
$$

と
and 

$$
\alpha \ket{00} + \beta \ket{01} + \gamma \ket{10} + \delta \ket{11} \sim e^{i\theta} (\alpha \ket{00} + \beta \ket{01} + \gamma \ket{10} + \delta \ket{11})
$$

となります。量子ビットがいくつあっても拘束条件は2つだけです。
No matter how many quantum bits there are, there are only two constraints.

つまり、量子ビット$n$個のレジスタでは、基底の数が$2^n$個で、それぞれに複素数の振幅がかかるので、実数$2 \times 2^n - 2$個分の情報が記録できることになります。これが量子計算に関して「指数関数的」という表現がよく用いられる所以です。
In other words, a register with n quantum bits has $2^n$ basis states, and each has the amplitude of a complex number, so the amount of real numbers of information that can be recorded is $2 \times 2^n - 2$. This is why the word "exponential" is often used when discussing quantum calculation.

量子レジスタの計算基底状態の表記法としては、上に書いたようにケットを$n$個並べたり$n$個の0/1を一つのケットの中に並べたりする方法がありますが、さらにコンパクトなのが、0/1の並び（ビット列）を二進数とみなして、対応する（十進数の）数字で表現する方法です。例えば4量子ビットのレジスタで状態$\ket{0000}$と$\ket{1111}$はそれぞれ$\ket{0}$と$\ket{15}$と書けます。
The computational basis state of the quantum register can be expressed in various ways, such as by lining up $n$ kets, as described above, or by placing $n$ 0/1s within a single ket. An even more compact approach is to look at the string of 0/1s as a binary string (bit string) and representing it with the corresponding (decimal) number. For example, the four quantum bit register states $\ket{0000}$ and $\ket{1111}$ can be expressed as $\ket{0}$ and $\ket{15}$, respectively.

ただし、ここで注意すべきなのは、左右端のどちらが「1の位」なのか事前に約束しないといけないことです。$\ket{0100}$を$\ket{4}$（右端が1の位）とするか$\ket{2}$（左端が1の位）とするかは約束次第です。このワークブックでは、Qiskitでの定義に従って、右端を1の位とします。同時に、レジスタの最初の量子ビットが1の位に対応するようにしたいので、ケットや0/1を並べて計算基底を表現するときは、右から順にレジスタの量子ビットを並べていくことにします。
However, when doing so, care must be taken to first indicate which end, the left or the right, is the ones' place. Whether $\ket{0100}$ becomes $\let{4}$ (because the rightmost place is the ones' place) or $\ket{2}$ (because the leftmost place is the ones' place) depends on which convention is used. In this workbook, in accordance with Qiskit's definition, the rightmost place is the ones' place. We want to make the first quantum bit of the register correspond to the ones' place, so when expressing a computational basis with a string of kets or 0/1s, the register's quantum bits will be arranged in order starting from the right.

Qiskitには量子レジスタオブジェクトがあり、
Qiskit has quantum register objects.
```{code-block} python
from qiskit import QuantumRegister
register = QuantumRegister(4, 'myregister')
```
のように量子ビット数（この場合4）と名前（`'myregister'`）を指定して初期化します。初期状態では、量子ビットはすべて$\ket{0}$状態にあります。レジスタオブジェクトはこのままではあまり使い道がなく、基本的には次に紹介する量子回路の一部として利用します。
These objects are initialized by specifying the number of quantum bits (in this case, 4) and the name ('myregister'), as shown above. By default, all quantum bits will be in the |0⟩ state. There aren't many uses for a register object by itself. Instead, they are, as a general rule, used as parts in quantum circuits, which are introduced below.

[^mixed_state]: 正確には、状態がケットで表せるのはこのレジスタが他のレジスタとエンタングルしていないときに限られますが、ここでは詳細を割愛します。
[^basis]: ここで言う「基底」は線形代数での意味（basis）で、「線形空間中の任意のベクトルを要素の線形和で表せる最小の集合」です。基底となる量子状態だから「基底状態」と呼びます。化学や量子力学で言うところのエネルギーの最も低い状態「基底状態」（ground state）とは関係ありません。
[^complexarray]: 実際に量子計算のシミュレーションをコンピュータ上で行う時などは、量子レジスタの状態を複素数の配列で表すので、この表記の方がよく対応します。
[^mixed_state] Strictly speaking, states can be expressed with kets only when this register is not entangled with other registers, but we'll skip the details here.
[^basis] The "basis state" here is being used in its linear algebra sense, as the "minimum set in which any given vector in linear space can be expressed as the linear sum of its elements." It is called the basis state because it is the base quantum state. This is unrelated to the ground state, used in chemistry and quantum mechanics to refer to the lowest possible energy state.
[^complexarray]] When actually performing quantum calculation simulations on computers, quantum register states are expressed using arrays of complex numbers, so this notation method corresponds closely.


+++

### ゲート、回路、測定

量子計算とは、端的に言えば、量子レジスタに特定の状態を生成し、その振幅を利用することと言えます。
One could even go so far as to say that quantum calculation consists of generating a certain state in a quantum register and then using its amplitude.

とは言っても、いきなり「えいや」と好きな量子状態を作れるわけではなく、パターンの決まった単純操作（$\ket{0}$と$\ket{1}$を入れ替える、ブロッホ球表現での位相角度$\phi$を増減させる、など）を順番に組み合わせて複雑な状態を作っていきます。この単純操作のオペレーションのことを一般に量子**ゲート**といい、ゲートの種類や順番を指定したプログラムに相当するものを量子**回路**と呼びます。
However, you can't simply up and create whatever quantum state you want. Instead, complex states are created by combining, in order, simple operations with defined patterns (such as swapping $\ket{0}$ and $\ket{1}$, amplifying the phase angle $\phi$ in Bloch sphere representation, etc.). These simple operations are generally referred to as quantum **gates**, and programs which specify types and sequences of these gates are called quantum **circuits**.

Qiskitでは、量子回路を`QuantumCircuit`オブジェクトで表します。
Qiskit represents quantum circuits using `QuantumCircuit` objects. Below is an example.
```{code-block} python
from qiskit import QuantumCircuit, QuantumRegister
register = QuantumRegister(4, 'myregister')
circuit = QuantumCircuit(register)
```
という具合です。

作られた量子回路は、量子ビットの数が決まっているもののゲートが一つもない「空っぽ」の状態なので、そこにゲートをかけていきます。例えば下で説明するアダマールゲートをレジスタの2個目の量子ビットに作用させるには
The quantum circuit that was created has a defined number of quantum bits, but it contains no gates -- it is empty. You need to add gates. For example, the following is used to apply a Hadamard gate, explained below, to the second quantum bit of a register.
```{code-block} python
circuit.h(register[1])
```
とします。

上で「振幅を利用する」という曖昧な表現をしましたが、それはいろいろな利用の仕方があるからです。しかし、どんな方法であっても、必ず量子レジスタの**測定**という操作を行います。量子コンピュータから何かしらの情報を得るための唯一の方法が測定です。Qiskitでは`measure_all`というメソッドを使って測定を行います。
Above, we used the vague expression "using its amplitude." The reason for this vagueness is that there are many ways to use the amplitude. However, no matter the method of using the amplitude, the quantum register is always **measured**. Measurement is the only way to obtain information from a quantum computer.
```{code-block} python
circuit.measure_all()
```

測定は量子レジスタの状態を「覗き見る」ような操作ですが、一回の測定操作で具体的に起きることは、各量子ビットに対して0もしくは1という値が得られるというだけです。つまり、量子状態が$2^n$個の計算基底の複雑な重ね合わせであったとしても、測定をすると一つの計算基底に対応するビット列が出てくるだけということになります。しかも、一度測定してしまった量子ビットはもう状態を変えてしまっていて、複雑な重ね合わせは失われてしまいます。
Measurement is like "peeking" at the state of the quantum register, but what specifically happens in each measurement operation is simply obtaining a 0 or a 1 for each quantum bit. In other words, even if there is a computational basis with a complex superposition of 2n quantum states, when measurement is performed, a bit sequence that corresponds to a single computational basis is output. What's more, when a quantum bit is measured, its state changes, and it loses its complex superposition.

ではこの「一つの計算基底」がどの基底なのかというと、実は特殊な場合を除いて決まっていません。全く同じ回路を繰り返し実行して測定すると、毎回ランダムにビット列が決まります。ただし、このランダムさには法則があって、**特定のビット列が得られる確率は、対応する計算基底の振幅の絶対値自乗**となっています。つまり、$n$ビットレジスタの状態$\sum_{j=0}^{2^n-1} c_j \ket{j}$があるとき、測定でビット列$k$が得られる確率は$|c_k|^2$です。根本的には、この確率の分布$|c_0|^2, |c_1|^2, \dots, |c_{2^n-1}|^2$こそが量子計算の結果です。
So what basis state is this "one computational basis"? Actually, except in special cases, it isn't fixed. You can perform measurement repeatedly on the exact same circuit, and the bit sequence will be decided randomly each time. However, this randomness is constrained by rules, and **the percentage likelihood of obtaining a specific bit sequence is the square of the absolute value of the amplitude of the corresponding computational basis**. In other words, when the state of an $n$-bit register is $\sum_{j=0}^{2^n-1} c_j \ket{j}$, the likelihood of bit string $|c_k|^2$ being obtained is $|c_k |^2$.

+++

### 量子計算結果の解析

回路の実行と測定を何度も繰り返して、それぞれのビット列が現れる頻度を記録すれば、だんだん$|c_j|^2$の値がわかっていきます。例えば、2量子ビットの回路を1000回実行・測定して、ビット列00、01、10、11がそれぞれ246回、300回、103回、351回得られたとすれば、統計誤差を考慮して$|c_0|^2=0.24 \pm 0.01$、$|c_1|^2=0.30 \pm 0.01$、$|c_2|^2=0.11 \pm 0.01$、$|c_3|^2=0.35 \pm 0.01$という具合です。しかし、わかるのは$c_j$の絶対値だけで、複素位相については知る術なしです。どうもすっきりしませんが、これが量子コンピュータから情報を得る方法です。
Therefore, if you repeatedly run the circuit, perform measurements, and record the frequency with which each bit sequence occurs, you can gradually determine the value of $|c_j|^2$ and see the quantum state of the register. However, what you will have determined is only the absolute value of $c_j$. You will have no way of knowing the complex phase. That may be unsatisfying, but that's how you obtain information from quantum computers.

逆に、指数関数的な内部の情報量をうまく使って計算を行いつつ、測定という限定的な方法でも答えが読み出せるように工夫するのが、量子アルゴリズム設計の真髄ということになります。例えば理想的には、何か計算の答えが整数$k$であり、それを計算する回路の終状態が単純に$\ket{k}$となるようであれば、一度の測定で答えがわかる（上でいった特殊な場合に相当）わけです。単純に$\ket{k}$でなくても、重ね合わせ$\sum_{j=0}^{2^n-1} c_j \ket{j}$において$|c_k| \gg |c_{j \neq k}|$を実現できれば、数回の測定で答えが高確率でわかります。{doc}`shor`で紹介する位相推定アルゴリズムはその好例です。
Conversely, the true essence of quantum algorithm design lies in skillfully performing calculations using the exponential amount of internal information and using creative techniques to produce results through the limited method of measurement. For example, ideally, if the answer to a computation is the integer k, and the final state of a circuit used to perform that computation is simply |k⟩, then you will know the answer after a single measurement (except in the special cases mentioned above). Even if it is not simply |k⟩, if ∑_(j=0)^(2^((n-1) ))▒〖c_j |j⟩ 〗is possible with superposition |c_k |≫|c_(j≠k) |, the answer can be determined with a high likelihood after several measurements The phase estimation algorithm introduced in the "Learning about the prime factorization algorithm" section is a good example of this.

一度の測定で答えがわかるケースを除いて、基本的には多数回の試行から確率分布を推定することになるので、量子回路を量子コンピュータの実機やシミュレータに送って実行させる時には必ず繰り返し数（「ショット数」と呼びます）を指定します。ショット数$S$でビット列$k$が$n_k$回得られた時、$|c_k|^2$の推定値は$z_k = n_k/S$、その統計誤差は（$S, n_k, S-n_k$が全て十分大きい場合）$\sqrt{z_k (1-z_k) / S}$で与えられます。

+++

(common_gates)=
### よく使うゲート

IBM Q System Oneのような超電導振動子を利用した量子コンピュータでは、実際に使用できるゲートは量子ビット1つにかかるものと2つにかかるものに限定されます。しかし、それらを十分な数組み合わせれば、$n$量子ビットレジスタにおいてどのような状態も実現できることが、数学的に証明されています。
In quantum computers that use superconducting oscillators, like the IBM Q System One, the only gates that can be used are gates that apply to one quantum bit and gates that apply to two. However, it has been mathematically proven that, if these are sufficiently combined, they can create any state for an n-quantum bit register.

#### 1量子ビットの操作

1量子ビットの操作でよく使われるゲートには、以下のようなものがあります。（表中コードの`i`, `j`は量子ビットの番号）
The following gates are often used with single quantum bit operations. (`i` and `j` in the code are the quantum bit number)

```{list-table}
:header-rows: 1
* - Gate name
  - Explanation
  - Qiskit code
* - $X$
  - Switches \ket{0} and \ket{1}.
  - `circuit.x(i)`
* - $Z$
  - Multiplies the amplitude of \ket{1} by -1.
  - `circuit.z(i)`
* - $H$（Hadamard gate）
  - Applies the following transformation to each computational basis.
    ```{math}
    H\ket{0} = \frac{1}{\sqrt{2}} (\ket{0} + \ket{1}) \\
    H\ket{1} = \frac{1}{\sqrt{2}} (\ket{0} - \ket{1})
    ```
    （「量子状態にゲートを作用させる」ことをケットの記法で書くときは、ゲートに対応する記号をケットに左からかけます。）<br/>
    例えば状態$\ket{\psi} = \alpha\ket{0} + \beta\ket{1}$に対しては、
     (When using ket notation to indicate that a gate is applied to a quantum state, the symbol of the gate is written to the left of the ket.)
     For example, $\ket{\psi} = \alpha\ket{0} + \beta\ket{1}$, it would be as follows.

    ```{math}
    \begin{align}
    H\ket{\psi} & = \alpha \frac{1}{\sqrt{2}} (\ket{0} + \ket{1}) + \beta \frac{1}{\sqrt{2}} (\ket{0} - \ket{1}) \\
                & = \frac{1}{\sqrt{2}} (\alpha + \beta) \ket{0} + \frac{1}{\sqrt{2}} (\alpha - \beta) \ket{1}
    \end{align}
    ```
    となる。
  - `circuit.h(i)`
* - $R_{y}$
  - Takes parameter $\theta$ and applies the following transformation to each computational basis.
    ```{math}
    R_{y}(\theta)\ket{0} = \cos\frac{\theta}{2}\ket{0} + \sin\frac{\theta}{2}\ket{1} \\
    R_{y}(\theta)\ket{1} = -\sin\frac{\theta}{2}\ket{0} + \cos\frac{\theta}{2}\ket{1}
    ```
  - `circuit.ry(theta, i)`
* - $R_{z}$
  - Takes parameter $\phi$ and applies the following transformation to each computational basis.
    ```{math}
    R_{z}(\phi)\ket{0} = e^{-i\phi/2}\ket{0} \\
    R_{z}(\phi)\ket{1} = e^{i\phi/2}\ket{1}
  - `circuit.rz(phi, i)`
```

それでは、2量子ビットレジスタの第0ビットに$H, R_y, X$の順にゲートをかけて、最後に測定をする回路をQiskitで書いてみましょう。
Let's create a circuit with Qiskit that applies $H$, $R_y$, and $X$ gates to the 0th bit of a 2-quantum bit register, in that order, and then measures it.

```{code-cell} ipython3
:tags: [remove-output]

# First, import all the necessary python modules
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_provider.accounts import AccountNotFoundError
# qc_workbookはこのワークブック独自のモジュール（インポートエラーが出る場合はPYTHONPATHを設定するか、sys.pathをいじってください）
from qc_workbook.utils import operational_backend

print('notebook ready')
```

```{code-cell} ipython3
circuit = QuantumCircuit(2) # You can also create a circuit by specifying the number of bits, without using a register
circuit.h(0) # In that case, directly specify the number of the quantum bit for the gate, not register[0]
circuit.ry(np.pi / 2., 0) #　θ = π/2
circuit.x(0)
# 実際の回路では出力を得るためには必ず最後に測定を行う
circuit.measure_all()

print(f'This circuit has {circuit.num_qubits} qubits and {circuit.size()} operations')
```

最後のプリント文で、ゲートが3つなのにも関わらず "5 operations" と出てくるのは、各量子ビットの測定も一つのオペレーションとして数えられるからです。
The reason the last print statement says "5 operations" despite there only being three gates is that each quantum bit measurement is also counted as an operation.

量子計算に慣れる目的で、この$H, R_y(\pi/2), X$という順の操作で第0ビットに何が起こるかを辿ってみましょう。初期状態は$\ket{0}$で、ケット記法では操作は左からかけていく（ゲート操作が右から順に並ぶ）ので、$X R_y(\pi/2) H \ket{0}$を計算することになります。
Let's look, in order, at what happens to the 0th bit through these $H$, $R_y(\pi/2)$, $X$ operations in order to become more accustomed to quantum calculation. From the default \ket{0} state, the gates are applied from the left, according to gate notation, so $X R_y(\pi/2) H \ket{0}$ is calculated.

$$
\begin{align}
X R_y\left(\frac{\pi}{2}\right) H \ket{0} & = X R_y\left(\frac{\pi}{2}\right) \frac{1}{\sqrt{2}}(\ket{0} + \ket{1}) \\
& = \frac{1}{\sqrt{2}} X \left[\left(\cos\left(\frac{\pi}{4}\right)\ket{0} + \sin\left(\frac{\pi}{4}\right)\ket{1}\right) + \left(-\sin\left(\frac{\pi}{4}\right)\ket{0} + \cos\left(\frac{\pi}{4}\right)\ket{1}\right)\right] \\
& = \frac{1}{\sqrt{2}} X \frac{1}{\sqrt{2}} \left[\left(\ket{0} + \ket{1}\right) + \left(-\ket{0} + \ket{1}\right)\right] \\
& = X \ket{1} \\
& = \ket{0}
\end{align}
$$

なので、結局$\ket{0}$状態に戻る操作でした。
These operations eventually take us back to the $\ket{0}$ state.

#### 2量子ビットの操作

2量子ビットの操作は、量子ビットの超電導素子での実装の都合上、全て「制御ゲート」（controlled gates）という方式で行われます。この方式では、2つのビットのうち片方を制御（control）、もう片方を標的（target）として、制御ビットが1の時だけ標的ビットに何らかの操作がかかります。
Operations on 2-quantum bits are always performed using controlled gates for reasons related to how quantum bits are implemented using superconducting elements. With this method, one of the two bits is called the control and the other is called the target. An operation is performed on the target bit only when the control bit's value is 1.

例として、任意の1ビットゲート$U$を制御ゲート化した$C^i_j[U]$を考えます。ここで$i$が制御、$j$が標的ビットとします。ケットの添字でビットの番号を表して（reminder: 並べて書くときは右から順に番号を振ります）
For example, consider a 1-bit gate, U, as $C^i_j[U]$, a controlled gate. Here, $i$ is the control bit and $j$ is the target bit. Representing the bit number with a gate subscript produces the following. (Reminder: when writing in a line, numbers are assigned in order starting from the right.)

$$
\begin{align}
C^1_0[U](\ket{0}_1\ket{0}_0) & = \ket{0}_1\ket{0}_0 \\
C^1_0[U](\ket{0}_1\ket{1}_0) & = \ket{0}_1\ket{1}_0 \\
C^1_0[U](\ket{1}_1\ket{0}_0) & = \ket{1}_1U\ket{0}_0 \\
C^1_0[U](\ket{1}_1\ket{1}_0) & = \ket{1}_1U\ket{1}_0
\end{align}
$$

です。

上で紹介した頻出する1ビットゲート$X, Z, H, R_y, R_z$のうち、$H$以外は制御ゲート化バージョンもよく使われます。特に$C[X]$はCXやCNOTとも呼ばれ、量子計算の基本要素として多様されます。実際、全ての2量子ビット制御ゲートはCNOTと1量子ビットゲートの組み合わせに分解できます。
Of the $X$, $Z$, $H$, $R_y$, and $R_z$ frequently used 1-bit gates introduced above, all other than the $H$ gate also have frequently used controlled gate versions. $C[X]$, also known as CX and CNOT, is particularly often used as a basic element in quantum calculation. In fact, all 2-quantum bit controlled gates can be broken down into combinations of CNOT gates and 1-quantum bit gates.

```{list-table}
:header-rows: 1
* - Gate name
  - Explanation
  - Qiskit code
* - $C^i_j[X]$, CX, CNOT
  - Performs the operation of gate $X$ on bit $j$ for a computational basis in which the value of bit $i$ is 1. 
  - `circuit.cx(i, j)`
* - $C^i_j[Z]$
  - Reverses the sign of the computational basis when the values of bits $i$ and $j$ are 1.
  - `circuit.cz(i, j)`
* - $C^i_j[R_{y}]$
  - Obtains parameter $\theta$ and performs the operation of gate $R_y$ on bit $j$ for a computational basis in which the value of bit $i$ is 1.
  - `circuit.cry(theta, i, j)`
* - $C^i_j[R_{z}]$
  - Obtains parameter $\phi$ and performs the operation of gate $R_z$ on bit $j$ for a computational basis in which the value of bit $i$ is 1.
  - `circuit.crz(phi, i, j)`
```

Qiskitで2ビットレジスタに制御ゲートを用い、計算基底$\ket{0}, \ket{1}, \ket{2}, \ket{3}$の振幅の絶対値自乗が$1:2:3:4$の比になるような状態を作ってみましょう。さらに$C^0_1[Z]$ゲートを使って$\ket{3}$だけ振幅の符号が他と異なるようにします。
Let's use Qiskit to apply a controlled gate to a 2-bit register and make the squared absolute value of the amplitudes of computational bases $\ket{0}$, $\ket{1}$, $\ket{2}, and $\ket{3}$ have the ratio $1:2:3:4$. Furthermore, let's use a $C_1^0[Z]$ gate to make the sign of the amplitude of $\ket{3}$ alone different than the others.

```{code-cell} ipython3
theta1 = 2. * np.arctan(np.sqrt(7. / 3.))
theta2 = 2. * np.arctan(np.sqrt(2.))
theta3 = 2. * np.arctan(np.sqrt(4. / 3))

circuit = QuantumCircuit(2)
circuit.ry(theta1, 1)
circuit.ry(theta2, 0)
circuit.cry(theta3 - theta2, 1, 0) # C[Ry] 1 is the control and 0 is the target
circuit.cz(0, 1) # C[Z] 0 is the control and 1 is the target (in reality, for C[Z], the results are the same regardless of which the control is)

circuit.measure_all()

print(f'This circuit has {circuit.num_qubits} qubits and {circuit.size()} operations')
```

This is a little complex, but let's follow the computation steps in order. First, given the definitions of angles $θ_1$,$θ_2$, and $θ_3$, the following relationships are satisfied.

$$
\begin{align}
R_y(\theta_1)\ket{0} & = \sqrt{\frac{3}{10}} \ket{0} + \sqrt{\frac{7}{10}} \ket{1} \\
R_y(\theta_2)\ket{0} & = \sqrt{\frac{1}{3}} \ket{0} + \sqrt{\frac{2}{3}} \ket{1} \\
R_y(\theta_3 - \theta_2)R_y(\theta_2)\ket{0} & = R_y(\theta_3)\ket{0} = \sqrt{\frac{3}{7}} \ket{0} + \sqrt{\frac{4}{7}} \ket{1}.
\end{align}
$$

したがって、
Therefore:

$$
\begin{align}
& C^1_0[R_y(\theta_3 - \theta_2)]R_{y1}(\theta_1)R_{y0}(\theta_2)\ket{0}_1\ket{0}_0 \\
= & C^1_0[R_y(\theta_3 - \theta_2)]\left(\sqrt{\frac{3}{10}} \ket{0}_1 + \sqrt{\frac{7}{10}} \ket{1}_1\right) R_y(\theta_2)\ket{0}_0\\
= & \sqrt{\frac{3}{10}} \ket{0}_1 R_y(\theta_2)\ket{0}_0 + \sqrt{\frac{7}{10}} \ket{1}_1 R_y(\theta_3)\ket{0}_0 \\
= & \sqrt{\frac{3}{10}} \ket{0}_1 \left(\sqrt{\frac{1}{3}} \ket{0}_0 + \sqrt{\frac{2}{3}} \ket{1}_0\right) + \sqrt{\frac{7}{10}} \ket{1}_1 \left(\sqrt{\frac{3}{7}} \ket{0}_0 + \sqrt{\frac{4}{7}} \ket{1}_0\right) \\
= & \sqrt{\frac{1}{10}} \ket{00} + \sqrt{\frac{2}{10}} \ket{01} + \sqrt{\frac{3}{10}} \ket{10} + \sqrt{\frac{4}{10}} \ket{11}
\end{align}
$$

最初の行で、ビット0と1にかかる$R_y$ゲートをそれぞれ$R_{y0}, R_{y1}$と表しました。
On the first line, the $R_y$ gates that are applied to bits 0 and 1 are denoted as $R_{y0}$ and $R_{y1}$. 

最後に$C[Z]$をかけると、$\ket{11}$だけ符号が反転します。
When $C[Z]$ is applied at the end, only the sign of $\ket{11}$ is reversed.

+++

### 回路図の描き方と読み方

量子回路を可視化する方法として、「回路図」の標準的な描き方が決まっています。Qiskitでは`QuantumCircuit`オブジェクトの`draw()`というメソッドを使って自動描画できます。
There is a standard way of drawing the circuit diagrams that are used to visualize quantum circuits. With Qiskit, you can use the QuantumCircuit object draw() method to automatically draw circuit diagrams.

```{code-cell} ipython3
circuit.draw('mpl')
```

　ここで`draw()`の引数`'mpl'`はmatplotlibライブラリを使ってカラーで描くことを指定しています。実行環境によっては対応していないこともあるので、その場合は引数なしの`draw()`を使います。結果は`mpl`の場合に比べて見劣りしますが、内容は同じです。
  Here, the `draw()` argument `'mpl'` is used to draw the circuit diagram in color, using the matplotlib library. Some operating environments may not support this. In those cases, use `draw()` by itself, with no argument. The result will not be as visually appealing as the circuit diagrams produced by `mpl`, but the content will be the same.
```{code-cell} ipython3
circuit.draw()
```

回路図は左から右に読んでいきます。水平の2本の実線が上からそれぞれ第0、第1量子ビットに対応し、その上にかぶさっている四角がゲート、最後にある矢印が下に伸びている箱が測定を表します。1ビットゲートから伸びている先端の丸い縦線は制御を表します。一番下の二重線は「古典レジスタ」（量子現象のない物理学を「古典物理学」と呼ぶので、量子でない通常のコンピュータにまつわる概念にはよく「古典 classical」という接頭辞をつけます）に対応し、測定結果の0/1が記録される部分です。
Circuit diagrams are read from left to right. The two horizontal solid lines represent, from top to bottom, quantum bits 0 and 1. The squares on top of the lines are gates. The boxes at the end, with arrows extending downwards, represent measurements. The vertical lines with circles at their ends extending from the 1 bit gate represent control. The double line at the very bottom corresponds to the "classical register," and is the portion where measurement results of 0 or 1 are recorded.

+++

## CHSH不等式を計算する回路を書く

それではいよいよ本題に入りましょう。CHSH不等式を「ベル状態」$1/\sqrt{2}(\ket{00} + \ket{11})$で検証します。ベル状態は「どちらの量子ビットについても$\ket{0}$でも$\ket{1}$でもない状態」つまり、全体としては一つの定まった（純粋）状態であるにも関わらず、部分を見ると純粋でない状態です。このような時、**二つの量子ビットはエンタングルしている**といいます。エンタングルメントの存在は量子力学の非常に重要な特徴です。
Let us now get to the main part of this exercise. The CHSH inequality is an inequality involving four observables of a two-body system, so we will prepare four 2-bit circuits. Each will represent the Bell state $1/\sqrt{2}(\ket{00} + \ket{11})$. The Bell state is one in which "the states of both of the quantum bits are neither $\ket{0}$ nor $\ket{1}$." In other words, despite the fact that the overall state is pure, its individual parts are not. In situations such as this, we say that the two quantum bits are entangled. The existence of entanglement is an extremely important feature of quantum mechanics.

ベル状態はアダマールゲートとCNOTゲートを組み合わせて作ります。詳しい説明は{doc}`課題 <nonlocal_correlations>`に譲りますが、CHSH不等式の検証用の観測量を作るために、4つの回路I, II, III, IVを使います。回路IとIIIでは量子ビット1に対し測定の直前に$R_y(-\pi/4)$、IIとIVでは同様に$R_y(-3\pi/4)$を作用させます。また回路IIIとIVでは量子ビット0に$R_y(-\pi/2)$を同じく測定の直前に作用させます。4つの回路を一度にIBMQに送るので、`circuits`というリストに回路を足していきます。
Let's create a Bell state by combining Hadamard gates and CNOT gates. We will create four circuits, so we will use a loop to add circuits to the circuits array.

```{code-cell} ipython3
circuits = []

# 回路I - H, CX[0, 1], Ry(-π/4)[1]をかける
circuit = QuantumCircuit(2, name='circuit_I')
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(-np.pi / 4., 1)
circuit.measure_all()
# 回路リストに追加
circuits.append(circuit)

# 回路II - H, CX[0, 1], Ry(-3π/4)[1]をかける
circuit = QuantumCircuit(2, name='circuit_II')
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(-3. * np.pi / 4., 1)
circuit.measure_all()
# 回路リストに追加
circuits.append(circuit)

# 回路III - H, CX[0, 1], Ry(-π/4)[1], Ry(-π/2)[0]をかける
circuit = QuantumCircuit(2, name='circuit_III')
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(-np.pi / 4., 1)
circuit.ry(-np.pi / 2., 0)
circuit.measure_all()
# 回路リストに追加
circuits.append(circuit)

# 回路IV - H, CX[0, 1], Ry(-3π/4)[1], Ry(-π/2)[0]をかける
circuit = QuantumCircuit(2, name='circuit_IV')
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(-3. * np.pi / 4., 1)
circuit.ry(-np.pi / 2., 0)
circuit.measure_all()
# 回路リストに追加
circuits.append(circuit)

# draw()にmatplotlibのaxesオブジェクトを渡すと、そこに描画してくれる
# 一つのノートブックセルで複数プロットしたい時などに便利
fig, axs = plt.subplots(2, 2, figsize=[12., 6.])
for circuit, ax in zip(circuits, axs.reshape(-1)):
    circuit.draw('mpl', ax=ax)
    ax.set_title(circuit.name)
```

それぞれの回路で2ビットレジスタの基底$\ket{00}, \ket{01}, \ket{10}, \ket{11}$が現れる確率を計算してみましょう。
Let's calculate the likelihood of basis states $\ket{00}$, $\ket{01}$, $\ket{10}$, and $\ket{11}$ appearing in the 2-bit register of each circuit. Circuit 0's state is as follows:

回路Iの状態は
The state for circuit 1 is as follows:

$$
\begin{align}
R_{y1}\left(-\frac{\pi}{4}\right) C^0_1[X] H_0 \ket{0}_1\ket{0}_0 = & R_{y1}\left(-\frac{\pi}{4}\right) \frac{1}{\sqrt{2}} (\ket{0}_1\ket{0}_0 + \ket{1}_1\ket{1}_0) \\
= & \frac{1}{\sqrt{2}} \big[(c\ket{0}_1 - s\ket{1}_1)\ket{0}_0 + (s\ket{0}_1 + c\ket{1}_1)\ket{1}_0\big]\\
= & \frac{1}{\sqrt{2}} (c\ket{00} + s\ket{01} - s\ket{10} + c\ket{11}).
\end{align}
$$

簡単のため$c = \cos(\pi/8), s = \sin(\pi/8)$とおきました。
For simplicity's sake, we have set $c = \cos(\pi/8)$ and $s = \sin(\pi/8)$.

したがって回路Iでの確率$P^{\rmI}_{l} \, (l=00,01,10,11)$は
Therefore probability $P^{\rmI}_{l} \, (l=00,01,10,11)$ for circuit I is as follows:

$$
P^{\rmI}_{00} = P^{\rmI}_{11} = \frac{c^2}{2} \\
P^{\rmI}_{01} = P^{\rmI}_{10} = \frac{s^2}{2}
$$

同様に、回路IIの状態は
Likewise, the state for circuit II is as follows:

```{math}
:label: eqn-circuit1
R_{y1}\left(-\frac{3\pi}{4}\right) \frac{1}{\sqrt{2}} (\ket{0}_1\ket{0}_0 + \ket{1}_1\ket{1}_0) = \frac{1}{\sqrt{2}} (s\ket{00} + c\ket{01} - c\ket{10} + s\ket{11})
```

で確率$P^{\rmII}_{l}$は
Therefore probability $P^{\rmII}_{l}$ is:

$$
P^{\rmII}_{00} = P^{\rmII}_{11} = \frac{s^2}{2} \\
P^{\rmII}_{01} = P^{\rmII}_{10} = \frac{c^2}{2}
$$

です。回路IIIの状態は
Circuit III's state is as follows:

$$
\begin{align}
& R_{y1}\left(-\frac{\pi}{4}\right) R_{y0}\left(-\frac{\pi}{2}\right) \frac{1}{\sqrt{2}} (\ket{0}_1\ket{0}_0 + \ket{1}_1\ket{1}_0) \\
= & \frac{1}{\sqrt{2}} \left[ \frac{1}{\sqrt{2}} (c\ket{0}_1 - s\ket{1}_1) (\ket{0}_0 - \ket{1}_0) + \frac{1}{\sqrt{2}} (s\ket{0}_1 + c\ket{1}_1) (\ket{0}_0 + \ket{1}_0) \right] \\
= & \frac{1}{2} \big[ (s+c)\ket{00} + (s-c)\ket{01} - (s-c)\ket{10} + (s+c)\ket{11} \big]
\end{align}
$$

で確率$P^{\rmIII}_{l}$は
Probability $P^{\rmIII}_{l}$ is:

$$
P^{\rmIII}_{00} = P^{\rmIII}_{11} = \frac{(s + c)^2}{4} \\
P^{\rmIII}_{01} = P^{\rmIII}_{10} = \frac{(s - c)^2}{4}
$$

同様に回路IVの状態と確率$P^{\rmIV}_l$は
Likewise, the state and probability $P^{\rmIV}_l$ of circuit IV are:

$$
\begin{align}
& R_{y1}\left(-\frac{3\pi}{4}\right) R_{y0}\left(-\frac{\pi}{2}\right) \frac{1}{\sqrt{2}} (\ket{0}_1\ket{0}_0 + \ket{1}_1\ket{1}_0) \\
= & \frac{1}{2} \big[ (s+c)\ket{00} - (s-c)\ket{01} + (s-c)\ket{10} + (s+c)\ket{11} \big]
\end{align}
$$

$$
P^{\rmIV}_{00} = P^{\rmIV}_{11} = \frac{(s + c)^2}{4} \\
P^{\rmIV}_{01} = P^{\rmIV}_{10} = \frac{(s - c)^2}{4}
$$

となります。

それぞれの回路でビット0と1で同じ値が観測される確率$P^{i}_{00} + P^{i}_{11}$から異なる値が観測される確率$P^{i}_{01} + P^{i}_{10}$を引いた値を$C^{i}$と定義します。
The probability $P^{i}_{00} + P^{i}_{11}$ of the same value being observed for bits 0 and 1 on each circuit minus the probability $P^{i}_{01} + P^{i}_{10}$ of different values being observed is defined as $C^i$.

$$
C^{\rmI} = c^2 - s^2 = \cos\left(\frac{\pi}{4}\right) = \frac{1}{\sqrt{2}} \\
C^{\rmII} = s^2 - c^2 = -\frac{1}{\sqrt{2}} \\
C^{\rmIII} = 2sc = \sin\left(\frac{\pi}{4}\right) = \frac{1}{\sqrt{2}} \\
C^{\rmIV} = 2sc = \frac{1}{\sqrt{2}}
$$

なので、これらの組み合わせ$S = C^{\rmI} - C^{\rmII} + C^{\rmIII} + C^{\rmIV}$の値は$2\sqrt{2}$です。
Therefore, combining these, the value of $S = C^{\rmI} - C^{\rmII} + C^{\rmIII} + C^{\rmIV}$ is $2\sqrt{2}$.

実は、エンタングルメントが起こらない場合、この観測量$S$の値は2を超えられないことが知られています。例えば$R_y$ゲートをかける前の状態がベル状態ではなく、確率$\frac{1}{2}$で$\ket{00}$、確率$\frac{1}{2}$で$\ket{11}$という「混合状態」である場合、
Actually, if entanglement does not occur, the value of this observable $S$ is known to not exceed 2. For example, if the state before the $R_y$ gate is applied is not a Bell state, there is a $\frac{1}{2}$ probability that the value is $\ket{00}$ and a $\frac{1}{2}$ probability that it is $\ket{11}$, a mixed state.

$$
C^{\rmI} = \frac{1}{\sqrt{2}} \\
C^{\rmII} = -\frac{1}{\sqrt{2}} \\
C^{\rmIII} = 0 \\
C^{\rmIV} = 0
$$

となり、$S = \sqrt{2} < 2$です。これがCHSH不等式です。
Therefore, $S = \sqrt{2} < 2$. This is the CHSH inequality.

それでは、IBMQの「量子コンピュータ」が実際にエンタングル状態を生成できるのか、上の四つの回路から$S$の値を計算して確認してみましょう。
So let's check if the IBMQ "quantum computer" really generates entangled states by using the four circuits above to calculate the value of $S$.

+++

## 回路を実機で実行する

まずはIBMQに認証・接続します。IBM Quantum Experience (IBM Quantumウェブサイト上のJupyter Lab)で実行している、もしくは自分のラップトップなどローカルの環境ですでに{ref}`認証設定が保存されている <install_token>`場合は`provider = IBMProvider()`で接続ができます。設定がない場合は`IBMProvider`のコンストラクタに{ref}`トークン <install_token>`を渡してIBMQに接続します。

```{code-cell} ipython3
:tags: [remove-output, raises-exception]

# 利用できるインスタンスが複数ある場合（Premium accessなど）はここで指定する
# instance = 'hub-x/group-y/project-z'
instance = None

try:
    provider = IBMProvider(instance=instance)
except AccountNotFoundError:
    provider = IBMProvider(token='__paste_your_token_here__', instance=instance)
```

認証が済んだら、利用する量子コンピュータ（「バックエンド」と呼びます）を選びます。
Once authentication has been completed, choose the quantum computer you wish to use (called a "backend").

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

# バックエンド（実機）のうち量子ビット数2個以上のもののリストをプロバイダから取得する
# A list of backends (actual devices) with 2 or more quantum bits is acquired from the provider
# operational_backendはこのワークブック用にqc_workbook.utilsで定義された関数
backend_list = provider.backends(filters=operational_backend(min_qubits=2))

# リストの中から一番空いているものを選ぶ
# The one with the highest availability is selected
backend = least_busy(backend_list)

print(f'Jobs will run on {backend.name()}')
```

回路をバックエンドに送るには、`transpile`という関数とバックエンドの`run`というメソッドを使います。`transpile`については次回{ref}`transpilation`で説明するので、今は「おまじない」だと思ってください。`run`で回路を送るとき、前述したように同時にショット数を指定します。バックエンドごとに一度のジョブでの最大ショット数が決められており、8192、30000、100000などとさまざまです。回路をバックエンドに渡し、`shots`回実行させることをジョブと呼びます。
Use the execute function to send a circuit to the backend. Use the shots argument to specify how many times the circuit is to be run and measured. The maximum number of shots per job performed on each backend is defined. In most cases, it is 8192 (=213).

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

# バックエンドごとに決められている最大ショット数
shots = backend.configuration().max_shots
print(f'Running four circuits, {shots} shots each')

# transpileの説明は次回の実習にて
circuits = transpile(circuits, backend=backend)
# バックエンドで回路をshots回実行させ、測定結果を返させる
job = backend.run(circuits, shots=shots)

# ジョブが終了するまで状態を表示しながら待つ（正常に完了、エラーで停止、など終了する理由は一つではない）
job_monitor(job, interval=2)
```

これで回路がバックエンドに送られ、キューに入りました。ジョブの実行結果は`run`メソッドの返り値であるジョブオブジェクトから参照します。
This will send the circuits to the backend as a job, which is added to the queue. The job execution results is checked using the job object, which is the return value of the execute function.

IBMQのバックエンドは世界中からたくさんのユーザーに利用されているため、場合によっては予約されているジョブが多数あってキューにかなりの待ち時間が生じることがあります。
IBMQ backends are used by many users around the world, so in some cases there may be many jobs in the queue and it may take a long time for your job to be executed.

バックエンドごとのキューの長さは<a href="https://quantum-computing.ibm.com/services?services=systems" target="_blank">IBM Quantumのバックエンド一覧ページ</a>から確認できます。バックエンドを一つクリックすると詳細が表示され、現在の全ジョブ数が Total pending jobs として表示されます。また、一番下の Your access providers という欄でバックエンドのジョブあたりの最大ショット数と最大回路数を確認できます。
The length of the queue for each backend can be seen at right on the IBM Quantum Experience website. Clicking one of the backends in the column at right will display details about the backend. In the bottommost field, "Your access providers," you can see the maximum number of shots and the maximum number of jobs for the backend.

また、自分の投じたジョブのステータスは<a href="https://quantum-computing.ibm.com/jobs" target="_blank">ジョブ一覧ページ</a>から確認できます。
Use the <a href="https://quantum-computing.ibm.com/jobs" target="_blank">"job list page"</a> to see the status of jobs you have submitted.

Qiskitプログラム中からもジョブのステータスを確認できます。いくつか方法がありますが、シンプルに一つのジョブをテキストベースでモニターするだけなら上のように`job_monitor`を使います。
You can check the status of your jobs from within the Qiskit program. There are several methods for doing so. If you simply want to monitor the status of a single job, in text form, you can use `job_monitor`. 

+++

## 量子測定結果の解析

ジョブオブジェクトの`result()`というメソッドを呼ぶと、ジョブが完了して結果が帰ってくるまでコードの実行が止まります。実行結果はオブジェクトとして返され、それの`get_counts`というメソッドを使うと、各ビット列が何回観測されたかというヒストグラムデータがPythonのdictとして得られます。
Calling the `result()` method for a job object will stop the execution of code until the job is complete and a result arrives. The execution results will be returned as an object. Use the `get_counts method` on this object to obtain histogram data on how many times each bit sequence was observed in the form of a Python dict.

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

result = job.result()

# 4つの回路のヒストグラムデータを入れるリスト
counts_list = []

# 回路ごとの結果をresultから抽出する
for idx in range(4):
    # get_counts(i)で回路iのヒストグラムデータが得られる
    counts = result.get_counts(idx)
    # データをリストに足す
    counts_list.append(counts)

print(counts_list)
```

```{code-cell} ipython3
:tags: [remove-cell]

# テキスト作成用のダミーセルなので無視してよい
try:
    counts_list
except NameError:
    counts_list = [
        {'00': 3339, '01': 720, '10': 863, '11': 3270},
        {'00': 964, '01': 3332, '10': 3284, '11': 612},
        {'00': 3414, '01': 693, '10': 953, '11': 3132},
        {'00': 3661, '01': 725, '10': 768, '11': 3038}
    ]

    shots = 8192
```

````{tip}
ノートブックの接続が切れてしまったり、過去に走らせたジョブの結果を再び解析したくなったりした場合は、ジョブIDを使って`retrieve_job`というメソッドでジョブオブジェクトを再構成することができます。過去に走らせたジョブはIBM Quantumのホームページにリストされているので、そこにあるジョブID（cgr3kaemln50ss91pj10のような）をコピーし、

```{code-block} python
backend = provider.get_backend('__backend_you_used__')
job = backend.retrieve_job('__job_id__')
```

とすると、`backend.run`によって返されたのと同じようにジョブオブジェクトが生成されます。
````

Qiskitから提供されている`plot_histogram`関数を使って、この情報を可視化できます。プロットの縦軸は観測回数を全測定数で割って、観測確率に規格化してあります。
This information can be visualized using Qiskit's `plot_histogram` function. The vertical axis of the histogram is a standardized representation of the observation probability, determined by dividing the number of observations by the total number of measurements.

```{code-cell} ipython3
fig, axs = plt.subplots(2, 2, sharey=True, figsize=[12., 8.])
for counts, circuit, ax in zip(counts_list, circuits, axs.reshape(-1)):
    plot_histogram(counts, ax=ax)
    ax.set_title(circuit.name)
    ax.yaxis.grid(True)
```

$c^2/2 = (s + c)^2/4 = 0.427$, $s^2/2 = (s - c)^2 / 4 = 0.073$なので、得られた確率は当たらずとも遠からずというところでしょうか。
$c^2/2 = (s + c)^2/4 = 0.427$, $s^2/2 = (s - c)^2 / 4 = 0.073$, so the probability was fairly close to the mark.

実は現在の量子コンピュータにはまだ様々なノイズやエラーがあり、計算結果は往々にして理論的な値から統計誤差の範囲を超えてずれます。特定のエラーに関しては多少の緩和法も存在しますが、全て防げるわけでは決してありません。現在の量子コンピュータを指して "*Noisy* intermediate-scale quantum (NISQ) device" と呼んだりしますが、このNoisyの部分はこのような簡単な実験でもすでに顕著に現れるわけです。
The reality is that modern quantum computers still have various types of noise and errors, so calculation results sometimes deviate from theoretical values by amounts that exceed the bounds of statistical error. There are methods for mitigating specific errors, to some degree, but there is no way to prevent all errors. Modern quantum computers are known as "Noisy intermediate-scale quantum (NISQ) devices." This "noisy" aspect is very evident even in simple experiments like this.

逆に、NISQデバイスを有効活用するには、ノイズやエラーがあっても意味のある結果が得られるようなロバストな回路が求められます。{doc}`vqe`で紹介する変分量子回路を用いた最適化などがその候補として注目されています。
To use NISQ devices effectively requires robust circuits that produce meaningful results despite noise and errors. A great deal of attention is being turned to methods for achieving this, such as optimization using the variational quantum circuits introduced in the "Learning about the variational method and variational quantum eigensolver method" section.

さて、それでは最後にCHSH不等式の破れを確認してみましょう。$C^{\rmI}, C^{\rmII}, C^{\rmIII}, C^{\rmIV}$を計算して$S$を求めます。
Let us finish this section by confirming that the CHSH inequality has been violated. Let us determine $S$ by calculating $C^{\rmI}$, $C^{\rmII}$, $C^{\rmIII}$, and $C^{\rmIV}$.

下のコードで`counts`という辞書オブジェクトからキー`'00'`などに対応する値を取り出す際に`counts['00']`ではなく`counts.get('00', 0)`としています。二つの表現は`counts`に`'00'`というキーが定義されていれば全く同義ですが、キーが定義されていないときは、前者の場合エラーとして実行が止まるのに対して、後者ではデフォルト値として2個目の引数で指定されている`0`が返ってきます。qiskitの結果データは一度も測定されなかったビット列についてキーを持たないので、常に`get`でカウント数を抽出するようにしましょう。

```{code-cell} ipython3
# C^I, C^II, C^III, C^IVを一つのアレイにする
#（今の場合ただのリストにしてもいいが、純粋な数字の羅列にはnumpy arrayを使うといいことが多い）
C = np.zeros(4, dtype=float)

# enumerate(L)でリストのインデックスと対応する要素に関するループを回せる
for ic, counts in enumerate(counts_list):
    # counts['00'] でなく counts.get('00', 0) - 上のテキストを参照
    C[ic] = counts.get('00', 0) + counts.get('11', 0) - counts.get('01', 0) - counts.get('10', 0)

# 4つの要素を同時にshotsで規格化（リストではこういうことはできない）
C /= shots

S = C[0] - C[1] + C[2] + C[3]

print('C:', C)
print('S =', S)
if S > 2.:
    print('Yes, we are using a quantum computer!')
else:
    print('Armonk, we have a problem.')
```

無事、$S$が2を超えました。
$S$ was, indeed, greater than 2.