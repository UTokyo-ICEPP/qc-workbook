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

# 予備知識：数値表現

```{contents} 目次
---
local: true
---
```

$\newcommand{\ket}[1]{|#1\rangle}$

+++

(signed_binary)=
## 符号付き二進数

（計算機中のバイナリ表現に慣れていない人のために）
(This information is presented for readers that are unfamiliar with the binary notation used in computers)

$-2^{n-1}$から$2^{n}-1$までの整数$X$を$n$ビットレジスタで表現する方法は何通りもありえますが、標準的なのは最高位ビットの0を+、1を-の符号に対応させ、残りの$n-1$ビットで絶対値を表現する方法です。このとき、$X$が正なら通常の（符号なし）整数と同じ表現となり、負なら場合は$n-1$ビット部分が$2^{n-1} - |X|$になるようにとります。つまり、$100 \dots 0$は$-2^{n-1}$、$111 \dots 1$は$-1$に対応します。これは、別の言い方をすれば「$[-2^{n-1}, -1]$の負の数$X$に、$X + 2^n$という符号なし整数を対応させる」とも見て取れます。
There are several ways to represent an integer $X$ with a value of $-2^{n-1}$ to $2^{n}-1$ with an $n$-bit register. The most typical way is to make the highest bit 0 to indicate + and 1 to indicate -, and to use the remaining $n-1$ bits to represent the absolute value. When doing this, if $X$ is positive, normally it is represented by a (unsigned) integer, and if it is negative, the $n-1$ bit portion takes the value of $2^{n-1} - |X|$. In other words, $100 \dots 0$ corresponds to $-2^{n-1}$ and $111 \dots 1$ corresponds to $-1$. This could be rephrased as "For $[-2^{n-1}, -1]$, negative number $X$ corresponds to the unsigned integer $X + 2^n$.

正の数同士の足し算などの結果、符号付き$n$ビットレジスタに$2^{n-1}$以上の値が入ってしまうと、最高位ビットが1となり符号が反転して急に小さい数が現れてしまうことがあります。例えば、形式上$2^{n-1} - 1 = (011 \dots 1)_2$に1を足すと、$(100 \dots 0)_2 = -2^{n-1}$です。このような現象をオーバーフローと呼びます。
If adding positive numbers would result in a value of $2^{n-1}$ or more being entered into a signed $n$-bit register, the uppermost bit would become 1, which would cause the sign to flip and the number to suddenly become an extremely small number. For example, if 1 were added to $2^{n-1} - 1 = (011 \dots 1)_2$, it would become $(100 \dots 0)_2 = -2^{n-1}$. This is called an overflow.

+++

(nonintegral_fourier)=
## 負・非整数のフーリエ変換

{doc}`extreme_simd`では、$n$ビットレジスタでの逆フーリエ変換$1/\sqrt{2^n} \sum_{k} \exp (2 \pi i j k / 2^n) \ket{k} \rightarrow \ket{j}$を扱いました。ここでは$k$も$j$も$0$以上$2^{n-1}-1$以下の整数に限定していました。それでは、$j$が負であったり非整数であったりする場合の逆フーリエ変換はどのようになるでしょうか。
In the {doc}`extreme_simd`, we worked with the inverse Fourier transform $1/\sqrt{2^n} \sum_{k} \exp (2 \pi i j k / 2^n) \ket{k} \rightarrow \ket{j}$, used on an $n$-bit register. Here, both $k$ and $j$ were limited to integers with values between 0 and $2^{n-1}-1$. What would happen to the inverse Fourier transform if $j$ were a negative number or a non-integer number?

まず、$j$が負の場合、正の整数$a$で必ず
First, if $j$ were negative, with positive integer $a$ the following would always be true.

$$
\exp (2 \pi i j k / 2^n) = \exp [2 \pi i (j + 2^n a) k / 2^n]
$$

が成り立つので、$0 \leq j + 2^n a < 2^n$となる$a$を選べば、逆フーリエ変換は
Therefore, if you selected a value for $a$ such that $0 \leq j + 2^n a < 2^n$, the inverse Fourier transform would be as shown below.

$$
1/\sqrt{2^n} \sum_{k} \exp (2 \pi i j k / 2^n) \ket{k} \rightarrow \ket{j + 2^n a}
$$

です。特に$2^{n-1} \leq j < 0$の場合、右辺は$\ket{j + 2^n}$であり、上の符号付き二進数の標準的な表現法に合致することがわかります。
In particular, for $2^{n-1} \leq j < 0$, the right side becomes $\ket{j + 2^n}$. As you can see, this matches the standard representation method for signed binary numbers indicated above.

次に$j$が非整数の場合ですが、このときは逆フーリエ変換の結果が一つの計算基底で表現されず、計算基底の重ね合わせ$\sum_{l} f_{jl} \ket{l}$となります。ここで$f_{jl}$は$j$に最も近い$l$において$|f_{jl}|^2$が最も大きくなるような分布関数です。次のセルで$j$の値を変えて色々プロットして、感覚を掴んでみてください。
Next, if $j$ is a non-integer, the result of the inverse Fourier transform cannot be expressed with a single computation basis state, instead becoming the computation basis state superposition $\sum_{l} f_{jl} \ket{l}$ . Here, $f_{jl}$ is a distribution function that maximizes the value of $|f_{jl}|^2$ when $l$ is closest to $j$. In the next cell, change the value of $j$ and plot the values to get a better sense of the relationship.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

n = 4
j = 7.8

N = 2 ** n

# Array jk = j*[0, 1, ..., N-1]
jk = np.arange(N) * j
phase_jk = np.exp(2. * np.pi * 1.j * jk / N)

# Array kl = [[0, 0, ..., 0], [0, 1, ..., N-1], ..., [0, N-1, ..., (N-1)(N-1)]]
kl = np.outer(np.arange(N), np.arange(N))
phase_minuskl = np.exp(-2. * np.pi * 1.j * kl / N)

# Inverse Fourier transform
f_jl = (phase_jk @ phase_minuskl) / N

# Plot Re(f_jl), Im(f_jl), and |f_jl|
plt.plot(np.arange(N), np.real(f_jl), 'o', label='real')
plt.plot(np.arange(N), np.imag(f_jl), 'o', label='imag')
plt.plot(np.arange(N), np.abs(f_jl), 'o', label='norm')
plt.legend()
```
