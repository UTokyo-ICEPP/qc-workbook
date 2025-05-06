---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 量子多体問題における誤り抑制

+++

{doc}`前節 <qem_general>`では、一般の量子回路中のノイズを抑制する手段として外挿法を説明しました。
ここでは、量子コンピュータの重要な応用先である、量子多体問題に焦点を当て、特に基底状態計算における誤り抑制手法を紹介します。

```{contents} 目次
---
local: true
---
```

$\newcommand{\ket}[1]{| #1 \rangle}$
$\newcommand{\bra}[1]{\langle #1 |}$
$\newcommand{\braket}[2]{\langle #1 | #2 \rangle}$

+++

## そもそも基底状態とは
興味のある量子多体系$\mathcal{S}$に対応する$d$次元のHilbert空間を$\mathcal{H}=\mathbb{C}^{d}$とします。
$\mathcal{S}$が外界から孤立しているとすると、時刻$t$における系の量子状態$|\psi(t)\rangle$は、以下のSchrödinger方程式に従います：

$$
i\frac{d}{dt}|\psi(t)\rangle = H(t) |\psi(t)\rangle.
$$

ここで、$H(t)$は時刻$t$におけるハミルトニアンです。外場などの影響がない場合には、のように時間に依存しないものと仮定できます。
以下ではかんたんのため、$N$量子ビット系の時間非依存な系のハミルトニアン$H(t)=H$を考えることにしましょう。
孤立系の時間発展はユニタリ演算子で記述されることから、ハミルトニアンは$H=H^\dagger$を満たすエルミート演算子です。
したがって、固有状態は実の固有値を持ちます。特に、固有値$E_i$をもつ$i$番目の固有状態$|E_i\rangle$は以下のような固有方程式を満たします。

$$
H|E_i\rangle = E_i |E_i\rangle.
$$

ここで、$E_0 \leq E_1 \leq \cdots \leq E_{d-1}$のように並べた時、最も小さい固有値$E_0$に対応する状態$|E_0\rangle$を **基底状態 (Ground state)** と呼びます。

+++

### 基底状態の計算例 1 : Heisenbergハミルトニアン、$N=2$サイト

```{code-cell} ipython3
import numpy as np

# パウリ行列
x = np.array([[0, 1], [1, 0]])
y = np.array([[0, -1j], [1j, 0]])
z = np.array([[1, 0], [0, -1]])

# ハミルトニアン H = X1 X2 + Y1 Y2 + Z1 Z2（2サイト）
ham = (np.kron(x, x) + np.kron(y, y) + np.kron(z, z)).real

# 固有値と基底状態を求める
eigvals, eigvecs = np.linalg.eigh(ham)
print("基底エネルギー:", eigvals[0])
print("基底ベクトル:", eigvecs[:, 0]) # numpyの仕様に注意, 確かに singlet状態
```

### 基底状態の計算例 2 : Heisenbergハミルトニアン、$N=6$サイト

次に、Heisenberg相互作用をする量子ビットが1次元的に繋がった系を考えることにします。
周期境界条件のもとで、ハミルトニアンは

$$
 H = \sum_{i=0}^{N-1} X_i X_{i+1} + Y_i Y_{i+1} + Z_{i} Z_{i+1}~(Z_{N}=Z_0)
$$

のように与えられているものとしましょう。この場合、$N$が十分小さい場合には、依然として厳密対角化が実行できます。

```{code-cell} ipython3
import scipy
from scipy.sparse import kron, identity

def heisenberg_chain(L):
    """LサイトのHeisenbergハミルトニアンを構築"""
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])

    def kron_all(op_list):
        res = op_list[0]
        for op in op_list[1:]:
            res = kron(res, op) # sparse matrixとして情報を保持
        return res

    dim = 2 ** L
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(L - 1):
        for S in [x, y, z]:
            ops = [identity(2)] * L
            ops[i] = S
            ops[i + 1] = S
            H += kron_all(ops)
    return H.real

# 12サイトハミルトニアンの構築と固有値計算
n_qubit = 10
hamiltonian = heisenberg_chain(n_qubit)
eigvals, eigvecs = scipy.sparse.linalg.eigsh(hamiltonian, k=1, which='SA')  # 最小固有値を求める
gs_en = eigvals[0]
print(f"{n_qubit}サイト基底エネルギー:", gs_en)
gsvec = eigvecs[:,0]
print(f"{n_qubit}サイト基底状態:", gsvec)
```

## 基底状態計算における誤り抑制： 量子選択配置間相互作用法(QSCI)/サンプルベース量子対角化法(SQD)

以前にみたように、量子回路ではノイズが発生してしまうため、量子回路で実現しようとした量子状態は、意図したものとは別物になってしまいます。
そのような場合には、エネルギーを測定したとしても、全く別の値が得られることがしばしばあります。

+++

### ノイズのある基底状態

{doc}`前節 <qem_general>`では各量子ビットにおけるデコヒーレンスの影響を考えましたが、ここではノイズを最も抽象化したモデルとして、大域的脱分局ノイズを考えます。
具体的に、その作用は、入力の量子状態$\rho$に対して

$$
\mathcal{N}_{\rm wn}(\rho) = (1-p) \rho + p \frac{I}{2^N}
$$

のように表されます。ここで、$p$はノイズ率です。第2項の$I/2^N$は完全混合状態なので、口語的には「確率$p$で入力状態を完全混合状態と入れ替えるノイズ」と説明できます。

```{code-cell} ipython3
p = 0.3
rho = (1-p) * np.outer(gsvec, gsvec.conj()) + p * (np.identity(2**n_qubit)/2**n_qubit)
gs_noisy = np.trace(rho @ hamiltonian).real

print("Exact energy:")
print(gs_en)
print("Noisy energy:")
print(gs_noisy)
```

### QSCIもしくはSQD

上で見たように、ノイズの影響は物理量の評価に大きな影響を与えてしまいます。

(qsci_fig)=
```{image} figs/72_qsci_sqd.png
:alt: qsci
:width: 700px
:align: center
```

```{code-cell} ipython3
probs = rho.diagonal().real

n_sample = 200
states_sampled = np.random.choice(len(probs), size = n_sample, p = probs)
unique_bases = np.unique(states_sampled)
H_sub = hamiltonian[:, unique_bases][unique_bases, :] # 部分行列を抽出

energy_sub = np.linalg.eigvalsh(H_sub)[0]
print("reconstructed energy :")
print(energy_sub)
```

```{code-cell} ipython3
nsamp_array =[20, 100, 200, 500, 1000, 2000]
energy_array = []

for n_sample in nsamp_array:
    states_sampled = np.random.choice(len(probs), size = n_sample, p = probs)
    unique_bases = np.unique(states_sampled)
    H_sub = hamiltonian[:, unique_bases][unique_bases, :] # 部分行列を抽出
    
    energy_sub = np.linalg.eigvalsh(H_sub)[0]
    energy_array.append(energy_sub)
    
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(nsamp_array, energy_array, "-o")

plt.hlines(y = gs_noisy, xmin = min(nsamp_array), xmax = max(nsamp_array), linestyle = "--", color = "gray", label = "raw noisy")
plt.hlines(y=gs_en, xmin = min(nsamp_array), xmax = max(nsamp_array), linestyle = "-", color = "red", label = "exact")

plt.xlabel("# of samples")
plt.ylabel("energy")
```

```{code-cell} ipython3

```
