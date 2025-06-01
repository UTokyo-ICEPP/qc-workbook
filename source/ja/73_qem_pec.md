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

# PEC（確率的誤りキャンセル法）の使い方

+++

{doc}`前前節 <qem_general>`で紹介した外挿法は、ハードウェアでの実行性に優れているものの、ノイズのバイアスに関する理論保証がない、という問題を抱えていました。
ここでは、ノイズチャネル$\mathcal{E}$に関する表式が得られているとの仮定のもとで、不偏推定を可能にする量子エラー抑制の手法である、 **確率的誤りキャンセル法 (Probabilistic Error Cancellation, 以下PEC)** を紹介します。


```{contents} 目次
---
local: true
---
```

$\newcommand{\ket}[1]{| #1 \rangle}$
$\newcommand{\bra}[1]{\langle #1 |}$
$\newcommand{\braket}[2]{\langle #1 | #2 \rangle}$

```{code-cell} ipython3
# notebookを実行する前に、以下のライブラリをインストールしてください。
# mitiqは量子エラー抑制用のライブラリです。
# 
# !pip install mitiq ply cirq
```

PEC（Probabilistic Error Cancellation）とは、エラーに対応するノイズチャネル$\mathcal{E}$の逆演算$\mathcal{E}^{-1}$を、物理的な量子チャネルの線形和として実現するものです。

$$
\mathcal{E}^{-1} = \sum_k q_k \mathcal{B}_k,~\sum_k q_k =1
$$

ここで、$\{\mathcal{B}_k\}$はノイズなく実行できる量子操作の集合で、$\{q_k\}$はその符号つき重みに対応しています。

```{code-cell} ipython3
import mitiq
mitiq.SUPPORTED_PROGRAM_TYPES.keys()
frontend = "qiskit"
```

## 問題設定

1量子ビットの深さ2の量子回路を定義します。

```{code-cell} ipython3
from qiskit import QuantumCircuit
import numpy as np

circuit = QuantumCircuit(1)

circuit.h(0)
circuit.ry(np.pi/8, 0)

print(circuit)
```

```{code-cell} ipython3
from cirq import DensityMatrixSimulator, depolarize
from mitiq.interface import convert_to_mitiq

noise_rate = 0.3
mitiq_circuit, _ = convert_to_mitiq(circuit)
noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_rate))

print(noisy_circuit)
```

## 実行器の定義とノイズ付き期待値計算

```{code-cell} ipython3
import cirq
import numpy as np

def estimate_Z_noisy(circuit, noise_level=0.3):
    """Returns Tr[ρ Z] where ρ is the state prepared by the circuit
    executed with depolarizing noise.
    """
    # Replace with code based on your frontend and backend.
    mitiq_circuit, _ = convert_to_mitiq(circuit)
    # We add a simple noise model to simulate a noisy backend.
    noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real - rho[1, 1].real
```

```{code-cell} ipython3
noisy_value = estimate_Z_noisy(circuit, noise_level = noise_rate)
ideal_value = estimate_Z_noisy(circuit, noise_level=0.0)
print(f"誤差（補正なし）: {abs(ideal_value - noisy_value):.5f}")
```

## 擬確率表現の生成とPECの適用

ノイズチャネルは脱分局ノイズで、以下のように与えられています。

$$
\mathcal{E}_{\rm dep}=(1-p)\mathcal{I} + \frac{p}{3}(\mathcal{X}+ \mathcal{Y}+ \mathcal{Z})
$$

これに対して、逆写像を計算すると、以下のように与えられることがわかります。

$$
\mathcal{E}_{\rm dep}^{-1} = \frac{1-\frac{p}{3}}{1-\frac{4p}{3}}\mathcal{I} - \frac{p}{3(1-\frac{4p}{3})}(\mathcal{X}+\mathcal{Y}+\mathcal{Z})
$$

例えば、 $p=0.3$とおくと、

$$
\mathcal{E}_{\rm dep}^{-1} = 1.5 \mathcal{I} - \frac{1}{6}(\mathcal{X} + \mathcal{Y} + \mathcal{Z})
$$

であり、$\gamma = \sum_k|q_k| = 1.5 + 3 * (1/6) = 2$であることから、物理量の推定を実行する際には$\gamma^{N_g}=4$倍された値を用いる。
例えばパウリ演算子の期待値推定を行う際には、 測定によって$0, 1$が得られた場合、推定値を$\pm4$に割り当てる。

```{code-cell} ipython3
from mitiq.pec.representations.depolarizing import represent_operations_in_circuit_with_local_depolarizing_noise

#reps_ideal = represent_operations_in_circuit_with_local_depolarizing_noise(circuit, 0)
reps_noisy = represent_operations_in_circuit_with_local_depolarizing_noise(circuit, noise_rate)

print(f"ゲートのノイズ除去のための擬確率表現:")
print(reps_noisy[0])
print(reps_noisy[1])
```

```{code-cell} ipython3
from mitiq import pec
sampled_circuits = pec.construct_circuits(circuit, representations=reps_noisy, num_samples = 300)

print(f"Number of sample circuits:    {len(sampled_circuits)}")

for i in range(4):
    print(f"\n{i}番目の量子回路: ")
    print(sampled_circuits[i])
```

```{code-cell} ipython3
from mitiq import pec

noisy_value = estimate_Z_noisy(circuit, noise_level = noise_rate)
print(f"誤差（PECなし）: {abs(ideal_value - noisy_value):.5f}")

for num_shots in [10, 100, 1000]:

    pec_value = pec.execute_with_pec(circuit, estimate_Z_noisy, representations=reps_noisy, num_samples = num_shots)


    print(f"\n測定回数 = {num_shots}")
    print(f"誤差（PECあり）: {abs(ideal_value - pec_value):.5f}")
```
