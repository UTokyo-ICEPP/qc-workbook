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

# 【参考】足し算を実機で行う

実習の内容の延長です。ここでは並列足し算回路を実機で実行します。

## 効率化前後の回路の比較

実習のおさらいをすると、まずもともとの足し算のロジックをそのまま踏襲した回路を作り、それではゲート数が多すぎるので効率化した回路を作成しました。

実は効率化した回路でもまだゲートの数が多すぎて、4ビット+4ビットの計算では答えがスクランブルされてしまいます。回路が小規模になればそれだけ成功確率も上がりますので、$(n_1, n_2)$の値として(4, 4)以外に(3, 3)、(2, 2)、(1, 1)も同時に試すことにしましょう。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output]
---
# まずは全てインポート
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.accounts import AccountNotFoundError
from qiskit_experiments.library import CorrelatedReadoutError
from qc_workbook.optimized_additions import optimized_additions
from qc_workbook.utils import operational_backend, find_best_chain

print('notebook ready')
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output, raises-exception]
---
# 利用できるインスタンスが複数ある場合（Premium accessなど）はここで指定する
# instance = 'hub-x/group-y/project-z'
instance = None

try:
    service = QiskitRuntimeService(channel='ibm_quantum', instance=instance)
except AccountNotFoundError:
    service = QiskitRuntimeService(channel='ibm_quantum', token='__paste_your_token_here__', instance=instance)

backend = service.least_busy(min_num_qubits=13, filters=operational_backend())

print(f'Using backend {backend.name}')
```

実習と全く同じ`setup_addition`関数と、次のセルで効率化前の回路を返す`make_original_circuit`関数を定義します。

```{code-cell} ipython3
def setup_addition(circuit, reg1, reg2, reg3):
    """Set up an addition subroutine to a circuit with three registers
    """

    # Equal superposition in register 3
    circuit.h(reg3)

    # Smallest unit of phi
    dphi = 2. * np.pi / (2 ** reg3.size)

    # Loop over reg1 and reg2
    for reg_ctrl in [reg1, reg2]:
        # Loop over qubits in the control register (reg1 or reg2)
        for ictrl, qctrl in enumerate(reg_ctrl):
            # Loop over qubits in the target register (reg3)
            for itarg, qtarg in enumerate(reg3):
                # C[P(phi)], phi = 2pi * 2^{ictrl} * 2^{itarg} / 2^{n3}
                circuit.cp(dphi * (2 ** (ictrl + itarg)), qctrl, qtarg)

    # Insert a barrier for better visualization
    circuit.barrier()

    # Inverse QFT
    for j in range(reg3.size // 2):
        circuit.swap(reg3[j], reg3[-1 - j])

    for itarg in range(reg3.size):
        for ictrl in range(itarg):
            power = ictrl - itarg - 1 + reg3.size
            circuit.cp(-dphi * (2 ** power), reg3[ictrl], reg3[itarg])

        circuit.h(reg3[itarg])

def make_original_circuit(n1, n2):
    """A function to define a circuit with the original implementation of additions given n1 and n2
    """
    n3 = np.ceil(np.log2((2 ** n1) + (2 ** n2) - 1)).astype(int)

    reg1 = QuantumRegister(n1, 'r1')
    reg2 = QuantumRegister(n2, 'r2')
    reg3 = QuantumRegister(n3, 'r3')

    # QuantumCircuit can be instantiated from multiple registers
    circuit = QuantumCircuit(reg1, reg2, reg3)

    # Set register 1 and 2 to equal superpositions
    circuit.h(reg1)
    circuit.h(reg2)

    setup_addition(circuit, reg1, reg2, reg3)

    circuit.measure_all()

    return circuit
```

(4, 4)から(1, 1)までそれぞれオリジナルと効率化した回路の二通りを作り、全てリストにまとめてバックエンドに送ります。

```{code-cell} ipython3
# count_ops()の結果をテキストにする関数
def display_nops(circuit):
    nops = circuit.count_ops()
    text = []
    for key in ['rz', 'x', 'sx', 'cx']:
        text.append(f'N({key})={nops.get(key, 0)}')

    return ', '.join(text)

# オリジナルと効率化した回路を作る関数
def make_circuits(n1, n2, backend):
    print(f'Original circuit with n1, n2 = {n1}, {n2}')
    circuit_orig = make_original_circuit(n1, n2)

    print('  Transpiling..')
    circuit_orig = transpile(circuit_orig, backend=backend, optimization_level=3)

    print(f'  Done. Ops: {display_nops(circuit_orig)}')
    circuit_orig.name = f'original_{n1}_{n2}'

    print(f'Optimized circuit with n1, n2 = {n1}, {n2}')
    circuit_opt = optimized_additions(n1, n2)

    n3 = np.ceil(np.log2((2 ** n1) + (2 ** n2) - 1)).astype(int)

    print('  Transpiling..')
    initial_layout = find_best_chain(backend, n1 + n2 + n3)
    circuit_opt = transpile(circuit_opt, backend=backend, routing_method='basic',
                            initial_layout=initial_layout, optimization_level=3)

    print(f'  Done. Ops: {display_nops(circuit_opt)}')
    circuit_opt.name = f'optimized_{n1}_{n2}'

    return [circuit_orig, circuit_opt]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input]
---
# テキスト作成用のセル
import pickle

shots = 2000
with open('data/quantum_computation_fake_data.pkl', 'rb') as source:
    counts_list = pickle.load(source)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output, raises-exception]
---
# List of circuits
circuits = []
for n1, n2 in [(4, 4), (3, 3), (2, 2), (1, 1)]:
    circuits += make_circuits(n1, n2, backend)

# バックエンドで定められた最大のショット数だけ各回路を実行
shots = min(backend.max_shots, 2000)

print(f'Submitting {len(circuits)} circuits to {backend.name}, {shots} shots each')

job = backend.run(circuits, shots=shots)
counts_list = job.result().get_counts()
```

ジョブが返ってきたら、正しい足し算を表しているものの割合を調べてみましょう。

```{code-cell} ipython3
:tags: [remove-output]

def count_correct_additions(counts, n1, n2):
    """Extract the addition equation from the counts dict key and tally up the correct ones."""

    correct = 0

    for key, value in counts.items():
        # cf. plot_counts() from the SIMD lecture
        x1 = int(key[-n1:], 2)
        x2 = int(key[-n1 - n2:-n1], 2)
        x3 = int(key[:-n1 - n2], 2)

        if x1 + x2 == x3:
            correct += value

    return correct


icirc = 0
for n1, n2 in [(4, 4), (3, 3), (2, 2), (1, 1)]:
    for ctype in ['Original', 'Optimized']:
        n_correct = count_correct_additions(counts_list[icirc], n1, n2)
        r_correct = n_correct / shots
        print(f'{ctype} circuit ({n1}, {n2}): {n_correct} / {shots} = {r_correct:.3f} +- {np.sqrt(r_correct * (1. - r_correct) / shots):.3f}')
        icirc += 1
```

ちなみに、`ibm_kawasaki`というマシンで同じコードを走らせると、下のような結果が得られます。

<pre>
Original circuit with n1, n2 = 4, 4
  Transpiling..
  Done. Ops: N(rz)=170, N(x)=3, N(sx)=67, N(cx)=266
Optimized circuit with n1, n2 = 4, 4
  Transpiling..
  Done. Ops: N(rz)=175, N(x)=1, N(sx)=64, N(cx)=142
Original circuit with n1, n2 = 3, 3
  Transpiling..
  Done. Ops: N(rz)=120, N(x)=5, N(sx)=57, N(cx)=90
Optimized circuit with n1, n2 = 3, 3
  Transpiling..
  Done. Ops: N(rz)=117, N(x)=0, N(sx)=48, N(cx)=84
Original circuit with n1, n2 = 2, 2
  Transpiling..
  Done. Ops: N(rz)=50, N(x)=0, N(sx)=20, N(cx)=56
Optimized circuit with n1, n2 = 2, 2
  Transpiling..
  Done. Ops: N(rz)=67, N(x)=0, N(sx)=32, N(cx)=41
Original circuit with n1, n2 = 1, 1
  Transpiling..
  Done. Ops: N(rz)=25, N(x)=0, N(sx)=15, N(cx)=13
Optimized circuit with n1, n2 = 1, 1
  Transpiling..
  Done. Ops: N(rz)=27, N(x)=0, N(sx)=16, N(cx)=13
</pre>

+++ {"tags": ["remove-input"]}

<pre>
Original circuit (4, 4): 990 / 32000 = 0.031 +- 0.001
Optimized circuit (4, 4): 879 / 32000 = 0.027 +- 0.001
Original circuit (3, 3): 2435 / 32000 = 0.076 +- 0.001
Optimized circuit <b>(3, 3)</b>: 2853 / 32000 = <b>0.089 +- 0.002</b>
Original circuit (2, 2): 3243 / 32000 = 0.101 +- 0.002
Optimized circuit <b>(2, 2)</b>: 7994 / 32000 = <b>0.250 +- 0.002</b>
Original circuit <b>(1, 1)</b>: 25039 / 32000 = <b>0.782 +- 0.002</b>
Optimized circuit <b>(1, 1)</b>: 26071 / 32000 = <b>0.815 +- 0.002</b>
</pre>

+++ {"editable": true, "slideshow": {"slide_type": ""}}

回路が均一にランダムに$0$から$2^{n_1 + n_2 + n_3} - 1$までの数を返す場合、レジスタ1と2のそれぞれの値の組み合わせに対して正しいレジスタ3の値が一つあるので、正答率は$2^{n_1 + n_2} / 2^{n_1 + n_2 + n_3} = 2^{-n_3}$となります。実機では、(4, 4)と(3, 3)でどちらの回路も正答率がほとんどこの値に近くなっています。(2, 2)では効率化回路で明らかにランダムでない結果が出ています。(1, 1)では両回路とも正答率8割です。

フェイクバックエンドでは実機のエラーもシミュレートされていますが、エラーのモデリングが甘い部分もあり、$2^{-n_3}$よりは遥かに良い成績が得られています。いずれのケースも、回路が短い効率化バージョンの方が正答率が高くなっています。

+++ {"tags": ["remove-output", "raises-exception"], "editable": true, "slideshow": {"slide_type": ""}}

(measurement_error_mitigation)=
## 測定エラーの緩和

{doc}`extreme_simd`でも軽く触れましたが、現状ではCNOTゲートのエラー率は1量子ビットゲートのエラー率より一桁程度高くなっています。CNOTを含むゲートで発生するエラーは本格的なエラー訂正が可能になるまでは何も対処しようがなく、そのため上ではCNOTを極力減らす回路を書きました。しかし、そのようなアプローチには限界があります。

一方、測定におけるエラーは、エラー率が実は決して無視できない高さであると同時に、統計的にだいたい再現性がある（あるビット列$x$が得られるべき状態から別のビット列$y$が得られる確率が、状態の生成法に依存しにくい）という性質があります。そのため、測定エラーは事後的に緩和（部分的補正）できます。そのためには$n$ビットレジスタの$2^n$個すべての計算基底状態について、相当するビット列が100%の確率で得られるべき回路を作成し、それを測定した結果を利用します。

例えば$n=2$で状態$\ket{x} \, (x = 00, 01, 10, 11)$を測定してビット列$y$を得る確率が$\epsilon^x_y$だとします。このとき実際の量子計算をして測定で得られた確率分布が$\{p_y\}$であったとすると、その計算で本来得られるべき確率分布$\{P_x\}$は連立方程式

$$
p_{00} = P_{00} \epsilon^{00}_{00} + P_{01} \epsilon^{01}_{00} + P_{10} \epsilon^{10}_{00} + P_{11} \epsilon^{11}_{00} \\
p_{01} = P_{00} \epsilon^{00}_{01} + P_{01} \epsilon^{01}_{01} + P_{10} \epsilon^{10}_{01} + P_{11} \epsilon^{11}_{01} \\
p_{10} = P_{00} \epsilon^{00}_{10} + P_{01} \epsilon^{01}_{10} + P_{10} \epsilon^{10}_{10} + P_{11} \epsilon^{11}_{10} \\
p_{11} = P_{00} \epsilon^{00}_{11} + P_{01} \epsilon^{01}_{11} + P_{10} \epsilon^{10}_{11} + P_{11} \epsilon^{11}_{11}
$$

を解けば求まります。つまり、行列$\epsilon^x_y$の逆をベクトル$p_y$にかければいいわけです[^actually_fits]。

Qiskitでは測定エラー緩和用の関数やクラスが提供されているので、それを使って実際にエラーを求め、上の足し算の結果の改善を試みましょう。

[^actually_fits]: 実際には数値的安定性などの理由から、単純に逆行列をかけるわけではなくフィッティングが行われますが、発想はここで書いたものと変わりません。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [raises-exception, remove-output]
---
qubits = find_best_chain(backend, 4)

# 測定エラー緩和の一連の操作（2^4通りの回路生成、ジョブの実行、結果の解析）がCorrelatedReadoutErrorクラスの内部で行われる
experiment = CorrelatedReadoutError(qubits)
experiment.analysis.set_options(plot=True)
result = experiment.run(backend)

# mitigatorオブジェクトが上でいうε^x_yの逆行列を保持している
mitigator = result.analysis_results('Correlated Readout Mitigator').value
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output, raises-exception]
---
# counts_list[1] = (4, 4)の最適化された回路での結果を得る
raw_counts = counts_list[1]
shots = backend.configuration().max_shots
# ここから下はおまじない
quasiprobs = mitigator.quasi_probabilities(raw_counts, shots=shots)
mitigated_probs = quasiprobs.nearest_probability_distribution().binary_probabilities()
mitigated_counts = dict((key, value * shots) for key, value in mitigated_probs.items())

n_correct = count_correct_additions(mitigated_counts, n1, n2)
r_correct = n_correct / shots
print(f'Optimized circuit with error mitigation ({n1}, {n2}): {n_correct} / {shots} = {r_correct:.3f} +- {np.sqrt(r_correct * (1. - r_correct) / shots):.3f}')
```
