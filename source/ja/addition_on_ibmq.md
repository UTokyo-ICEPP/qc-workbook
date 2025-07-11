---
jupytext:
  notebook_metadata_filter: all
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  name: python
  version: 3.12.3
  mimetype: text/x-python
  codemirror_mode:
    name: ipython
    version: 3
  pygments_lexer: ipython3
  nbconvert_exporter: python
  file_extension: .py
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# 【参考】足し算を実機で行う

実習の内容の延長です。ここでは並列足し算回路を実機で実行します。

## 効率化前後の回路の比較

実習のおさらいをすると、まずもともとの足し算のロジックをそのまま踏襲した回路（標準回路）を作り、量子ビットが一列に並んでいることを仮定してSWAPを最小化した回路（効率化回路）を作成しました。ただし、トランスパイラの高性能化によって、標準回路も`optimization_level=3`でトランスパイルすれば2量子ビットゲートの数に効率化回路と大差がなくなることを知りました。

2量子ビットゲートの数を極力減らせたとはいえ、これでもまだ実機での実行ではエラーで答えがスクランブルされてしまいます。回路が小規模になればそれだけ成功確率も上がりますので、$(n_1, n_2)$の値として(1, 1)から(8, 8)まで試してみることにしましょう。

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
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.accounts import AccountNotFoundError
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
channel = 'ibm_quantum_platform'
# 環境設定時に作成したインスタンスの名前を入れてください
instance = '__your_instance_name__'
# API keyをローカルに保存していない場合は、ここに文字列として貼り付けてください
token = None

service = QiskitRuntimeService(channel=channel, instance=instance, token=token)

backend = service.least_busy(filters=operational_backend())

print(f'Using backend {backend.name}')
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

実習と全く同じ`setup_addition`関数と、次のセルで効率化前の回路を返す`make_original_circuit`関数を定義します。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
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

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(1, 1)から(8, 8)までそれぞれ標準回路と効率化回路を作り、全てリストにまとめてバックエンドに送ります。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output, raises-exception]
---
entangling_gate = next(g.name for g in backend.gates if g.name in ['cz', 'ecr'])

# count_ops()の結果をテキストにする関数
def display_nops(circuit):
    nops = circuit.count_ops()
    text = []
    for key in ['rz', 'x', 'sx', entangling_gate]:
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
tags: [remove-output, raises-exception]
---
# List of circuits
circuits = []
for nq in range(1, 9):
    circuits += make_circuits(nq, nq, backend)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output, raises-exception]
---
# バックエンドで定められた最大のショット数だけ各回路を実行
shots = min(backend.max_shots, 2000)

print(f'Submitting {len(circuits)} circuits to {backend.name}, {shots} shots each')

sampler = Sampler(backend)
job = sampler.run(circuits, shots=shots)
counts_list = [result.data.meas.get_counts() for result in job.result()]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

ジョブが返ってきたら、正しい足し算を表しているものの割合を調べてみましょう。

```{code-cell} ipython3
---
tags: [remove-output, raises-exception]
editable: true
slideshow:
  slide_type: ''
---
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
for nq in range(1, 9):
    print(f'({nq}, {nq}):')
    for ctype in ['Original', 'Optimized']:
        n_correct = count_correct_additions(counts_list[icirc], nq, nq)
        r_correct = n_correct / shots
        print(f'  {ctype} circuit: {n_correct} / {shots} = {r_correct:.3f} +- {np.sqrt(r_correct * (1. - r_correct) / shots):.3f}')
        icirc += 1
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

トランスパイル後の標準回路と効率化回路とのCZの数の差が計算結果の精度に直接影響します。トランスパイルをするセルから先を何度か実行してみて、差が大きい時と小さい時を比べるのも参考になるでしょう。

ちなみに、`ibm_torino`というマシンでの試行では、以下のような結果が得られました。

<pre>
Original circuit with n1, n2 = 1, 1
  Transpiling..
  Done. Ops: N(rz)=25, N(x)=4, N(sx)=23, N(cz)=11
Optimized circuit with n1, n2 = 1, 1
  Transpiling..
  Done. Ops: N(rz)=25, N(x)=1, N(sx)=33, N(cz)=13
Original circuit with n1, n2 = 2, 2
  Transpiling..
  Done. Ops: N(rz)=56, N(x)=5, N(sx)=83, N(cz)=41
Optimized circuit with n1, n2 = 2, 2
  Transpiling..
  Done. Ops: N(rz)=79, N(x)=1, N(sx)=105, N(cz)=41
Original circuit with n1, n2 = 3, 3
  Transpiling..
  Done. Ops: N(rz)=114, N(x)=11, N(sx)=185, N(cz)=88
Optimized circuit with n1, n2 = 3, 3
  Transpiling..
  Done. Ops: N(rz)=150, N(x)=3, N(sx)=203, N(cz)=84
Original circuit with n1, n2 = 4, 4
  Transpiling..
  Done. Ops: N(rz)=174, N(x)=15, N(sx)=303, N(cz)=148
Optimized circuit with n1, n2 = 4, 4
  Transpiling..
  Done. Ops: N(rz)=245, N(x)=3, N(sx)=340, N(cz)=142
Original circuit with n1, n2 = 5, 5
  Transpiling..
  Done. Ops: N(rz)=253, N(x)=17, N(sx)=490, N(cz)=238
Optimized circuit with n1, n2 = 5, 5
  Transpiling..
  Done. Ops: N(rz)=362, N(x)=11, N(sx)=516, N(cz)=215
Original circuit with n1, n2 = 6, 6
  Transpiling..
  Done. Ops: N(rz)=342, N(x)=15, N(sx)=694, N(cz)=333
Optimized circuit with n1, n2 = 6, 6
  Transpiling..
  Done. Ops: N(rz)=512, N(x)=7, N(sx)=718, N(cz)=303
Original circuit with n1, n2 = 7, 7
  Transpiling..
  Done. Ops: N(rz)=447, N(x)=25, N(sx)=941, N(cz)=457
Optimized circuit with n1, n2 = 7, 7
  Transpiling..
  Done. Ops: N(rz)=661, N(x)=16, N(sx)=974, N(cz)=406
Original circuit with n1, n2 = 8, 8
  Transpiling..
  Done. Ops: N(rz)=577, N(x)=21, N(sx)=1244, N(cz)=599
Optimized circuit with n1, n2 = 8, 8
  Transpiling..
  Done. Ops: N(rz)=851, N(x)=19, N(sx)=1250, N(cz)=524
</pre>

+++ {"tags": ["remove-input"]}

<pre>
(1, 1):
  Original circuit: 1764 / 2000 = 0.882 +- 0.007
  Optimized circuit: 1823 / 2000 = 0.911 +- 0.006
(2, 2):
  Original circuit: 1161 / 2000 = 0.581 +- 0.011
  Optimized circuit: 1544 / 2000 = 0.772 +- 0.009
(3, 3):
  Original circuit: 1035 / 2000 = 0.517 +- 0.011
  Optimized circuit: 1100 / 2000 = 0.550 +- 0.011
(4, 4):
  Original circuit: 689 / 2000 = 0.344 +- 0.011
  Optimized circuit: 720 / 2000 = 0.360 +- 0.011
(5, 5):
  Original circuit: 120 / 2000 = 0.060 +- 0.005
  Optimized circuit: 406 / 2000 = 0.203 +- 0.009
(6, 6):
  Original circuit: 216 / 2000 = 0.108 +- 0.007
  Optimized circuit: 337 / 2000 = 0.169 +- 0.008
(7, 7):
  Original circuit: 73 / 2000 = 0.036 +- 0.004
  Optimized circuit: 18 / 2000 = 0.009 +- 0.002
(8, 8):
  Original circuit: 10 / 2000 = 0.005 +- 0.002
  Optimized circuit: 22 / 2000 = 0.011 +- 0.002
</pre>

+++ {"editable": true, "slideshow": {"slide_type": ""}}

系が大きくなるに従ってトランスパイラが効率的なルーティングを見つけにくくなる傾向があるようです。(5, 5)以降で標準回路と効率化回路に2量子ビットゲートの数の差が出ています。

回路が均一にランダムに$0$から$2^{n_1 + n_2 + n_3} - 1$までの数を返す場合、レジスタ1と2のそれぞれの値の組み合わせに対して正しいレジスタ3の値が一つあるので、正答率は$2^{n_1 + n_2} / 2^{n_1 + n_2 + n_3} = 2^{-n_3}$となります。(1, 1)では正答率が9割を超えていますが、(8, 8)まで行くとランダムに近い結果が出てしまっているのが見て取れます。
