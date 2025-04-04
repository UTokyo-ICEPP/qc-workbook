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

```{code-cell} ipython3

```
