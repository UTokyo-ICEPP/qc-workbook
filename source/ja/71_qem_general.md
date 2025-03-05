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

# 外挿法による誤りの抑制

+++

ここでは、**量子誤り抑制**の考え方と、その代表例である外挿法(Zero-Noise Extrapolation, ZNE)を紹介します。外挿法の概要を説明した後に、Qiskitを使用して、量子ダイナミクスにおけるノイズの効果を抑制できることを確認します。

```{contents} 目次
---
local: true
---
```

$\newcommand{\ket}[1]{| #1 \rangle}$
$\newcommand{\bra}[1]{\langle #1 |}$
$\newcommand{\braket}[2]{\langle #1 | #2 \rangle}$

+++

## 量子誤り抑制と量子誤り訂正

## 外挿法

## 外挿法の実装
