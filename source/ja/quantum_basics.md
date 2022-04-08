---
jupytext:
  notebook_metadata_filter: all
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
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
  version: 3.8.5
---

# 量子力学の基礎

+++

## 測定

偏光の決まった（良質のレーザーなどから出る）光を偏光板に通す実験を考えましょう。偏光板はA, Bの2枚用意し、それぞれ縦（角度0）と横（角度$\pi/2$）の光が100%透過するように設置します。レーザーの偏光角が0であれば、Aは光を透過させますがBは遮断します。レーザーを$\pi/4$だけ回すと、今度はA, Bともに元の50%の強度で光を透過させるようになります。偏向角を$\pi/6$にすると、透過率はAで75%, Bで25%となります。

Image

実はこの偏光板による光の透過実験は、量子力学における「測定」という操作に対応しています。偏光板を透過する光の強度は、光のビームを成す個々の光子が偏光板を透過する確率に比例しています。
さらに、光子を偏光板に当てることは光子の偏光を測定することと同じで、
測定の結果、偏光が偏光板の向きと一致していれば光子は透過し、

```{code-cell} ipython3

```
