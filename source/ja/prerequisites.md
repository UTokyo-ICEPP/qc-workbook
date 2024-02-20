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

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# 実習の準備

```{contents} 目次
---
local: true
---
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## IBM Quantum

### IBMidを取得し、IBM Quantumにログインする

IBM Quantumを利用するには、IBMidというアカウントを作り、サービストークンを取得する必要があります。<a href="https://quantum.ibm.com/" target="_blank">IBM Quantum</a>ウェブサイトからIDを取得し、サービスにログインしてください。

(install_token)=
### （ローカル環境）IBM Quantum APIトークンを取得し、Qiskit設定に保存する

IBM Quantum Lab（IBM Quantumウェブサイト上のJupyter Lab）でプログラムを実行する場合、以下の手続きは不要です。

ログインしたらホーム画面のYour API tokenという欄からトークンをコピーできます。
```{image} figs/ibmq_home.png
:height: 400px
:name: My Account
```

アカウントごとに発行されるサービストークンは、ユーザー名＋パスワードの代わりとしてPythonプログラム中でIBMQに接続するために使用されます。ローカルディスクに書き込める環境にある場合は、一度トークンを設定ファイルに保存することで、以降の認証を自動化できます。下のコードセルの`__paste_your_token_here__`のところにIBM Quantumからコピーしたトークンを貼り付け、実行してください。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [raises-exception, remove-output]
---
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account('__paste_your_token_here__')
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

トークンを保存することで、プログラム中でのIBM Quantumへの認証（QiskitRuntimeServiceの取得）は

```{code-block} python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel='ibm_quantum')
```

のようになります。ちなみにIBM Quantum Labでは最初からトークンが保存されている状態なので、このコードで認証が行なえます。

ローカルディスクに書き込める環境でない場合（このワークブックをインタラクティブに使っている場合など）は、Pythonプログラムを実行するたびに（Jupyterのカーネルを再起動するたびに）手動で認証を行う必要があります。

```{code-block} python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel='ibm_quantum', token='__paste_your_token_here__')
```

## ワークブックの使い方

### インタラクティブHTML

このワークブックの各ページにあるプログラムの書かれたセルは、そのまま<a href="https://jupyter.org/" target="_blank">Jupyter Notebook</a>のようにブラウザ上で実行することができます。ページの右上の<i class="fas fa-rocket"></i>にカーソルを乗せ、現れるメニューから<span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><i class="fas fa-play" style="margin-left: .4em;"></i> <span style="margin: 0 .4em 0 .4em;">Live Code</span></span>をクリックしてください。ページのタイトルの下にステータス表示が現れるので、<span style="color: green; font-family: monospace; font-weight: bold; font-size: 1em;">ready</span>と表示されるまで待ちます。

```{image} figs/toggle_interactive.jpg
:height: 400px
:name: Turn interactive contents on
```

ページがインタラクティブになると、コード・セルに<span style="background-color:#dddddd; font-family:'Roboto', sans-serif; margin:0 1em 0 1em;">run</span>および<span style="background-color:#dddddd; font-family:'Roboto', sans-serif; margin:0 1em 0 1em;">restart</span>というボタンが現れ、直下にセルの出力が表示されるようになります。

```{image} figs/interactive_cell.jpg
:height: 200px
:name: Interactive code cell
```

この状態になったら、入力セルの内容を自由に書き換えて、runボタンをクリックして（もしくはShift + Enterで）Pythonコードを実行することができます。このときいくつか注意すべき点があります。

- restartを押すまでページ全体が一つのプログラムになっているので、定義された変数などはセルをまたいで利用される。
- しばらく何もしないでページを放置していると、実行サーバーとの接続が切れてしまう。その場合ページを再度読み込んで、改めてインタラクティブコンテンツを起動する必要がある。
- コードは<a href="https://mybinder.org/" target="_blank">mybinder.org</a>という外部サービス上で実行されるので、個人情報等センシティブな内容の送信は極力避ける。<br/>
  （通信は暗号化されていて、mybinder.org中ではそれぞれのユーザーのプログラムは独立のコンテナ中で動くので、情報が外に筒抜けということではないはずですが、念の為。）<br/>
  ただし上で出てきたように、IBM Quantumのサービストークンだけはどうしても送信する必要があります。

### Jupyter Notebook

インタラクティブHTMLのセキュリティの問題が気になったり、編集したコードを保存したいと考えたりする場合は、ページの元になったノートブックファイルをダウンロードし、自分のローカルの環境で実行することもできます。右上の<i class="fas fa-download"></i>のメニューの<span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><span style="margin: 0 .4em 0 .4em;">.ipynb</span></span>をクリックするか、もしくは<i class="fab fa-github"></i>のメニューの<span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><span style="margin: 0 .4em 0 .4em;">repository</span></span>からリンクされている<a href="https://github.com/UTokyo-ICEPP/qc-workbook" target="_blank">githubレポジトリ</a>をクローンしてください。

ノートブックをローカルに実行するためには、Pythonバージョン3.10以上が必要です。また、`pip`を使って以下のパッケージをインストールする必要があります。

```{code-block}
pip install qiskit qiskit-aer qiskit-ibm-runtime qiskit-experiments qiskit-machine-learning qiskit-optimization matplotlib pylatexenc pandas tabulate
```
