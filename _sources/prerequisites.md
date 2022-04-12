---
jupytext:
  notebook_metadata_filter: all
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
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
  version: 3.8.10
---

# 実習の準備

```{contents} 目次
---
local: true
---
```

+++

## IBM Quantum

### IBMidを取得し、IBM Quantumにログインする

IBM Q System Oneを利用するには、IBMidというアカウントを作り、サービストークンを取得する必要があります。[IBM Quantum](https://quantum-computing.ibm.com/)ウェブサイトからIDを取得し、サービスにログインしてください。

(copy_token)=
### IBM Quantum APIトークンを取得する

ログインしたらホーム画面のYour API tokenという欄からトークンをコピーできます。
```{image} figs/ibmq_home.png
:height: 400px
:name: My Account
```

(install_token)=
### Qiskitにトークンを登録する

アカウントごとに発行されるサービストークンは、ユーザー名＋パスワードの代わりとしてPythonプログラム中でIBMQに接続するために使用されます。

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

from qiskit import IBMQ

IBMQ.enable_account('__paste_your_token_here__')
```

上のように`enable_account`を利用する場合は、Pythonプログラムを実行するたびに（Jupyter notebookのカーネルを再起動するたびに）認証を行います。

ローカルディスクに書き込める環境にあれば、
```{code-block} python
IBMQ.save_account('__paste_your_token_here__')
```
とすることでトークンが保存され、以降はPythonプログラムを実行するたびに行う手続きが
```{code-block} python
IBMQ.load_account()
```
に変わります。

## ワークブックの使い方

### インタラクティブHTML

このワークブックの各ページにあるプログラムの書かれたセルは、そのまま[Jupyter Notebook](https://jupyter.org/)のようにブラウザ上で実行することができます。ページの右上の<i class="fas fa-rocket"></i>にカーソルを乗せ、現れるメニューから<span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><i class="fas fa-play" style="margin-left: .4em;"></i> <span style="margin: 0 .4em 0 .4em;">Live Code</span></span>をクリックしてください。ページのタイトルの下にステータス表示が現れるので、<span style="color: green; font-family: monospace; font-weight: bold; font-size: 1em;">ready</span>と表示されるまで待ちます。

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
- コードは[mybinder.org](https://mybinder.org/)という外部サービス上で実行されるので、個人情報等センシティブな内容の送信は極力避ける。<br/>
  （通信は暗号化されていて、mybinder.org中ではそれぞれのユーザーのプログラムは独立のコンテナ中で動くので、情報が外に筒抜けということではないはずですが、念の為。）<br/>
  ただし上で出てきたように、IBM Quantum Experienceのサービストークンだけはどうしても送信する必要があります。
  
### Jupyter Notebook
  
インタラクティブHTMLのセキュリティの問題が気になったり、編集したコードを保存したいと考えたりする場合は、ページの元になったノートブックファイルをダウンロードし、自分のローカルの環境で実行することもできます。右上の<i class="fas fa-download"></i>のメニューの<span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><span style="margin: 0 .4em 0 .4em;">.ipynb</span></span>をクリックするか、もしくは<i class="fab fa-github"></i>のメニューの<span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><span style="margin: 0 .4em 0 .4em;">repository</span></span>からリンクされている[githubレポジトリ](https://github.com/UTokyo-ICEPP/qc-workbook)をクローンしてください。

ノートブックをローカルに実行するためには、Pythonバージョン3.8以上が必要です。また、`pip`を使って以下のパッケージをインストールする必要があります。

```{code-block}
pip install qiskit matplotlib pylatexenc tabulate
```
