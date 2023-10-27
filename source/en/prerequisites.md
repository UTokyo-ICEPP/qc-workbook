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

# 実習の準備

```{contents} 目次
---
local: true
---
```

+++

## IBM Quantum

### IBMidを取得し、IBM Quantumにログインする

IBM Quantumを利用するには、IBMidというアカウントを作り、サービストークンを取得する必要があります。<a href="https://quantum-computing.ibm.com/" target="_blank">IBM Quantum</a>ウェブサイトからIDを取得し、サービスにログインしてください。
TO use IBM Quantum, you must first create an IBMid account and receive a service token. Acquire an ID from the <a href="https://quantum-computing.ibm.com/" target="_blank">IBM Quantum</a> website and log into the service.

(install_token)=
### （ローカル環境）IBM Quantum APIトークンを取得し、Qiskit設定に保存する

IBM Quantum Lab（IBM Quantumウェブサイト上のJupyter Lab）でプログラムを実行する場合、以下の手続きは不要です。

ログインしたらホーム画面のYour API tokenという欄からトークンをコピーできます。
On the main screen shown after logging in, copy the token displayed in the "Your API token" area.
```{image} figs/ibmq_home.png
:height: 400px
:name: My Account
```

アカウントごとに発行されるサービストークンは、ユーザー名＋パスワードの代わりとしてPythonプログラム中でIBMQに接続するために使用されます。ローカルディスクに書き込める環境にある場合は、一度トークンを設定ファイルに保存することで、以降の認証を自動化できます。下のコードセルの`__paste_your_token_here__`のところにIBM Quantumからコピーしたトークンを貼り付け、実行してください。
Service tokens are issued on a per-account basis and are used by the Python program to connect to IBMQ. They play the roles of user IDs and passwords.

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

from qiskit_ibm_provider import IBMProvider

IBMProvider.save_account('__paste_your_token_here__')
```

トークンを保存することで、プログラム中でのIBM Quantumへの認証（IBMProviderの取得）は

```{code-block} python
from qiskit_ibm_provider import IBMProvider

provider = IBMProvider()
```

のようになります。ちなみにIBM Quantum Labでは最初からトークンが保存されている状態なので、このコードで認証が行なえます。

ローカルディスクに書き込める環境でない場合（このワークブックをインタラクティブに使っている場合など）は、Pythonプログラムを実行するたびに（Jupyterのカーネルを再起動するたびに）手動で認証を行う必要があります。

```{code-block} python
from qiskit_ibm_provider import IBMProvider

provider = IBMProvider(token='__paste_your_token_here__')
```

## ワークブックの使い方

### インタラクティブHTML

このワークブックの各ページにあるプログラムの書かれたセルは、そのまま<a href="https://jupyter.org/" target="_blank">Jupyter Notebook</a>のようにブラウザ上で実行することができます。ページの右上の<i class="fas fa-rocket"></i>にカーソルを乗せ、現れるメニューから<span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><i class="fas fa-play" style="margin-left: .4em;"></i> <span style="margin: 0 .4em 0 .4em;">Live Code</span></span>をクリックしてください。ページのタイトルの下にステータス表示が現れるので、<span style="color: green; font-family: monospace; font-weight: bold; font-size: 1em;">ready</span>と表示されるまで待ちます。
The cells in which programs are written in in this workbook can be executed directly from the browser, like using <a href="https://jupyter.org/" target="_blank">Jupyter Notebook</a>. Mouse over <i class="fas fa-rocket"></i> at the top right of the page and click <span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><i class="fas fa-play" style="margin-left: .4em;"></i> <span style="margin: 0 .4em 0 .4em;">Live Code</span></span> on the menu that appears. A status indicator will appear below the page title. Wait until the status is <span style="color: green; font-family: monospace; font-weight: bold; font-size: 1em;">ready</span>.

```{image} figs/toggle_interactive.jpg
:height: 400px
:name: Turn interactive contents on
```

ページがインタラクティブになると、コード・セルに<span style="background-color:#dddddd; font-family:'Roboto', sans-serif; margin:0 1em 0 1em;">run</span>および<span style="background-color:#dddddd; font-family:'Roboto', sans-serif; margin:0 1em 0 1em;">restart</span>というボタンが現れ、直下にセルの出力が表示されるようになります。
When the page becomes interactive, <span style="background-color:#dddddd; font-family:'Roboto', sans-serif; margin:0 1em 0 1em;">run</span> and <span style="background-color:#dddddd; font-family:'Roboto', sans-serif; margin:0 1em 0 1em;">restart</span> buttons will appear in code cells, and the output of running the code will be displayed immediately below them.

```{image} figs/interactive_cell.jpg
:height: 200px
:name: Interactive code cell
```

この状態になったら、入力セルの内容を自由に書き換えて、runボタンをクリックして（もしくはShift + Enterで）Pythonコードを実行することができます。このときいくつか注意すべき点があります。
In this state, you can directly edit the contents of the cell and click the "run" button (or press Shift + Enter) to run the Python code. There are several important points to note when doing so.

- restartを押すまでページ全体が一つのプログラムになっているので、定義された変数などはセルをまたいで利用される。
- しばらく何もしないでページを放置していると、実行サーバーとの接続が切れてしまう。その場合ページを再度読み込んで、改めてインタラクティブコンテンツを起動する必要がある。
- コードは<a href="https://mybinder.org/" target="_blank">mybinder.org</a>という外部サービス上で実行されるので、個人情報等センシティブな内容の送信は極力避ける。<br/>
  （通信は暗号化されていて、mybinder.org中ではそれぞれのユーザーのプログラムは独立のコンテナ中で動くので、情報が外に筒抜けということではないはずですが、念の為。）<br/>
  ただし上で出てきたように、IBM Quantumのサービストークンだけはどうしても送信する必要があります。

- The entire page will be treated as one program, so, for examples, the values of variables defined in one cell will be carried over into other cells until "restart" is pressed.The 
- If you do not perform any actions on the page for some time, you will lose the connection to the execution server. If this happens, you must reload the page and start up the interactive content again.
•- The code is executed using an external service, <a href="https://mybinder.org/" target="_blank">mybinder.org</a>, so whenever possible avoid sending any sensitive information such as personal information. 
(Transmission is encrypted, and interactive content is executed in a separate container for each user on mybinder.org, so the information is not exposed, but it is best to avoid sending any sensitive information just in case.) 
Of course, as explained above, the IBM Quantum service token must be sent in order to use the service.

### Jupyter Notebook

インタラクティブHTMLのセキュリティの問題が気になったり、編集したコードを保存したいと考えたりする場合は、ページの元になったノートブックファイルをダウンロードし、自分のローカルの環境で実行することもできます。右上の<i class="fas fa-download"></i>のメニューの<span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><span style="margin: 0 .4em 0 .4em;">.ipynb</span></span>をクリックするか、もしくは<i class="fab fa-github"></i>のメニューの<span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><span style="margin: 0 .4em 0 .4em;">repository</span></span>からリンクされている<a href="https://github.com/UTokyo-ICEPP/qc-workbook" target="_blank">githubレポジトリ</a>をクローンしてください。
If you are concerned about security issues with interactive HTML, or if you would like to save code that you have modified, you can download the notebook file that each page is based on and execute the code in a local environment. From the <i class="fas fa-download"></i> menu at top right, click <span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><span style="margin: 0 .4em 0 .4em;">.ipynb</span></span>, or in the <i class="fab fa-github"></i> menu click on <span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><span style="margin: 0 .4em 0 .4em;">repository</span></span> to clone the <a href="https://github.com/UTokyo-ICEPP/qc-workbook" target="_blank">GitHub repository</a>.

ノートブックをローカルに実行するためには、Pythonバージョン3.8以上が必要です。また、`pip`を使って以下のパッケージをインストールする必要があります。
Python version 3.8 or above is required to run a notebook locally. You must also use `pip` to install the following packages.

```{code-block}
pip install qiskit qiskit-ibm-provider qiskit-ibm-runtime matplotlib pylatexenc tabulate
```
