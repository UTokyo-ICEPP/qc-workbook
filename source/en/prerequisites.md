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

# Preparation of Hands-on Exercise

```{contents} Contents
---
local: true
---
```

+++

## IBM Quantum

### Obtain IBMid and log in IBM Quantum

To use IBM Quantum, you must first create an IBMid account and receive a service token. Obtain an IBMid from the <a href="https://quantum.ibm.com/" target="_blank">IBM Quantum</a> website and log into the service.

(install_token)=
### （Local environment）Get IBM Quantum API token and store it in Qiskit setting

You can skip the following step if you are going to execute program on IBM Quantum Lab (Jupyter Lab on IBM Quantum website).

On the main screen shown after logging in, copy the token displayed in the "Your API token" area.
```{image} figs/ibmq_home.png
:height: 400px
:name: My Account
```

Service tokens issued on an account basis are used for username+password in Python program to connect to IBM Quantum. If you have a write access to local disks, the authentication to access IBM Quantum can be processed transparently by saving the token into a setting file. Copy the token from IBM Quantum and paste it to `__paste_your_token_here__` in the following code cell and execute.

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account('__paste_your_token_here__')
```

By saving the token in a setting file, your access to IBM Quantum can be authenticated through IBMProvider, as follows:

```{code-block} python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel='ibm_quantum')
```

In the IBM Quantum Lab, the access token is automatically stored, therefore this code will work as it is.

If you do not have a write access to local disks (e.g, you are using this notebook interactively), you need to authenticate by hand every time you execute a python program (restarting a Jupyter kernel) as follows:

```{code-block} python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel='ibm_quantum', token='__paste_your_token_here__')
```

## How to use this workbook

### Interactive HTML

You can execute the programs written in each cell of this workbook directly on the browser, just like using <a href="https://jupyter.org/" target="_blank">Jupyter Notebook</a>. Move the cursor over <i class="fas fa-rocket"></i> at the top-right of the page and click <span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><i class="fas fa-play" style="margin-left: .4em;"></i> <span style="margin: 0 .4em 0 .4em;">Live Code</span></span> from the menu. A status indicator will appear below the page title, and wait until the status becomes <span style="color: green; font-family: monospace; font-weight: bold; font-size: 1em;">ready</span>.


```{image} figs/toggle_interactive.jpg
:height: 400px
:name: Turn interactive contents on
```

When the page becomes interactive, <span style="background-color:#dddddd; font-family:'Roboto', sans-serif; margin:0 1em 0 1em;">run</span> and <span style="background-color:#dddddd; font-family:'Roboto', sans-serif; margin:0 1em 0 1em;">restart</span> buttons will appear in a code cell, and the output of the code will be displayed just below.

```{image} figs/interactive_cell.jpg
:height: 200px
:name: Interactive code cell
```

In this state, you can directly edit the contents of the cell and click the "run" button (or press Shift + Enter) to run the Python code. There are several points to be aware of when doing this.

- Since the entire page is treated as a single program, the values of variables defined in a cell are carried over into other cells until the "restart" button is pressed.
- If you do not perform any actions on the page for some time, the connection to the executing server is lost. If this happens, you must reload the page and start up the interactive content again.
- The code is executed using an external service called <a href="https://mybinder.org/" target="_blank">mybinder.org</a>, therefore, you should avoid sending any sensitive information such as personal information whenever possible (since the communication is encrypted and individual user programs are executed in a separate container on mybinder.org, the information is somewhat protected, but it is the best practice to avoid sending any sensitive information). However, as explained above, the service token of IBM Quantum needs to be sent to use the service.

### Jupyter Notebook

If you are concerned about the security of interactive HTML or you want to save codes that you have modified, you can download the original notebook file from each page and execute the codes in a local environment. Click <span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><span style="margin: 0 .4em 0 .4em;">.ipynb</span></span> from the menu of <i class="fas fa-download"></i> at the top-right, or make a clone of <a href="https://github.com/UTokyo-ICEPP/qc-workbook" target="_blank">GitHub repository</a> linked from <span style="background-color:#5a5a5a; color:white; font-family:Lato, sans-serif; font-weight:400; font-size:15px;"><span style="margin: 0 .4em 0 .4em;">repository</span></span> of the menu of <i class="fab fa-github"></i>.

Python version 3.8 or above is required to execute the notebook locally. You need to install the following packages using `pip`.


```{code-block}
pip install qiskit qiskit-aer qiskit-ibm-runtime qiskit-experiments qiskit-machine-learning qiskit-optimization matplotlib pylatexenc pandas tabulate
```
