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

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-input", "remove-output"]}

# 【Exercise】Find Tracks of Charged Particles Produced in High-Energy Physics Experiment

+++

In this assignment, you will think about how to apply the variational quantum eigensolver to physics experiment. Specifically, we focuse on high-energy physics (HEP) experiment and attempt to **reconstruct tracks of charged particles**, which is essential for HEP experiments, using variational quantum eigensolver.

```{contents} Contents
---
local: true
---
```

+++

## Introduction
In the {doc}`notebook<vqe>` that introduced **Variational Quantum Eigensolver** (VQE), we learned about the VQE and the basic method to implement a variational quantum circuit. In this section, we consider how we can apply VQE to high-energy physics.

In HEP experiments, particles with electric charges (such as protons) are accelerated through magnetic field to high energy and collide with each other. The collision produces a numerous number of secondary particles, which are measured using a detector surrounding a collision point. Through these experiments, we invetigate the fundamental properties of the particles and their underlying interaction mechanism. To do this, the detector signals are processed to identify the secondary particles and measure their energies and momenta precisely. In this exercise, you will learn if we can use VQE to **reconstruct tracks of charged particles** (generally called **tracking**) as the important first step to identify the particles.

+++

(hep)=
## High-Energy Physics Experiment
(hep_LHC)=
### Overview of LHC Experiment

```{image} figs/LHC_ATLAS.png
:alt: LHC_ATLAS
:width: 1000px
:align: center
```

The Large Hadron Collider (LHC) is a circular collider that lies at the border of Switzerland and France, operated by European Organization for Nuclear Research (CERN). It is placed inside a tunnel, located about 100 meters underground, with a 27 kilometer circumference. Currently the LHC can accelerate protons up to 6.5 TeV in energy (1 TeV is $10^12$ eV). The accelerated protons collide head-on at the world's highest energy of 13.6 TeV (see the picture at the upper left). The picture at the upper right shows the LHC in the underground tunnel.

Four experiments, ATLAS, CMS, ALICE, and LHCb, are carried out at the LHC. Of these, the ATLAS and CMS experiments use large general-purpose detectors (the ATLAS detector is shown at the bottom left). In ATLAS and CMS, secondary particles generated by proton collisions are observed by high-precision detectors arranged surrounding the collision point, making it possible to observe various reactions and explore new phenomena. The bottom right figure shows an actual event observed with ATLAS detector, and it is a candidate of Higgs boson which was first observed in 2012 by the ATLAS and CMS experiments. The Higgs boson itself is observed not as a single particle but as a collection of particles produced from the decay of Higgs boson.

+++

(hep_detect)=
### Measurement of Charged Particles

The detectors used in ATLAS and CMS experiments consist of multiple detectors with differing characteristics, arranged outward in concentric layers. The innermost detector is used to reconstruct and identify charged particles, and is one of the most important detectors in the experiments. The detector itself is made up of roughly 10 layers and sends multiple detector signals when a single charged particle passes through it. For example, as in the lefft figure, a single proton collision produces many secondary particles, and leave the detector signal called "hits" (corresponding to colored points in the figure). From the collection of these hits, a set of hits produced by a charged particle traversing the detector is selected to reconstruct the particle track. The colored lines in the right figure correspond to reconstructed tracks. The reconstruction of charged particle tracks, called tracking, is one of the most important experimental techniques in HEP experiment.

The charged particle tracking is essentially an optimization problem of detector hits to tracks, and the computational cost often grows exponentially in the number of particles produced in collisions. The LHC is expected to be upgraded to higher beam intensity (number of protons to be accelerated in a single beam) or higher beam energy, and the current reconstruction technique could suffer from increasing computational complexity. Various new techniques are being examined, including quantum computing, to reduce the computational time or resources required for future HEP experiments.

```{image} figs/tracking.png
:alt: tracking
:width: 1000px
:align: center
```

+++

(ML_challenge)=
### TrackML challenge

CERN has plans to upgrade the LHC to "High-Luminosity LHC" or HL-LHC (expected to start operation in 2029), in which the beam intensity is significantly increased. The increased beam intensity will result in 10-fold increase in collision rate and hence produced particle density, making charged particle tracking even more challenging.

Unfortunately it is hard to imagine that quantum computing can be used in real experiments from 2029. However, some researchers thought that the capability of classical reconstruction algorithm can be enhanced using advanced machine learning techniques and held a competition called <a href="https://www.kaggle.com/c/trackml-particle-identification" target="_blank">TrackML Particle Tracking challenge</a> in 2018. In this competition, the public data composed of detector hits simulated in HL-LHC environment are provided and the participants attempted to apply their own algorithms to these data to compete with each other in speed and accuracy.

+++

(tracking)=
## Exercise: Tracking with VQE

Our goal in this exercise is to perform charged particle tracking with VQE using this TrackML challenge dataset. Since it is still difficult to solve large-scale tracking problem, we only consider the case of small number of particles produced in collision.

+++

Fisrt, import the necessary libraries.

```{code-cell} ipython3
---
editable: true
jupyter:
  outputs_hidden: false
pycharm:
  name: '#%%

    '
slideshow:
  slide_type: ''
tags: [remove-input, remove-output]
---
import pprint
import numpy as np
import h5py
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import BackendEstimator
from qiskit_algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_optimization.applications import OptimizationApplication
from qiskit_aer import AerSimulator
```

(hamiltonian_form)=
### Hamiltonian and VQE

In order to use VQE for optimization, the problem will need to be formulated in the form of Hamiltonian. If the problem is formulated such that the solution corresponds to the lowest energy state of the Hamiltonian, the VQE could solve the problem by finding such state.

+++

(pre_processing)=
#### Preparation

This assignment will use data from the TrackML challenge, but because original data is difficult to work with, we will use data that has been preprocessed to make it easier to use in quantum calculation, based on {cite}`bapst2019pattern`.

First, three consecutive layer hits are selected as shown in the figure below (a set of three-layer hits surrounded by a dashed line is called 'segment' here). Think of colored dots as hits. These segments are constructd simply by connecting hits, therefore they do not necessarily originate from real particles. Some of them may arise from connecting hits from different particles or from misidentifying detector noise as hits and grouping them to form segments. However, those "fake" segments can point to any directions while "genuine" segments originating from real particles should point to the center of the detector where collisions occur.
Given this, each segment is assigned *score* depending on how well the direction of the segment is consistent with that from the detector center. As explained later, the score is assigned a smaller value with increasing consistency of the segment pointing to the detector center. The segments obviously identified as fakes are ignored from the first place, and only segments with scores less than certain threshold are considered below.

```{image} figs/track_segment.png
:alt: track_segment
:width: 350px
:align: center
```

Next, all pairings of selected segments are taken. For each pairing, a score is then assigned depending on how the paired segments are **consistent with track originating from a single particle**. This score is designed to have value closer to -1 as the paired segments get closer to forming an identical track.

For example, if one of three hits in a segment is shared by two segments (such as those two red segments in the figure), it is not considered a track candidate and the score becomes +1. This is because these tracks with branched or merged segments are not consistent with charged particles produced at the center of the detector.

Furthermore, the segments shown in orange (one of the hits is missing in intermediate layer) and in brown (a zigzag pattern) are not tracks of our interest either, therefore the score is set value larger than -1. The reason why a zigzag pattern is not preferred is that if a charged particle enters perpendicularly into a uniform magnetic field, the trajectory of the particle is bent with a constant curvature on the incident plane due to Lorentz force.

+++

#### QUBO Format

Under this setup, the next step is whether a given segment is adopted as part of particle tracks or rejected as fake. In a sample of $N$ segments, the adoptation or rejection of $i$-th segment is associated to 1 or 0 of a binary variable $T_i$, and the variable $T_i$ is determined such that the objective function defined as

$$
O(b, T) = \sum_{i=1}^N a_{i} T_i + \sum_{i=1}^N \sum_{j<i}^N b_{ij} T_i T_j
$$

is minimized. Here $a_i$ is the score of $i$-th segment and $b_{ij}$ is the score of the pair of $i$- and $j$-th segments. The objective function becomes smaller by selecting segments that have smaller $a_i$ values (pointing towards the detector center) and are paired with other segments with smaller $b_{ij}$ values (more consistent with a real track) and rejecting otherwise. Once correct segments are identified, the corresponding tracks can be reconstructed with high efficiency. Therefore, solving this minimization problem is the key to tracking.

The optimization problem of the form illustrated above is called **QUBO**（*Quadratic Unconstrained Binary Optimization*). The form looks unique at first glance, but it is known that various optimization problems (for example, the famous travelling salesman problem) can be converted to QUBO format. This exercise considers only gate-based quantum computer, but different type of quantum computer called quantum annealing is designed to solve QUBO problem primarily.

Let us first read out the scores $a_i$ and $b_{ij}$.

```{code-cell} ipython3
# Reading out scores
with h5py.File('data/QUBO_05pct_input.h5', 'r') as source:
    a_score = source['a_score'][()]
    b_score = source['b_score'][()]

print(f'Number of segments: {a_score.shape[0]}')
# Print out the first 5x5
print(a_score[:5])
print(b_score[:5, :5])
```

#### Ising Format

The QUBO objective function is not the form of Hamiltonian (i.e, not Hermitian operator). Therefore, the objective function needs to be transformed before solving with VQE. Given that $T_i$ takes a binary value $\{0, 1\}$, a new variable $s_i$ with values of $\{+1, -1\}$ can be defined by

$$
T_i = \frac{1}{2} (1 - s_i).
$$

Note that $\{+1, -1\}$ is the eigenvalue of Pauli operator. By replacing $s_i$ with Pauli $Z$ operator acting on $i$-th qubit, the following objective Hamiltonian for which computational basis states in $N$-qubit system correspond to the eigenstates that encode adoptation or rejection of the segments is obtained.

$$
H(h, J, s) = \sum_{i=1}^N h_i Z_i + \sum_{i=1}^N \sum_{j<i}^N J_{ij} Z_i Z_j + \text{(constant)}
$$

The form of this Hamiltonian is the same as Ising model Hamiltonian, which often appears in various fields of natural science. The $\text{constant}$ is constant and has no impact in variational method, hence is ignored in the rest of this exercise.

By following the above prescription, please calculate the coefficients $h_i$ and $J_{ij}$ of the Hamiltonian in the next cell.

```{code-cell} ipython3
num_qubits = a_score.shape[0]

coeff_h = np.zeros(num_qubits)
coeff_J = np.zeros((num_qubits, num_qubits))

##################
### EDIT BELOW ###
##################

# Calculate coeff_h and coeff_J from b_ij

##################
### EDIT ABOVE ###
##################
```

Next, let us define the Hamiltonian used in VQE as a SparsePauliOp object. In {ref}`vqe_imp` the SparsePauliOp was used to define a single Pauli string $ZXY$, but the same class can be used for the sum of Pauli strings. For example,

$$
H = 0.2 IIZ + 0.3 ZZI + 0.1 ZIZ
$$

can be expressed as

```python
H = SparsePauliOp(['IIZ', 'ZZI', 'ZIZ'], coeffs=[0.2, 0.3, 0.1])
```

Note that the qubits are ordered from right to left (the most right operator acts on the 0-th qubit) according to the rule in Qiskit.

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

##################
### EDIT BELOW ###
##################

# Pick up all Pauli strings with non-zero coefficients and make the array of corresponding coefficients

pauli_products = []
coeffs = []

##################
### EDIT ABOVE ###
##################

hamiltonian = SparsePauliOp(pauli_products, coeffs=coeffs)
```

(tracking_vqe)=
#### Executing VQE

Now we try to approximately obtain the lowest energy eigenvalues using VQE with the Hamiltonian defined above. But, before doing that, let us diagonalize the Hamiltonian matrix and calculate the exact energy eigenvalues and eigenstates.

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
# Diagonalize the Hamiltonian and calculate the energy eigenvalues and eigenstates
ee = NumPyMinimumEigensolver()
result_diag = ee.compute_minimum_eigenvalue(hamiltonian)

# Print out the combination of qubits corresponding to the lowest energy
print(f'Minimum eigenvalue (diagonalization): {result_diag.eigenvalue.real}')
# Expand the state with computational bases and select the one with the highest probability
optimal_segments_diag = OptimizationApplication.sample_most_likely(result_diag.eigenstate)
print(f'Optimal segments (diagonalization): {optimal_segments_diag}')
```

The qubits with 1 in `optimal_segments_diag` correspond to segments that make the value of the objective function smallest.

Next, the energy eigenvalues are obtained using VQE. The following code uses SPSA or COBYLA as optimizer.

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
backend = AerSimulator()
# Create Estimator instance
estimator = BackendEstimator(backend)

# Define variational form of VQE using a built-in function called TwoLocal.
ansatz = TwoLocal(num_qubits, 'ry', 'cz', 'linear', reps=1)

# Optimizer
optimizer_name = 'SPSA'

if optimizer_name == 'SPSA':
    optimizer = SPSA(maxiter=300)
    grad = ParamShiftEstimatorGradient(estimator)

elif optimizer_name == 'COBYLA':
    optimizer = COBYLA(maxiter=500)
    grad = None

# Initialize parameters
rng = np.random.default_rng()
init = rng.uniform(0., 2. * np.pi, size=len(ansatz.parameters))

# Make VQE object and find the ground state
vqe = VQE(estimator, ansatz, optimizer, gradient=grad, initial_point=init)
result_vqe = vqe.compute_minimum_eigenvalue(hamiltonian)

# Create state vector from ansatz using optimized parameters
optimal_state = Statevector(ansatz.bind_parameters(result_vqe.optimal_parameters))

# Print out the combination of qubits with the lowest energy
print(f'Minimum eigenvalue (VQE): {result_vqe.eigenvalue.real}')
optimal_segments_vqe = OptimizationApplication.sample_most_likely(optimal_state)
print(f'Optimal segments (VQE): {optimal_segments_vqe}')
```

+++ {"pycharm": {"name": "#%% md\n"}}

(omake)=
### A Giveaway

Even if the tracking works successfully, expressing the answer as a string of 0s and 1s is a bit dry. Run the following code to visually confirm if correct tracks are found.

This code viualizes the detector hits used in QUBO by projecting them onto a plane perpendicular to the beam axis and shows which detector hits are selected after optimization. The green lines correspond to found tracks and the blue lines, altogether with the green ones, correspond to all track candidates. In this exercise, only small number of qubits are used and therefore most of tracks are not found. However, it shows that a correct track is successfully found from the presence of green line.

```{code-cell} ipython3
---
pycharm:
  name: '#%%

    '
tags: [raises-exception, remove-output]
---
from hepqpr.qallse import DataWrapper, Qallse, TrackRecreaterD
from hepqpr.qallse.plotting import iplot_results, iplot_results_tracks
from hepqpr.qallse.utils import diff_rows

optimal_segments = optimal_segments_vqe
# optimal_segments = optimal_segments_diag

# Since each segment has ID, the data is passed to Qallse in the form of {ID: 0 or 1}
# Read out segment IDs (decode into UTF-8 because the data is stored in binary text data)
with h5py.File('data/QUBO_05pct_input.h5', 'r') as source:
    triplet_keys = map(lambda key: key.decode('UTF-8'), source['triplet_keys'][()])

# Dictionary in {ID: 0 or 1}
samples = dict(zip(triplet_keys, optimal_segments))

# get the results
all_doublets = Qallse.process_sample(samples)

final_tracks, final_doublets = TrackRecreaterD().process_results(all_doublets)

dw = DataWrapper.from_path('data/event000001000-hits.csv')

p, r, ms = dw.compute_score(final_doublets)
trackml_score = dw.compute_trackml_score(final_tracks)

print(f'SCORE  -- precision (%): {p * 100}, recall (%): {r * 100}, missing: {len(ms)}')
print(f'          tracks found: {len(final_tracks)}, trackml score (%): {trackml_score * 100}')

dims = ['x', 'y']
_, missings, _ = diff_rows(final_doublets, dw.get_real_doublets())
dout = 'plot-ising_found_tracks.html'
iplot_results(dw, final_doublets, missings, dims=dims, filename=dout)
```

**Items to submit**:
- Code used to implement the Hamiltonian
