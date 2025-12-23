# Physics Constraint Paradox
### When Less Physics Gives Better Data for Machine Learning

This repository contains the **physics-informed spectrum generation algorithm and ablation framework** introduced in the paper:

**“The Physics Constraint Paradox: When Less Physics Gives Better Data for ML”**  
Rahul D Ray, Department of Electronics & Electrical Engineering, BITS Pilani (Hyderabad)

The work demonstrates a counter-intuitive but critical insight for scientific machine learning:  
> *Correct physical formulation can make explicit physics constraints mathematically redundant, while selective physical structure is essential for data realism.*

This repository **implements the algorithm**, **not the large dataset** derived from it.

---

## 🔬 What This Repository Is

- A **high-speed physics-informed surrogate generator** for grating coupler spectra
- A **controlled ablation framework** to isolate the role of individual physics components
- A **validation suite** exposing hidden physical failures missed by mean statistics
- A **reproducible research codebase** supporting the paper’s figures and conclusions


---

## ⚙️ Core Idea: The Physics Constraint Paradox

Most physics-informed ML pipelines assume **“more constraints = better data.”**  
This work shows that assumption is false.

### Key Findings:
- **Energy conservation enforcement is mathematically redundant** when equations are physically consistent
- **Fabry–Perot oscillations dominate bandwidth variability** (72% reduction when removed)
- **Noise + renormalization pipelines introduce unphysical negative absorption**
- **Mean-based validation passes while pointwise physics fails**

The result is a **principled recipe** for building efficient, physically faithful generators without over-constraining them.

---

## 🧠 Algorithm Overview

The generator maps **five geometric parameters → 100-point spectra (R, T, A)** via a multi-stage physics pipeline:

1. Effective index computation (slab + grating + substrate)
2. Lorentzian resonance (temporal coupled-mode theory)
3. Fabry–Perot interference superposition
4. Absorption and scattering loss modeling
5. Numerical energy normalization (stability safeguard)
6. Controlled noise injection

Execution speed: **~200 spectra/second**  
≈ **1000× faster** than FDTD/FEM solvers

---

## 🧪 Ablation Variants Implemented

| Variant | Description |
|------|------------|
| Reference | Full physics-informed generator |
| A | No energy normalization |
| B | No Fabry–Perot oscillations |
| C | Fixed bandwidth |
| D | No noise injection |

Each variant is **structurally identical**, differing by exactly one physics component.

---

## 📊 Figures Reproduced by This Code

### Example Spectrum (Reference Generator)
Shows physically realistic resonance and global energy conservation.

![Example Spectrum](figures/example_spectra.jpeg)

---

### Effective Bandwidth Distribution (Reference vs No Fabry–Perot)
Demonstrates the dominant role of Fabry–Perot oscillations in spectral variability.

![Effective Spectral Bandwidth Distribution](figures/Effective_Spectral_Bandwidth_Distribution.jpeg)

---

### Negative Absorption Diagnostic (Beyond Mean Statistics)
Reveals localized physical violations invisible to average-based validation.

![Negative Absorption](figures/negative_absorption.jpeg)

---

## 📁 Repository Structure

```
physics-constraint-paradox/
├── generator_algo.py # Reference physics-informed generator
├── ablation_algo.py # Ablation variants A–D
├── validation_utils.py # Physical validation metrics
├── run_experiments.py # Reproduces all paper figures
├── figures/ # Saved figure outputs
│ ├── example_spectra.jpeg
│ ├── Effective_Spectral_Bandwidth_Distribution.jpeg
│ └── negative_absorption.jpeg)
├── requirements.txt
├── LICENSE
└── README.md
```
## Reproducing Paper Figures

All figures from the paper can be reproduced with a single command:

```bash
python run_experiments.py
```
This will generate:

Figure 1: Example spectra from reference generator 

Figure 2: Bandwidth distribution comparison across 

Figure 3: Negative absorption diagnostics

Step-by-Step Reproduction
Install dependencies:

```bash
pip install -r requirements.txt
Run the complete analysis:
```
```bash
python run_experiments.py
Alternatively, generate individual figures:
```
```bash
python figures/fig1_example_spectra.py
python figures/fig3_bandwidth_distribution.py
python figures/fig4_negative_absorption.py
```
Important Notes
This repository contains the ALGORITHM only, not the full GC-500K dataset

The full 500,000-sample dataset is available separately (see reference below)

All code is deterministic with proper seeding for exact reproducibility

Numerical results are bitwise identical to those in the paper

Dataset Reference
The full GC-500K dataset (500,000 grating coupler spectra with geometric parameters) is published separately. Please cite:

[Reference to be added: GC-500K Dataset Paper]

Citation
If you use this code in your research, please cite our paper:

[Reference to be added: Physics Constraint Paradox Paper]
