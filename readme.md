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

![Example Spectrum](figures/figure1_example_spectra.png)

---

### Effective Bandwidth Distribution (Reference vs No Fabry–Perot)
Demonstrates the dominant role of Fabry–Perot oscillations in spectral variability.

![Bandwidth Distribution](figures/figure3_bandwidth_distribution.png)

---

### Negative Absorption Diagnostic (Beyond Mean Statistics)
Reveals localized physical violations invisible to average-based validation.

![Negative Absorption](figures/figure4_negative_absorption.png)

---

## 📁 Repository Structure

```
physics-constraint-paradox/
├── generator_algo.py # Reference physics-informed generator
├── ablation_algo.py # Ablation variants A–D
├── validation_utils.py # Physical validation metrics
├── run_experiments.py # Reproduces all paper figures
├── figures/ # Saved figure outputs
│ ├── fig1_example_spectra.py
│ ├── fig3_bandwidth_distribution.py
│ └── fig4_negative_absorption.py
├── requirements.txt
├── LICENSE
└── README.md
```
