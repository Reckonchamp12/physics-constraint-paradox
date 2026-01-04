# Physics Constraint Paradox in Grating Coupler Dataset Generation

This repository contains the physics-informed spectrum generation algorithm and ablation framework introduced in the paper:

**"The Physics Constraint Paradox: When Less Physics Gives Better Data for ML"**  
Rahul D Ray, Department of Electronics & Electrical Engineering, BITS Pilani (Hyderabad)

## 🔬 What This Repository Is

A high-speed physics-informed surrogate generator for grating coupler spectra  
A controlled ablation framework to isolate the role of individual physics components  
A validation suite exposing hidden physical failures missed by mean statistics  
A reproducible research codebase supporting the paper's figures and conclusions  

## ⚙️ Core Idea: The Physics Constraint Paradox

Most physics-informed ML pipelines assume "more constraints = better data."  
This work demonstrates that assumption is false.

### Key Findings:
- **Energy conservation enforcement is mathematically redundant** when equations are physically consistent
- **Fabry-Perot oscillations dominate bandwidth variability** (72% reduction when removed)
- **Noise + renormalization pipelines introduce unphysical negative absorption**
- **Mean-based validation passes while pointwise physics fails**

The result is a principled recipe for building efficient, physically faithful generators without over-constraining them.

## 🧠 Algorithm Overview

The generator maps five geometric parameters → 100-point spectra (R, T, A) via a multi-stage physics pipeline:

1. Effective index computation (slab + grating + substrate)
2. Lorentzian resonance (temporal coupled-mode theory)
3. Fabry-Perot interference superposition
4. Absorption and scattering loss modeling
5. Numerical energy normalization (stability safeguard)
6. Controlled noise injection

**Execution speed:** ~200 spectra/second  
**Speedup:** ≈ 1000× faster than FDTD/FEM solvers

## 🧪 Ablation Variants Implemented

| Variant | Description |
|---------|-------------|
| Reference | Full physics-informed generator |
| A | No energy normalization |
| B | No Fabry-Perot oscillations |
| C | Fixed bandwidth |
| D | No noise injection |

Each variant is structurally identical, differing by exactly one physics component.

## 📊 Figures Reproduced by This Code

### Example Spectrum (Reference Generator)
Shows physically realistic resonance and global energy conservation.

### Effective Bandwidth Distribution (Reference vs No Fabry-Perot)
Demonstrates the dominant role of Fabry-Perot oscillations in spectral variability.

### Negative Absorption Diagnostic (Beyond Mean Statistics)
Reveals localized physical violations invisible to average-based validation.

## 📁 Repository Structure
```
physics-constraint-paradox/
├── generator_algo.py # Reference physics-informed generator
├── ablation_algo.py # Ablation variants A-D
├── validation_utils.py # Physical validation metrics
├── run_experiments.py # Reproduces all paper figures
├── figures/ # Figure generation scripts
│ ├── example_spectra.py
│ ├── bandwidth_distribution.py
│ └── negative_absorption.py
├── requirements.txt # Minimal dependencies
├── LICENSE # MIT License
└── README.md # This file
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt

```
## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```
## Reproduce All Paper Figures
```
python run_experiments.py
This will generate:
-figures/example_spectra.png
-figures/bandwidth_distribution.png
-figures/negative_absorption.png
```
## Dataset Reference
The full GC-500K dataset (500,000 grating coupler spectra with geometric parameters) is published separately. This repository contains the algorithm only, not the dataset.

## Citation
If you use this code in your research, please cite:
[Ray, R. D., The Physics Constraint Paradox: When Removing Explicit Constraints Improves Physics-Informed Data for Machine Learning, (https://arxiv.org/pdf/2512.22261)]

📄 License
MIT License - see LICENSE file for details.
