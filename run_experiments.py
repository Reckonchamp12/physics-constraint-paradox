"""
Main entry point to reproduce all paper figures.

This script orchestrates the generation of all figures and analyses
presented in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time
import json
from typing import Dict, List

# Import local modules
from generator_algo import ReferenceGenerator
from ablation_algo import (
    AblationA_NoEnergyConservation,
    AblationB_NoFabryPerot, 
    AblationC_FixedBandwidth,
    AblationD_NoNoise
)
from validation_utils import (
    compute_energy_error,
    compute_negative_absorption,
    compute_validation_summary
)

# Set up matplotlib for publication-quality plots
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['font.size'] = 11
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


def generate_common_parameters(n_samples: int = 1000, seed: int = 42) -> List[Dict]:
    """
    Generate common parameter set for all variants.
    
    Ensures fair comparison across generator variants.
    """
    generator = ReferenceGenerator(seed=seed)
    return generator.generate_parameters_batch(n_samples, 0)


def run_ablation_study(n_samples: int = 1000) -> Dict:
    """
    Run the complete ablation study.
    
    Generates samples from all variants and computes validation metrics.
    """
    print("="*80)
    print("RUNNING COMPLETE ABLATION STUDY")
    print("="*80)
    print(f"Samples per variant: {n_samples}")
    
    # Generate common parameters
    print("\nGenerating common parameter set...")
    common_params = generate_common_parameters(n_samples)
    
    # Initialize all generators
    generators = {
        'Reference': ReferenceGenerator(seed=42),
        'A_NoEnergyConservation': AblationA_NoEnergyConservation(seed=42),
        'B_NoFabryPerot': AblationB_NoFabryPerot(seed=42),
        'C_FixedBandwidth': AblationC_FixedBandwidth(seed=42),
        'D_NoNoise': AblationD_NoNoise(seed=42)
    }
    
    results = {}
    
    for name, generator in generators.items():
        print(f"\nGenerating samples for {name}...")
        start_time = time.time()
        
        # Generate samples
        samples = []
        for i, params in enumerate(common_params):
            add_noise = (name != 'D_NoNoise')  # Only add noise if not ablation D
            sample = generator.generate_sample(params, i, add_noise=add_noise)
            samples.append(sample)
        
        # Extract data
        R = np.array([s['R'] for s in samples])
        T = np.array([s['T'] for s in samples])
        A = np.array([s['A'] for s in samples])
        wavelengths = generator.wavelengths_um
        
        # Compute validation metrics
        validation = compute_validation_summary(R, T, A, wavelengths)
        
        results[name] = {
            'R': R,
            'T': T, 
            'A': A,
            'wavelengths': wavelengths,
            'validation': validation,
            'generation_time': time.time() - start_time,
            'samples': samples
        }
        
        print(f"  Completed in {results[name]['generation_time']:.2f} seconds")
        print(f"  Energy error: {validation['energy_conservation']['mean_error']:.2e}")
        print(f"  Negative absorption: {validation['negative_absorption']['negative_fraction']:.3f}%")
    
    return results


def save_results(results: Dict, output_dir: str = 'results'):
    """
    Save ablation study results to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary statistics
    summary = {}
    for name, data in results.items():
        summary[name] = {
            'validation': data['validation'],
            'generation_time': data['generation_time']
        }
    
    with open(os.path.join(output_dir, 'ablation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save data for each variant
    for name, data in results.items():
        np.savez_compressed(
            os.path.join(output_dir, f'{name}_data.npz'),
            R=data['R'],
            T=data['T'],
            A=data['A'],
            wavelengths=data['wavelengths']
        )
    
    print(f"\nResults saved to '{output_dir}' directory")


def main():
    """
    Main function to reproduce all paper figures.
    """
    print("="*80)
    print("PHYSICS CONSTRAINT PARADOX - REPRODUCING PAPER FIGURES")
    print("="*80)
    
    # Create output directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run ablation study
    results = run_ablation_study(n_samples=1000)
    
    # Save results
    save_results(results)
    
    # Generate figures
    print("\n" + "="*80)
    print("GENERATING PAPER FIGURES")
    print("="*80)
    
    # Import and run figure generation scripts
    print("\nGenerating Figure 1: Example Spectra...")
    from figures.fig1_example_spectra import create_figure1
    create_figure1(results)
    
    print("\nGenerating Figure 3: Bandwidth Distribution...")
    from figures.fig3_bandwidth_distribution import create_figure3
    create_figure3(results)
    
    print("\nGenerating Figure 4: Negative Absorption Diagnostics...")
    from figures.fig4_negative_absorption import create_figure4
    create_figure4(results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("- figures/fig1_example_spectra.png")
    print("- figures/fig3_bandwidth_distribution.png")
    print("- figures/fig4_negative_absorption.png")
    print("- results/ablation_summary.json")
    print("- results/*_data.npz (data files for each variant)")
    print("\nAll paper figures have been reproduced successfully!")


if __name__ == "__main__":
    main()
