"""
Figure: Example spectra from reference generator and ablation variants.

Shows representative spectra to illustrate differences between variants.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict


def create_example_spectra(results: Dict, save_path: str = 'figures/example_spectra.png'):
    """
    Create figure showing example spectra.
    
    Parameters:
    -----------
    results : dict
        Results from ablation study
    save_path : str
        Path to save the figure
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Variant order for display
    variants = ['Reference', 'A_NoEnergyConservation', 'B_NoFabryPerot', 
                'C_FixedBandwidth', 'D_NoNoise']
    titles = ['Reference', 'A: No Energy Norm', 'B: No Fabry-Perot', 
              'C: Fixed Bandwidth', 'D: No Noise']
    
    # Plot example spectra for each variant
    for i, (variant, title) in enumerate(zip(variants, titles)):
        if i >= len(axes) - 1:  # Last subplot is for legend/key
            break
            
        ax = axes[i]
        data = results[variant]
        
        # Plot 5 random spectra
        n_spectra = min(5, data['T'].shape[0])
        indices = np.random.choice(data['T'].shape[0], n_spectra, replace=False)
        
        for idx in indices:
            ax.plot(data['wavelengths'], data['T'][idx], alpha=0.7, linewidth=1.5)
        
        # Customize subplot
        ax.set_xlabel('Wavelength (μm)', fontweight='bold')
        ax.set_ylabel('Transmittance', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        
        # Add statistics annotation
        stats = data['validation']
        annotation = (f"ε = {stats['energy_conservation']['mean_error']:.1e}\n"
                     f"A_min < 0: {stats['negative_absorption']['negative_fraction']:.2f}%")
        ax.text(0.95, 0.95, annotation, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Last subplot: comparison of all variants for one sample
    ax = axes[-1]
    sample_idx = 0  # Use first sample for comparison
    
    for variant in variants:
        data = results[variant]
        ax.plot(data['wavelengths'], data['T'][sample_idx], 
                label=variant, alpha=0.8, linewidth=2)
    
    ax.set_xlabel('Wavelength (μm)', fontweight='bold')
    ax.set_ylabel('Transmittance', fontweight='bold')
    ax.set_title('Direct Comparison (Same Parameters)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_ylim([-0.05, 1.05])
    
    # Adjust layout
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle('Example Spectra Across Generator Variants', 
                fontsize=14, fontweight='bold', y=1.02)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Example spectra figure saved to {save_path}")


if __name__ == "__main__":
    # For standalone execution, load results
    import sys
    sys.path.append('..')
    from run_experiments import run_ablation_study
    
    print("Running ablation study for example spectra...")
    results = run_ablation_study(n_samples=100)
    create_example_spectra(results)
