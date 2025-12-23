"""
Figure: Bandwidth distribution comparison across variants.

Shows how different ablation variants affect the bandwidth distribution.
Demonstrates the dominant role of Fabry-Perot oscillations in spectral variability.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict
from validation_utils import compute_effective_bandwidth


def create_bandwidth_distribution(results: Dict, save_path: str = 'figures/bandwidth_distribution.png'):
    """
    Create figure showing bandwidth distributions.
    
    Parameters:
    -----------
    results : dict
        Results from ablation study
    save_path : str
        Path to save the figure
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Variant order
    variants = ['Reference', 'A_NoEnergyConservation', 'B_NoFabryPerot', 
                'C_FixedBandwidth', 'D_NoNoise']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Compute bandwidths for all variants
    bandwidths = {}
    for variant in variants:
        data = results[variant]
        bandwidths[variant] = compute_effective_bandwidth(
            data['T'], data['wavelengths']
        )
    
    # Panel 1: Histogram of bandwidths
    ax = axes[0]
    bins = np.linspace(0, 500, 100)  # Bandwidth range in nm
    
    for variant, color in zip(variants, colors):
        ax.hist(bandwidths[variant], bins=bins, alpha=0.5, 
                label=variant, color=color, density=True)
    
    ax.set_xlabel('Bandwidth (nm)', fontweight='bold')
    ax.set_ylabel('Probability Density', fontweight='bold')
    ax.set_title('(a) Bandwidth Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Panel 2: Box plot comparison
    ax = axes[1]
    box_data = [bandwidths[variant] for variant in variants]
    box = ax.boxplot(box_data, labels=variants, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Bandwidth (nm)', fontweight='bold')
    ax.set_title('(b) Statistical Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    # Panel 3: Bandwidth vs Peak Transmission
    ax = axes[2]
    
    for variant, color in zip(variants, colors):
        data = results[variant]
        peak_T = np.max(data['T'], axis=1)
        bw = bandwidths[variant]
        
        # Subsample for clarity
        subsample = np.random.choice(len(bw), min(200, len(bw)), replace=False)
        ax.scatter(peak_T[subsample], bw[subsample], alpha=0.5, 
                  s=10, color=color, label=variant)
    
    ax.set_xlabel('Peak Transmission', fontweight='bold')
    ax.set_ylabel('Bandwidth (nm)', fontweight='bold')
    ax.set_title('(c) Bandwidth vs Peak Transmission', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, markerscale=2)
    
    # Panel 4: Cumulative distribution
    ax = axes[3]
    
    for variant, color in zip(variants, colors):
        bw = bandwidths[variant]
        sorted_bw = np.sort(bw)
        y = np.arange(1, len(sorted_bw) + 1) / len(sorted_bw)
        ax.plot(sorted_bw, y, color=color, linewidth=2, label=variant)
    
    ax.set_xlabel('Bandwidth (nm)', fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontweight='bold')
    ax.set_title('(d) Cumulative Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add statistics table
    stats_text = "Bandwidth Statistics (nm):\n"
    for variant in variants:
        bw = bandwidths[variant]
        stats_text += f"\n{variant[:15]:<15}: mean={np.mean(bw):.1f}, std={np.std(bw):.1f}"
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, 
             verticalalignment='bottom', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add overall title
    fig.suptitle('Effective Spectral Bandwidth Distribution', 
                fontsize=14, fontweight='bold', y=1.02)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Bandwidth distribution figure saved to {save_path}")
    
    # Print summary statistics
    print("\nBandwidth Statistics (nm):")
    print("-" * 50)
    for variant in variants:
        bw = bandwidths[variant]
        print(f"{variant:<25}: mean = {np.mean(bw):6.1f}, std = {np.std(bw):6.1f}, "
              f"median = {np.median(bw):6.1f}")
    
    # Calculate Fabry-Perot impact
    ref_bw = np.mean(bandwidths['Reference'])
    no_fp_bw = np.mean(bandwidths['B_NoFabryPerot'])
    fp_impact = (ref_bw - no_fp_bw) / ref_bw * 100
    
    print(f"\nFabry-Perot oscillations impact on bandwidth: {fp_impact:.1f}% reduction")


if __name__ == "__main__":
    # For standalone execution
    import sys
    sys.path.append('..')
    from run_experiments import run_ablation_study
    
    print("Running ablation study for bandwidth analysis...")
    results = run_ablation_study(n_samples=1000)
    create_bandwidth_distribution(results)
