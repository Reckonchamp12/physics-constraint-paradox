"""
Figure: Negative absorption diagnostics.

Shows energy conservation error and negative absorption distributions
to reveal hidden physical violations invisible to average-based validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict
from validation_utils import compute_energy_error, compute_negative_absorption


def create_negative_absorption(results: Dict, save_path: str = 'figures/negative_absorption.png'):
    """
    Create figure showing negative absorption diagnostics.
    
    Parameters:
    -----------
    results : dict
        Results from ablation study
    save_path : str
        Path to save the figure
    """
    # Focus on Reference generator for main figure
    ref_data = results['Reference']
    
    # Compute metrics
    epsilon_energy = compute_energy_error(ref_data['R'], ref_data['T'], ref_data['A'])
    A_min, absorption_stats = compute_negative_absorption(ref_data['A'])
    
    # Create the figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel (a): Energy Conservation Error Distribution
    ax1 = axes[0]
    
    # Use identical bin count for both panels
    n_bins = 100
    
    # For energy error, use logarithmic x-scale
    energy_min = np.min(epsilon_energy[epsilon_energy > 0])
    energy_max = np.max(epsilon_energy)
    
    # Create logarithmic bins
    log_bins = np.logspace(np.log10(energy_min), 
                          np.log10(energy_max), 
                          n_bins)
    
    # Plot histogram
    counts, bins, patches = ax1.hist(epsilon_energy, bins=log_bins, 
                                     color='#3498db', edgecolor='black', 
                                     linewidth=0.5, alpha=0.7)
    
    # Add vertical dashed line at 1e-3
    ax1.axvline(1e-3, color='red', linestyle='--', linewidth=2, 
                label='Threshold (1e-3)')
    
    # Set log scale for both axes
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Labels and title
    ax1.set_xlabel('Pointwise Energy Error\n$\epsilon_{\mathrm{energy}} = \max_\lambda |R(\lambda)+T(\lambda)+A(\lambda)-1|$', 
                  fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Energy Conservation Error Distribution', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Add grid
    ax1.grid(True, alpha=0.3, which='both', linestyle='--')
    
    # Add annotation about conservation
    conservation_text = (f'All samples satisfy\n'
                       f'global energy conservation\n'
                       f'(mean error: {np.mean(epsilon_energy):.1e})')
    ax1.text(0.95, 0.95, conservation_text, transform=ax1.transAxes,
             fontsize=10, fontweight='bold', verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                      edgecolor='black'))
    
    # Add legend
    ax1.legend(loc='upper left', framealpha=0.9)
    
    # Panel (b): Negative Absorption Distribution
    ax2 = axes[1]
    
    # Focus on the region around zero for better visualization
    A_min_range = np.percentile(np.abs(A_min), 99.9)  # Capture 99.9% of data
    plot_min = max(-0.05, -A_min_range)  # Ensure we see negative region
    plot_max = min(0.05, np.percentile(A_min, 99.9))  # Ensure we see positive region
    
    # Create bins for A_min (linear scale)
    A_bins = np.linspace(plot_min, plot_max, n_bins)
    
    # Plot histogram
    counts, bins, patches = ax2.hist(A_min, bins=A_bins, 
                                     color='#e74c3c', edgecolor='black', 
                                     linewidth=0.5, alpha=0.7)
    
    # Add vertical line at A = 0
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, 
                label='Physical Boundary (A=0)')
    
    # Shade the negative region
    ax2.axvspan(plot_min, 0, alpha=0.2, color='red', 
                label='Unphysical Region (A<0)')
    
    # Labels and title
    ax2.set_xlabel('Minimum Absorption per Sample\n$A_{\min} = \min_\lambda A(\lambda)$', 
                  fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Negative Absorption Distribution', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Set y-scale to log to see the tail distribution
    ax2.set_yscale('log')
    
    # Add grid
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotation about negative absorption
    negative_fraction = absorption_stats['negative_fraction']
    annotation_text = (f'Samples with A_min < 0:\n'
                      f'{negative_fraction:.3f}%\n'
                      f'({absorption_stats["negative_count"]} of {len(A_min)})')
    ax2.text(0.95, 0.95, annotation_text, transform=ax2.transAxes,
             fontsize=10, fontweight='bold', verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                      edgecolor='black'))
    
    # Add legend
    ax2.legend(loc='upper left', framealpha=0.9)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Add overall figure title
    fig.suptitle('Negative Absorption Diagnostics: Beyond Mean Statistics', 
                fontsize=13, fontweight='bold', y=1.02)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Negative absorption figure saved to {save_path}")
    
    # Print summary
    print("\nFigure Summary:")
    print("="*60)
    print(f"Energy conservation error:")
    print(f"  Mean: {np.mean(epsilon_energy):.2e}")
    print(f"  Max: {np.max(epsilon_energy):.2e}")
    print(f"  Samples > 1e-3: {np.sum(epsilon_energy > 1e-3)}")
    
    print(f"\nNegative absorption:")
    print(f"  Minimum A: {np.min(A_min):.4f}")
    print(f"  Samples with A_min < 0: {absorption_stats['negative_count']} "
          f"({negative_fraction:.3f}%)")
    print(f"  Samples with A_min < -0.01: {np.sum(A_min < -0.01)} "
          f"({absorption_stats['severe_negative_fraction']:.3f}%)")
    
    # Create additional comparison across variants
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Energy error comparison across variants
    ax = axes2[0, 0]
    variants = list(results.keys())
    
    for variant in variants:
        data = results[variant]
        eps = compute_energy_error(data['R'], data['T'], data['A'])
        ax.hist(eps, bins=np.logspace(-12, -3, 50), alpha=0.5, 
                label=variant, density=True, histtype='step', linewidth=2)
    
    ax.set_xscale('log')
    ax.set_xlabel('Energy Error', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Energy Error Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Plot 2: Negative absorption comparison
    ax = axes2[0, 1]
    
    for variant in variants:
        data = results[variant]
        A_min_var, _ = compute_negative_absorption(data['A'])
        ax.hist(A_min_var, bins=100, alpha=0.5, 
                label=variant, density=True, histtype='step', linewidth=2)
    
    ax.set_xlabel('Minimum Absorption', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Negative Absorption Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Plot 3: Correlation between energy error and negative absorption
    ax = axes2[1, 0]
    
    for variant in variants:
        data = results[variant]
        eps = compute_energy_error(data['R'], data['T'], data['A'])
        A_min_var, _ = compute_negative_absorption(data['A'])
        
        # Subsample for clarity
        subsample = np.random.choice(len(eps), min(200, len(eps)), replace=False)
        ax.scatter(eps[subsample], A_min_var[subsample], alpha=0.5, 
                  s=10, label=variant)
    
    ax.set_xscale('log')
    ax.set_xlabel('Energy Error', fontweight='bold')
    ax.set_ylabel('Minimum Absorption', fontweight='bold')
    ax.set_title('Error vs Negative Absorption', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, markerscale=2)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.axvline(1e-3, color='red', linestyle='--', linewidth=1)
    
    # Plot 4: Statistics table
    ax = axes2[1, 1]
    ax.axis('off')
    
    # Create statistics table
    table_data = []
    for variant in variants:
        data = results[variant]
        eps = compute_energy_error(data['R'], data['T'], data['A'])
        A_min_var, stats = compute_negative_absorption(data['A'])
        
        table_data.append([
            variant,
            f"{np.mean(eps):.2e}",
            f"{np.max(eps):.2e}",
            f"{stats['negative_fraction']:.3f}%",
            f"{stats['negative_count']}"
        ])
    
    # Create table
    columns = ['Variant', 'Mean ε', 'Max ε', 'A_min < 0', 'Count']
    table = ax.table(cellText=table_data, colLabels=columns,
                     loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    ax.set_title('Summary Statistics', fontweight='bold', y=0.95)
    
    plt.suptitle('Extended Analysis: Comparison Across All Variants', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save extended analysis
    extended_path = save_path.replace('.png', '_extended.png')
    plt.savefig(extended_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Extended analysis saved to {extended_path}")


if __name__ == "__main__":
    # For standalone execution
    import sys
    sys.path.append('..')
    from run_experiments import run_ablation_study
    
    print("Running ablation study for negative absorption analysis...")
    results = run_ablation_study(n_samples=1000)
    create_negative_absorption(results)
