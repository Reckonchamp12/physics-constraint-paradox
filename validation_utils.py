"""
Physical validation utilities for grating coupler spectra.

Contains functions for computing:
- Energy conservation error
- Effective bandwidth
- Negative absorption detection
"""

import numpy as np
from typing import Tuple, Dict


def compute_energy_error(R: np.ndarray, T: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Compute pointwise energy error per sample.
    
    Parameters:
    -----------
    R, T, A : numpy arrays
        Reflectance, transmittance, absorptance spectra (n_samples x n_wavelengths)
    
    Returns:
    --------
    epsilon_energy : numpy array
        Pointwise energy error per sample: max|R(λ) + T(λ) + A(λ) - 1|
    """
    # Pointwise Energy Error: max|R(λ) + T(λ) + A(λ) - 1| per sample
    return np.max(np.abs(R + T + A - 1), axis=1)


def compute_negative_absorption(A: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Compute negative absorption statistics.
    
    Parameters:
    -----------
    A : numpy array
        Absorptance spectra (n_samples x n_wavelengths)
    
    Returns:
    --------
    A_min : numpy array
        Minimum absorption value per sample
    stats : dict
        Statistics about negative absorption
    """
    # Minimum absorption per sample
    A_min = np.min(A, axis=1)
    
    # Calculate statistics
    stats = {
        'mean': float(np.mean(A_min)),
        'median': float(np.median(A_min)),
        'min': float(np.min(A_min)),
        'max': float(np.max(A_min)),
        'negative_fraction': float(np.sum(A_min < 0) / len(A_min) * 100),
        'severe_negative_fraction': float(np.sum(A_min < -0.01) / len(A_min) * 100),
        'negative_count': int(np.sum(A_min < 0))
    }
    
    return A_min, stats


def compute_effective_bandwidth(T: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """
    Compute effective bandwidth at half maximum for each sample.
    
    Parameters:
    -----------
    T : numpy array
        Transmittance spectra (n_samples x n_wavelengths)
    wavelengths : numpy array
        Wavelength grid in consistent units
    
    Returns:
    --------
    bandwidths : numpy array
        Effective bandwidth for each sample
    """
    n_samples = T.shape[0]
    bandwidths = np.zeros(n_samples)
    
    for i in range(n_samples):
        peak_T = np.max(T[i])
        if peak_T > 0:
            half_max = peak_T / 2.0
            above_half = T[i] >= half_max
            
            if np.any(above_half):
                # Find wavelength range where transmission is above half max
                bandwidths[i] = (wavelengths[above_half][-1] - 
                               wavelengths[above_half][0])
    
    return bandwidths


def compute_validation_summary(R: np.ndarray, T: np.ndarray, A: np.ndarray, 
                             wavelengths: np.ndarray) -> Dict:
    """
    Compute comprehensive validation summary.
    
    Parameters:
    -----------
    R, T, A : numpy arrays
        Spectra arrays (n_samples x n_wavelengths)
    wavelengths : numpy array
        Wavelength grid
    
    Returns:
    --------
    summary : dict
        Comprehensive validation statistics
    """
    # Energy conservation
    epsilon_energy = compute_energy_error(R, T, A)
    
    # Negative absorption
    A_min, absorption_stats = compute_negative_absorption(A)
    
    # Bandwidth
    bandwidths = compute_effective_bandwidth(T, wavelengths)
    
    # Peak transmission
    peak_transmission = np.max(T, axis=1)
    
    # Compile summary
    summary = {
        'energy_conservation': {
            'mean_error': float(np.mean(epsilon_energy)),
            'max_error': float(np.max(epsilon_energy)),
            'median_error': float(np.median(epsilon_energy)),
            'samples_gt_1e-3': int(np.sum(epsilon_energy > 1e-3)),
            'samples_gt_1e-4': int(np.sum(epsilon_energy > 1e-4))
        },
        'negative_absorption': absorption_stats,
        'bandwidth': {
            'mean': float(np.mean(bandwidths)),
            'std': float(np.std(bandwidths)),
            'median': float(np.median(bandwidths))
        },
        'peak_transmission': {
            'mean': float(np.mean(peak_transmission)),
            'std': float(np.std(peak_transmission)),
            'median': float(np.median(peak_transmission))
        },
        'physical_bounds': {
            'R_range': [float(np.min(R)), float(np.max(R))],
            'T_range': [float(np.min(T)), float(np.max(T))],
            'A_range': [float(np.min(A)), float(np.max(A))]
        }
    }
    
    return summary


def validate_physical_constraints(R: np.ndarray, T: np.ndarray, A: np.ndarray) -> Dict:
    """
    Validate all physical constraints.
    
    Returns pass/fail status for each constraint.
    """
    # Energy conservation constraint
    epsilon_energy = compute_energy_error(R, T, A)
    energy_pass = np.all(epsilon_energy < 1e-3)
    
    # Non-negative absorption constraint
    A_min = np.min(A, axis=1)
    absorption_pass = np.all(A_min >= 0)
    
    # Reflectance bounds
    R_min, R_max = np.min(R), np.max(R)
    R_bounds_pass = (R_min >= -1e-6) and (R_max <= 1 + 1e-6)
    
    # Transmittance bounds
    T_min, T_max = np.min(T), np.max(T)
    T_bounds_pass = (T_min >= -1e-6) and (T_max <= 1 + 1e-6)
    
    return {
        'energy_conservation': {
            'pass': bool(energy_pass),
            'max_violation': float(np.max(epsilon_energy)),
            'samples_failing': int(np.sum(epsilon_energy >= 1e-3))
        },
        'non_negative_absorption': {
            'pass': bool(absorption_pass),
            'min_value': float(np.min(A_min)),
            'samples_failing': int(np.sum(A_min < 0))
        },
        'reflectance_bounds': {
            'pass': bool(R_bounds_pass),
            'min': float(R_min),
            'max': float(R_max)
        },
        'transmittance_bounds': {
            'pass': bool(T_bounds_pass),
            'min': float(T_min),
            'max': float(T_max)
        },
        'all_constraints_pass': bool(energy_pass and absorption_pass and 
                                   R_bounds_pass and T_bounds_pass)
    }
