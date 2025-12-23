"""
Ablation variants for the grating coupler generator.

Contains four modified versions of the reference generator:
A: No Energy Conservation Enforcement
B: No Fabry-Perot Oscillations  
C: Fixed Bandwidth (No Geometry Dependence)
D: No Measurement Noise

Each variant is structurally identical, differing by exactly one physics component.
"""

import numpy as np
from typing import Dict, List, Tuple
from generator_algo import ReferenceGenerator


class AblationA_NoEnergyConservation(ReferenceGenerator):
    """
    Ablation A: No Energy Conservation Enforcement
    
    Removes the energy conservation enforcement step to test:
    - Importance of physical feasibility
    - Impact on ML model training stability
    - ECE-ML metric sensitivity
    """
    
    def _enforce_energy_conservation(self, R: np.ndarray, T: np.ndarray, 
                                   A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """NO ENERGY CONSERVATION - Return spectra without normalization."""
        # Simply return the spectra as-is, allowing R+T+A ≠ 1
        return R.astype(np.float32), T.astype(np.float32), A.astype(np.float32)


class AblationB_NoFabryPerot(ReferenceGenerator):
    """
    Ablation B: No Fabry-Perot Oscillations
    
    Removes Fabry-Perot oscillations to test:
    - Importance of realistic spectral fine structure
    - Impact on spectral correlation metrics
    - RLE sensitivity to spectral features
    """
    
    def _compute_coupling_efficiency(self, params: Dict[str, float], n_eff: float) -> np.ndarray:
        """Compute coupling efficiency WITHOUT Fabry-Perot oscillations."""
        period = params['period_nm']
        ff = params['fill_factor']
        etch_depth = params['etch_depth_nm']
        
        # Phase-matching condition (simplified grating equation)
        lambda_center = period * n_eff
        
        # Lorentzian resonance
        delta_lambda = self.wavelengths_nm - lambda_center
        
        # Bandwidth depends on parameters
        bandwidth = (self.model_params['bandwidth_base'] + 
                    self.model_params['bandwidth_ff_factor'] * (1 - ff) +
                    self.model_params['bandwidth_etch_factor'] * (etch_depth / 100.0))
        
        coupling = bandwidth**2 / (bandwidth**2 + delta_lambda**2)
        
        # NO FABRY-PEROT OSCILLATIONS ADDED
        
        # Ensure physical bounds
        return np.clip(coupling, 0.0, 0.95)


class AblationC_FixedBandwidth(ReferenceGenerator):
    """
    Ablation C: Fixed Bandwidth (No Geometry Dependence)
    
    Uses constant bandwidth regardless of geometry to test:
    - Importance of geometry-spectrum coupling
    - Impact on scalar metrics (bandwidth prediction)
    - Physics-informed model advantage
    """
    
    def _compute_coupling_efficiency(self, params: Dict[str, float], n_eff: float) -> np.ndarray:
        """Compute coupling efficiency with FIXED bandwidth (no geometry dependence)."""
        period = params['period_nm']
        
        # Phase-matching condition (simplified grating equation)
        lambda_center = period * n_eff
        
        # Lorentzian resonance
        delta_lambda = self.wavelengths_nm - lambda_center
        
        # FIXED BANDWIDTH (no geometry dependence)
        bandwidth = self.model_params['bandwidth_base']
        
        coupling = bandwidth**2 / (bandwidth**2 + delta_lambda**2)
        
        # Add Fabry-Perot oscillations (waveguide cavity resonances)
        fp_period = 2 * n_eff * params['si_thickness_nm']  # Round-trip phase
        
        for i, amplitude in enumerate(self.model_params['fp_amplitudes'], 1):
            fp_term = amplitude * np.sin(2 * np.pi * self.wavelengths_nm / (fp_period / i))**2
            coupling += fp_term
        
        # Ensure physical bounds
        return np.clip(coupling, 0.0, 0.95)


class AblationD_NoNoise(ReferenceGenerator):
    """
    Ablation D: No Measurement Noise
    
    Generates clean spectra without noise to test:
    - ML model robustness to experimental realism
    - Overfitting potential
    - Generalization gap
    """
    
    def generate_sample(self, params: Dict[str, float], 
                       sample_id: int = 0,
                       add_noise: bool = True) -> Dict:
        """Generate sample WITHOUT measurement noise."""
        # Override to always set add_noise=False
        return super().generate_sample(params, sample_id, add_noise=False)
