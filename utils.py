"""
Utility Functions for CNN-based PAA Compensation

This module provides helper functions for:
- Array factor computation
- Phase manipulation
- Pattern statistics
- Data visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def compute_array_factor_4x4(phases: np.ndarray, 
                             frequency: float = 28e9,
                             element_spacing: float = None,
                             theta_range: tuple = (0, np.pi),
                             phi_range: tuple = (0, 2*np.pi),
                             n_points: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute array factor for 4x4 planar array.
    
    This is a simplified array factor model for quick pattern estimation.
    For accurate results, use CST Microwave Studio.
    
    Args:
        phases: 4x4 phase matrix (radians)
        frequency: Operating frequency in Hz (default 28 GHz)
        element_spacing: Element spacing in meters (default lambda/2)
        theta_range: (min, max) elevation angles in radians
        phi_range: (min, max) azimuth angles in radians
        n_points: Number of points in each dimension
        
    Returns:
        Tuple of (theta_grid, phi_grid, array_factor)
    """
    # Wavelength
    c = 3e8  # Speed of light
    wavelength = c / frequency
    
    # Element spacing (default lambda/2)
    if element_spacing is None:
        element_spacing = wavelength / 2
    
    # Wave number
    k = 2 * np.pi / wavelength
    
    # Create angle grids
    theta = np.linspace(theta_range[0], theta_range[1], n_points)
    phi = np.linspace(phi_range[0], phi_range[1], n_points)
    THETA, PHI = np.meshgrid(theta, phi)
    
    # Direction cosines
    u = np.sin(THETA) * np.cos(PHI)
    v = np.sin(THETA) * np.sin(PHI)
    
    # Element positions (4x4 grid centered at origin)
    positions = []
    for i in range(4):
        for j in range(4):
            x = (j - 1.5) * element_spacing
            y = (i - 1.5) * element_spacing
            positions.append((x, y))
    
    # Compute array factor
    AF = np.zeros_like(THETA, dtype=complex)
    
    for idx, (x, y) in enumerate(positions):
        i, j = idx // 4, idx % 4
        phase = phases[i, j]
        
        # Phase contribution from element position and excitation
        phase_contrib = k * (x * u + y * v) + phase
        AF += np.exp(1j * phase_contrib)
    
    # Normalize
    AF = np.abs(AF) / np.max(np.abs(AF))
    
    return THETA, PHI, AF


def db_to_linear(gain_db: np.ndarray) -> np.ndarray:
    """
    Convert dB gain to linear scale.
    
    Args:
        gain_db: Gain in dB
        
    Returns:
        Gain in linear scale
    """
    return 10 ** (gain_db / 10)


def linear_to_db(gain_linear: np.ndarray, min_db: float = -100) -> np.ndarray:
    """
    Convert linear gain to dB scale.
    
    Args:
        gain_linear: Gain in linear scale
        min_db: Minimum dB value (for clipping)
        
    Returns:
        Gain in dB
    """
    gain_db = 10 * np.log10(gain_linear + 1e-10)
    return np.maximum(gain_db, min_db)


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """
    Wrap phase to [-pi, pi] range.
    
    Args:
        phase: Phase values (any range)
        
    Returns:
        Wrapped phase values in [-pi, pi]
    """
    return (phase + np.pi) % (2 * np.pi) - np.pi


def phase_difference(phase1: np.ndarray, phase2: np.ndarray) -> np.ndarray:
    """
    Compute wrapped phase difference.
    
    Args:
        phase1: First phase array
        phase2: Second phase array
        
    Returns:
        Wrapped phase difference in [-pi, pi]
    """
    diff = phase1 - phase2
    return wrap_phase(diff)


def compute_pattern_statistics(pattern: np.ndarray, 
                                angle_grid: Optional[np.ndarray] = None) -> dict:
    """
    Compute statistics of a radiation pattern.
    
    Args:
        pattern: 2D radiation pattern array
        angle_grid: Optional angle grid for the pattern
        
    Returns:
        Dictionary of pattern statistics
    """
    stats = {
        'max_gain': np.max(pattern),
        'min_gain': np.min(pattern),
        'mean_gain': np.mean(pattern),
        'std_gain': np.std(pattern),
        'dynamic_range': np.max(pattern) - np.min(pattern),
    }
    
    # Find peak location
    max_idx = np.unravel_index(np.argmax(pattern), pattern.shape)
    stats['peak_location'] = max_idx
    
    # Compute 3dB beamwidth (approximate)
    max_val = stats['max_gain']
    threshold = max_val - 3  # 3dB below peak
    above_threshold = pattern >= threshold
    stats['beam_area_3db'] = np.sum(above_threshold)
    
    return stats


def plot_phase_distribution(phases: np.ndarray, title: str = "Phase Distribution"):
    """
    Plot histogram of phase values.
    
    Args:
        phases: Phase values (radians)
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.hist(phases.flatten(), bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Phase (radians)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14)
    plt.axvline(np.mean(phases), color='red', linestyle='--', 
                label=f'Mean: {np.mean(phases):.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def compare_patterns(pattern1: np.ndarray, 
                     pattern2: np.ndarray,
                     title1: str = "Pattern 1",
                     title2: str = "Pattern 2",
                     diff_title: str = "Difference"):
    """
    Compare two radiation patterns side by side.
    
    Args:
        pattern1: First pattern
        pattern2: Second pattern
        title1: Title for first pattern
        title2: Title for second pattern
        diff_title: Title for difference plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmax = max(pattern1.max(), pattern2.max())
    vmin = min(pattern1.min(), pattern2.min())
    
    # Pattern 1
    im1 = axes[0].imshow(pattern1, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(title1, fontsize=14)
    plt.colorbar(im1, ax=axes[0])
    
    # Pattern 2
    im2 = axes[1].imshow(pattern2, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title(title2, fontsize=14)
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    diff = pattern1 - pattern2
    im3 = axes[2].imshow(diff, cmap='RdBu_r', origin='lower')
    axes[2].set_title(diff_title, fontsize=14)
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()


def create_comparison_table(metrics_dict: dict) -> str:
    """
    Create a formatted comparison table.
    
    Args:
        metrics_dict: Dictionary of metrics for different methods
        
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("="*70)
    lines.append(f"{'Method':<30} {'MAE (rad)':<15} {'RMSE (rad)':<15} {'R2':<10}")
    lines.append("="*70)
    
    for method, metrics in metrics_dict.items():
        mae = metrics.get('mae', 0)
        rmse = metrics.get('rmse', 0)
        r2 = metrics.get('r2', 0)
        lines.append(f"{method:<30} {mae:<15.6f} {rmse:<15.6f} {r2:<10.4f}")
    
    lines.append("="*70)
    
    return "\n".join(lines)


def save_phases_to_file(phases: np.ndarray, filepath: str, 
                        format: str = 'degrees'):
    """
    Save phase values to a text file.
    
    Args:
        phases: Phase values (radians or degrees)
        filepath: Output file path
        format: 'degrees' or 'radians'
    """
    if format == 'degrees':
        phases_out = np.rad2deg(phases)
        unit = 'degrees'
    else:
        phases_out = phases
        unit = 'radians'
    
    with open(filepath, 'w') as f:
        f.write(f"Phase Values ({unit})\n")
        f.write("="*40 + "\n\n")
        
        if phases.ndim == 1:
            for i, phase in enumerate(phases_out):
                f.write(f"Element {i+1:2d}: {phase:.4f}\n")
        else:
            # 4x4 matrix
            for i in range(phases.shape[0]):
                for j in range(phases.shape[1]):
                    f.write(f"({i},{j}): {phases_out[i,j]:.4f}  ")
                f.write("\n")


def load_phases_from_file(filepath: str, format: str = 'degrees') -> np.ndarray:
    """
    Load phase values from a text file.
    
    Args:
        filepath: Input file path
        format: 'degrees' or 'radians'
        
    Returns:
        Phase values in radians
    """
    phases = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('Phase'):
                continue
            
            # Try to extract numeric value
            parts = line.split(':')
            if len(parts) >= 2:
                try:
                    value = float(parts[-1].strip())
                    phases.append(value)
                except ValueError:
                    continue
    
    phases = np.array(phases)
    
    if format == 'degrees':
        phases = np.deg2rad(phases)
    
    return phases


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test phase wrapping
    phases = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, 5*np.pi/2])
    wrapped = wrap_phase(phases)
    print(f"\nPhase wrapping:")
    print(f"  Original: {np.rad2deg(phases)}")
    print(f"  Wrapped:  {np.rad2deg(wrapped)}")
    
    # Test dB conversion
    gain_db = np.array([-10, -5, 0, 5, 10])
    gain_lin = db_to_linear(gain_db)
    gain_db_back = linear_to_db(gain_lin)
    print(f"\ndB conversion:")
    print(f"  Original (dB): {gain_db}")
    print(f"  Linear:        {gain_lin}")
    print(f"  Back to dB:    {gain_db_back}")
    
    # Test array factor computation
    print("\nComputing sample array factor...")
    phases_4x4 = np.random.rand(4, 4) * 2 * np.pi
    theta, phi, af = compute_array_factor_4x4(phases_4x4, n_points=32)
    print(f"  Array factor shape: {af.shape}")
    print(f"  Max gain: {20*np.log10(af.max()):.2f} dBi")
    
    print("\nUtility tests completed!")
