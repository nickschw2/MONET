"""
Script to run a single NIF simulation
"""

from src.simulation import NIFSimulation
from src.analysis import DataProcessor
import matplotlib.pyplot as plt
from labellines import labelLines
import numpy as np

if __name__ == "__main__":
    print("Setting up NIF simulation...")
    # Create and run simulation
    sim = NIFSimulation(output_dir='data/niv_scan')
    source_kwargs = {'pulse_fwhm': 100e-12}
    
    # Scan a number of NIV configurations
    geometry_parameters = [
        {'niv_fill_thickness': [7.3], 'niv_fill_material': [None]},
        {'niv_fill_thickness': [7.3], 'niv_fill_material': ['ch2']},
        {'niv_fill_thickness': [7.3], 'niv_fill_material': ['steel']},
        {'niv_fill_thickness': [7.3], 'niv_fill_material': ['graphite']},
        {'niv_fill_thickness': [7.3], 'niv_fill_material': ['lithium_hydride']},
        {'niv_fill_thickness': [7.3], 'niv_fill_material': ['zirconium_hydride']},
    ]
    
    # Create figure for partial current
    fig, ax = plt.subplots(constrained_layout=True, figsize=(10,6))
    # Create figure for cumulative partial current
    fig2, ax2 = plt.subplots(constrained_layout=True, figsize=(10,6))
    
    for i, geometry_kwargs in enumerate(geometry_parameters):
        print(f"Running simulation {i+1}/{len(geometry_parameters)}...")
        model = sim.setup_simulation(
            geometry_type='niv',
            convergence_ratio=5.0,
            source_kwargs=source_kwargs,
            geometry_kwargs=geometry_kwargs,
            fuel_fraction=0.0, # DD fuel
            particles_per_batch=int(1e6)
        )
        run_dir = sim.run_simulation(model=model, run_name=f'niv_{i}', reset=False)
        print(f"Simulation completed. Results in: {run_dir}")
        
        # Add processor to list
        processor = DataProcessor(run_dir)
        
        # Plot surface tally results
        tally_df = processor.extract_surface_data()
        energy = tally_df['energy low [eV]'] / 1e6  # MeV
        partial_current = tally_df['mean']  # neutrons/source
        std = tally_df['std. dev.']  # neutrons/source
        
        # Plot partial current
        if geometry_kwargs['niv_fill_material'][0] is None:
            label = "Vacuum"
        else:
            label = f"{geometry_kwargs['niv_fill_material'][0]}"
        ax.step(energy, partial_current, label=label)
        
        # Plot std dev as shaded region
        positive = partial_current - std > 0
        ax.fill_between(
            energy,
            partial_current - std,
            partial_current + std,
            alpha=0.3,
            where=positive,
            step='pre'
        )
        
        # Plot cumulative tally on secondary axis
        cumulative = partial_current.cumsum()
        # Calculate uncertainty propagation for cumulative sum
        cumsum_var = np.nancumsum(std**2)  # Sum of variances
        cumsum_std = np.sqrt(cumsum_var)   # Standard deviation
        
        ax2.step(energy, cumulative, label=label)
        
        # Plot std dev as shaded region
        positive = cumulative - cumsum_std > 0
        ax2.fill_between(
            energy,
            cumulative - cumsum_std,
            cumulative + cumsum_std,
            alpha=0.3,
            where=positive,
            step='pre'
        )

    ax.set_xlabel('Energy [MeV]')
    ax.set_ylabel('Current [neutrons/source]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((0.01, 5.0))
    ax.grid(True, which='major', lw=0.5)
    
    ax2.set_xlabel('Energy [MeV]')
    ax2.set_ylabel('Cumulative Current [neutrons/source]')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim((0.01, 5.0))
    ax2.set_ylim((1e-9, 2e-5))
    ax2.grid(True, which='major', lw=0.5)
    
    # Add labels to lines
    labelLines(ax.get_lines(), align=False, fontsize=20)
    labelLines(ax2.get_lines(), align=False, fontsize=20)
    
    # Save figures
    fig.savefig(f'{sim.output_dir}/niv_scan_partial_current.png', dpi=300)
    fig2.savefig(f'{sim.output_dir}/niv_scan_cumulative_partial_current.png', dpi=300)