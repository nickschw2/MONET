"""
Script to run a single NIF simulation
"""

from src.simulation import NIFSimulation
from src.analysis import DataProcessor, ResultsPlotter

if __name__ == "__main__":
    print("Setting up NIF simulation...")
    # Create and run simulation
    sim = NIFSimulation(output_dir='data/single_runs')
    source_kwargs = {'pulse_fwhm': 100e-12}
    geometry_kwargs = {
        'primary_geometry_type': 'standard',
        'secondary_geometry_type': 'coronal',
        'moderator_thickness': [0.3,0.3, 0.3],
        'moderator_material': ['tungsten', 'beryllium', 'ch2'],
        # 'layered_moderator_material': ['lead','tungsten', 'beryllium'],
        # 'layered_moderator_thickness': [0.1, 0.1, 0.1],
        # 'layered_moderator_primary_gap': 0.1,
        # 'source_gap': 1.0,
        # 'hohlraum_wall_thickness': 0.07
    }
    model = sim.setup_simulation(
        geometry_type='dual_hohlraum_coronal',
        convergence_ratio={'primary': 20.0, 'secondary': 1.0},
        source_kwargs=source_kwargs,
        geometry_kwargs=geometry_kwargs
    )
    
    print("Running simulation...")
    run_dir = sim.run_simulation(model=model, run_name='dual_hohlraum_coronal', reset=True)
    print(f"Simulation completed. Results in: {run_dir}")
    
    # Process results
    print("Processing results...")
    processor = DataProcessor(run_dir)
    
    # Calculate moderation efficiency
    moderation_data = processor.calculate_moderation_efficiency()
    
    # Create plots
    print("Creating plots...")
    plotter = ResultsPlotter(processor)
    
    # Plot all standard plots
    plotter.plot_all()
    print(f"Plots saved in: {plotter.save_dir}")