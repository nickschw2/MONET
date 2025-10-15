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
        'primary_geometry_type': 'coronal',
        'secondary_geometry_type': 'coronal',
    }
    model = sim.setup_simulation(
        geometry_type='dual_filled_hohlraum',
        convergence_ratio=1.0,
        source_kwargs=source_kwargs,
        geometry_kwargs=geometry_kwargs
    )

    print("Running simulation...")
    run_dir = sim.run_simulation(model=model, run_name='dual_filled_hohlraum', reset=True)
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