import openmc
import numpy as np
from typing import List, Optional
from .geometry import DualSourceUniverse, DualFilledHohlraum

class NIFTallies(openmc.Tallies):
    """Class to handle all tally definitions for NIF simulations"""
    
    def __init__(
        self,
        geometry: openmc.Geometry,
        tallies: Optional[List[openmc.Tally]] = None,):
        """Initialize tallies collection
        
        Parameters:
        -----------
        geometry : openmc.Geometry
            Geometry object containing cells and materials
        tallies : List[openmc.Tally], optional
            Initial list of tallies
        """
        super().__init__(tallies or [])
        self.geometry = geometry
    
    def create_tallies(
        self,
        low_energy_threshold:  List[float] = [1e-3, 1e-2, 1e-1, 1.0],
        energy_min: float = 1e-3,
        energy_max: float = 16.0,
        n_energy_bins: int = 200,
        pulse_fwhm: Optional[float] = None,
        generate_mesh: bool = True,
        mesh_pixels: int = 200,
        trace_nuclide: Optional[str] = 'Tm171',
    ):
        """
        Create all tallies for the simulation
        low_energy_threshold : List[float]
            Energy threshold for moderation analysis in MeV, provided as a list for multiple thresholds.
        energy_min : float
            Minimum energy for energy bins in MeV
        energy_max : float
            Maximum energy for energy bins in MeV
        n_energy_bins : float
            Number of energy bins
        pulse_fwhm : float
            FWHM of gaussian pulse in seconds
        generate_mesh : bool
            Whether to generate a mesh tally
        mesh_pixels : int
            Number of pixels in spatial mesh
        trace_nuclide : str
            Nuclide to trace for (n,gamma) reactions
        """
        print("Creating all tallies...")
        
        ### SPECTRAL TALLIES ###
        energy_bins = np.logspace(np.log10(energy_min), np.log10(energy_max), n_energy_bins) * 1e6  # eV
        
        # Define time bins based on fwhm of the pulse
        if pulse_fwhm:
            sigma = pulse_fwhm / (2 * np.sqrt(2 * np.log(2)))
            # TODO: refine the time definition
            # TODO: the max time should correspond to whichever of the two dual sources has the latest and longest pulse
            time_max = 50 * sigma
        else:
            time_max = 1e-11
        
        # TODO: change arbitrary definition of time bins
        time_bins = np.linspace(0, time_max, 200)  # seconds
        
        energy_filter = openmc.EnergyFilter(energy_bins)
        time_filter = openmc.TimeFilter(time_bins)
        
        if trace_nuclide:            
            # Trace spectra
            trace_tally = openmc.Tally(name='trace_tally')
            trace_tally.filters = [energy_filter, time_filter]
            trace_tally.scores = ['(n,gamma)']
            trace_tally.nuclides = [trace_nuclide]
            self.append(trace_tally)
        
        # Neutron energy spectrum in fuel
        if isinstance(self.geometry.root_universe, (DualSourceUniverse, DualFilledHohlraum)):
            # If it's a dual source universe, get secondary fuel
            fuel_cell = self.geometry.get_cells_by_name('fuel_secondary')
        else:
            # Else get the primary fuel
            fuel_cell = self.geometry.get_cells_by_name('fuel')
            
        if fuel_cell:
            fuel_filter = openmc.CellFilter(fuel_cell)
            
            # Total neutron flux in fuel by energy
            fuel_tally = openmc.Tally(name='fuel_tally')
            fuel_tally.filters = [energy_filter, time_filter, fuel_filter]
            fuel_tally.scores = ['flux']
            self.append(fuel_tally)
        
        self.low_energy_threshold = low_energy_threshold # Save for data storage
        
        ### MESH TALLY ###
        if generate_mesh:
            # Create mesh
            mesh = openmc.RegularMesh()
            thickness = 0.1
            mesh.lower_left = np.array([
                self.geometry.bounding_box.lower_left[0],
                -thickness/2,
                self.geometry.bounding_box.lower_left[2]
            ])
            mesh.upper_right = np.array([
                self.geometry.bounding_box.upper_right[0],
                thickness/2,
                self.geometry.bounding_box.upper_right[2],
            ])
            mesh.dimension = [mesh_pixels, 1, mesh_pixels]
            mesh_filter = openmc.MeshFilter(mesh)
            
            # Binary energy filter for low energy threshold
            bins = np.array([0.] + low_energy_threshold + [energy_max]) * 1e6 # eV
            low_energy_threshold_filter = openmc.EnergyFilter(bins)
                
            
            # mesh tally
            mesh_tally = openmc.Tally(name='mesh_tally')
            mesh_tally.filters = [low_energy_threshold_filter, mesh_filter]
            mesh_tally.scores = ['flux', 'scatter', '(n,2n)', '(n,3n)', '(n,gamma)']
            self.append(mesh_tally)