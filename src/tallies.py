import openmc
import numpy as np
from typing import List, Optional, Union, Sequence, Tuple

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
        cells: Optional[List[openmc.Cell]] = None,
        partial_current_surfaces: Optional[List[Tuple[openmc.Surface, openmc.Cell]]] = None,
        low_energy_threshold:  List[float] = [1e-3, 1e-2, 1e-1, 1.0],
        energy_min: float = 1e-3,
        energy_max: float = 16.0,
        n_energy_bins: int = 200,
        generate_mesh: bool = True,
        mesh_pixels: int = 200,
        track_nuclides: Optional[Union[str, Sequence[str]]] = 'Tm171',
        reactions: Optional[Union[str, Sequence[str]]] = '(n,gamma)',
    ):
        """
        Create all tallies for the simulation
        
        Parameters:
        -----------
        cells : openmc.Cell, optional
            Specific cell to tally in; if None, uses fuel cells
        partial_current_surfaces : List[Tuple[openmc.Surface, openmc.Cell]], optional
            List of tuples of surface and cell to tally partial current on
        low_energy_threshold : List[float]
            Energy threshold for moderation analysis in MeV, provided as a list for multiple thresholds.
        energy_min : float
            Minimum energy for energy bins in MeV
        energy_max : float
            Maximum energy for energy bins in MeV
        n_energy_bins : float
            Number of energy bins
        generate_mesh : bool
            Whether to generate a mesh tally
        mesh_pixels : int
            Number of pixels in spatial mesh
        track_nuclides : str
            Nuclide to track for reactions
        reactions : str
            Reaction to tally for track nuclide
        """
        print("Creating all tallies...")
        
        ### SPECTRAL TALLIES ###
        energy_bins = np.logspace(np.log10(energy_min), np.log10(energy_max), n_energy_bins) * 1e6  # eV
        
        # Find maximum extent to set maximum time for time bins
        max_extent = np.max(self.geometry.bounding_box.width) * 0.01 # m
        min_neutron_energy = energy_min * 1e6  # eV
        neutron_mass = 1.04540751e-8  # eV/(m/s)^2
        min_neutron_velocity = np.sqrt(2 * min_neutron_energy / neutron_mass)  # m/s
        max_time = max_extent / min_neutron_velocity  # s
        time_bins = np.linspace(0, max_time, 200)  # seconds
        
        energy_filter = openmc.EnergyFilter(energy_bins)
        time_filter = openmc.TimeFilter(time_bins)
        
        if track_nuclides and reactions:
            # Create a tally for each nuclide/reaction pair
            if isinstance(track_nuclides, str) and isinstance(reactions, str):
                track_nuclides = [track_nuclides]
                reactions = [reactions]
            elif isinstance(track_nuclides, Sequence) and isinstance(reactions, Sequence) and not isinstance(track_nuclides, str):
                if len(track_nuclides) != len(reactions):
                    raise ValueError("Tally nuclide and reactions must be the same length")
            else:
                raise ValueError("Tally nuclide and reactions must be the same type")
            for nuclide, react in zip(track_nuclides, reactions):
                # Nuclide spectra
                nuclide_tally = openmc.Tally(name=f'nuclide_tally_{nuclide}_{react}')
                nuclide_tally.filters = [energy_filter, time_filter]
                nuclide_tally.scores = [react] if isinstance(react, str) else react
                nuclide_tally.nuclides = [nuclide] if isinstance(nuclide, str) else nuclide
                self.append(nuclide_tally)
        
        # Neutron energy spectrum in cell of interest
        # Default to fuel cells if none provided
        fuel_cells = self.geometry.get_cells_by_name('fuel')
        if cells is None:
            cells = fuel_cells
        else:
            cells += fuel_cells
        
        # Create cell filter    
        cell_filter = openmc.CellFilter(cells)
        
        # Total neutron flux in cells by energy and time
        cell_tally = openmc.Tally(name='cell_tally')
        cell_tally.filters = [energy_filter, time_filter, cell_filter]
        cell_tally.scores = ['flux']
        self.append(cell_tally)
        
        # Total neutron flux in cells by energy
        self.low_energy_threshold = low_energy_threshold # Save for data storage
        
        ### SURFACE TALLIES ###
        if partial_current_surfaces is not None:
            for surface, cell in partial_current_surfaces:
                surface_filter = openmc.SurfaceFilter(surface)
                cell_from_filter = openmc.CellFromFilter(cell)
                surface_tally = openmc.Tally(name=f'surface_tally_{surface.id}')
                surface_tally.filters = [energy_filter, surface_filter, cell_from_filter]
                surface_tally.scores = ['current']
                self.append(surface_tally)
        
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
            mesh_tally.scores = ['flux', 'scatter', '(n,2n)', '(n,3n)', '(n,gamma)', '(n,t)']
            self.append(mesh_tally)