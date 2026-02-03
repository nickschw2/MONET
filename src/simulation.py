import openmc
from openmc import stats
import os
import json
from typing import Optional, Dict, Any, Union, Literal
from pathlib import Path

from .materials import NIFMaterials
from .sources import SphericalSource
from .geometry import *
from .tallies import NIFTallies

class NIFModel(openmc.Model):
    """Main simulation model that coordinates all NIF components"""
    
    def __init__(self,
                 geometry: Optional[openmc.Geometry] = None,
                 materials: Optional[openmc.Materials] = None,
                 settings: Optional[openmc.Settings] = None,
                 tallies: Optional[NIFTallies] = None):
        """Initialize NIF model
        
        Parameters:
        -----------
        geometry : openmc.Geometry, optional
            Geometry object
        materials : NIFMaterials, optional
            Materials collection
        settings : openmc.Settings, optional
            Simulation settings
        tallies : NIFTallies, optional
            Tallies collection
        """
        # Initialize materials first
        if materials is None:
            materials = NIFMaterials()
        
        super().__init__(
            geometry=geometry,
            materials=materials,
            settings=settings, 
            tallies=tallies
        )
        
        self.setup_params: Dict[str, Any] = {}

class NIFSimulation:
    """Factory class for creating and running NIF simulations"""
    
    def __init__(self, output_dir: Union[str, Path] = 'data/raw'):
        """Initialize simulation manager
        
        Parameters:
        -----------
        output_dir : str or Path
            Base directory for simulation outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_simulation(
        self,
        geometry_type: Literal['indirect', 'double_shell', 'coronal', 'nrv', 'dual_source', 'dual_filled_hohlraum', 'dual_indirect_coronal'] = 'indirect',
        # Simulation parameters
        batches: int = 100,
        particles_per_batch: int = int(1e5),
        convergence_ratio: Union[float, Dict[Literal['primary', 'secondary'], float]] = 1.0,
        trace_nuclide: str = 'Tm171',
        fuel_fraction: float = 0.5,
        geometry_kwargs: Optional[Dict[str, Any]] = None,
        fuel_kwargs: Optional[Dict[str, Any]] = None,
        source_kwargs: Optional[Dict[str, Any]] = None,
        tally_kwargs: Optional[Dict[str, Any]] = None,
        settings_kwargs: Optional[Dict[str, Any]] = None,
    ) -> NIFModel:
        """Set up standard NIF simulation
        
        Parameters:
        -----------
        geometry_type : str
            Geometry type, one of ['indirect', 'double_shell', 'coronal', 'nrv', 'dual_source', 'dual_filled_hohlraum', 'dual_indirect_coronal']
        batches : int
            Number of batches
        particles_per_batch : int
            Particles per batch
        convergence_ratio : float
            Convergence ratio
        trace_nuclide : str
            Trace nuclide for (n,gamma) reactions
        fuel_fraction : float
            Fraction of tritium in DT fuel
        geometry_kwargs : dict, optional
            Geometry parameters
        fuel_kwargs : dict, optional
            Fuel material parameters
        source_kwargs : dict, optional
            Source parameters
        tally_kwargs : dict, optional
            Tally parameters
        settings_kwargs : dict, optional
            Additional settings parameters
            
        Returns:
        --------
        NIFModel
            Configured simulation model
        """
        # Create materials
        materials = NIFMaterials()

        # Create settings
        settings = openmc.Settings(**settings_kwargs or {})
        settings.run_mode = "fixed source"
        settings.batches = batches
        settings.particles = particles_per_batch
        settings.output = {'tallies': True}
                
        # Create geometry
        if geometry_type in ('dual_source', 'dual_filled_hohlraum', 'dual_indirect_coronal'):
            if not isinstance(convergence_ratio, dict):
                raise ValueError("For dual source geometries, convergence_ratio must be a dict with 'primary' and 'secondary' keys")
            
            # Primary fuel without trace nuclide
            primary_fuel = materials.create_dt_fuel(
                name='dt_fuel_primary',
                trace_nuclide=trace_nuclide,
                trace_concentration=0.0,  # No trace in primary
                fuel_fraction=fuel_fraction,
                **fuel_kwargs or {}
            )
            # Secondary fuel with trace nuclide
            secondary_fuel = materials.create_dt_fuel(
                name='dt_fuel_secondary',
                trace_nuclide=trace_nuclide,
                fuel_fraction=fuel_fraction,
                **fuel_kwargs or {}
            )
            fuel = secondary_fuel  # Use secondary fuel for setup_params tracking
            
            # Pass geometry kwargs to primary and secondary geometries
            primary_geometry_kwargs = {'fuel_material': primary_fuel}
            secondary_geometry_kwargs = {'fuel_material': secondary_fuel}
            
            # Set up dual source geometry
            universe = self._setup_dual_source_geometry(
                geometry_type,
                materials,
                convergence_ratio,
                primary_geometry_kwargs=primary_geometry_kwargs,
                secondary_geometry_kwargs=secondary_geometry_kwargs,
                **geometry_kwargs or {}
            )
            primary_fuel_radius, primary_fuel_cell = universe.primary_geom.get_fuel_params()
            secondary_fuel_radius, secondary_fuel_cell = universe.secondary_geom.get_fuel_params()
            primary_fuel_radius_compressed = primary_fuel_radius / universe.primary_geom.convergence_ratio
            secondary_fuel_radius_compressed = secondary_fuel_radius / universe.secondary_geom.convergence_ratio
            
            # TODO: assuming the two sources have the same strength, change this to be configurable
            # TODO: assume the two sources have the same pulse width and start time, change this to be configurable
            # Create source
            primary_origin = tuple(universe.axis_vector * universe.primary_translation)
            primary_source = SphericalSource(
                fuel_radius=primary_fuel_radius_compressed,
                constraints={'domains': [primary_fuel_cell]},
                fuel_fraction=fuel_fraction,
                origin=primary_origin,
                **source_kwargs or {}
            )
            
            pulse_fwhm = primary_source.pulse_fwhm
            plasma_temperature = primary_source.plasma_temperature
            
            # Place secondary source in correct location
            secondary_origin = tuple(universe.axis_vector * universe.secondary_translation)
            secondary_source = SphericalSource(
                fuel_radius=secondary_fuel_radius_compressed,
                constraints={'domains': [secondary_fuel_cell]},
                fuel_fraction=fuel_fraction,
                origin=secondary_origin,
                **source_kwargs or {}
            )
            
            # Set source in settings
            settings.source = [primary_source]
            # settings.source = [primary_source, secondary_source]
            
        else:
            if not isinstance(convergence_ratio, float):
                raise ValueError("For single source geometries, convergence_ratio must be a float")
            # Create fuel material
            fuel = materials.create_dt_fuel(
                trace_nuclide=trace_nuclide,
                fuel_fraction=fuel_fraction,
                **fuel_kwargs or {}
            )
            geometry_kwargs = geometry_kwargs or {}
            geometry_kwargs['fuel_material'] = fuel
            
            # Set up single source geometry
            universe = self._setup_geometry(geometry_type, materials, convergence_ratio, **geometry_kwargs)
            
            # Calculate fuel radius after compression
            fuel_radius, fuel_cell = universe.get_fuel_params()
            fuel_radius_compressed = fuel_radius / universe.convergence_ratio
            
            # If using NRV, set azimuthal angle for source
            if isinstance(universe, NuclearReactionVesselUniverse):
                source_kwargs = source_kwargs or {}
                source_kwargs['angle'] = stats.PolarAzimuthal(
                    mu=stats.Uniform(a=0, b=universe.cos_cone_angle),
                    phi=stats.Uniform(a=0, b=2*np.pi),
                    reference_vwu=(1.0, 0, 0)
                )
                # Need to set source strength to account for solid angle subtended by NRV
                source_kwargs['strength'] = universe.solid_angle / (4 * np.pi)
            
            # Create source
            source = SphericalSource(
                fuel_radius=fuel_radius_compressed,
                constraints={'domains': [fuel_cell]},
                fuel_fraction=fuel_fraction,
                **source_kwargs or {}
            )
            
            pulse_fwhm = source.pulse_fwhm
            plasma_temperature = source.plasma_temperature
            
            settings.source = source
        
        geometry = openmc.Geometry(root=universe)
        
        # Create tallies
        tallies = NIFTallies(geometry=geometry)
        # Determine cell for tallying
        if isinstance(universe, NuclearReactionVesselUniverse):
            cells = universe.tally_cells
            partial_current_surfaces = [universe.tally_partial_current_surfaces]
            track_nuclides = universe.tally_nuclides
            reactions = universe.tally_reactions
            
        else:
            # Default to fuel cell and n,gamma reaction
            cells = None
            partial_current_surfaces = None
            track_nuclides = [trace_nuclide]
            reactions = ['(n,gamma)']
        
        tallies.create_tallies(
            cells=cells,
            partial_current_surfaces=partial_current_surfaces,
            track_nuclides=track_nuclides,
            reactions=reactions,
            **tally_kwargs or {}
        )
        
        # Create model
        model = NIFModel(
            geometry=geometry,
            settings=settings,
            tallies=tallies,
            materials=get_materials(universe)
        )
        
        # Store parameters and universe for later use
        model.setup_params = {
            'geometry_type': geometry_type,
            'convergence_ratio': convergence_ratio,
            'fuel_fraction': fuel_fraction,
            'fuel_pressure': fuel.fuel_pressure,
            'track_nuclides': track_nuclides,
            'reactions': reactions,
            'trace_concentration': fuel.trace_concentration,
            'dopant_nuclide': fuel.dopant_nuclide,
            'dopant_concentration': fuel.dopant_concentration,
            'plasma_temperature': plasma_temperature,
            'pulse_fwhm': pulse_fwhm,
            'batches': batches,
            'particles_per_batch': particles_per_batch,
            'low_energy_threshold': tallies.low_energy_threshold
        }
        
        # Get universe parameters
        universe_params = {}
        for param, value in vars(universe).items():
            # Only intrerested in storing numeric or string parameters
            # Ignore private parameters that start with '_'
            if isinstance(value, (int, float, str)) and not param.startswith('_'):
                universe_params[param] = value
        
        model.setup_params.update(universe_params)
        
        return model
    
    def _setup_geometry(
        self,
        geometry_type: Literal['indirect', 'double_shell', 'coronal', 'nrv'],
        materials: NIFMaterials,
        convergence_ratio: float,
        tag: Literal['primary', 'secondary'] = 'primary',
        **kwargs
    ) -> BaseImplosionUniverse:
        """
        Helper function for setting up the geometry.
        
        Parameters:
        geometry_type : str
            Geometry type, one of ['indirect', 'double_shell', 'coronal', 'nrv']
        materials : NIFMaterials
            Material database
        convergence_ratio : float
            Compression ratio for geometry and materials
        tag : str
            Tag to add to cell names and compressed materials
        **kwargs
            Additional parameters for BaseImplosionUniverse
        
        Returns:
        BaseImplosionUniverse
            Configured universe
        """
        
        if geometry_type == 'indirect':
            return IndirectDriveUniverse(
                materials=materials,
                convergence_ratio=convergence_ratio,
                tag=tag,
                **kwargs or {}
            )
            
        elif geometry_type == 'double_shell':            
            return DoubleShellUniverse(
                materials=materials,
                convergence_ratio=convergence_ratio,
                tag=tag,
                **kwargs or {}
            )
            
        elif geometry_type == 'coronal':
            return CoronalUniverse(
                materials=materials,
                convergence_ratio=convergence_ratio,
                tag=tag,
                **kwargs or {}
            )
            
        elif geometry_type == 'nrv':
            return NuclearReactionVesselUniverse(
                materials=materials,
                convergence_ratio=convergence_ratio,
                tag=tag,
                **kwargs or {}
            )
            
        else:
            raise ValueError(f"Invalid geometry type: {geometry_type}. Options are ['indirect', 'double_shell', 'coronal', 'nrv'].")
        
    def _setup_dual_source_geometry(
        self,
        type: Literal['dual_source', 'dual_filled_hohlraum', 'dual_indirect_coronal'],
        materials: NIFMaterials,
        convergence_ratio: Dict[Literal['primary', 'secondary'], float],
        primary_geometry_type: Literal['indirect', 'double_shell', 'coronal'],
        secondary_geometry_type: Literal['indirect', 'double_shell', 'coronal'],
        primary_geometry_kwargs: Optional[Dict[str, Any]] = None,
        secondary_geometry_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> DualSourceUniverse:
        """
        Helper function for setting up the dual source geometry.
        
        Parameters:
        type : str
            Geometry type, one of ['dual_source', 'dual_filled_hohlraum', 'dual_indirect_coronal']
        materials : NIFMaterials
            Material database
        primary_geometry_type : str
            Primary geometry type, one of ['indirect', 'double_shell', 'coronal']
        secondary_geometry_type : str
            Secondary geometry type, one of ['indirect', 'double_shell', 'coronal']
        primary_geometry_kwargs : dict, optional
            Additional parameters for primary BaseImplosionUniverse
        secondary_geometry_kwargs : dict, optional
            Additional parameters for secondary BaseImplosionUniverse
        **kwargs
            Additional parameters for DualSourceUniverse
        
        Returns:
        DualSourceUniverse
            Configured universe
        """
        primary_geom = self._setup_geometry(
            geometry_type=primary_geometry_type,
            materials=materials,
            convergence_ratio=convergence_ratio['primary'],
            tag='primary',
            **primary_geometry_kwargs or {}
        )

        secondary_geom = self._setup_geometry(
            geometry_type=secondary_geometry_type,
            materials=materials,
            convergence_ratio=convergence_ratio['secondary'],
            tag='secondary',
            **secondary_geometry_kwargs or {}
        )
        
        if type == 'dual_source':
            universe = DualSourceUniverse(
                materials=materials,
                primary_geom=primary_geom,
                secondary_geom=secondary_geom,
                **kwargs or {}
            )
        elif type == 'dual_filled_hohlraum':
            if not isinstance(primary_geom, CoronalUniverse) or not isinstance(secondary_geom, CoronalUniverse):
                raise ValueError("Both primary and secondary geometries must be 'coronal' for 'dual_filled_hohlraum' type")
            
            universe = DualFilledHohlraum(
                primary_coronal=primary_geom,
                secondary_coronal=secondary_geom,
                materials=materials,
                **kwargs or {}
            )
        elif type == 'dual_indirect_coronal':
            if not isinstance(primary_geom, IndirectDriveUniverse) or not isinstance(secondary_geom, CoronalUniverse):
                raise ValueError("Both primary and secondary geometries must be 'coronal' for 'dual_indirect_coronal' type")
            
            universe = DualIndirectCoronal(
                primary_hohlraum=primary_geom,
                secondary_coronal=secondary_geom,
                materials=materials,
                **kwargs or {}
            )
        else:
            raise ValueError(f"Invalid geometry type: {type}")
        
        return universe
    
    def run_simulation(
        self, 
        model: NIFModel, 
        run_name: str = 'simulation',
        reset: bool = False) -> Path:
        """Run the simulation
        
        Parameters:
        -----------
        model : NIFModel
            Configured simulation model
        run_name : str
            Name for this simulation run
            
        Returns:
        --------
        Path
            Directory containing simulation results
        """
        # Set output directory
        original_dir = os.getcwd()
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        if reset:
            try:
                os.chdir(run_dir)
                
                # Export model to XML
                model.export_to_xml()
                model.export_to_model_xml()

                # Run simulation
                openmc.run()
                
                # Save simulation parameters
                params_to_save = {}
                for key, value in model.setup_params.items():
                    # Only save JSON-serializable values
                    if isinstance(value, (int, float, str, bool, type(None))):
                        params_to_save[key] = value
                    elif isinstance(value, (list, tuple)):
                        if all(isinstance(x, (int, float, str, bool)) for x in value):
                            params_to_save[key] = list(value)
                
                with open('simulation_params.json', 'w') as f:
                    json.dump(params_to_save, f, indent=2)
                    
            finally:
                os.chdir(original_dir)
        
        return run_dir