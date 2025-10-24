import openmc
from openmc.model import RectangularParallelepiped as BOX
from openmc.model import RightCircularCylinder as RCC
import numpy as np
from typing import Optional, Literal, Tuple, Dict, Any
from abc import abstractmethod
from .materials import NIFMaterials

class NIFUniverse(openmc.Universe):
    """Base class for NIF target geometries"""
    
    def __init__(
        self,
        materials: Optional[NIFMaterials] = None,
        convergence_ratio: float = 1.0,
        **kwargs):
        """Initialize NIF Universe
        
        Parameters:
        -----------
        materials : NIFMaterials, optional
            Materials collection
        convergence_ratio : float
            Compression ratio for implosion modeling
        **kwargs
            Additional parameters for openmc.Universe
        """
        super().__init__(**kwargs)
        self.materials = materials or NIFMaterials()
        self.convergence_ratio = convergence_ratio
    
    def compress_dimension(self, original_dim: float) -> float:
        """Apply compression to a dimension
        
        Parameters:
        -----------
        original_dim : float
            Original dimension before compression
            
        Returns:
        --------
        float
            Compressed dimension
        """
        return original_dim / self.convergence_ratio
    
    def remove_vacuum_boundaries(self):
        """
        Recursively remove vacuum boundary conditions from all surfaces
        in a universe and its nested universes.
        """
        # Get all cells in this universe
        for cell in self.cells.values():
            # Check if cell region has surfaces
            if cell.region is not None:
                # Get all surfaces from the region
                surfaces = cell.region.get_surfaces()
                
                # Remove vacuum boundary from each surface
                for surface in surfaces.values():
                    if surface.boundary_type == 'vacuum':
                        surface.boundary_type = 'transmission'
            
            # If the cell is filled with a universe, recurse
            if isinstance(cell.fill, NIFUniverse):
                cell.fill.remove_vacuum_boundaries()
    
    def add_tag_to_cells(self, tag: str):
        """
        Add a tag to all cells in this universe and its nested universes.
        
        Parameters:
        -----------
        tag : str
            Tag to add to all cells
        """
        for cell in self.cells.values():
            # Add suffix to cell name
            if cell.name:
                cell.name = f'{cell.name}_{tag}'
            
            # If cell is filled with a universe, recurse
            if isinstance(cell.fill, NIFUniverse):
                cell.fill.add_tag_to_cells(tag)
    
    @abstractmethod
    def get_fuel_params(self) -> Tuple[float, openmc.Cell]:
        """
        Get parameters for the fuel cell
        
        Returns:
        --------
        Tuple[float, openmc.Cell]
            Tuple of (radius, cell)
        """
        # Must be implemented in subclasses
        pass
    
    @abstractmethod
    def get_outer_region(self) -> openmc.Region:
        """
        Get the outer region of the geometry so that it can be manipulated
        as a fill for a cell.
        
        Returns:
        --------
        openmc.Region
            Outer region
        """
        # Must be implemented in subclass
        pass

class StandardNIFUniverse(NIFUniverse):
    """Standard NIF indirect-drive target geometry"""
    
    def __init__(
        self,
        fuel_radius_original: float = 0.1,
        shell_thickness_original: float = 100e-4,
        hohlraum_length: float = 1.0,
        hohlraum_radius: float = 0.3,
        hohlraum_thickness: float = 300e-4,
        ablator_material: str = 'ch2',
        hohlraum_material: str = 'gold',
        hohlraum_lining_thickness: Optional[float] = None,
        hohlraum_lining_material: Optional[str] = None,
        moderator_material: Optional[str] = None,
        moderator_thickness: float = 0.0,
        **kwargs):
        """Initialize standard NIF geometry
        
        Parameters:
        -----------
        fuel_radius_original : float
            Original fuel radius in cm
        shell_thickness_original : float
            Original shell thickness in cm
        hohlraum_length : float
            Hohlraum length in cm
        hohlraum_radius : float
            Hohlraum radius in cm
        hohlraum_thickness : float
            Hohlraum wall thickness in cm
        ablator_material : str
            Name of ablator material
        hohlraum_material : str
            Name of hohlraum material
        hohlraum_lining_thickness : float
            Thickness of gold lining if using gold-lined uranium hohlraum in cm
        moderator_material : str, optional
            Name of moderator material
        moderator_thickness : float
            Thickness of moderator layer in cm
        **kwargs
            Additional parameters for NIFUniverse
        """
        super().__init__(**kwargs)
        
        self.fuel_radius_original = fuel_radius_original
        self.shell_thickness_original = shell_thickness_original
        self.hohlraum_length = hohlraum_length
        self.hohlraum_radius = hohlraum_radius
        self.hohlraum_thickness = hohlraum_thickness
        self.hohlraum_lining_material = hohlraum_lining_material
        self.ablator_material = ablator_material
        self.hohlraum_material = hohlraum_material
        self.hohlraum_lining_thickness = hohlraum_lining_thickness
        self.moderator_material = moderator_material
        self.moderator_thickness = moderator_thickness
        
        self._create_geometry()
    
    def _create_geometry(self) -> None:
        """Create the standard NIF geometry"""
        # Apply compression to fuel and shell only
        self.fuel_radius = self.compress_dimension(self.fuel_radius_original)
        ablator_outer_radius = self.compress_dimension(self.fuel_radius_original + self.shell_thickness_original)
        
        # Hohlraum dimensions remain unchanged
        hohlraum_inner_radius = self.hohlraum_radius
        hohlraum_outer_radius = self.hohlraum_radius + self.hohlraum_thickness
        
        # Create surfaces
        # Fuel sphere
        fuel_surface = openmc.Sphere(r=self.fuel_radius)
        
        # Ablator outer surface
        ablator_surface = openmc.Sphere(r=ablator_outer_radius)
        
        # Hohlraum surfaces
        hohlraum_inner_cyl = openmc.ZCylinder(r=hohlraum_inner_radius)
        hohlraum_outer_cyl = openmc.ZCylinder(r=hohlraum_outer_radius, boundary_type='vacuum')
        hohlraum_bottom = openmc.ZPlane(z0=-self.hohlraum_length/2, boundary_type='vacuum')
        hohlraum_top = openmc.ZPlane(z0=self.hohlraum_length/2, boundary_type='vacuum')
        
        ### Regions ###
        fuel_region = -fuel_surface
        ablator_region = -ablator_surface & +fuel_surface
        hohlraum_region = +hohlraum_inner_cyl & -hohlraum_outer_cyl & +hohlraum_bottom & -hohlraum_top
        self.outer_region = -hohlraum_outer_cyl & +hohlraum_bottom & -hohlraum_top
        vacuum_region = -hohlraum_inner_cyl & +ablator_surface
        
        ### Cells ###
        self.fuel_cell = openmc.Cell(
            name='fuel',
            fill=self.materials['dt_fuel'],
            region=fuel_region
        )
        
        ablator_cell = openmc.Cell(
            name='ablator',
            fill=self.materials[self.ablator_material],
            region=ablator_region
        )
        
        hohlraum_cell = openmc.Cell(
            name='hohlraum',
            fill=self.materials[self.hohlraum_material],
            region=hohlraum_region
        )
        
        vacuum_cell = openmc.Cell(
            name='vacuum_cell',
            fill=None,
            region=vacuum_region
        )
        
        self.add_cell(self.fuel_cell)
        self.add_cell(ablator_cell)
        self.add_cell(hohlraum_cell)
        self.add_cell(vacuum_cell)
        
        # Optional features
        if self.hohlraum_lining_thickness and self.hohlraum_lining_material:
            """Create lined hohlraum"""
            # Lining surface
            lining_inner_radius = hohlraum_inner_radius - self.hohlraum_lining_thickness
            lining_inner_cyl = openmc.ZCylinder(r=lining_inner_radius)
            
            lining_region = (+lining_inner_cyl & -hohlraum_inner_cyl & +hohlraum_bottom & -hohlraum_top)
            
            # Lining cell
            lining_cell = openmc.Cell(
                name='hohlraum_lining',
                fill=self.materials[self.hohlraum_lining_material],
                region=lining_region
            )
            self.add_cell(lining_cell)
            
            # Modify vacuum region
            vacuum_cell.region &= -lining_inner_cyl
        
        # Moderator if present
        if self.moderator_material and self.moderator_thickness > 0:
            # Remove vacuum boundary from hohlraum outer surface
            self.remove_vacuum_boundaries()
            
            # Surfaces
            moderator_radius = hohlraum_outer_radius + self.moderator_thickness
            # Assume moderator is the same length as the hohlraum
            # TODO: make this configurable, will need to change vacuum definition as well
            moderator_length = self.hohlraum_length
            
            moderator_cyl = openmc.ZCylinder(r=moderator_radius, boundary_type='vacuum')
            moderator_bottom = openmc.ZPlane(z0=-moderator_length/2, boundary_type='vacuum')
            moderator_top = openmc.ZPlane(z0=moderator_length/2, boundary_type='vacuum')
            
            # Region
            moderator_region = -moderator_cyl & +hohlraum_outer_cyl & +moderator_bottom & -moderator_top
            
            """Create moderator cell around hohlraum"""
            moderator_cell = openmc.Cell(
                name='moderator',
                fill=self.materials[self.moderator_material],
                region=moderator_region
            )
            self.add_cell(moderator_cell)
            
            # Define outer region
            self.outer_region = -moderator_cyl & +moderator_bottom & -moderator_top
            
        self.add_cell(vacuum_cell)
        
    def get_fuel_params(self) -> Tuple[float, openmc.Cell]:
        return self.fuel_radius, self.fuel_cell
    
    def get_outer_region(self) -> openmc.Region:
        
        return self.outer_region

class DoubleShellUniverse(NIFUniverse):
    """Double-shell target geometry"""
    
    def __init__(
        self,
        fuel_radius_original: float = 271e-4,
        pusher_radius_original: float = 321e-4,
        tamper_radius_original: float = 372e-4,
        foam_radius_original: float = 1191e-4,
        ablator_radius_original: float = 1386e-4,
        pusher_material: str = 'tungsten',
        tamper_material: str = 'ch2',
        foam_material: str = 'ch',
        ablator_material: str = 'aluminum',
        **kwargs
    ):
        """Initialize double-shell geometry
        
        Parameters:
        -----------
        fuel_radius_original : float
            Original fuel radius in cm
        pusher_radius_original : float
            Original pusher radius in cm
        tamper_radius_original : float
            Original tamper radius in cm
        foam_radius_original : float
            Original foam radius in cm
        ablator_radius_original : float
            Original ablator radius in cm
        pusher_material : str
            Name of pusher material
        tamper_material : str
            Name of tamper material
        foam_material : str
            Name of foam material
        ablator_material : str
            Name of ablator material
        **kwargs
            Additional parameters for NIFUniverse
        """
        super().__init__(**kwargs)
        
        self.fuel_radius_original = fuel_radius_original
        self.pusher_radius_original = pusher_radius_original
        self.tamper_radius_original = tamper_radius_original
        self.foam_radius_original = foam_radius_original
        self.ablator_radius_original = ablator_radius_original
        self.pusher_material = pusher_material
        self.tamper_material = tamper_material
        self.foam_material = foam_material
        self.ablator_material = ablator_material
        
        self._create_geometry()
    
    def _create_geometry(self) -> None:
        """Create the double-shell geometry"""
        # Apply compression
        self.fuel_radius = self.compress_dimension(self.fuel_radius_original)
        pusher_radius = self.compress_dimension(self.pusher_radius_original)
        tamper_radius = self.compress_dimension(self.tamper_radius_original)
        foam_radius = self.compress_dimension(self.foam_radius_original)
        ablator_radius = self.compress_dimension(self.ablator_radius_original)
        
        # Create surfaces
        fuel_surface = openmc.Sphere(r=self.fuel_radius)
        pusher_surface = openmc.Sphere(r=pusher_radius)
        tamper_surface = openmc.Sphere(r=tamper_radius)
        foam_surface = openmc.Sphere(r=foam_radius)
        ablator_surface = openmc.Sphere(r=ablator_radius, boundary_type='vacuum')
        
        # Define outer region
        self.outer_region = -ablator_surface
        
        # Create cells
        # Fuel
        self.fuel_cell = openmc.Cell(
            name='fuel',
            region=-fuel_surface,
            fill=self.materials['dt_fuel']
        )
        self.add_cell(self.fuel_cell)
        
        # Pusher
        pusher_cell = openmc.Cell(
            name='pusher',
            region=+fuel_surface & -pusher_surface,
            fill=self.materials[self.pusher_material]
        )
        self.add_cell(pusher_cell)
        
        # Tamper
        tamper_cell = openmc.Cell(
            name='tamper',
            region=+pusher_surface & -tamper_surface,
            fill=self.materials[self.tamper_material]
        )
        self.add_cell(tamper_cell)
        
        # Foam
        foam_cell = openmc.Cell(
            name='foam',
            region=+tamper_surface & -foam_surface,
            fill=self.materials[self.foam_material]
        )
        self.add_cell(foam_cell)
        
        # Ablator
        ablator_cell = openmc.Cell(
            name='ablator',
            region=+foam_surface & -ablator_surface,
            fill=self.materials[self.ablator_material]
        )
        self.add_cell(ablator_cell)
        
    def get_fuel_params(self) -> Tuple[float, openmc.Cell]:
        return self.fuel_radius, self.fuel_cell
    
    def get_outer_region(self) -> openmc.Region:
        return self.outer_region

class CoronalUniverse(NIFUniverse):
    """Coronal source target geometry"""
    
    def __init__(
        self,
        capsule_radius_original: float = 0.1,
        capsule_thickness_original: float = 0.01,
        capsule_material: str = 'ch',
        lining_thickness_original: Optional[float] = None,
        lining_material: Optional[str] = None,
        ice_thickness_original: Optional[float] = None,
        ice_material: Optional[str] = None,
        hole_radius_original: float = 0.05,
        n_holes: Literal[1, 2] = 1,
        **kwargs
    ):
        """Initialize coronal source geometry
        
        Parameters:
        -----------
        capsule_radius_original : float
            Capsule radius in cm
        capsule_thickness_original : float
            Capsule thickness in cm
        capsule_material : str
            Name of capsule material
        lining_thickness_original : float, optional
            lining thickness in cm
        lining_material : str, optional
            Name of lining material
        ice_thickness_original : float, optional
            DT ice layer thickness in cm
        ice_material : str, optional
            Name of ice material
        hole_radius_original : float
            Laser entrance hole radius in cm
        n_holes : int
            Number of laser entrance holes (1 or 2)
        **kwargs
            Additional parameters for NIFUniverse
        """
        super().__init__(**kwargs)
        
        self.capsule_radius_original = capsule_radius_original
        self.capsule_thickness_original = capsule_thickness_original
        self.capsule_material = capsule_material
        self.lining_thickness_original = lining_thickness_original or 0.0
        self.lining_material = lining_material
        self.ice_thickness_original = ice_thickness_original or 0.0
        self.ice_material = ice_material
        self.hole_radius_original = hole_radius_original
        self.n_holes = n_holes
        
        if self.hole_radius_original >= self.capsule_radius_original:
            raise ValueError("Hole radius must be less than capsule radius")
        
        self._create_geometry()
    
    def _create_geometry(self) -> None:
        """Create the coronal source geometry"""
        # Apply compression
        capsule_radius = self.compress_dimension(self.capsule_radius_original)
        capsule_thickness = self.compress_dimension(self.capsule_thickness_original)
        lining_thickness = self.compress_dimension(self.lining_thickness_original)
        ice_thickness = self.compress_dimension(self.ice_thickness_original)
        hole_radius = self.compress_dimension(self.hole_radius_original)
        
        # Define fuel radius
        self.fuel_radius = capsule_radius - capsule_thickness - lining_thickness - ice_thickness
        
        # Create surfaces
        fuel_surface = openmc.Sphere(r=self.fuel_radius)
        capsule_surface = openmc.Sphere(r=capsule_radius, boundary_type='vacuum')            
        
        # Laser entrance hole (plane)
        z_loc = -np.sqrt(capsule_radius**2 - hole_radius**2) # negative z location to point hole upwards
        hole1_plane = openmc.ZPlane(z0=z_loc, boundary_type='vacuum')
        
        # Regions
        fuel_region = -fuel_surface & +hole1_plane
        capsule_region = -capsule_surface & +fuel_surface & +hole1_plane
        self.outer_region = -capsule_surface & +hole1_plane
        
        # Cells
        self.fuel_cell = openmc.Cell(
            name='fuel',
            region=fuel_region,
            fill=self.materials['dt_fuel']
        )
        self.add_cell(self.fuel_cell)
        
        capsule_cell = openmc.Cell(
            name='capsule',
            region=capsule_region,
            fill=self.materials[self.capsule_material]
        )
        self.add_cell(capsule_cell)
        
        ### OPTIONAL FEATURES ###
        if self.lining_thickness_original > 0.0 and self.lining_material:
            lining_radius = capsule_radius - capsule_thickness
            lining_surface = openmc.Sphere(r=lining_radius)
            lining_region = -lining_surface & +fuel_surface & +hole1_plane
            lining_cell = openmc.Cell(
                name='lining',
                region=lining_region,
                fill=self.materials[self.lining_material]
            )
            self.add_cell(lining_cell)
            
            # Remove lining from other regions
            capsule_cell.region &= +lining_surface
            
        if ice_thickness > 0.0 and self.ice_material:
            ice_radius = capsule_radius - capsule_thickness - lining_thickness
            ice_surface = openmc.Sphere(r=ice_radius)
            ice_region = -ice_surface & +fuel_surface & +hole1_plane
            ice_cell = openmc.Cell(
                name='ice',
                region=ice_region,
                fill=self.materials[self.ice_material]
            )
            self.add_cell(ice_cell)
            
            # Remove ice from other regions
            capsule_cell.region &= +ice_surface
            if self.lining_thickness_original > 0.0:
                lining_cell.region &= +ice_surface
                    
        if self.n_holes == 2:
            hole2_plane = openmc.ZPlane(z0=-z_loc, boundary_type='vacuum')
            
            self.fuel_cell.region &= -hole2_plane
            capsule_cell.region &= -hole2_plane
            if self.lining_thickness_original > 0.0:
                lining_cell.region &= -hole2_plane
            if self.ice_thickness_original > 0.0:
                ice_cell.region &= -hole2_plane
                
            # Add constraint to outer region
            self.outer_region &= -hole2_plane
                
    def get_fuel_params(self) -> Tuple[float, openmc.Cell]:
        return self.fuel_radius, self.fuel_cell
    
    def get_outer_region(self) -> openmc.Region:
        return self.outer_region
                
class DualSourceUniverse(NIFUniverse):
    def __init__(
        self,
        primary_geom: NIFUniverse,
        secondary_geom: NIFUniverse,
        center_distance: float = 0.5,
        moderator_thickness: float = 0.1,
        moderator_radius: float = 1.0,
        moderator_material: str = 'ch2',
        moderator_distance: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize dual source geometry. Geometry is centered on the moderator in z.
        
        Parameters:
        -----------
        primary_geom : NIFUniverse
            Primary source geometry
        secondary_geom : NIFUniverse
            Secondary source geometry
        center_distance : float
            Distance between source centers in cm
        moderator_thickness : float
            Moderator thickness in cm
        moderator_radius : float
            Moderator diameter in cm
        moderator_material : str, optional
            Moderator material name
        moderator_distance : float, optional
            Distance between moderator center and primary source center in cm
        """
        
        super().__init__(**kwargs)
        self.primary_geom = primary_geom
        self.secondary_geom = secondary_geom
        self.center_distance = center_distance
        self.moderator_thickness = moderator_thickness
        self.moderator_radius = moderator_radius
        self.moderator_material = moderator_material
        self.moderator_distance = moderator_distance or center_distance / 2 # default to midway between sources
        
        if self.moderator_distance > center_distance:
            raise ValueError("Moderator distance must be less than center distance")
        
        # Remove vacuum boundary conditions from sources
        self.primary_geom.remove_vacuum_boundaries()
        self.secondary_geom.remove_vacuum_boundaries()
        
        # Add tags to cell names
        self.primary_geom.add_tag_to_cells('primary')
        self.secondary_geom.add_tag_to_cells('secondary')
        
        # Check moderator width
        primary_fuel_radius = self.primary_geom.get_fuel_params()[0]
        secondary_fuel_radius = self.secondary_geom.get_fuel_params()[0]
        if self.moderator_radius < max(primary_fuel_radius, secondary_fuel_radius):
            raise ValueError("Moderator radius must be greater than the maximum of the primary and secondary source fuel radii")
        
        self._create_geometry()
        
    def _create_geometry(self):       
        ### Surfaces ###
        self.moderator_rcc = RCC(
            center_base=(0, 0, -self.moderator_thickness / 2),
            radius=self.moderator_radius,
            height=self.moderator_thickness,
            axis='z'
        )
        
        # Translation of primary and secondary regions
        self.primary_translation = -self.moderator_distance
        self.secondary_translation = self.center_distance - self.moderator_distance
        
        # Use bounding boxes to find the extent of the geometry
        primary_min_z = self.primary_geom.bounding_box.lower_left[2] + self.primary_translation
        # The secondary geometry is located at the top of the moderator
        secondary_max_z = self.secondary_geom.bounding_box.upper_right[2] + self.secondary_translation
        # Add a bit of padding to all sides
        vacuum_rcc = RCC(
            center_base=(0, 0, primary_min_z * 1.1),
            radius=self.moderator_radius * 1.1,
            height=(secondary_max_z - primary_min_z) * 1.1,
            axis='z',
            boundary_type='vacuum'
        )
        
        ### Regions ###
        primary_geom_region = self.primary_geom.get_outer_region()
        secondary_geom_region = self.secondary_geom.get_outer_region()
        
        # Rotate secondary geometry by 180 degrees to make a mirror
        transformed_secondary_geom_region = secondary_geom_region.rotate((180, 0, 0))
        # Translate source regions into position
        transformed_primary_geom_region = primary_geom_region.translate((0, 0, self.primary_translation))
        transformed_secondary_geom_region = transformed_secondary_geom_region.translate((0, 0, self.secondary_translation))
        
        # The primary and secondary source could interset the moderator region
        moderator_region = -self.moderator_rcc & ~transformed_primary_geom_region & ~transformed_secondary_geom_region
        vacuum_region = -vacuum_rcc & ~transformed_primary_geom_region & ~transformed_secondary_geom_region & + self.moderator_rcc
        
        self.outer_region = -vacuum_rcc
        
        ### Cells ###
        primary_geom_cell = openmc.Cell(
            name='primary_geom_cell',
            fill=self.primary_geom,
            region=transformed_primary_geom_region
        )
        # Translate primary source cell into position
        primary_geom_cell.translation = (0, 0, self.primary_translation)
        
        secondary_geom_cell = openmc.Cell(
            name='secondary_geom_cell',
            fill=self.secondary_geom,
            region=transformed_secondary_geom_region
        )
        # Rotate and translate secondary source cell into position
        secondary_geom_cell.rotation = (180, 0, 0)
        secondary_geom_cell.translation = (0, 0, self.secondary_translation)
        
        self.moderator_cell = openmc.Cell(
            name='moderator_cell',
            fill=self.materials[self.moderator_material],
            region=moderator_region
        )
        
        vacuum_cell = openmc.Cell(
            name='vacuum_cell',
            fill=None,
            region=vacuum_region
        )
        
        self.add_cells([primary_geom_cell, secondary_geom_cell, self.moderator_cell, vacuum_cell])
        
    def get_outer_region(self) -> openmc.Region:
        return self.outer_region
    
    def get_fuel_params(self) -> Tuple[Tuple[float, openmc.Cell], Tuple[float, openmc.Cell]]:
        return self.primary_geom.get_fuel_params(), self.secondary_geom.get_fuel_params()
    
class DualFilledHohlraum(DualSourceUniverse):
    """
    Specialized dual source geometry with coronal sources positioned at opposite 
    ends of a CH2-filled hohlraum with heavy metal lining for neutron reflection.
    """
    
    def __init__(
        self,
        primary_coronal: CoronalUniverse,
        secondary_coronal: CoronalUniverse,
        source_gap: float = 0.5,
        hohlraum_inner_radius: float = 0.3,
        hohlraum_material: str = 'gold',
        fill_material: str = 'ch2',
        hohlraum_wall_thickness: float = 0.03,
        multiplier_material: Optional[str] = None,
        multiplier_thickness: Optional[float] = None,
        multiplier_primary_gap: Optional[float] = None,
        hohlraum_lining_thickness: Optional[float] = None,
        hohlraum_lining_material: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize filled hohlraum dual source geometry
        
        Parameters:
        -----------
        primary_coronal : CoronalUniverse
            Primary coronal source geometry
        secondary_coronal : CoronalUniverse  
            Secondary coronal source geometry
        source_gap : float
            Distance between outermost radii of coronal sources in cm
        hohlraum_inner_radius : float
            Inner radius of hohlraum in cm
        hohlraum_material : str
            Name of hohlraum material
        fill_material : str
            Name of hohlraum fill material
        hohlraum_wall_thickness : float
            Wall thickness of hohlraum in cm
        multiplier_material: str, optional
            Name of neutron multiplier material
        multiplier_thickness: float, optional
            Thickness of neutron multiplier layer
        multiplier_primary_gap: float, optional
            Gap between the edge of the primary source and the start of the multiplier layer
        hohlraum_lining_thickness : float, optional
            Wall thickness of hohlraum lining in cm
        hohlraum_lining_material : str, optional
            Name of hohlraum lining material
        **kwargs
            Additional parameters passed to DualSourceUniverse
        """
        
        # Store hohlraum-specific parameters
        self.primary_coronal = primary_coronal
        self.secondary_coronal = secondary_coronal
        self.hohlraum_inner_radius = hohlraum_inner_radius
        self.hohlraum_material = hohlraum_material
        self.hohlraum_wall_thickness = hohlraum_wall_thickness
        self.multiplier_material = multiplier_material
        self.multiplier_thickness = multiplier_thickness
        self.multiplier_primary_gap = multiplier_primary_gap
        self.hohlraum_lining_thickness = hohlraum_lining_thickness
        self.hohlraum_lining_material = hohlraum_lining_material
                
        # Calculate center distance based on source size and gap
        self.primary_radius = primary_coronal.capsule_radius_original
        secondary_radius = secondary_coronal.capsule_radius_original
        center_distance = self.primary_radius + secondary_radius + source_gap
        
        # Calculate moderator thickness based on distance between flats of LEH
        primary_LEH_z = np.sqrt(self.primary_radius**2 - primary_coronal.hole_radius_original**2)
        secondary_LEH_z = np.sqrt(secondary_radius**2 - secondary_coronal.hole_radius_original**2)
        moderator_thickness = primary_LEH_z + secondary_LEH_z + center_distance
        
        # Check to make sure that the moderator does not intersect the secondary source
        if self.multiplier_primary_gap and self.multiplier_thickness:
            if self.multiplier_primary_gap + self.multiplier_thickness > source_gap:
                raise ValueError('The multiplier is intersecting the secondary source.')
        
        # Initialize parent class
        # We'll override _create_geometry to build our specialized version
        super().__init__(
            primary_geom=primary_coronal,
            secondary_geom=secondary_coronal,
            center_distance=center_distance,
            moderator_thickness=moderator_thickness,
            moderator_radius=hohlraum_inner_radius,
            moderator_material=fill_material,
            convergence_ratio=1.0, # No compression
            **kwargs
        )
        
        # Remove vacuum boundary conditions from parent class
        self.remove_vacuum_boundaries()
        
        self._create_dual_filled_hohlraum_geometry()
    
    def _create_dual_filled_hohlraum_geometry(self):
        """Override parent's geometry creation with filled hohlraum design"""
        
        # The parent __init__ already handled, need to remove vacuum cell
        for cell in self.cells.values():
            if cell.name == 'vacuum_cell':
                self.remove_cell(cell)
                break
        
        # Calculate surfaces for hohlraum
        hohlraum_outer_cylinder = openmc.ZCylinder(r=self.hohlraum_inner_radius + self.hohlraum_wall_thickness)
        primary_LEH_cylinder = openmc.ZCylinder(r=self.primary_coronal.hole_radius_original)
        secondary_LEH_cylinder = openmc.ZCylinder(r=self.secondary_coronal.hole_radius_original)
        hohlraum_bottom = openmc.ZPlane(z0=-(self.moderator_thickness/2 + self.hohlraum_wall_thickness))
        hohlraum_top = openmc.ZPlane(z0=self.moderator_thickness/2 + self.hohlraum_wall_thickness)
        moderator_bottom = self.moderator_rcc.top
        moderator_top = self.moderator_rcc.bottom
        
        # Vacuum is slightly larger than hohlraum to avoid overlap
        vacuum_center_base = (0, 0, -(self.moderator_thickness/2 + self.hohlraum_wall_thickness) * 1.1)
        vacuum_radius = (self.hohlraum_inner_radius + self.hohlraum_wall_thickness) * 1.1
        vacuum_height = (self.moderator_thickness + 2 * self.hohlraum_wall_thickness) * 1.1
        
        # Define hohlraum region
        hohlraum_bottom_LEH_region = -hohlraum_outer_cylinder & +primary_LEH_cylinder & +hohlraum_bottom & -moderator_bottom
        hohlraum_top_LEH_region = -hohlraum_outer_cylinder & +secondary_LEH_cylinder & +moderator_top & -hohlraum_top
        hohlraum_shell_region = -hohlraum_outer_cylinder & +self.moderator_rcc.cyl & +moderator_bottom & -moderator_top
        hohlraum_region = hohlraum_bottom_LEH_region | hohlraum_top_LEH_region | hohlraum_shell_region
        
        # Define outer region
        self.outer_region = hohlraum_region | -self.moderator_rcc
        
        # Create cells
        hohlraum_cell = openmc.Cell(
            name='hohlraum',
            region=hohlraum_region,
            fill=self.materials[self.hohlraum_material]
        )
        self.add_cell(hohlraum_cell)
        
        # Add lining
        if self.hohlraum_lining_material and self.hohlraum_lining_thickness:
            # Remove vacuum boundary from hohlraum outer surface
            self.remove_vacuum_boundaries()
            
            hohlraum_lining_outer_cylinder = openmc.ZCylinder(r=self.hohlraum_inner_radius + self.hohlraum_wall_thickness + self.hohlraum_lining_thickness)
            hohlraum_lining_bottom = openmc.ZPlane(z0=-(self.moderator_thickness/2 + self.hohlraum_wall_thickness + self.hohlraum_lining_thickness))
            hohlraum_lining_top = openmc.ZPlane(z0=self.moderator_thickness/2 + self.hohlraum_wall_thickness + self.hohlraum_lining_thickness)
            
            # Define lining region, excluding LEH regions
            hohlraum_lining_bottom_LEH_region = -hohlraum_lining_outer_cylinder & +primary_LEH_cylinder & +hohlraum_lining_bottom & -hohlraum_bottom
            hohlraum_lining_top_LEH_region = -hohlraum_lining_outer_cylinder & +secondary_LEH_cylinder & +hohlraum_top & -hohlraum_lining_top
            hohlraum_lining_shell_region = -hohlraum_lining_outer_cylinder & +hohlraum_outer_cylinder & +hohlraum_lining_bottom & -hohlraum_lining_top
            hohlraum_lining_region = hohlraum_lining_bottom_LEH_region | hohlraum_lining_top_LEH_region | hohlraum_lining_shell_region
            
            # Redefine vacuum region to exclude lining
            vacuum_center_base = (0, 0, -(self.moderator_thickness/2 + self.hohlraum_wall_thickness + self.hohlraum_lining_thickness) * 1.1)
            vacuum_radius = (self.hohlraum_inner_radius + self.hohlraum_wall_thickness + self.hohlraum_lining_thickness) * 1.1
            vacuum_height = (self.moderator_thickness + 2 * (self.hohlraum_wall_thickness + self.hohlraum_lining_thickness)) * 1.1
            
            # Create hohlraum lining cell
            hohlraum_lining_cell = openmc.Cell(
                name='hohlraum_lining',
                region=hohlraum_lining_region,
                fill=self.materials[self.hohlraum_lining_material]
            )
            self.add_cell(hohlraum_lining_cell)
            
            # Redefine outer region
            self.outer_region = -hohlraum_lining_outer_cylinder & +hohlraum_lining_bottom & -hohlraum_lining_top
            
        if self.multiplier_material and self.multiplier_thickness and self.multiplier_primary_gap:
            multiplier_bottom_plane = openmc.ZPlane(self.primary_translation + self.primary_radius + self.multiplier_primary_gap)
            multiplier_top_plane = openmc.ZPlane(self.primary_translation + self.primary_radius + self.multiplier_primary_gap + self.multiplier_thickness)
            
            multiplier_region = -self.moderator_rcc.cyl & +multiplier_bottom_plane & -multiplier_top_plane
            multiplier_cell = openmc.Cell(
                name='multiplier_layer',
                region=multiplier_region,
                fill=self.materials[self.multiplier_material]
            )
            self.add_cell(multiplier_cell)
            
            # Exclude multiplier from other regions
            self.moderator_cell.region &= ~multiplier_region
            hohlraum_cell.region &= ~multiplier_region
            
        # Create vacuum cell
        vacuum_rcc = RCC(
            center_base=vacuum_center_base,
            radius=vacuum_radius,
            height=vacuum_height,
            axis='z',
            boundary_type='vacuum'
        )
        vacuum_region = -vacuum_rcc & ~hohlraum_region & +self.moderator_rcc
        vacuum_cell = openmc.Cell(
            name='vacuum_cell',
            region=vacuum_region
        )
        self.add_cell(vacuum_cell)
        
            
    def get_outer_region(self) -> openmc.Region:
        return self.outer_region
    