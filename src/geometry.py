import openmc
from openmc.model import RectangularParallelepiped as BOX
from openmc.model import RightCircularCylinder as RCC
import numpy as np
from typing import Optional, Literal, Tuple, Union, Sequence
from abc import abstractmethod
from .materials import NIFMaterials, FuelMaterial

# Helper function for getting materials from universes
def get_materials(universe: openmc.Universe) -> openmc.Materials:
        """
        Recursively get materials
        """
        materials = openmc.Materials()
        # Get all cells in this universe
        for cell in universe.cells.values():
            if cell.fill:
                # If the cell is filled with a universe, recurse
                if isinstance(cell.fill, openmc.Universe):
                    for material in get_materials(cell.fill):
                        materials.append(material)
                else:
                    # Skip over materials that have already been added
                    if cell.fill not in materials:
                        materials.append(cell.fill)
        return materials

class BaseImplosionUniverse(openmc.Universe):
    """Base class for spherical implosion geometry"""
    
    def __init__(
        self,
        materials: Optional[NIFMaterials] = None,
        convergence_ratio: float = 1.0,
        fuel_radius_original: float = 0.1,
        ablator_thickness_original: float = 0.01,
        fuel_material: FuelMaterial = FuelMaterial(),
        ablator_material: str = 'ch2',
        tag: Optional[Literal['primary', 'secondary']] = None,
        **kwargs):
        """Initialize NIF Universe
        
        Parameters:
        -----------
        materials : NIFMaterials, optional
            Materials collection
        convergence_ratio : float
            Compression ratio for implosion modeling
        fuel_radius_original : float
            Original fuel radius in cm
        ablator_thickness_original : float
            Original ablator thickness in cm
        fuel_material : FuelMaterial
            Fuel material, defaults to standard DT fuel with trace nuclide
        ablator_material : str
            Ablator material
        tag : str, optional
            Tag to add to cell names
        **kwargs
            Additional parameters for openmc.Universe
        """
        super().__init__(**kwargs)
        self.materials = materials or NIFMaterials()
        self.convergence_ratio = convergence_ratio
        self.fuel_radius_original = fuel_radius_original
        self.ablator_thickness_original = ablator_thickness_original
        self.fuel_material = fuel_material
        self.ablator_material = self.materials[ablator_material]
        self.tag = tag
        
        self._create_base_geometry()
      
    def _create_base_geometry(self) -> None:
        # Compress fuel and ablator
        self.fuel_radius, self.fuel_density = self.compress_dimension(self.fuel_radius_original, self.fuel_material)
        self.ablator_thickness, ablator_density = self.compress_dimension(self.ablator_thickness_original, self.ablator_material)
        
        # Create surfaces, regions, and cells
        fuel_surface = openmc.Sphere(r=self.fuel_radius)
        ablator_surface = openmc.Sphere(r=self.fuel_radius + self.ablator_thickness, boundary_type='vacuum')
        
        fuel_region = -fuel_surface
        ablator_region = -ablator_surface & +fuel_surface
        
        # Define outer_region
        self.outer_region = -ablator_surface
        
        self.fuel_cell = openmc.Cell(
            name='fuel',
            region=fuel_region,
            fill=self.fuel_material
        )
        self.fuel_cell.density = self.fuel_density
        self.fuel_cell.volume = 4/3 * np.pi * self.fuel_radius**3
        
        self.ablator_cell = openmc.Cell(
            name='ablator',
            region=ablator_region,
            fill=self.ablator_material
        )
        self.ablator_cell.density = ablator_density
        
        self.add_cells([self.fuel_cell, self.ablator_cell])
        
    @abstractmethod
    def _create_geometry(self) -> None:
        """Create the specific geometry"""
        pass
    
    def compress_dimension(self, original_dim: float, material: Optional[openmc.Material] = None) -> Tuple[float, Optional[float]]:
        """Apply compression to a dimension
        
        Parameters:
        -----------
        original_dim : float
            Original dimension before compression
        original_density : float
            Original density before compression
            
        Returns:
        --------
        float
            Compressed dimension
        """
        new_dim = original_dim / self.convergence_ratio
        if material is None:
            new_density = None
        else:
            if material.density:
                new_density = material.density * (self.convergence_ratio ** 3)
            else:
                raise ValueError(f"Material {material.name} has no density defined.")
        return new_dim, new_density
    
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
            if isinstance(cell.fill, BaseImplosionUniverse):
                cell.fill.remove_vacuum_boundaries()
    
    def add_tag_to_cells(self, tag: Optional[str]=None):
        """
        Add a tag to all cells in this universe and its nested universes.
        
        Parameters:
        -----------
        tag : str
            Tag to add to all cells
        """
        # Do nothing if tag is None
        if tag is None:
            return
        for cell in self.cells.values():
            # Add suffix to cell name
            if cell.name:
                cell.name = f'{cell.name}_{tag}'
            
            # If cell is filled with a universe, recurse
            if isinstance(cell.fill, BaseImplosionUniverse):
                cell.fill.add_tag_to_cells(tag)
    
    def get_fuel_params(self) -> Tuple[float, openmc.Cell]:
        """
        Get parameters for the fuel cell
        
        Returns:
        --------
        Tuple[float, openmc.Cell]
            Tuple of (radius, cell)
        """
        return self.fuel_radius, self.fuel_cell
    
    def get_outer_region(self) -> openmc.Region:
        """
        Get the outer region of the geometry so that it can be manipulated
        as a fill for a cell.
        
        Returns:
        --------
        openmc.Region
            Outer region
        """
        return self.outer_region

class IndirectDriveUniverse(BaseImplosionUniverse):
    """Standard NIF indirect-drive target geometry"""
    
    def __init__(
        self,
        hohlraum_length: float = 1.0,
        hohlraum_radius: float = 0.3,
        hohlraum_thickness: float = 300e-4,
        leh_radius: float = 0.05,
        hohlraum_material: str = 'gold',
        hohlraum_lining_thickness: Optional[float] = None,
        hohlraum_lining_material: Optional[str] = None,
        moderator_material: Optional[str] = None,
        moderator_thickness: float = 0.0,
        **kwargs):
        """Initialize standard NIF geometry
        
        Parameters:
        -----------
        hohlraum_length : float
            Hohlraum length in cm
        hohlraum_radius : float
            Hohlraum outer radius in cm
        hohlraum_thickness : float
            Hohlraum wall thickness in cm
        leh_radius : float
            Laser entrance hole radius in cm
        hohlraum_material : str
            Name of hohlraum material
        hohlraum_lining_thickness : float
            Thickness of gold lining if using gold-lined uranium hohlraum in cm
        moderator_material : str, optional
            Name of moderator material
        moderator_thickness : float
            Thickness of moderator layer in cm
        **kwargs
            Additional parameters for BaseImplosionUniverse
        """
        super().__init__(**kwargs)
        
        self.hohlraum_length = hohlraum_length
        self.hohlraum_radius = hohlraum_radius
        self.hohlraum_thickness = hohlraum_thickness
        self.leh_radius = leh_radius
        self.hohlraum_lining_material = self.materials[hohlraum_lining_material]
        self.hohlraum_material = self.materials[hohlraum_material]
        self.hohlraum_lining_thickness = hohlraum_lining_thickness
        self.moderator_material = self.materials[moderator_material]
        self.moderator_thickness = moderator_thickness
        
        # Remove vacuum boundaries from surfaces
        self.remove_vacuum_boundaries()
        
        self._create_geometry()
    
    def _create_geometry(self) -> None:
        """Create the standard NIF geometry"""       
        # Hohlraum dimensions remain unchanged     
        hohlraum_inner_radius = self.hohlraum_radius - self.hohlraum_thickness
        
        # Hohlraum surfaces
        hohlraum_inner_cyl = openmc.ZCylinder(r=hohlraum_inner_radius)
        hohlraum_outer_cyl = openmc.ZCylinder(r=self.hohlraum_radius, boundary_type='vacuum')
        hohlraum_bottom = openmc.ZPlane(z0=-self.hohlraum_length/2, boundary_type='vacuum')
        hohlraum_top = openmc.ZPlane(z0=self.hohlraum_length/2, boundary_type='vacuum')
        
        # Laser entrance hole
        leh_cylinder = openmc.ZCylinder(r=self.leh_radius)
        leh_lower_inside = openmc.ZPlane(z0=-self.hohlraum_length/2 + self.hohlraum_thickness)
        leh_upper_inside = openmc.ZPlane(z0=self.hohlraum_length/2 - self.hohlraum_thickness)
        
        ### Regions ###
        hohlraum_walls_region = +hohlraum_inner_cyl & -hohlraum_outer_cyl & +hohlraum_bottom & -hohlraum_top
        leh_lower_region = -hohlraum_inner_cyl & +leh_cylinder & -leh_lower_inside & +hohlraum_bottom
        leh_upper_region = -hohlraum_inner_cyl & +leh_cylinder & +leh_upper_inside & -hohlraum_top
        hohlraum_region = hohlraum_walls_region | leh_lower_region | leh_upper_region
        
        # Define vacuum region with parent's outer_region
        parent_outer_region = self.outer_region
        vacuum_region = -hohlraum_outer_cyl & ~parent_outer_region & +hohlraum_bottom & -hohlraum_top & ~hohlraum_region
        
        # Redefine outer_region
        self.outer_region = -hohlraum_outer_cyl & +hohlraum_bottom & -hohlraum_top
        
        ### Cells ###
        hohlraum_cell = openmc.Cell(
            name='hohlraum',
            fill=self.hohlraum_material,
            region=hohlraum_region
        )
        
        vacuum_cell = openmc.Cell(
            name='vacuum_cell',
            fill=None,
            region=vacuum_region
        )
        
        self.add_cells([hohlraum_cell, vacuum_cell])
        
        # Optional features
        if self.hohlraum_lining_thickness and self.hohlraum_lining_material:
            """Create lined hohlraum"""
            # Lining surface
            lining_inner_radius = hohlraum_inner_radius - self.hohlraum_lining_thickness
            lining_inner_cyl = openmc.ZCylinder(r=lining_inner_radius)
            lining_lower_inside = openmc.ZPlane(z0=-self.hohlraum_length/2 + self.hohlraum_thickness + self.hohlraum_lining_thickness)
            lining_upper_inside = openmc.ZPlane(z0=self.hohlraum_length/2 - self.hohlraum_thickness - self.hohlraum_lining_thickness)
            
            # Lining region
            lining_walls = -hohlraum_inner_cyl & +lining_inner_cyl & +leh_lower_inside & -leh_upper_inside
            lining_lower_leh = -lining_inner_cyl & +leh_cylinder & -lining_lower_inside & +leh_lower_inside
            lining_upper_leh = -lining_inner_cyl & +leh_cylinder & +lining_upper_inside & -leh_upper_inside
            lining_region = lining_walls | lining_lower_leh | lining_upper_leh
            
            # Lining cell
            lining_cell = openmc.Cell(
                name='hohlraum_lining',
                fill=self.hohlraum_lining_material,
                region=lining_region
            )
            self.add_cell(lining_cell)
            
            # Modify vacuum region
            vacuum_cell.region &= ~lining_region
        
        # Moderator if present
        if self.moderator_material and self.moderator_thickness > 0:
            # Remove vacuum boundary from hohlraum outer cylinder
            hohlraum_outer_cyl.boundary_type = 'transmission'
            
            # Moderator surfaces
            moderator_outer_radius = self.hohlraum_radius + self.moderator_thickness
            moderator_outer_cyl = openmc.ZCylinder(r=moderator_outer_radius, boundary_type='vacuum')
            
            # Moderator region
            moderator_region = -moderator_outer_cyl & +hohlraum_outer_cyl & +hohlraum_bottom & -hohlraum_top
            
            """Create moderator cell around hohlraum"""
            moderator_cell = openmc.Cell(
                name='moderator',
                fill=self.moderator_material,
                region=moderator_region
            )
            self.add_cell(moderator_cell)
            
            # Define outer region
            self.outer_region = -moderator_outer_cyl & +hohlraum_bottom & -hohlraum_top
            
        self.add_cell(vacuum_cell)

class DoubleShellUniverse(BaseImplosionUniverse):
    """Double-shell target geometry"""
    
    def __init__(
        self,
        pusher_radius_original: float = 321e-4,
        tamper_radius_original: float = 372e-4,
        foam_radius_original: float = 1191e-4,
        pusher_material: str = 'tungsten',
        tamper_material: str = 'ch2',
        foam_material: str = 'ch',
        **kwargs
    ):
        """Initialize double-shell geometry
        
        Parameters:
        -----------
        pusher_radius_original : float
            Original pusher radius in cm
        tamper_radius_original : float
            Original tamper radius in cm
        foam_radius_original : float
            Original foam radius in cm
        pusher_material : str
            Name of pusher material
        tamper_material : str
            Name of tamper material
        foam_material : str
            Name of foam material
        **kwargs
            Additional parameters for BaseImplosionUniverse
        """
        super().__init__(**kwargs)
        
        self.pusher_radius_original = pusher_radius_original
        self.tamper_radius_original = tamper_radius_original
        self.foam_radius_original = foam_radius_original
        self.pusher_material = self.materials[pusher_material]
        self.tamper_material = self.materials[tamper_material]
        self.foam_material = self.materials[foam_material]
        
        self._create_geometry()
    
    def _create_geometry(self) -> None:
        """Create the double-shell geometry"""
        # Compress geometry
        pusher_radius, pusher_density = self.compress_dimension(self.pusher_radius_original, self.pusher_material)
        tamper_radius, tamper_density = self.compress_dimension(self.tamper_radius_original, self.tamper_material)
        foam_radius, foam_density = self.compress_dimension(self.foam_radius_original, self.foam_material)
        
        # Create surfaces
        pusher_surface = openmc.Sphere(r=pusher_radius)
        tamper_surface = openmc.Sphere(r=tamper_radius)
        foam_surface = openmc.Sphere(r=foam_radius)
        
        # Create regions
        pusher_region = -pusher_surface & ~self.fuel_cell.region
        tamper_region = -tamper_surface & +pusher_surface
        foam_region = -foam_surface & +tamper_surface
        
        # Modify parent region(s)
        self.ablator_cell.region &= +foam_surface
        
        # Create cells
        # Pusher
        pusher_cell = openmc.Cell(
            name='pusher',
            region=pusher_region,
            fill=self.pusher_material
        )
        pusher_cell.density = pusher_density
        
        # Tamper
        tamper_cell = openmc.Cell(
            name='tamper',
            region=tamper_region,
            fill=self.tamper_material
        )
        tamper_cell.density = tamper_density
        
        # Foam
        foam_cell = openmc.Cell(
            name='foam',
            region=foam_region,
            fill=self.foam_material
        )
        foam_cell.density = foam_density
        
        self.add_cells([
            pusher_cell,
            tamper_cell,
            foam_cell
        ])

class CoronalUniverse(BaseImplosionUniverse):
    """Coronal source target geometry"""
    
    def __init__(
        self,
        ice_thickness_original: Optional[float] = None,
        ice_material: Optional[str] = None,
        hole_radius_original: float = 0.05,
        n_holes: Literal[1, 2] = 1,
        **kwargs
    ):
        """Initialize coronal source geometry
        
        Parameters:
        -----------
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
            Additional parameters for BaseImplosionUniverse
        """
        super().__init__(**kwargs)
        
        if self.convergence_ratio != 1.0:
            raise ValueError(f'Convergence ratio of coronal target cannot be {self.convergence_ratio}, as it is always 1 because they are not compressed.')
        
        self.ice_thickness_original = ice_thickness_original or 0.0
        self.ice_material = self.materials[ice_material]
        self.hole_radius_original = hole_radius_original
        self.n_holes = n_holes
        
        if self.hole_radius_original >= self.fuel_radius_original + self.ablator_thickness_original:
            raise ValueError("Hole radius must be less than capsule radius")
        
        self._create_geometry()
    
    def _create_geometry(self) -> None:
        """Create the coronal source geometry"""
        # Compress geometry
        ice_thickness, ice_density = self.compress_dimension(self.ice_thickness_original, self.ice_material)
        self.hole_radius, _ = self.compress_dimension(self.hole_radius_original) # density not needed for hole
        
        # Create surfaces
        self.capsule_radius = self.fuel_radius + self.ablator_thickness + ice_thickness
        # Laser entrance hole (plane)
        z_loc = -np.sqrt(self.capsule_radius**2 - self.hole_radius**2) # negative z location to point hole upwards
        hole1_plane = openmc.ZPlane(z0=z_loc, boundary_type='vacuum')
        
        # Regions
        self.fuel_cell.region &= +hole1_plane
        self.ablator_cell.region &= +hole1_plane
        
        # Redefine outer_region
        self.outer_region &= +hole1_plane
        
        # Recalculate the volume of fuel cell
        cap_height = self.fuel_radius + z_loc
        cap_volume = np.pi / 3 * cap_height**2 * (3 * self.fuel_radius - cap_height)
        self.fuel_cell.volume = 4 / 3 * np.pi * self.fuel_radius**3 - cap_volume
        
        ### OPTIONAL FEATURES ###           
        if ice_thickness > 0.0 and self.ice_material:
            # Construct geometry
            ice_outer_radius = self.fuel_radius + ice_thickness
            ice_outer_surface = openmc.Sphere(r=ice_outer_radius)
            ice_region = -ice_outer_surface & ~self.fuel_cell.region & +hole1_plane
            ice_cell = openmc.Cell(
                name='ice',
                region=ice_region,
                fill=self.ice_material
            )
            ice_cell.density = ice_density
            self.add_cell(ice_cell)
            
            # Remove ice from other regions
            self.ablator_cell.region &= +ice_outer_surface
                    
        if self.n_holes == 2:
            hole2_plane = openmc.ZPlane(z0=-z_loc, boundary_type='vacuum')
            
            self.fuel_cell.region &= -hole2_plane
            self.ablator_cell.region &= -hole2_plane
            if ice_thickness > 0.0:
                ice_cell.region &= -hole2_plane
                
            # Add constraint to outer region
            self.outer_region &= -hole2_plane
            
            # Subtract second hole volume from fuel cell
            self.fuel_cell.volume -= cap_volume
            
class NuclearInteractionVesselUniverse(BaseImplosionUniverse):
    def __init__(
        self,
        cone_length: float = 7.5,
        small_diameter: float = 0.9,
        large_diameter: float = 4.0,
        wall_thickness: float = 0.1,
        distance_from_source: float = 9.0,
        niv_wall_material: str = 'aluminum',
        niv_fill_thickness: Optional[Union[float, Sequence[float]]] = None,
        niv_fill_material: Optional[Union[str, Sequence[str]]] = None,
        tally_nuclides: Optional[Union[str, Sequence[str]]] = None,
        tally_reactions: Optional[Union[str, Sequence[str]]] = None,
        no_shielding: bool = False,
        **kwargs):
        """
        Initialize nuclear interaction vessel geometry. Details of NIV can be found here https://doi.org/10.1016/j.nima.2018.01.072
        
        Parameters:
        -----------
        cone_length : float
            Length of cone in cm
        small_diameter : float
            Small diameter of cone in cm
        large_diameter : float
            Large diameter of cone in cm
        wall_thickness : float
            Wall thickness in cm
        distance_from_source : float
            Distance from source in cm
        niv_wall_material : str
            Name of niv material
        niv_fill_thickness : float, optional
            Thickness(es) of niv fill, should sum to `cone_length - 2 * wall_thickness` if provided
        niv_fill_material : str, optional
            Name of niv fill material(s)
        tally_nuclides : str, optional
            Nuclide to tally
        tally_reactions : str, optional
            Reaction(s) to tally, corresponding in order to `tally_nuclides`
        no_shielding : bool
            If True, set fill to None for any material that does not contain a tally nuclide
        **kwargs
            Additional parameters for BaseImplosionUniverse
        """
        super().__init__(**kwargs)
        
        self.cone_length = cone_length
        self.small_radius = small_diameter / 2
        self.large_radius = large_diameter / 2
        self.wall_thickness = wall_thickness
        self.distance_from_source = distance_from_source
        self.niv_wall_material = self.materials[niv_wall_material]
        
        # Find solid angle of cone for later use
        self.cos_cone_angle = (self.distance_from_source + self.cone_length) / np.sqrt((self.distance_from_source + self.cone_length)**2 + self.large_radius**2)
        self.solid_angle = 2 * np.pi * (1 - self.cos_cone_angle)
        
        # Handle if niv fill is iterable vs not
        if isinstance(niv_fill_thickness, float) and isinstance(niv_fill_material, str):
            self.niv_fill_thickness = [niv_fill_thickness]
            self.niv_fill_material = [self.materials[niv_fill_material]]
        elif isinstance(niv_fill_thickness, Sequence) and isinstance(niv_fill_material, Sequence) and not isinstance(niv_fill_material, str):
            if len(niv_fill_thickness) != len(niv_fill_thickness):
                raise ValueError("Fill thicknesses and materials must be the same length")
            self.niv_fill_thickness = niv_fill_thickness
            self.niv_fill_material = [self.materials[mat] for mat in niv_fill_material]
        elif niv_fill_thickness is None and niv_fill_material is None:
            self.niv_fill_thickness = niv_fill_thickness
            self.niv_fill_material = niv_fill_material
        else:
            raise ValueError("Fill thickness and material must be the same type")
        
        # Handle if nuclide and reactions are iterable vs not
        if isinstance(tally_nuclides, str) and isinstance(tally_reactions, str):
            self.tally_nuclides = [tally_nuclides]
            self.tally_reactions = [tally_reactions]
        elif isinstance(tally_nuclides, Sequence) and isinstance(tally_reactions, Sequence) and not isinstance(tally_nuclides, str):
            if len(tally_nuclides) != len(tally_reactions):
                raise ValueError("Tally nuclides and reactions must be the same length")
            self.tally_nuclides = tally_nuclides
            self.tally_reactions = tally_reactions
        elif tally_nuclides is None and tally_reactions is None:
            self.tally_nuclides = tally_nuclides
            self.tally_reactions = tally_reactions
        else:
            raise ValueError("Tally nuclide and reactions must be the same type")

        self.no_shielding = no_shielding
        
        # If total thickness is not equal to cone length - 2 * wall_thickness, raise error
        if self.niv_fill_thickness:
            total_fill_thickness = np.sum(self.niv_fill_thickness)
            expected_thickness = self.cone_length - 2 * self.wall_thickness
            if total_fill_thickness != expected_thickness:
                raise ValueError(f"Total fill thickness {total_fill_thickness} does not equal expected thickness {expected_thickness}")
        
        self.remove_vacuum_boundaries()
        self._create_geometry()
        
    def _create_geometry(self) -> None:
        # Create conical surfaces
        slope = (self.large_radius - self.small_radius) / self.cone_length
        apex_outer = self.distance_from_source - self.small_radius / slope
        apex_inner = apex_outer + self.wall_thickness / (slope / np.sqrt(1 + slope**2))  # Trig to find inner apex location
        niv_outer_cone = openmc.XCone(x0=apex_outer, r2=slope**2)
        niv_inner_cone = openmc.XCone(x0=apex_inner, r2=slope**2)
        outer_bottom_plane = openmc.XPlane(x0=self.distance_from_source)
        outer_top_plane = openmc.XPlane(x0=self.distance_from_source + self.cone_length)
        
        # Create niv_fill stack
        # Create planes
        self.niv_fill_planes = [
            openmc.XPlane(x0=self.distance_from_source + self.wall_thickness), # start at beginning of fill region
            openmc.XPlane(x0=self.distance_from_source + self.cone_length - self.wall_thickness) # end at end of fill region
        ]
        
        # Need to also define perpendicular planes because bounding_box cannot be defined automatically for cone surfaces
        outer_ymin_plane = openmc.YPlane(y0=-self.large_radius)
        outer_ymax_plane = openmc.YPlane(y0=self.large_radius)
        outer_zmin_plane = openmc.ZPlane(z0=-self.large_radius)
        outer_zmax_plane = openmc.ZPlane(z0=self.large_radius)
        perpendicular_boundary_region = +outer_ymin_plane & -outer_ymax_plane & +outer_zmin_plane & -outer_zmax_plane
        
        total_niv_fill_region = -niv_inner_cone & +self.niv_fill_planes[0] & -self.niv_fill_planes[-1] & perpendicular_boundary_region
        
        # Add niv_fill planes
        if self.niv_fill_material and self.niv_fill_thickness:
            for i in range(len(self.niv_fill_thickness) - 1):
                self.niv_fill_planes.insert(
                    i + 1,
                    openmc.XPlane(self.distance_from_source + self.wall_thickness + np.sum(self.niv_fill_thickness[:i+1]))
                )
                
            # Create niv_fill cells
            self.tally_cells = []
            for i, material in enumerate(self.niv_fill_material):
                niv_fill_region = -niv_inner_cone & +self.niv_fill_planes[i] & -self.niv_fill_planes[i+1] & perpendicular_boundary_region

                # Check if this material contains a tally nuclide
                contains_tally_nuclide = self.tally_nuclides and material and any(nuclide in material.get_nuclides() for nuclide in self.tally_nuclides)

                # If no_shielding is True, only fill with material if it contains a tally nuclide
                if self.no_shielding and not contains_tally_nuclide:
                    fill_material = None
                else:
                    fill_material = material

                niv_fill_cell = openmc.Cell(
                    name=f'niv_fill_cell_{i}',
                    region=niv_fill_region,
                    fill=fill_material
                )
                self.add_cell(niv_fill_cell)
                
                # Set volume of cell
                r1 = slope * (self.niv_fill_planes[i].x0 - apex_inner)
                r2 = slope * (self.niv_fill_planes[i+1].x0 - apex_inner)
                niv_fill_cell.volume = (1/3) * np.pi * self.niv_fill_thickness[i] * (r1**2 + r1 * r2 + r2**2)

                # Store the cell for tallying if it contains a tally nuclide
                if contains_tally_nuclide:
                    self.tally_cells.append(niv_fill_cell)
        
        # Create wall region
        niv_outer_region = -niv_outer_cone & +outer_bottom_plane & -outer_top_plane & perpendicular_boundary_region
        niv_wall_region = niv_outer_region & ~total_niv_fill_region
        
        # Extend vacuum boundary slightly
        epsilon = 1e-3
        vacuum_box = BOX(
            xmin=-(self.fuel_radius + self.ablator_thickness + epsilon),
            xmax=self.distance_from_source + self.cone_length + epsilon,
            ymin=-(self.large_radius + epsilon),
            ymax=self.large_radius + epsilon,
            zmin=-(self.large_radius + epsilon),
            zmax=self.large_radius + epsilon,
            boundary_type='vacuum'
        )
        vacuum_region = -vacuum_box & ~self.fuel_cell.region & ~niv_outer_region
        
        # Redefine outer region
        self.outer_region |= niv_outer_region
        
        # Cells
        niv_wall_cell = openmc.Cell(
            name='niv_wall_cell',
            region=niv_wall_region,
            fill=self.niv_wall_material,
        )
        self.add_cell(niv_wall_cell)
        
        vacuum_cell = openmc.Cell(
            name='vacuum_cell',
            region=vacuum_region,
            fill=None,
        )
        self.add_cell(vacuum_cell)
        
        # Add back surface of NIV to tally, use niv_wall_cell for partial current tally
        self.tally_partial_current_surfaces = (outer_top_plane, niv_wall_cell)
                
class DualSourceUniverse(openmc.Universe):
    def __init__(
        self,
        primary_geom: BaseImplosionUniverse,
        secondary_geom: BaseImplosionUniverse,
        materials: Optional[NIFMaterials] = None,
        center_distance: float = 0.5,
        moderator_radius: float = 1.0,
        moderator_thickness: Union[float, Sequence[float]] = 0.1,
        moderator_material: Union[str, Sequence[str]] = 'ch2',
        moderator_distance: Optional[float] = None,
        secondary_orientation: Literal['parallel', 'perpendicular'] = 'parallel',
        no_shielding: bool = False,
        **kwargs
    ):
        """
        Initialize dual source geometry. Geometry is centered on the moderator in z.

        Parameters:
        -----------
        primary_geom : BaseImplosionUniverse
            Primary source geometry
        secondary_geom : BaseImplosionUniverse
            Secondary source geometry
        materials : NIFMaterials, optional
            Material database
        center_distance : float
            Distance between source centers in cm
        moderator_radius : float
            Moderator diameter in cm
        moderator_thickness : float, Sequence[float]
            Moderator thickness in cm
        moderator_material : str, Sequence[str]
            Moderator material name
        moderator_distance : float, optional
            Distance between moderator center and primary source center in cm
        secondary_orientation : str, optional
            Orientation of secondary source, one of ['parallel', 'perpendicular']
        no_shielding : bool
            If True, set all moderator materials to None (vacuum)
        """
        
        super().__init__(**kwargs)
        self.primary_geom = primary_geom
        self.secondary_geom = secondary_geom
        self.materials = materials or NIFMaterials()
        self.center_distance = center_distance
        self.moderator_radius = moderator_radius
        self.moderator_distance = moderator_distance or center_distance / 2 # default to midway between sources
        self.secondary_orientation = secondary_orientation
        self.no_shielding = no_shielding
        
        # Handle if moderator is iterable vs not
        if isinstance(moderator_thickness, float) and isinstance(moderator_material, str):
            self.moderator_thickness = [moderator_thickness]
            self.moderator_material = [self.materials[moderator_material]]
        elif isinstance(moderator_thickness, Sequence) and isinstance(moderator_material, Sequence) and not isinstance(moderator_material, str):
            if len(moderator_thickness) != len(moderator_material):
                raise ValueError("Moderator thickness and material must be the same length")
            self.moderator_thickness = moderator_thickness
            self.moderator_material = [self.materials[mat] for mat in moderator_material]
        else:
            raise ValueError("Moderator thickness and material must be the same type")

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
        # Define axes based on parallel or perpendicular orientation
        if self.secondary_orientation == 'parallel':
            axis = 'z'
            axis_int = 2
            self.axis_vector = np.array([0, 0, 1])
            rotation = np.array([180, 0, 0])
            self.moderator_cyl = openmc.ZCylinder(r=self.moderator_radius)
        elif self.secondary_orientation == 'perpendicular':
            axis = 'x'
            axis_int = 0
            self.axis_vector = np.array([1, 0, 0])
            rotation = np.array([0, -90, 0])
            self.moderator_cyl = openmc.XCylinder(r=self.moderator_radius)
        else:
            raise ValueError("secondary_orientation must be one of ['parallel', 'perpendicular']")
        
        # Translation of primary and secondary regions
        self.primary_translation = -self.moderator_distance
        self.secondary_translation = self.center_distance - self.moderator_distance
        
        # Use bounding boxes to find the extent of the geometry
        primary_min_pos = self.primary_geom.bounding_box.lower_left[axis_int] + self.primary_translation
        self.primary_max_pos = self.primary_geom.bounding_box.upper_right[axis_int] + self.primary_translation
        # The secondary geometry is located at the top of the moderator
        secondary_max_pos = self.secondary_geom.bounding_box.upper_right[axis_int] + self.secondary_translation
        self.secondary_min_pos = self.secondary_geom.bounding_box.lower_left[axis_int] + self.secondary_translation
        # Add a bit of padding to all sides
        vacuum_rcc = RCC(
            center_base=self.axis_vector * primary_min_pos * 1.1,
            radius=self.moderator_radius * 1.1,
            height=(secondary_max_pos - primary_min_pos) * 1.1,
            axis=axis,
            boundary_type='vacuum'
        )
        
        ### Regions ###
        primary_geom_region = self.primary_geom.get_outer_region()
        secondary_geom_region = self.secondary_geom.get_outer_region()
        
        # Rotate secondary geometry by 180 degrees to make a mirror
        self.transformed_secondary_geom_region = secondary_geom_region.rotate(rotation)
        # Need to rotate primary in opposite direction if secondary is perpendicular and coronal
        if self.secondary_orientation == 'perpendicular' and isinstance(self.primary_geom, CoronalUniverse):
            primary_geom_region = primary_geom_region.rotate(-rotation)
        
        # Translate source regions into position
        self.transformed_primary_geom_region = primary_geom_region.translate(self.axis_vector * self.primary_translation)
        self.transformed_secondary_geom_region = self.transformed_secondary_geom_region.translate(self.axis_vector * self.secondary_translation)
        
        # Create moderator stack
        total_thickness = np.sum(self.moderator_thickness)
        # Create planes
        self.moderator_planes = [
            openmc.Plane(
                *tuple(self.axis_vector),
                -total_thickness / 2
            )
        ]
        for i in range(len(self.moderator_thickness)):
            self.moderator_planes.append(
                openmc.Plane(
                    *tuple(self.axis_vector),
                    -total_thickness / 2 + np.sum(self.moderator_thickness[:i+1])
                )
            )
            
        # Create moderator cells
        for i, material in enumerate(self.moderator_material):
            moderator_region = -self.moderator_cyl & +self.moderator_planes[i] & -self.moderator_planes[i+1]
            # Exclude the sources from the moderator if it overlaps
            if self.primary_max_pos > self.moderator_planes[i].d:
                moderator_region &= ~self.transformed_primary_geom_region
            if self.secondary_min_pos < self.moderator_planes[i+1].d:
                moderator_region &= ~self.transformed_secondary_geom_region
            moderator_cell = openmc.Cell(
                name=f'moderator_cell_{i}',
                region=moderator_region,
                fill=None if self.no_shielding else material
            )
            self.add_cell(moderator_cell)
            
        total_moderator_region = -self.moderator_cyl & +self.moderator_planes[0] & -self.moderator_planes[-1]
        
        # The primary and secondary source could interset the moderator region
        vacuum_region = -vacuum_rcc & ~self.transformed_primary_geom_region & ~self.transformed_secondary_geom_region & ~total_moderator_region
        
        self.outer_region = self.transformed_primary_geom_region | self.transformed_secondary_geom_region | total_moderator_region
        
        ### Cells ###
        primary_geom_cell = openmc.Cell(
            name='primary_geom_cell',
            fill=self.primary_geom,
            region=self.transformed_primary_geom_region
        )
        
        # Need to rotate primary in opposite direction if secondary is perpendicular and coronal
        if self.secondary_orientation == 'perpendicular' and isinstance(self.primary_geom, CoronalUniverse):
            primary_geom_cell.rotation = -rotation
        # Translate primary source cell into position
        primary_geom_cell.translation = self.axis_vector * self.primary_translation
        self.add_cell(primary_geom_cell)
        
        secondary_geom_cell = openmc.Cell(
            name='secondary_geom_cell',
            fill=self.secondary_geom,
            region=self.transformed_secondary_geom_region
        )
        # Rotate and translate secondary source cell into position
        secondary_geom_cell.rotation = rotation
        secondary_geom_cell.translation = self.axis_vector * self.secondary_translation
        self.add_cell(secondary_geom_cell)
        
        vacuum_cell = openmc.Cell(
            name='vacuum_cell',
            fill=None,
            region=vacuum_region
        )
        self.add_cell(vacuum_cell)
    
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
        layered_moderator_material: Optional[Union[str, Sequence[str]]] = None,
        layered_moderator_thickness: Optional[Union[float, Sequence[float]]] = None,
        layered_moderator_primary_gap: Optional[float] = None,
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
        layered_moderator_material: str | Sequence[str], optional
            Name of neutron layered moderator material
        layered_moderator_thickness: float | Sequence[float], optional
            Thickness of neutron layered moderator
        layered_moderator_primary_gap: float, optional
            Gap between the edge of the primary source and the start of the layered moderator
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
        self.hohlraum_wall_thickness = hohlraum_wall_thickness
        self.layered_moderator_primary_gap = layered_moderator_primary_gap
        self.hohlraum_lining_thickness = hohlraum_lining_thickness
        
        # Calculate center distance based on source size and gap
        self.primary_radius = primary_coronal.capsule_radius
        secondary_radius = secondary_coronal.capsule_radius
        center_distance = self.primary_radius + secondary_radius + source_gap
        
        # Calculate moderator thickness based on distance between flats of LEH
        primary_LEH_z = np.sqrt(self.primary_radius**2 - primary_coronal.hole_radius**2)
        secondary_LEH_z = np.sqrt(secondary_radius**2 - secondary_coronal.hole_radius**2)
        self.fill_length = primary_LEH_z + secondary_LEH_z + center_distance
        
        # Initialize parent class
        # We'll override _create_geometry to build our specialized version
        super().__init__(
            primary_geom=primary_coronal,
            secondary_geom=secondary_coronal,
            center_distance=center_distance,
            moderator_thickness=self.fill_length,
            moderator_radius=hohlraum_inner_radius,
            moderator_material=fill_material,
            **kwargs
        )
        
        # Assign materials
        self.hohlraum_material = self.materials[hohlraum_material]
        self.hohlraum_lining_material = self.materials[hohlraum_lining_material]
        
        # Handle if layered moderator is iterable vs not
        if layered_moderator_material and layered_moderator_thickness:
            if isinstance(layered_moderator_thickness, float) and isinstance(layered_moderator_material, str):
                self.layered_moderator_thickness = [layered_moderator_thickness]
                self.layered_moderator_material = [self.materials[layered_moderator_material]]
            elif isinstance(layered_moderator_thickness, Sequence) and isinstance(layered_moderator_material, Sequence) and not isinstance(layered_moderator_material, str):
                if len(layered_moderator_thickness) != len(layered_moderator_material):
                    raise ValueError("Moderator thickness and material must be the same length")
                self.layered_moderator_thickness = layered_moderator_thickness
                self.layered_moderator_material = [self.materials[mat] for mat in layered_moderator_material]
            else:
                raise ValueError("Moderator thickness and material must be the same type")
        
        # Remove vacuum boundary conditions from parent classes
        self.primary_coronal.remove_vacuum_boundaries()
        self.secondary_coronal.remove_vacuum_boundaries()

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
        primary_LEH_cylinder = openmc.ZCylinder(r=self.primary_coronal.hole_radius)
        secondary_LEH_cylinder = openmc.ZCylinder(r=self.secondary_coronal.hole_radius)
        hohlraum_bottom = openmc.ZPlane(z0=-(self.fill_length/2 + self.hohlraum_wall_thickness))
        hohlraum_top = openmc.ZPlane(z0=self.fill_length/2 + self.hohlraum_wall_thickness)
        moderator_bottom = self.moderator_planes[0]
        moderator_top = self.moderator_planes[-1]
        
        # Vacuum is slightly larger than hohlraum to avoid overlap
        vacuum_center_base = (0, 0, -(self.fill_length/2 + self.hohlraum_wall_thickness) * 1.1)
        vacuum_radius = (self.hohlraum_inner_radius + self.hohlraum_wall_thickness) * 1.1
        vacuum_height = (self.fill_length + 2 * self.hohlraum_wall_thickness) * 1.1
        
        # Define hohlraum region
        hohlraum_bottom_LEH_region = -hohlraum_outer_cylinder & +primary_LEH_cylinder & +hohlraum_bottom & -moderator_bottom
        hohlraum_top_LEH_region = -hohlraum_outer_cylinder & +secondary_LEH_cylinder & +moderator_top & -hohlraum_top
        hohlraum_wall_region = -hohlraum_outer_cylinder & +self.moderator_cyl & +moderator_bottom & -moderator_top
        hohlraum_region = hohlraum_bottom_LEH_region | hohlraum_top_LEH_region | hohlraum_wall_region
        
        # Define outer region
        self.outer_region = hohlraum_region | (-self.moderator_cyl & +moderator_bottom & -moderator_top)
        
        # Create cells
        hohlraum_cell = openmc.Cell(
            name='hohlraum',
            region=hohlraum_region,
            fill=self.hohlraum_material
        )
        self.add_cell(hohlraum_cell)
        
        # Add lining
        if self.hohlraum_lining_material and self.hohlraum_lining_thickness:
            # Remove vacuum boundary from hohlraum outer surface
            self.primary_coronal.remove_vacuum_boundaries()
            self.secondary_coronal.remove_vacuum_boundaries()
            
            hohlraum_lining_outer_cylinder = openmc.ZCylinder(r=self.hohlraum_inner_radius + self.hohlraum_wall_thickness + self.hohlraum_lining_thickness)
            hohlraum_lining_bottom = openmc.ZPlane(z0=-(self.fill_length/2 + self.hohlraum_wall_thickness + self.hohlraum_lining_thickness))
            hohlraum_lining_top = openmc.ZPlane(z0=self.fill_length/2 + self.hohlraum_wall_thickness + self.hohlraum_lining_thickness)
            
            # Define lining region, excluding LEH regions
            hohlraum_lining_bottom_LEH_region = -hohlraum_lining_outer_cylinder & +primary_LEH_cylinder & +hohlraum_lining_bottom & -hohlraum_bottom
            hohlraum_lining_top_LEH_region = -hohlraum_lining_outer_cylinder & +secondary_LEH_cylinder & +hohlraum_top & -hohlraum_lining_top
            hohlraum_lining_wall_region = -hohlraum_lining_outer_cylinder & +hohlraum_outer_cylinder & +hohlraum_lining_bottom & -hohlraum_lining_top
            hohlraum_lining_region = hohlraum_lining_bottom_LEH_region | hohlraum_lining_top_LEH_region | hohlraum_lining_wall_region
            
            # Redefine vacuum region to exclude lining
            vacuum_center_base = (0, 0, -(self.fill_length/2 + self.hohlraum_wall_thickness + self.hohlraum_lining_thickness) * 1.1)
            vacuum_radius = (self.hohlraum_inner_radius + self.hohlraum_wall_thickness + self.hohlraum_lining_thickness) * 1.1
            vacuum_height = (self.fill_length + 2 * (self.hohlraum_wall_thickness + self.hohlraum_lining_thickness)) * 1.1
            
            # Create hohlraum lining cell
            hohlraum_lining_cell = openmc.Cell(
                name='hohlraum_lining',
                region=hohlraum_lining_region,
                fill=self.hohlraum_lining_material
            )
            self.add_cell(hohlraum_lining_cell)
            
            # Redefine outer region
            self.outer_region = -hohlraum_lining_outer_cylinder & +hohlraum_lining_bottom & -hohlraum_lining_top
        
        # Create layered moderator   
        if self.layered_moderator_material and self.layered_moderator_thickness and self.layered_moderator_primary_gap:
            # Create planes
            start_z = self.primary_translation + self.primary_radius + self.layered_moderator_primary_gap
            layered_moderator_planes = [openmc.ZPlane(start_z)]
            self.layered_moderator_thickness
            for i in range(len(self.layered_moderator_thickness)):
                layered_moderator_planes.append(
                    openmc.ZPlane(start_z + np.sum(self.layered_moderator_thickness[:i+1]))
                )
            
            # Create moderator cells
            for i, material in enumerate(self.layered_moderator_material):
                moderator_region = -self.moderator_cyl & +layered_moderator_planes[i] & -layered_moderator_planes[i+1]
                # Exclude the sources from the moderator if it overlaps
                if self.primary_max_pos > layered_moderator_planes[i].d:
                    moderator_region &= ~self.transformed_primary_geom_region
                if self.secondary_min_pos < layered_moderator_planes[i+1].d:
                    moderator_region &= ~self.transformed_secondary_geom_region
                moderator_cell = openmc.Cell(
                    name=f'layered_moderator_cell_{i}',
                    region=moderator_region,
                    fill=material
                )
                self.add_cell(moderator_cell)
                
            total_layered_moderator_region = -self.moderator_cyl & +layered_moderator_planes[0] & -layered_moderator_planes[-1]
            
            # Exclude layered moderator from other regions
            for cell in self.cells.values():
                # Exclude layered moderator cells, only interested in primary hohlraum_fill_moderator
                if 'moderator' in cell.name and 'layered' not in cell.name:
                    cell.region &= ~total_layered_moderator_region
            hohlraum_cell.region &= ~total_layered_moderator_region
            
        # Create vacuum cell
        vacuum_rcc = RCC(
            center_base=vacuum_center_base,
            radius=vacuum_radius,
            height=vacuum_height,
            axis='z',
            boundary_type='vacuum'
        )
        vacuum_region = -vacuum_rcc & ~hohlraum_region & ~(-self.moderator_cyl & +moderator_bottom & -moderator_top)
        vacuum_cell = openmc.Cell(
            name='vacuum_cell',
            region=vacuum_region
        )
        self.add_cell(vacuum_cell)
    
class DualIndirectCoronal(DualSourceUniverse):
    """
    Specialized dual source universe for a standard hohlraum implosion as the primary source
    and a inverted coronal as the secondary source.
    """
    def __init__(
        self,
        primary_hohlraum: IndirectDriveUniverse,
        secondary_coronal: CoronalUniverse,
        moderator_thickness: Union[float, Sequence[float]] = 0.6,
        moderator_material: Union[str, Sequence[str]] = 'ch2',
        reflector_thickness: float = 0.03,
        reflector_material: str = 'gold',
        **kwargs
    ):
        """
        Initialize dual hohlraum coronal source geometry
        
        Parameters:
        -----------
        primary_hohlraum : IndirectDriveUniverse
            Primary hohlraum source geometry
        secondary_coronal : CoronalUniverse
            Secondary coronal source geometry
        moderator_thickness : float, Sequence[float]
            Moderator thickness in cm
        moderator_material : str, Sequence[str]
            Moderator material name
        reflector_thickness : float
            Reflector thickness in cm
        reflector_material : str
            Reflector material name
        """
        self.primary_hohlraum = primary_hohlraum
        self.secondary_coronal = secondary_coronal
        if isinstance(moderator_thickness, float):
            self.total_thickness = moderator_thickness
        else:
            self.total_thickness = np.sum(moderator_thickness)
        self.reflector_thickness = reflector_thickness
        
        # Calculate distances
        secondary_LEH_z = np.sqrt(secondary_coronal.capsule_radius**2 - secondary_coronal.hole_radius**2)
        center_distance = primary_hohlraum.hohlraum_radius + self.total_thickness - secondary_LEH_z
        moderator_distance = primary_hohlraum.hohlraum_radius + self.total_thickness/2
        
        # Initialize parent class
        # TODO: Am I getting the compression ratio correct?
        super().__init__(
            primary_geom=primary_hohlraum,
            secondary_geom=secondary_coronal,
            center_distance=center_distance,
            moderator_thickness=moderator_thickness,
            moderator_distance=moderator_distance,
            moderator_radius=primary_hohlraum.hohlraum_length / 2,
            moderator_material=moderator_material,
            secondary_orientation='perpendicular',
            **kwargs
        )
        
        # Need to wait for parent init to complete to access materials
        self.reflector_material = self.materials[reflector_material]
        
        # Remove vacuum boundary conditions from parent class
        self.primary_hohlraum.remove_vacuum_boundaries()
        self.secondary_coronal.remove_vacuum_boundaries()

        self._create_dual_indirect_coronal_geometry()
        
    def _create_dual_indirect_coronal_geometry(self):
        # The parent __init__ already handled, need to remove vacuum cell
        for cell in self.cells.values():
            if cell.name == 'vacuum_cell':
                self.remove_cell(cell)
                break
            
        # Create reflector region around the moderator
        reflector_radius = self.moderator_radius + self.reflector_thickness
        reflector_outer_cylinder = openmc.XCylinder(r=reflector_radius)
        reflector_top = self.moderator_planes[0]
        reflector_bottom_inside = self.moderator_planes[-1]
        reflector_bottom_outside = openmc.XPlane(self.total_thickness / 2 + self.reflector_thickness)
        secondary_LEH_cylinder = openmc.XCylinder(r=self.secondary_coronal.hole_radius)
        
        reflector_wall_region = -reflector_outer_cylinder & +self.moderator_cyl & +reflector_top & -reflector_bottom_inside
        reflector_base_region = -reflector_outer_cylinder & +secondary_LEH_cylinder & +reflector_bottom_inside & -reflector_bottom_outside
        reflector_region = reflector_wall_region | reflector_base_region
        
        # Modify outer region to include reflector
        self.outer_region |= reflector_region
        
        # Add reflector cell
        reflector_cell = openmc.Cell(
            name='reflector_cell',
            region=reflector_region,
            fill=self.reflector_material
        )
        self.add_cell(reflector_cell)
            
        # Vacuum cell
        # slightly larger than geometry to account for rounding errors
        multiplier = 1.1
        vacuum_box = BOX(
            xmin=(self.primary_translation - self.primary_hohlraum.hohlraum_radius) * multiplier,
            xmax=(self.secondary_translation + self.secondary_coronal.capsule_radius + self.reflector_thickness) * multiplier,
            ymin=-reflector_radius * multiplier,
            ymax=reflector_radius * multiplier,
            zmin=-reflector_radius * multiplier,
            zmax=reflector_radius * multiplier,
            boundary_type='vacuum'
        )
        
        vacuum_region = -vacuum_box & ~self.outer_region
        vacuum_cell = openmc.Cell(
            name='vacuum_cell',
            region=vacuum_region,
            fill=None
        )
        self.add_cell(vacuum_cell)