import openmc
from typing import Optional, Dict

class NIFMaterial(openmc.Material):
    def __init__(
        self,
        name: str,
        color: str,
        density: float,
        elements: list[str],
        fractions: list[float],
        fraction_type: str = 'ao',
        is_compressed: bool = False,
        **kwargs
    ):
        """
        Class to represent a single NIF material
        
        Parameters:
        -----------
        name: str
            Name of material
        color: str
            Color for visualization
        density: float
            Density of material in g/cm3
        elements: list[str]
            List of elements in material
        fractions: list[float]
            List of element fractions in material
        fraction_type: str
            Fraction type, defaults to 'ao'
        is_compressed: bool
            True if material has been compressed, defaults to False
        kwargs: dict
            Additional arguments for openmc.Material
        """
        super().__init__(**kwargs)
        self.color = color
        self.is_compressed = is_compressed
        
        # Create the material
        self.name = name
        self.set_density("g/cm3", density)
        for element, fraction in zip(elements, fractions):
            if any(char.isdigit() for char in element):
                # It's a nuclide
                self.add_nuclide(element, fraction, fraction_type)
            else:
                # It's an element
                self.add_element(element, fraction, fraction_type)
                
class FuelMaterial(NIFMaterial):
    def __init__(
        self,
        fuel_pressure: float = 10, # atm
        fuel_fraction: float = 0.5,
        trace_nuclide: str = 'Tm171',
        trace_concentration: float = 1e-3,
        dopant_nuclide: Optional[str] = None,
        dopant_concentration: float = 0.0,
        **kwargs
    ):
        """
        Create DT fuel with optional trace doping and other dopants
        
        Parameters:
        -----------
        fuel_pressure: float
            Pressure of unimploded DT fuel in atm
        fuel_fraction: float
            Fraction of tritium in DT fuel
        trace_nuclide: str
            Name of trace nuclide to add
        trace_concentration: float
            Atomic fraction of trace nuclide
        dopant_nuclide: str, optional
            Nuclide symbol for additional dopant
        dopant_concentration: float
            Atomic fraction of dopant nuclide
        **kwargs : dict
            Additional arguments for NIFMaterial
        """
        self.fuel_pressure = fuel_pressure
        self.fuel_fraction = fuel_fraction
        self.trace_nuclide = trace_nuclide
        self.trace_concentration = trace_concentration
        self.dopant_nuclide = dopant_nuclide
        self.dopant_concentration = dopant_concentration
        
        # Base DT composition (50:50 mix)
        dt_fraction = 1.0 - dopant_concentration - trace_concentration
        
        # Initialize elements and fractions
        elements = ['H2', 'H3']
        fractions = [dt_fraction * (1 - fuel_fraction), dt_fraction * fuel_fraction]
        
        # Add trace doping
        if trace_concentration > 0:
            elements.append(trace_nuclide)
            fractions.append(trace_concentration)
        
        # Add dopant if specified
        if dopant_nuclide and dopant_concentration > 0:
            elements.append(dopant_nuclide)
            fractions.append(dopant_concentration)
        
        room_temperature = 293  # K
        atm2Pa = 101325  # Pa
        fuel_mass = 2.5 # g/mol for DT
        R = 8.314 # J/(mol*K)
        fuel_density = (fuel_pressure * atm2Pa * fuel_mass) / (R * room_temperature) * 1e-6 # g/cm3
        
        super().__init__(
            name='dt_fuel',
            color='tomato',
            density=fuel_density,
            elements=elements,
            fractions=fractions
        )

class NIFMaterials(openmc.Materials):
    """Class to handle all material definitions for NIF simulations"""


    library = {
        "dt_ice": NIFMaterial(
            name='dt_ice', 
            color='lightcyan',
            elements=['H2', 'H3'],
            fractions=[1.0, 1.0], 
            density=1.85,
        ),
        
        "beryllium": NIFMaterial(
            name='beryllium', 
            color='deeppink',
            elements=['Be'],
            fractions=[1.0], 
            density=1.85,
        ),
        "glass": NIFMaterial(
            name='glass',
            color='lightblue',
            elements=['Si', 'O'],
            fractions=[1, 2],
            density=2.2
        ),
        "ch2": NIFMaterial(
            name='ch2', 
            color='seagreen',
            elements=['C', 'H'],
            fractions=[1, 2], 
            density=0.94,
        ),
        "ch": NIFMaterial(
            name='ch', 
            color='darkgreen',
            elements=['C', 'H'],
            fractions=[1, 1], 
            density=0.94,
        ),
        "diamond": NIFMaterial(
            name='diamond', 
            color='white',
            elements=['C'],
            fractions=[1], 
            density=3.52,
        ),
        "boron_carbide": NIFMaterial(
            name='b4c', 
            color='lightblue',
            elements=['B', 'C'],
            fractions=[4, 1], 
            density=2.52,
        ),
        "gold": NIFMaterial(
            name='gold', 
            color='gold',
            elements=['Au'],
            fractions=[1], 
            density=19.32,
        ),
        "tungsten": NIFMaterial(
            name='tungsten', 
            color='darkorange',
            elements=['W'],
            fractions=[1], 
            density=19.3,
        ),
        "uranium": NIFMaterial(
            name='uranium', 
            color='darkgreen',
            elements=['U'],
            fractions=[1], 
            density=19.1,
        ),
        "aluminum": NIFMaterial(
            name='aluminum', 
            color='blue',
            elements=['Al'],
            fractions=[1], 
            density=2.7,
        ),
        "water": NIFMaterial(
            name='water', 
            color='royalblue',
            elements=['H', 'O'],
            fractions=[2, 1], 
            density=1.0,
        ),
        "graphite": NIFMaterial(
            name='graphite', 
            color='darkgray',
            elements=['C'],
            fractions=[1], 
            density=2.2,
        ),
        "lead": NIFMaterial(
            name='lead', 
            color='lightgray',
            elements=['Pb'],
            fractions=[1], 
            density=11.35,
        )
    }

    def __init__(self, convergence_ratio: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.convergence_ratio = convergence_ratio
        
        # Add all materials from library to the collection
        for material in self.library.values():
            self.append(material)
            
    def create_dt_fuel(
        self,
        **kwargs
    ) -> FuelMaterial:
        """
        Create DT fuel with optional trace doping and other dopants
        
        Parameters:
        -----------
        **kwargs: dict
            Keyword arguments for FuelMaterial
        
        Returns:
        --------
        FuelMaterial
            DT fuel material
        """        
        fuel = FuelMaterial(**kwargs)
        
        # Add to library     
        self.library['dt_fuel'] = fuel
        self.append(fuel)
        
        return fuel
    
    def __getitem__(self, name: str) -> openmc.Material:
        """Get material by name"""
        if name not in self.library:
            if name == 'dt_fuel':
                raise ValueError("DT fuel material not found. Use create_dt_fuel() to create it.")
            raise ValueError(f"Material '{name}' is not found in the library. Available materials: {list(self.library.keys())}")
        
        return self.library[name]
    
    def get_colors(self) -> Dict[str, str]:
        """Get a color definition dict with materials as keys and colors as values."""
        colors = {}
        for material in self.library.values():
            colors[material] = material.color
        return colors
    
    def compress_density(self, name: str) -> Optional[float]:
        """Calculate compressed density based on convergence ratio
        
        Parameters:
        -----------
        name: str
            Name of material for which density should be compressed
            
        Returns:
        --------
        float or None
            The new mass density after compression, or None if material not found
        """
        if name not in self.library:
            raise ValueError(f"Material '{name}' not found for density compression. Available materials: {list(self.library.keys())}")
        
        material = self.library[name]
        
        if material.is_compressed:
            print(f"Warning: Material '{name}' has already been compressed.")
            return material.get_mass_density()
        
        if material.density:
            new_density = material.density * (self.convergence_ratio ** 3)
            material.set_density(material.density_units, new_density)
            material.is_compressed = True
            return material.get_mass_density()
        else:
            raise ValueError(f"Material '{name}' does not have density set.")
        
    def get_colors(self) -> dict:
        """Get a color definition dict with material names as keys and colors as values."""
        colors = {}
        for material in self.library.values():
            colors[material] = material.color
        return colors