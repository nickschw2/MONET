import openmc
from typing import Optional, Dict, Union

class NIFMaterial(openmc.Material):
    def __init__(
        self,
        name: str,
        color: str,
        density: float,
        elements: list[str],
        fractions: list[Union[float, Dict]],
        fraction_type: str = 'ao',
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
        kwargs: dict
            Additional arguments for openmc.Material
        """
        super().__init__(**kwargs)
        self.color = color
        
        # Create the material
        self.name = name
        self.set_density("g/cm3", density)
        for element, fraction in zip(elements, fractions):
            if any(char.isdigit() for char in element):
                if isinstance(fraction, dict):
                    raise ValueError("Nuclide fractions must be provided as floats.")
                # It's a nuclide
                self.add_nuclide(element, fraction, fraction_type)
            else:
                # It's an element
                if isinstance(fraction, dict):
                    self.add_element(
                        element,
                        fraction['percent'],
                        percent_type=fraction_type,
                        enrichment=fraction['enrichment'],
                        enrichment_target=fraction['enrichment_target'],
                        enrichment_type=fraction_type
                    )
                else:
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
            fractions=fractions,
            **kwargs
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
        'steel': NIFMaterial(
            name='steel', 
            color='gray',
            elements=['Fe', 'C', 'Cr', 'Ni', 'Mn', 'Si'],
            fractions=[0.70, 0.002, 0.18, 0.10, 0.02, 0.01], 
            density=7.85,
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
        ),
        "lithium-6": NIFMaterial(
            name='lithium-6', 
            color='cyan',
            elements=['Li6'],
            fractions=[1], 
            density=0.534,
        ),
        "flibe": NIFMaterial(
            name='flibe', 
            color='orange',
            elements=['Li', 'Be', 'F'],
            fractions=[
                {'percent': 2.0,
                'enrichment': 60.0,
                'enrichment_target': 'Li6'},
                1,
                4
            ], 
            density=1.94,
        ),
        "lithium_hydride": NIFMaterial(
            name='lithium_hydride', 
            color='lightgreen',
            elements=['Li', 'H'],
            fractions=[1, 1], 
            density=0.793,
        ),
        "zirconium_hydride": NIFMaterial(
            name='zirconium_hydride', 
            color='pink',
            elements=['Zr', 'H'],
            fractions=[1, 1.6], 
            density=5.6,
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)        
        # Add all materials from library to the collection
        for material in self.library.values():
            self.append(material)
            
    def create_dt_fuel(
        self,
        name: str = 'dt_fuel',
        **kwargs
    ) -> FuelMaterial:
        """
        Create DT fuel with optional trace doping and other dopants

        Parameters:
        -----------
        name: str
            Name for the fuel material in the library (default: 'dt_fuel').
            Use different names to create distinct fuel variants (e.g., 'dt_fuel_primary', 'dt_fuel_secondary').
        **kwargs: dict
            Keyword arguments for FuelMaterial

        Returns:
        --------
        FuelMaterial
            DT fuel material
        """
        fuel = FuelMaterial(**kwargs)
        fuel.name = name

        # Add to library
        self.library[name] = fuel
        self.append(fuel)

        return fuel
    
    def __getitem__(self, name: Optional[str]) -> Optional[openmc.Material]:
        """Get material by name"""
        if name is None:
            print("Warning: Material name is None, returning None")
            return None
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