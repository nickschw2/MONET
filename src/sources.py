import openmc
import openmc.stats
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class SphericalSource(openmc.IndependentSource):
    """Neutron source with Muir energy distribution for DT fusion"""
    
    def __init__(
        self,
        fuel_radius: Optional[float] = None,
        fuel_fraction: float = 0.5,
        plasma_temperature: float = 10.0e3,
        pulse_fwhm: Optional[float] = None,
        origin: Optional[Tuple[float, float, float]] = None,
        **kwargs
    ):
        """Initialize Muir distribution source
        
        Parameters:
        -----------
        fuel_radius : float, optional
            Radius of fuel region in cm, otherwise point souce
        fuel_fraction : float
            Fraction of tritium in DT fuel
        plasma_temperature : float
            Plasma temperature in eV
        pulse_fwhm : float, optional
            FWHM of gaussian pulse in seconds, otherwise use delta function
        origin : Tuple[float, float, float], optional
            Origin of source in cm
        **kwargs : dict
            Additional arguments for openmc.IndependentSource
        """
        super().__init__(**kwargs)
        self.fuel_radius = fuel_radius
        self.fuel_fraction = fuel_fraction
        self.plasma_temperature = plasma_temperature
        self.pulse_fwhm = pulse_fwhm
        self.origin = origin or (0.0, 0.0, 0.0)
        
        self._setup_source()
    
    def _setup_source(self) -> None:
        """Set up the source spatial, angular, energy, and time distributions"""
        # Spatial distribution - uniform in fuel sphere
        # TODO: consider changing the radial distribution to a power law to simulate hotspot
        if self.fuel_radius:
            self.space = openmc.stats.SphericalIndependent(
                r=openmc.stats.Uniform(a=0.0, b=self.fuel_radius),
                cos_theta=openmc.stats.Uniform(a=-1, b=1),
                phi=openmc.stats.Uniform(a=0, b=2*np.pi),
                origin=self.origin
            )
        else:
            # Point source if radius is not provided
            self.space = openmc.stats.Point(xyz=self.origin)
        
        # Angular distribution - isotropic
        self.angle = openmc.stats.Isotropic()
        
        # Have two separate distributions for DD and DT neutrons
        dd_reactivity, dt_reactivity = self._get_reactivities()
        dd_probability = dd_reactivity / (dd_reactivity + dt_reactivity)
        dt_probability = 1.0 - dd_probability
        energy_dd = openmc.stats.muir(e0=2.45e6, m_rat=4.0, kt=self.plasma_temperature)
        energy_dt = openmc.stats.muir(e0=14.08e6, m_rat=5.0, kt=self.plasma_temperature)
        self.energy = openmc.stats.Mixture(
            [dd_probability, dt_probability],
            [energy_dd, energy_dt]
        )
        
        # Time distribution
        if self.pulse_fwhm:
            # Gaussian pulse
            sigma = self.pulse_fwhm / (2 * np.sqrt(2 * np.log(2)))
            # Pulse is centered at some amount away from t=0
            mean_time = 2 * sigma
            self.time = openmc.stats.Normal(mean_value=mean_time, std_dev=sigma)
            # Time bounds
            self.constraints['time_bounds'] = (0.0, mean_time + 2 * sigma)
        else:
            # Delta function at t=0
            self.time = openmc.stats.Discrete([0.0], [1.0])
            
        # self.plot_energy_spectrum()
        # self.plot_time_spectrum()
            
    def _get_reactivities(self) -> Tuple[float, float]:
        """
        Get reactivities for DD and DT neutrons
        
        Parameters and fitting function are taken from the D(d, n)3He and T(d, n)4He
        reactions fitting parameters from table VII and equations 12-14 of
        H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611. Assumes Maxwellian temperature.
        
        Returns:
        --------
        dd_reactivity : float
            Reactivity for DD neutrons
        dt_reactivity : float
            Reactivity for DT neutrons
        """
        
        def get_reactivity(fuel: str):
            if fuel == 'DD':
                B_G = 31.3970 # keV^(1/2)
                mrcc = 937814 # m_r*c^2, keV
                C1 = 5.43360E-12 # 1/keV
                C2 = 5.85778E-3 # 1/keV
                C3 = 7.68222E-3 # 1/keV
                C4 = 0 # 1/keV
                C5 = -2.96400E-6 # 1/keV
                C6 = 0 # 1/keV
                C7 = 0 # 1/keV
                delta_ij = 1 # For like-particles
                fuel_fraction_coefficient = (1 - self.fuel_fraction)**2
                
            elif fuel == 'DT':
                B_G = 34.3827 # keV^(1/2)
                mrcc = 1124656 # m_r*c^2, keV
                C1 = 1.17302e-9 # 1/keV
                C2 = 1.51361e-2 # 1/keV
                C3 = 7.51886e-2 # 1/keV
                C4 = 4.60643e-3 # 1/keV
                C5 = 1.35000e-2 # 1/keV
                C6 = -1.06750e-4 # 1/keV
                C7 = 1.36600e-5 # 1/keV
                delta_ij = 0 # For unlike-particles
                fuel_fraction_coefficient = self.fuel_fraction * (1 - self.fuel_fraction)
            
            Ti_keV = self.plasma_temperature / 1000
            if Ti_keV >= 0.2: # Cutoff for validity of formula
                theta = Ti_keV / (1 - Ti_keV * (C2 + Ti_keV * (C4 + Ti_keV * C6)) / (1 + Ti_keV * (C3 + Ti_keV * (C5 + Ti_keV * C7))))
                xi = (B_G**2 / (4*theta))**(1/3)
                sigma_v = C1 * theta * np.sqrt(xi / (mrcc * Ti_keV**3)) * np.exp(-3*xi)
            else:
                sigma_v = 0
                
            reactivity = sigma_v * fuel_fraction_coefficient / (1 + delta_ij)
                
            return reactivity
            
        dd_reactivity = get_reactivity('DD')
        dt_reactivity = get_reactivity('DT')
        
        return dd_reactivity, dt_reactivity
    
    def plot_energy_spectrum(self, num_samples: int = int(1e5), save_path: Optional[str] = None):
        """Plot the energy spectrum for the source"""
        # Randomly sample the energy distribution
        if isinstance(self.energy, openmc.stats.Mixture):
            energies = self.energy.sample(num_samples)
        else:
            raise NotImplementedError('Energy distribution has not been defined')
        
        # Plot the energy spectrum        
        fig, ax = plt.subplots(constrained_layout=True)
        
        ax.hist(energies / 1e6, bins=200, density=True, histtype='step', linewidth=3)
        ax.set_xlabel('Energy [MeV]')
        ax.set_ylabel('PDF')
        ax.set_yscale('log')
        # TODO: save to a better place and make plot prettier
        fig.savefig('./energy_spectrum.png')
        
    def plot_time_spectrum(self, num_samples: int = int(1e5), save_path: Optional[str] = None):
        """Plot the time spectrum for the source"""
        # Randomly sample the energy distribution
        if isinstance(self.time, openmc.stats.Normal):
            times = self.time.sample(num_samples)
        else:
            raise NotImplementedError('Time distribution has not been defined')
        
        # Plot the time spectrum        
        fig, ax = plt.subplots(constrained_layout=True)
        
        # Fix units
        if max(times) > 1e-6:
            multiplier = 1e6
            prefix = 'u'
        elif max(times) > 1e-9:
            multiplier = 1e9
            prefix = 'n'
        elif max(times) > 1e-12:
            multiplier = 1e12
            prefix = 'p'
        else:
            multiplier = 1
            prefix = ''
            
        
        # Plot constraints
        t_min, t_max = self.constraints['time_bounds']
        ax.axvline(t_min * multiplier, color='tab:red', linestyle='--', linewidth=3)
        ax.axvline(t_max * multiplier, color='tab:red', linestyle='--', linewidth=3)
        
        ax.hist(times * multiplier, bins=200, density=True, histtype='step', linewidth=3)
        ax.set_xlabel(f'Time [{prefix}s]')
        ax.set_ylabel('PDF')
        ax.set_yscale('log')
        # TODO: save to a better place and make plot prettier
        fig.savefig('./time_spectrum.png')