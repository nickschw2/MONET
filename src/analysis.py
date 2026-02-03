import openmc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
import numpy as np
import json
from typing import Optional, List, Tuple, Union, Dict, Literal
from dataclasses import dataclass
from labellines import labelLines

from .materials import NIFMaterials

# Set plotting params
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 3

@dataclass
class NuclideData:
    """Data structure for nuclide tally information"""
    tally: pd.DataFrame
    reaction: str

class DataProcessor:
    """Class to process OpenMC simulation results"""
    
    def __init__(self, simulation_dir: Path | str):
        if isinstance(simulation_dir, str):
            self.simulation_dir = Path(simulation_dir)
        else:
            self.simulation_dir = simulation_dir
        
        # Load simulation parameters
        simulation_params_file = self.simulation_dir / 'simulation_params.json'
        if simulation_params_file.exists():
            with open(simulation_params_file, 'r') as f:
                self.simulation_params = json.load(f)
            batches = self.simulation_params['batches']
        else:
            raise FileNotFoundError(f"No simulation_params.json file found in {simulation_dir}")
        
        # Open the statepoint file
        statepoint_file = self.simulation_dir / f'statepoint.{batches}.h5'
        
        if statepoint_file.exists():
            self.sp = openmc.StatePoint(str(statepoint_file))
        else:
            raise FileNotFoundError(f"No statepoint file found in {simulation_dir}")
        
        # Load model.xml to get geometry
        model_file = self.simulation_dir / 'model.xml'
        
        if model_file.exists():
            model = openmc.Model.from_model_xml(model_file)
            self.geometry = model.geometry
        else:
            raise FileNotFoundError(f"No model.xml file found in {simulation_dir}")
        
    def get_tally_cell(self, id: int) -> openmc.Cell:
        """Get cell object from model"""
        if self.geometry:
            return self.geometry.get_all_cells()[id]
        else:
            raise RuntimeError("Geometry not found")
    
    def extract_mesh_data(self):
        """Extract mesh tally data"""
        tally = self.sp.get_tally(name='mesh_tally')
        mesh_tally_df = tally.get_pandas_dataframe()
        
        # Initialize new columns
        mesh_tally_df['x'] = None
        mesh_tally_df['y'] = None
        mesh_tally_df['z'] = None
        
        # Extract coordinates
        mesh_tally_df['x'] = mesh_tally_df[('mesh 1', 'x')]
        mesh_tally_df['y'] = mesh_tally_df[('mesh 1', 'y')]  
        mesh_tally_df['z'] = mesh_tally_df[('mesh 1', 'z')]

        # Drop original MultiIndex mesh column
        mesh_tally_df = mesh_tally_df.drop('mesh 1', axis=1, level=0)

        # Normalize by voxel volume
        mesh_filter = tally.find_filter(openmc.MeshFilter)
        if mesh_filter is None:
            print("No mesh filter found in mesh_tally")
            voxel_volume = 1.0  # cm^3
        else:
            # All volumes are the same, so just take the first instance
            voxel_volume = mesh_filter.mesh.volumes[0][0][0]

        # Normalize by volume
        mesh_tally_df['mean'] /= voxel_volume
        mesh_tally_df['std. dev.'] /= voxel_volume
        
        self.mesh_tally_df = mesh_tally_df
        self.mesh = mesh_filter.mesh
        
    def extract_cell_data(self) -> Dict[str, pd.DataFrame]:
        # Initialize cell tally dataframe
        tally = self.sp.get_tally(name='cell_tally')
        cell_tallies_df = tally.get_pandas_dataframe()
        
        # Loop through all bins
        cell_tallies: Dict[str, pd.DataFrame] = {}
        for cell_id in tally.find_filter(openmc.CellFilter).bins:
            # Get cell and volume
            cell = self.get_tally_cell(cell_id)
            if cell is None:
                volume = 1.0  # cm^3
            else:
                volume = cell.volume if cell.volume else 1.0
            
            # Extract data for this cell
            cell_tally = cell_tallies_df[cell_tallies_df['cell'] == cell_id].copy()
            # Normalize by volume
            cell_tally['mean'] /= volume
            cell_tally['std. dev.'] /= volume
            # Store in dict
            cell_tallies[cell.name] = cell_tally
        
        return cell_tallies
    
    def extract_nuclide_data(self) -> Dict[str, NuclideData]:
        """Extract nuclide data"""
        nuclide_tallies: Dict[str, NuclideData] = {}
        for nuclide, reaction in zip(
            self.simulation_params['track_nuclides'],
            self.simulation_params['reactions']
        ):
            tally = self.sp.get_tally(name=f'nuclide_tally_{nuclide}_{reaction}')
            nuclide_tally_df = tally.get_pandas_dataframe()
            nuclide_tallies[nuclide] = NuclideData(tally=nuclide_tally_df, reaction=reaction)
        return nuclide_tallies
    
    def extract_surface_data(self) -> pd.DataFrame:
        """Extract surface tally data"""
        # TODO: There's only one surface for now, but expand later
        tally = self.sp.get_tally(scores=['current'])
        return tally.get_pandas_dataframe()
    
    def get_aggregate(self, tally_df: pd.DataFrame, groupby: List[str]) -> pd.DataFrame:
        """Aggregate tally data by specified columns"""
        # Group by the independent variable
        
        tally_group = tally_df.groupby(groupby)
        # Proper aggregation of means and standard deviations
        # When summing, std deviations add in quadrature (sqrt of sum of squares)
        tally_group_aggregate = pd.DataFrame({
            'mean': tally_group['mean'].sum(),
            'std. dev.': tally_group['std. dev.'].apply(lambda x: np.sqrt(np.sum(x**2)))
        }).reset_index()
        
        return tally_group_aggregate
    
    def calculate_moderation_efficiency(self, cutoff: float = 0.1) -> Dict[str,  Union[float, Dict[float, Tuple[float, float]]]]:
        """Calculate neutron moderation efficiency"""
        # Get cell tally
        cell_tallies = self.extract_cell_data()
        
        # Initialize results dictionaries
        moderation_efficiency = {}
        moderated_flux = {}
        total_flux = {}
        for cell_name, cell_tally_df in cell_tallies.items():
            
            # Get aggregated cell tally by energy
            cell_tally_agg = self.get_aggregate(cell_tally_df, ['energy low [eV]'])
            cell_tally_mean = cell_tally_agg['mean']
            cell_tally_std = cell_tally_agg['std. dev.']
            
            # Moderation efficiency defined as fraction of neutrons below energy threshold
            low_energy_threshold = np.array(self.simulation_params['low_energy_threshold']) * 1e6 # eV
            
            # Separate moderated and high-energy fluxes
            total_mean = cell_tally_mean.sum()
            total_std = np.sqrt((cell_tally_std**2).sum())
            total_flux[cell_name] = (total_mean, total_std)
            
            # Initialize dicts for this cell
            moderation_efficiency[cell_name] = {}
            moderated_flux[cell_name] = {}
            for energy in low_energy_threshold:
                # Find the efficiency of moderating below a given energy
                moderated_idx = cell_tally_agg['energy low [eV]'] <= energy
                moderated_mean = cell_tally_mean[moderated_idx].sum()
                moderated_std = np.sqrt((cell_tally_std[moderated_idx]**2).sum())
                
                efficiency = moderated_mean / total_mean if total_mean > 0 else 0
                efficiency_std = efficiency * np.sqrt(
                    (moderated_std / moderated_mean)**2 + (total_std / total_mean)**2
                ) if moderated_mean > 0 and total_mean > 0 else 0
                
                moderation_efficiency[cell_name][energy] = (efficiency, efficiency_std)
                moderated_flux[cell_name][energy] = (moderated_mean, moderated_std)
                
                # Print results
                print(f'Fraction below {int(energy)} eV: {efficiency} +/- {efficiency_std}')
                print(f"Target reached: {'YES' if efficiency > cutoff else 'NO'}")
                
        return {
            'moderation_efficiency': moderation_efficiency,
            'moderated_flux': moderated_flux,
            'total_flux': total_flux
        }

class ResultsPlotter:
    """Class to create plots from processed data"""
    
    def __init__(self, data_processor: DataProcessor | str, save_dir: Optional[str] = None):
        if isinstance(data_processor, DataProcessor):
            self.data_processor = data_processor
        elif isinstance(data_processor, str):
            self.data_processor = DataProcessor(data_processor)
        else:
            raise ValueError("data_processor must be a DataProcessor instance or a simulation directory path string")
        
        # Directory to save plots
        if save_dir is None:
            self.save_dir = self.data_processor.simulation_dir / 'plots'
        else:
            self.save_dir = Path(save_dir)
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
    def plot_mesh(
        self,
        score: str,
        corner: Literal['top_left', 'top_right', 'bottom_left', 'bottom_right'] = 'top_right'
    ):
        # Generate mesh data if not done already
        if not hasattr(self.data_processor, 'mesh'):
            self.data_processor.extract_mesh_data()
            
        fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
        self._plot_mesh_quadrant(fig, ax, score, corner)
        self._plot_geometry(ax)
        
        plt.savefig(self.save_dir / f'{score}_mesh.png', bbox_inches='tight')
        plt.close()
        
    def plot_combined_mesh(
        self,
        scores: List[str],
    ):
        if len(scores) > 4:
            raise ValueError('Cannot plot more than 4 scores at once since they are divided into quadrants')
        
        # Generate mesh data if not done already
        if not hasattr(self.data_processor, 'mesh'):
            self.data_processor.extract_mesh_data()
        
        fig, ax = plt.subplots(figsize=(10, 9), layout='constrained')
        corners = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        cmaps = ['magma', 'plasma', 'viridis', 'hot']
        for i, (score, cmap) in enumerate(zip(scores, cmaps)):
            self._plot_mesh_quadrant(fig, ax, score, corner=corners[i], cmap=cmap)
        self._plot_geometry(ax)
        
        plt.savefig(self.save_dir / f'combined_mesh.png', bbox_inches='tight')
        plt.close()
            
    def _plot_geometry(
        self,
        ax: plt.Axes
    ):
        # Plot geometry underlay
        geometry = self.data_processor.geometry
        bounding_box = geometry.bounding_box
        width = [bounding_box.width[0], bounding_box.width[2]]
        
        # Actually plot geometry
        pixels = 1000
        geometry.plot(
            axes=ax,
            origin=bounding_box.center,
            width=width,
            pixels=tuple([int(size * pixels / max(width)) for size in width]),
            basis='xz',
            color_by='material',
            colors=NIFMaterials().get_colors(),
            zorder=-1
        )
        
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('z (cm)')
        ax.set_aspect('equal')
    
    def _plot_mesh_quadrant(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        score: str,
        corner: Literal['top_left', 'top_right', 'bottom_left', 'bottom_right'] = 'top_right',
        cmap: Literal['magma', 'plasma', 'viridis', 'hot'] = 'magma'
    ):
        """Create 2D spatial plots"""
        mesh_df = self.data_processor.mesh_tally_df
        mesh = self.data_processor.mesh
        
        # Get score
        possible_scores = mesh_df['score'].unique()
        if score not in possible_scores:
            raise ValueError(f'{score} not in available scores: {possible_scores}')
        
        score_df = mesh_df[mesh_df['score'] == score].copy()
        
        if mesh.dimension:
            dim_x, dim_z = mesh.dimension[0], mesh.dimension[2]
        else:
            raise ValueError("Mesh dimensions not found")
        
        # Define the quadrants of the mesh
        mid_x = (mesh.lower_left[0] + mesh.upper_right[0]) / 2
        mid_z = (mesh.lower_left[2] + mesh.upper_right[2]) / 2
        
        # Find dimensions depending on corner
        if corner == 'top_left':
            quadrant_df = score_df[(score_df['x'] <= dim_x//2) & (score_df['z'] > dim_z//2)]
            extent = (mesh.lower_left[0], mid_x, mid_z, mesh.upper_right[2])
        elif corner == 'top_right':
            quadrant_df = score_df[(score_df['x'] > dim_x//2) & (score_df['z'] > dim_z//2)]
            extent = (mid_x, mesh.upper_right[0], mid_z, mesh.upper_right[2])
        elif corner == 'bottom_left':
            quadrant_df = score_df[(score_df['x'] <= dim_x//2) & (score_df['z'] <= dim_z//2)]
            extent = (mesh.lower_left[0], mid_x, mesh.lower_left[2], mid_z)
        elif corner == 'bottom_right':
            quadrant_df = score_df[(score_df['x'] > dim_x//2) & (score_df['z'] <= dim_z//2)]
            extent = (mid_x, mesh.upper_right[0], mesh.lower_left[2], mid_z)
        else:
            raise ValueError(f'Corner {corner} not recognized')
        
        print(f'Plotting {score} mesh')
        score_df_agg = self.data_processor.get_aggregate(quadrant_df, ['x', 'y', 'z'])
        # Reshape into quadrant size
        score_mean = score_df_agg['mean'].to_numpy().reshape((dim_x//2, dim_z//2), order='F')
        # Replace zeros with NaN to avoid log10(0) warning and hide empty cells
        score_mean = np.where(score_mean > 0, score_mean, np.nan)

        # Create plot for specified reaction
        im = ax.imshow(
            np.log10(score_mean),
            cmap=cmap,
            origin='lower',
            extent=extent,
        )
        cbar_location = 'right' if corner in ['top_right', 'bottom_right'] else 'left'
        cbar = fig.colorbar(im, orientation='vertical', location=cbar_location)
        if score == 'flux':
            label = f'log$_{{10}}$(neutrons/cm$^2$-source)'
        else:
            label = f'log$_{{10}}$({score}/cm$^3$-source)'
        cbar.set_label(label)
        
        # Add text label of score on top of quadrant
        ax.text(
            (extent[0] + extent[1])/2,
            (extent[2] + extent[3])/2,
            score,
            ha='center',
            va='center',
            color='white',
            fontsize=30
        )
    
    def plot_spectrum(
        self,
        tally_df: pd.DataFrame,
        variable: str,
        id: str,
        name: str,
        reaction: Optional[str] = None,
    ):
        # Establish the correct units
        if variable == 'energy':
            unit = 'eV'
            multiplier = 1e-6
            prefix = 'M'
        elif variable == 'time':
            unit = 's'
            if max(tally_df[f'{variable} low [{unit}]']) > 1e-6:
                multiplier = 1e6
                prefix = 'u'
            elif max(tally_df[f'{variable} low [{unit}]']) > 1e-9:
                multiplier = 1e9
                prefix = 'n'
            elif max(tally_df[f'{variable} low [{unit}]']) > 1e-12:
                multiplier = 1e12
                prefix = 'p'
            else:
                multiplier = 1
                prefix = ''
        else:
            raise ValueError('Please choose one of either energy or time for the independent variable')
        
        # Replace 0 values with None
        tally_df['mean'] = tally_df['mean'].replace(0, np.nan)
        tally_df['std. dev.'] = tally_df['std. dev.'].replace(0, np.nan)
        
        ### MAKE SPECTRUM PLOTS ###
        # Get aggregated data
        tally_aggregate = self.data_processor.get_aggregate(tally_df, [f'{variable} low [{unit}]'])
        mean = tally_aggregate['mean']
        std = tally_aggregate['std. dev.']
        x = tally_aggregate[f'{variable} low [{unit}]'] * multiplier
        
        # Create plots
        fig, ax = plt.subplots(figsize=(8,4), layout='constrained')
        ax.step(x, mean, color='tab:blue', where='pre', label='Total')
        
        # Plot std. dev. as shaded area
        positive = mean - std > 0
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.3,
            color='tab:blue',
            where=positive,
            step='pre'
        )
        
        # Calculate cumulative sum
        cumsum = np.nancumsum(mean)
        
        # Calculate uncertainty propagation for cumulative sum
        cumsum_var = np.nancumsum(std**2)  # Sum of variances
        cumsum_std = np.sqrt(cumsum_var)   # Standard deviation

        # Create secondary y-axis for CDF
        ax2 = ax.twinx()
        ax2.step(x, cumsum, color='black', where='pre', 
                label='Cumulative', linestyle='-', linewidth=2)

        # Plot uncertainty as shaded region
        positive = cumsum - cumsum_std > 0
        ax2.fill_between(
            x,
            cumsum - cumsum_std,
            cumsum + cumsum_std,
            alpha=0.2,
            color='black',
            where=positive,
            step='pre'
        )

        ax2.set_ylabel('Cumulative Tally', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(bottom=0)
        
        # Add features to plots depending on whether it's a time or energy spectrum
        low_energy_threshold = self.data_processor.simulation_params['low_energy_threshold']
        colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        for energy, color in zip(low_energy_threshold, colors):
            if variable == 'energy':
                # Plot vertical lines to show where the cutoff occurs
                ax.axvline(energy, color=color, linestyle='--',)
            elif variable == 'time':
                # Filter the time series by energy cutoff and plot contributions to each
                threshold_tally = tally_df[tally_df['energy low [eV]'] <= energy * 1e6] # eV
                
                # Get aggregated data
                threshold_tally_aggregate = self.data_processor.get_aggregate(threshold_tally, [f'{variable} low [{unit}]'])
                mean = threshold_tally_aggregate['mean']
                std = threshold_tally_aggregate['std. dev.']
                x = threshold_tally_aggregate[f'{variable} low [{unit}]'] * multiplier
                
                # Create plots
                energy_label = f'<{int(energy)} MeV' if energy >= 1 else f'<{int(energy*1e3)} keV'
                ax.step(x, mean, color=color, where='pre', label=energy_label)
                
                # Plot std. dev. as shaded area
                positive = mean - std > 0
                ax.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    alpha=0.3,
                    color=color,
                    where=positive,
                    step='pre'
                )
                
            else:
                raise ValueError('Please choose one of either energy or time for the independent variable')
                
        
        # Set axis parameters
        if variable == 'energy':
            ax.set_xscale('log')
        
        # Changes scale and sets axis limits
        self.setup_log_plot_axes(ax)
        if variable == 'time':
            ax.set_xlim(left=-0.25, right=10)
            # Add label lines
            labelLines(ax.get_lines(), align=True, fontsize=16, xvals=[0.55, 9.2, 1.7, 0.5, 5])
            labelLines(ax2.get_lines(), align=True, fontsize=16, xvals=[5])
            
        ax.set_xlabel(f'{variable.capitalize()} ({prefix}{unit})')
        # TODO: change ylabel based on tally type
        if name == 'cell':
            ylabel = '#/cm$^2$-source'
        elif name == 'nuclide':
            if reaction:
                if 'gamma' in reaction:
                    reaction = '(n,$\gamma$)'
                ylabel = f'{reaction}/source'
            else:
                ylabel = '(n,$\gamma$)/source'
        else:
            raise ValueError('Name must be either "cell" or "nuclide"')
        ax.set_ylabel(ylabel)
        ax.grid(which='major', lw=0.5)
        
        # Save figure
        fig.savefig(f'{self.save_dir}/{id}_{name}_{variable}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Print out expected values
        # Convert spectrum to pdf
        pdf = tally_aggregate.copy()
        pdf['mean'] /= pdf['mean'].sum()
        pdf['std. dev.'] /= pdf['mean'].sum()
        expected_value = (x * pdf['mean']).sum()
        second_moment = (x**2 * pdf['mean']).sum()
        expected_std = np.sqrt(second_moment - expected_value**2)
        print(f'Expected {variable} for {name} {id}: {expected_value} +/- {expected_std} {prefix}{unit}')
        
    @staticmethod
    def setup_log_plot_axes(ax: Axes):
        """Configure axes for logarithmic plots with proper limits."""
        ax.set_yscale('log')
        
        # Find global minimum across all data
        # Change axis limits to fit better
        y_min = None
        x_min = None
        x_max = None
        for line in ax.get_lines():
            # Skip vertical lines
            if isinstance(line.get_ydata(), list):
                continue
            
            # Find minimum
            if np.any(line.get_ydata() > 0):
                positive_idx = line.get_ydata() > 0
                y_min_new = line.get_ydata()[positive_idx].min()
                x_min_new = line.get_xdata()[positive_idx].min()
                x_max_new = line.get_xdata()[positive_idx].max()
                if y_min is None or y_min_new < y_min:
                    y_min = y_min_new
                if x_min is None or x_min_new < x_min:
                    x_min = x_min_new
                if x_max is None or x_max_new > x_max:
                    x_max = x_max_new
        
        if x_min is not None and x_max is not None:
            # Use multiplicative padding for log scale, additive for linear
            if ax.get_xscale() == 'log':
                padding_factor = 1.1
                ax.set_xlim(left=x_min / padding_factor, right=x_max * padding_factor)
            else:
                x_range = x_max - x_min
                ax.set_xlim(left=x_min - 0.05 * x_range, right=x_max + 0.05 * x_range)
        if y_min is not None:
            ax.set_ylim(bottom=0.5 * y_min)
        
    def plot_all(self):
        """Create all standard plots"""
        cell_tallies = self.data_processor.extract_cell_data()
        nuclide_tallies = self.data_processor.extract_nuclide_data()
        
        # Plot energy spectra
        for cell_name, cell_tally_df in cell_tallies.items():
            self.plot_spectrum(cell_tally_df, 'energy', cell_name, 'cell')
        for nuclide, nuclide_data in nuclide_tallies.items():
            self.plot_spectrum(nuclide_data.tally, 'energy', nuclide, 'nuclide', reaction=nuclide_data.reaction)
        # Plot time spectra
        for cell_name, cell_tally_df in cell_tallies.items():
            self.plot_spectrum(cell_tally_df, 'time', cell_name, 'cell')
        for nuclide, nuclide_data in nuclide_tallies.items():
            self.plot_spectrum(nuclide_data.tally, 'time', nuclide, 'nuclide', reaction=nuclide_data.reaction)
        # Plot 2D spatial maps
        self.plot_mesh('flux')
        self.plot_mesh('scatter')
        self.plot_mesh('(n,2n)')
        self.plot_mesh('(n,3n)')
        self.plot_mesh('(n,gamma)')
        self.plot_mesh('(n,t)')
        self.plot_combined_mesh(['flux', 'scatter', '(n,2n)'])