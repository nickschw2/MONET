import openmc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
import numpy as np
import json
from typing import Optional, List, Tuple, Union, Dict, Literal

from .materials import NIFMaterials

# Set plotting params
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 3

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
        
        # Load model.xml to get geometry
        model_file = self.simulation_dir / 'model.xml'
        geometry_file = self.simulation_dir / 'geometry.xml'
        materials_file = self.simulation_dir / 'materials.xml'
        if geometry_file.exists() and materials_file.exists():
            self.geometry = openmc.Geometry.from_xml(geometry_file, materials_file)
            # self.geometry = model.geometry
        else:
            raise FileNotFoundError(f"No geometry.xml or materials.xml file found in {simulation_dir}")
        
        # Open the statepoint file
        statepoint_file = self.simulation_dir / f'statepoint.{batches}.h5'
        
        if statepoint_file.exists():
            self.sp = openmc.StatePoint(str(statepoint_file))
        else:
            raise FileNotFoundError(f"No statepoint file found in {simulation_dir}")
        
    def get_fuel_cell(self) -> openmc.Cell | None:
        """Get fuel cell object from model"""
        if self.geometry:
            for cell in self.geometry.get_all_cells().values():
                if 'fuel' in cell.name.lower():
                    return cell
            print("Fuel cell not found in geometry")
            return None
        else:
            print("Geometry not found in model.xml")
            return None
    
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
        
    def extract_fuel_data(self) -> pd.DataFrame:
        tally = self.sp.get_tally(name='fuel_tally')
        fuel_tally_df = tally.get_pandas_dataframe()
        
        # Get fuel cell volume
        fuel_cell = self.get_fuel_cell()
        if fuel_cell is None:
            volume = 1.0  # cm^3
        else:
            volume = fuel_cell.volume if fuel_cell.volume else 1.0
            
        # Normalize by volume
        fuel_tally_df['mean'] /= volume
        fuel_tally_df['std. dev.'] /= volume
        
        return fuel_tally_df
    
    def extract_trace_data(self) -> pd.DataFrame:
        tally = self.sp.get_tally(name='trace_tally')
        trace_tally_df = tally.get_pandas_dataframe()
        return trace_tally_df
    
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
        # Get fuel tally
        fuel_tally_df = self.extract_fuel_data()
        
        # Get aggregated fuel tally by energy
        fuel_tally_agg = self.get_aggregate(fuel_tally_df, ['energy low [eV]'])
        fuel_tally_mean = fuel_tally_agg['mean']
        fuel_tally_std = fuel_tally_agg['std. dev.']
        
        # Moderation efficiency defined as fraction of neutrons below energy threshold
        low_energy_threshold = np.array(self.simulation_params['low_energy_threshold']) * 1e6 # eV
        
        # Separate moderated and high-energy fluxes
        total_mean = fuel_tally_mean.sum()
        total_std = np.sqrt((fuel_tally_std**2).sum())
        
        # Initialize dictionariess
        efficiency_dict = {}
        moderated_flux_dict = {}
        for energy in low_energy_threshold:
            # Find the efficiency of moderating below a given energy
            moderated_idx = fuel_tally_agg['energy low [eV]'] <= energy
            moderated_mean = fuel_tally_mean[moderated_idx].sum()
            moderated_std = np.sqrt((fuel_tally_std[moderated_idx]**2).sum())
            
            efficiency = moderated_mean / total_mean if total_mean > 0 else 0
            efficiency_std = efficiency * np.sqrt(
                (moderated_std / moderated_mean)**2 + (total_std / total_mean)**2
            ) if moderated_mean > 0 and total_mean > 0 else 0
            
            efficiency_dict[energy] = (efficiency, efficiency_std)
            moderated_flux_dict[energy] = (moderated_mean, moderated_std)
            
            # Print results
            print(f'Fraction below {int(energy)} eV: {efficiency} +/- {efficiency_std}')
            print(f"Target reached: {'YES' if efficiency > cutoff else 'NO'}")
            
        return {
            'moderation_efficiency': efficiency_dict,
            'moderated_flux': moderated_flux_dict,
            'total_flux': total_mean
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
        
        plt.savefig(self.save_dir / f'{score}_mesh.png')
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
        
        fig, ax = plt.subplots(figsize=(14, 10), layout='constrained')
        corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        cmaps = ['magma', 'plasma', 'viridis', 'hot']
        for i, (score, cmap) in enumerate(zip(scores, cmaps)):
            self._plot_mesh_quadrant(fig, ax, score, corner=corners[i], cmap=cmap)
        self._plot_geometry(ax)
        
        plt.savefig(self.save_dir / f'combined_mesh.png')
        plt.close()
            
    def _plot_geometry(
        self,
        ax: plt.Axes
    ):
        # Plot geometry underlay
        geometry = self.data_processor.geometry
        bounding_box = geometry.bounding_box
        origin = (
            (bounding_box.lower_left[0] + bounding_box.upper_right[0]) / 2,
            0,
            (bounding_box.lower_left[2] + bounding_box.upper_right[2]) / 2,
        )
        width = (
            (bounding_box.upper_right[0] - bounding_box.lower_left[0]),
            (bounding_box.upper_right[2] - bounding_box.lower_left[2]),
        )
        
        # Actually plot geometry
        pixels = 1000
        geometry.plot(
            axes=ax,
            origin=origin,
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
        
        score_df = mesh_df[mesh_df['score'] == score]
        
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
        name: str
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
                ax.step(x, mean, color=color, where='pre', label='Total')
                
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
        ax.set_xlabel(f'{variable.capitalize()} ({prefix}{unit})')
        # TODO: change ylabel based on tally type
        if name == 'fuel':
            ylabel = '#/cm$^2$-source'
        elif name == 'trace':
            ylabel = '(n,gamma)/source'
        else:
            raise ValueError('Name must be either "fuel" or "source"')
        ax.set_ylabel(ylabel)
        # ax.legend()
        
        # Save figure
        fig.savefig(f'{self.save_dir}/{name}_{variable}.png', dpi=300)
        plt.close(fig)
        
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
        
        if x_min or x_max:     
            ax.set_xlim(left=x_min - 0.05 * (x_max - x_min), right=x_max + 0.05 * (x_max - x_min))
        if y_min:
            ax.set_ylim(bottom=0.5 * y_min)
        
    def plot_all(self):
        """Create all standard plots"""
        fuel_tally_df = self.data_processor.extract_fuel_data()
        trace_tally_df = self.data_processor.extract_trace_data()
        
        # Plot energy spectra
        self.plot_spectrum(fuel_tally_df, 'energy', 'fuel')
        self.plot_spectrum(trace_tally_df, 'energy', 'trace')
        
        # Plot time spectra
        self.plot_spectrum(fuel_tally_df, 'time', 'fuel')
        self.plot_spectrum(trace_tally_df, 'time', 'trace')
        
        # Plot 2D spatial maps
        self.plot_mesh('flux')
        self.plot_mesh('scatter')
        self.plot_mesh('(n,2n)')
        self.plot_mesh('(n,3n)')
        self.plot_mesh('(n,gamma)')
        self.plot_combined_mesh(['flux', 'scatter', '(n,2n)'])