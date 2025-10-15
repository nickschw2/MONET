# MOderated NEutron Transport (MONET)

This simulation suite is designed to investigate neutron moderation strategies for producing low-energy neutrons (<1 MeV) that can react with <sup>171</sup>Tm through radiative capture at the National Ignition Facility (NIF).

## Overview

The goal of this simulation suite is to determine whether it's possible to moderate a significant portion (>10%) of neutrons from DT fusion reactions to energies below 1 MeV for enhanced <sup>171</sup>Tm capture reactions. The suite supports comprehensive parameter sweeps to optimize moderation efficiency.

## Features

### Geometry Options

The simulation suite supports multiple geometry options, including:

- **Standard NIF**: DT fuel sphere + ablator shell + hohlraum
- **Double-shell**: Two-shell implosion geometry
- **Coronal source**: DT ice layer with laser entrance holes
- **Dual sources**: Any combination of the above listed schemes on either side of a moderator
- **Dual filled hohlraum**: Filled hohlraum with moderator and two coronal sources on either side

The user defines the geometry and sources in the pre-ignition state, which are then subsquently compressed with the user-defined convergence ratios.

### Material Options

The simulation suite supports a wide range of material options, including:
TODO: add materials options for each
- **Ablators**:
- **Hohlraums**:
- **Moderators**:
- **Dopants**:

### Source Modeling

The simulation suite supports multiple source modeling options, including:

- Muir energy distribution for DD and DT fusion neutrons
- Delta function or Gaussian time dependence
- Point or spherical source

### Analysis Capabilities

The simulation suite supports comprehensive analysis capabilities, including:

- 2D spatial flux/reaction maps
- Energy and time spectra
- Nuclear reaction tallies
- Moderation efficiency calculations
- Comprehensive parameter sweeps

## Installation

To install the simulation suite, follow these steps:

1. Clone the repository:
```bash
git clone <repository_url>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure OpenMC is properly configured with nuclear data libraries.

## Quick Start

### Single Simulation

To run a single NIF simulation, use the following command:
```bash
python scripts/run_single.py
```

### Parameter Sweep

To run a parameter sweep, use the following command:
```bash
python scripts/run_sweep.py
```

## Usage Examples

### Basic Simulation

To set up and run a basic simulation, use the following code:
```python

```

### Parameter Sweep

To run a parameter sweep, use the following code:
```python
```

## Data Management

Results are automatically organized with:
TODO: add data organization
- **Simulation parameters** 
- **Tally data** 
- **Plots** 
- **Summary statistics** 
- **Parameter sweep databases**

## Customization

### Adding New Materials

To add a new material

```python

```

### Adding New Geometries

To add a new geometry

```python
```

### Custom Analysis

To add custom analysis

```python
```

## Contributing

Contributions are welcome! To contribute to the project, please follow these steps:

1. Create a new branch for your feature or bug fix
2. Make your changes and commit them
3. Push your changes to your forked repository
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project was made possible through funding from