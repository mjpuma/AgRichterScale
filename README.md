# AgRichter Scale

**Agricultural Disruption Magnitude Scale - Python Implementation**

A Python framework for quantifying and visualizing historical agricultural disruptions using the AgRichter Scale, analogous to the earthquake Richter scale.

## Overview

The AgRichter Scale provides a standardized way to measure and compare agricultural disruptions across time and space. Like the earthquake Richter scale, it uses a logarithmic magnitude scale based on the physical extent of disruption (harvest area affected).

### Key Features

- **AgRichter Scale**: Magnitude-based visualization (similar to earthquake Richter scale)
- **H-P Envelope**: Harvest-Production relationship analysis
- **Historical Events**: Analysis of 21+ major agricultural disruptions
- **Multiple Crops**: Support for wheat, rice, allgrain, and maize
- **SPAM 2020 Integration**: Uses latest global agricultural data
- **Publication-Quality Figures**: Multiple output formats (PNG, SVG, EPS, JPG)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mjpuma/AgRichterScale.git
cd AgRichterScale

# Install dependencies
pip install -r requirements.txt
```

### Generate Figures

```bash
# Generate AgRichter Scale for all crops
python generate_all_figures.py

# Specific crops only
python generate_all_figures.py --crops wheat rice

# Custom output directory
python generate_all_figures.py --output-dir my_outputs
```

### Output Structure

```
outputs/
├── wheat/
│   ├── data/
│   │   └── events_wheat_spam2020.csv
│   ├── figures/
│   │   ├── agririchter_scale_wheat.png
│   │   ├── agririchter_scale_wheat.svg
│   │   ├── agririchter_scale_wheat.eps
│   │   └── agririchter_scale_wheat.jpg
│   └── reports/
│       ├── pipeline_summary_wheat.txt
│       └── performance_wheat.txt
├── rice/
│   └── ...
└── allgrain/
    └── ...
```

## The AgRichter Scale

### Concept

The AgRichter Scale measures agricultural disruption magnitude using:

**M_D = log₁₀(A_H)**

Where:
- **M_D**: AgRichter Magnitude
- **A_H**: Harvest area disrupted (km²)

### Visualization

The AgRichter Scale plot shows:
- **X-axis**: Magnitude (M_D) - increases to the right
- **Y-axis**: Harvest Area Disrupted (km²) - logarithmic scale
- **Events**: Historical disruptions plotted as red circles
- **Thresholds**: Severity levels (Minimal, Stressed, Crisis, Emergency, Famine)

This orientation matches the familiar earthquake Richter scale convention.

### Dual-Scale Framework

The AgRichter Scale works with the H-P (Harvest-Production) Envelope:

| Visualization | Purpose | Shows |
|--------------|---------|-------|
| **AgRichter Scale** | Magnitude assessment | How big was the event? |
| **H-P Envelope** | Impact assessment | What was the actual food loss? |

Together they provide a complete picture of agricultural disruption.

## Historical Events Analyzed

The framework analyzes 21+ major historical agricultural disruptions:

- Great Famine (1315-1317)
- Laki Eruption (1783)
- Year Without a Summer (1816)
- Great Drought (1876-1878)
- Soviet Famine (1921-1922)
- Chinese Famine (1959-1961)
- Dust Bowl (1930s)
- Sahel Drought (2010)
- And more...

## Data Sources

- **SPAM 2020**: Global harvest area and production data
- **Event Definitions**: Historical disruption records
- **USDA PSD**: Production and trade statistics
- **Country Codes**: Multiple mapping systems (ISO3, FAOSTAT, GAUL, etc.)

## Project Structure

```
AgRichterScale/
├── agririchter/           # Main package
│   ├── analysis/          # Event calculation
│   ├── core/              # Configuration and constants
│   ├── data/              # Data loading and management
│   ├── pipeline/          # Analysis pipeline
│   ├── validation/        # Data validation
│   └── visualization/     # Figure generation
├── ancillary/             # Event definitions and mappings
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── tests/                 # Unit and integration tests
├── USDAdata/              # USDA production data
├── generate_all_figures.py  # Main figure generation script
└── README.md              # This file
```

## Documentation

- **[User Guide](docs/USER_GUIDE.md)**: Detailed usage instructions
- **[API Reference](docs/API_REFERENCE.md)**: Code documentation
- **[Data Requirements](docs/DATA_REQUIREMENTS.md)**: Required data files
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions
- **[AgRichter Scale Update](AGRICHTER_SCALE_UPDATE.md)**: Recent changes

## Development

### Running Tests

```bash
# All tests
pytest

# Specific test suite
pytest tests/unit/
pytest tests/integration/

# With coverage
pytest --cov=agririchter
```

### Code Style

```bash
# Format code
black agririchter/

# Lint
flake8 agririchter/
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- openpyxl
- xlrd
- adjustText (for label placement)
- pytest (for testing)

See `requirements.txt` for complete list.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{agrichter2025,
  title = {AgRichter Scale: Agricultural Disruption Magnitude Framework},
  author = {Puma, Michael J.},
  year = {2025},
  url = {https://github.com/mjpuma/AgRichterScale}
}
```

## License

[Add your license here]

## Contact

- **Author**: Michael J. Puma
- **Institution**: [Your institution]
- **Email**: [Your email]
- **GitHub**: https://github.com/mjpuma

## Acknowledgments

- SPAM 2020 data from MapSPAM
- USDA Production, Supply, and Distribution database
- Historical event records from multiple sources

## Recent Updates

### October 2025
- ✅ Renamed to "AgRichter Scale" (from "AgriRichter")
- ✅ Updated axis orientation to match Richter scale convention
- ✅ Fixed event loading from Excel files
- ✅ Implemented complete analysis pipeline
- ✅ Generated figures for wheat, rice, and allgrain
- ✅ Added comprehensive documentation

See [AGRICHTER_SCALE_UPDATE.md](AGRICHTER_SCALE_UPDATE.md) for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Status

🟢 **Active Development** - The framework is functional and producing results. Ongoing work includes:
- H-P Envelope visualization
- Global production maps
- Additional crop types
- MATLAB validation
- Performance optimization

---

**Note**: This is a Python reimplementation of the original MATLAB AgriRichter framework, with enhancements and modern visualization capabilities.
