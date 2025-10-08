# AgriRichter Documentation

Welcome to the AgriRichter documentation! This directory contains comprehensive guides for installing, using, and understanding the AgriRichter agricultural disruption analysis system.

## Documentation Overview

### 📚 Core Documentation

1. **[User Guide](USER_GUIDE.md)** - Start here!
   - Installation instructions
   - Quick start guide
   - Command-line usage
   - Output file descriptions
   - Advanced usage examples

2. **[Data Requirements](DATA_REQUIREMENTS.md)** - Essential reading
   - Required SPAM 2020 data files
   - Historical event definition files
   - Optional boundary data
   - Download sources and instructions
   - Directory structure recommendations

3. **[API Reference](API_REFERENCE.md)** - For developers
   - Complete API documentation
   - Class and method signatures
   - Type hints and parameters
   - Code examples
   - Best practices

4. **[Troubleshooting Guide](TROUBLESHOOTING.md)** - When things go wrong
   - Common errors and solutions
   - Installation issues
   - Data loading problems
   - Performance optimization
   - Debugging tips
   - FAQ

## Quick Links

### Getting Started
- [Installation Prerequisites](USER_GUIDE.md#installation)
- [Quick Start Example](USER_GUIDE.md#quick-start)
- [Download SPAM Data](DATA_REQUIREMENTS.md#1-spam-2020-gridded-data)

### Common Tasks
- [Run Analysis for Wheat](USER_GUIDE.md#basic-usage)
- [Process All Grains](USER_GUIDE.md#example-commands)
- [Generate Visualizations](USER_GUIDE.md#output-files)
- [Validate Results](USER_GUIDE.md#command-line-options)

### Troubleshooting
- [Installation Problems](TROUBLESHOOTING.md#installation-issues)
- [Data File Errors](TROUBLESHOOTING.md#data-loading-issues)
- [Memory Issues](TROUBLESHOOTING.md#performance-issues)
- [Visualization Problems](TROUBLESHOOTING.md#visualization-issues)

### Development
- [Using as Python Library](USER_GUIDE.md#using-as-a-python-library)
- [API Documentation](API_REFERENCE.md)
- [Adding Custom Events](DATA_REQUIREMENTS.md#adding-custom-events)

## Documentation Structure

```
docs/
├── README.md                    # This file - documentation overview
├── USER_GUIDE.md               # Installation and usage guide
├── DATA_REQUIREMENTS.md        # Data files and sources
├── API_REFERENCE.md            # Complete API documentation
└── TROUBLESHOOTING.md          # Common issues and solutions
```

## What is AgriRichter?

AgriRichter is a Python-based tool for analyzing historical agricultural disruption events and their impacts on global food production. It:

- Calculates production losses for 21 historical events (famines, droughts, conflicts)
- Uses SPAM 2020 gridded crop data at 5-arcminute resolution
- Generates publication-quality visualizations
- Provides AgriRichter magnitude scale (similar to earthquake magnitude)
- Enables food security impact assessment

### Key Features

✅ **Historical Event Analysis**: Calculate actual losses for 21 documented events  
✅ **Multiple Crops**: Analyze wheat, rice, or all grains  
✅ **Spatial Mapping**: Map events to gridded production data  
✅ **Publication Figures**: Generate H-P Envelope, AgriRichter Scale, and production maps  
✅ **Validation**: Compare with MATLAB reference outputs  
✅ **Performance**: Optimized for large datasets  

## Typical Workflow

1. **Install AgriRichter** → [Installation Guide](USER_GUIDE.md#installation)
2. **Download Data Files** → [Data Requirements](DATA_REQUIREMENTS.md)
3. **Run Analysis** → [Quick Start](USER_GUIDE.md#quick-start)
4. **Review Outputs** → [Output Files](USER_GUIDE.md#output-files)
5. **Validate Results** → [Validation](USER_GUIDE.md#command-line-options)

## Example Usage

### Basic Analysis

```bash
# Analyze wheat events
python scripts/run_agririchter_analysis.py --crop wheat

# Analyze all grains with validation
python scripts/run_agririchter_analysis.py --crop allgrain --validate
```

### Python API

```python
from agririchter.core.config import Config
from agririchter.pipeline.events_pipeline import EventsPipeline

# Configure and run
config = Config(crop_type='wheat')
pipeline = EventsPipeline(config, output_dir='outputs/wheat')
results = pipeline.run_complete_pipeline()

# Access results
events_df = results['events_dataframe']
print(f"Processed {len(events_df)} events")
```

## Output Examples

After running the analysis, you'll get:

### 📊 Data Files
- `events_wheat_spam2020.csv` - Event losses and magnitudes
- `summary_wheat.txt` - Analysis summary report
- `validation_wheat.txt` - Validation results

### 📈 Visualizations
- `hp_envelope_wheat.svg` - H-P Envelope with historical events
- `agririchter_scale_wheat.svg` - AgriRichter Scale with severity classifications
- `global_production_wheat.png` - Global production map

All figures are generated in multiple formats (SVG, EPS, PNG) for publication use.

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: 10GB (including data files)
- **OS**: macOS, Linux, or Windows

## Key Dependencies

- pandas, numpy - Data processing
- geopandas, shapely - Spatial operations
- matplotlib, cartopy - Visualization
- openpyxl - Excel file reading

See [Installation](USER_GUIDE.md#installation) for complete dependency list.

## Data Sources

AgriRichter uses publicly available datasets:

- **SPAM 2020**: Global gridded crop production data
  - Source: https://www.mapspam.info/
  - Resolution: 5-arcminute (~10km)
  - Coverage: Global, all major crops

- **Historical Events**: 21 documented agricultural disruptions
  - Famines, droughts, conflicts (1899-2012)
  - Country and state-level definitions
  - Compiled from academic literature

- **GDAM Boundaries**: Geographic boundary data
  - Source: https://gadm.org/
  - Country and state/province levels

See [Data Requirements](DATA_REQUIREMENTS.md) for detailed information.

## Support and Contributing

### Getting Help

- 📖 Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
- 🔍 Search existing issues in the repository
- 💬 Open a new issue with detailed information
- 📧 Contact package maintainers

### Contributing

We welcome contributions!

- 🐛 Report bugs with detailed reproduction steps
- 📝 Improve documentation
- ✨ Add new features
- 🧪 Write tests
- 🔧 Fix issues

## Citation

When using AgriRichter in publications, please cite:

**SPAM 2020 Data:**
> Yu, Q., You, L., Wood-Sichra, U., Ru, Y., Joglekar, A.K.B., Fritz, S., Xiong, W., Lu, M., Wu, W., Yang, P. (2020). A cultivated planet in 2010 – Part 2: The global gridded agricultural-production maps. Earth System Science Data, 12(4), 3545-3572.

**GDAM Boundaries:**
> Global Administrative Areas (2018). GADM database of Global Administrative Areas, version 2.0. [online] URL: www.gadm.org

## License

[Include license information here]

## Version History

- **v1.0.0** - Initial Python implementation
  - Complete MATLAB migration
  - SPAM 2020 integration
  - 21 historical events
  - Publication-quality figures

## Additional Resources

### External Documentation
- [SPAM Documentation](https://www.mapspam.info/)
- [GDAM Documentation](https://gadm.org/)
- [GeoPandas User Guide](https://geopandas.org/)
- [Cartopy Documentation](https://scitools.org.uk/cartopy/)

### Related Projects
- Original MATLAB implementation
- USDA PSD database
- FAO food security data

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Analyze wheat | `python scripts/run_agririchter_analysis.py --crop wheat` |
| Analyze rice | `python scripts/run_agririchter_analysis.py --crop rice` |
| Analyze all grains | `python scripts/run_agririchter_analysis.py --crop allgrain` |
| With validation | `python scripts/run_agririchter_analysis.py --crop wheat --validate` |
| Debug mode | `python scripts/run_agririchter_analysis.py --crop wheat --log-level DEBUG` |
| Custom output | `python scripts/run_agririchter_analysis.py --crop wheat --output-dir /path/to/output` |

---

**Need help?** Start with the [User Guide](USER_GUIDE.md) or check the [Troubleshooting Guide](TROUBLESHOOTING.md).

**Ready to dive deeper?** Explore the [API Reference](API_REFERENCE.md) for programmatic usage.

**Missing data files?** See [Data Requirements](DATA_REQUIREMENTS.md) for download instructions.
