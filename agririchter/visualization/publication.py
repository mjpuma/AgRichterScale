"""Publication-quality output formatting and styling for AgriRichter visualizations."""

import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import numpy as np

from ..core.config import Config


class PublicationFormatter:
    """Handles publication-quality formatting and output for all visualizations."""
    
    # Publication-quality style settings
    PUBLICATION_STYLE = {
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.transparent': False,
        
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        
        # Line and marker settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'patch.linewidth': 0.5,
        
        # Axes settings
        'axes.linewidth': 1.0,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
        
        # Color settings
        'axes.prop_cycle': plt.cycler('color', [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]),
        
        # Figure settings
        'figure.figsize': (8, 6),
        'figure.autolayout': False,
        'figure.constrained_layout.use': True,
    }
    
    # Supported output formats
    SUPPORTED_FORMATS = {
        'png': {'dpi': 300, 'transparent': False},
        'jpg': {'dpi': 300, 'transparent': False, 'quality': 95},
        'jpeg': {'dpi': 300, 'transparent': False, 'quality': 95},
        'svg': {'dpi': 300, 'transparent': False},
        'eps': {'dpi': 300, 'transparent': False},
        'pdf': {'dpi': 300, 'transparent': False},
        'tiff': {'dpi': 300, 'transparent': False, 'compression': 'lzw'},
    }
    
    def __init__(self, config: Config):
        """Initialize the publication formatter.
        
        Args:
            config: Configuration object with output settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._original_rcparams = None
        
    def __enter__(self):
        """Context manager entry - apply publication style."""
        self.apply_publication_style()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore original style."""
        self.restore_original_style()
        
    def apply_publication_style(self):
        """Apply publication-quality matplotlib style settings."""
        self.logger.info("Applying publication-quality style settings...")
        
        # Store original rcParams
        self._original_rcparams = rcParams.copy()
        
        # Apply publication style
        rcParams.update(self.PUBLICATION_STYLE)
        
        # Set backend if needed
        if mpl.get_backend() != 'Agg':
            mpl.use('Agg')  # Use non-interactive backend for publication output
            
    def restore_original_style(self):
        """Restore original matplotlib style settings."""
        if self._original_rcparams is not None:
            rcParams.update(self._original_rcparams)
            self.logger.info("Restored original matplotlib style settings")
            
    def save_figure(self, 
                   fig: plt.Figure,
                   base_path: Union[str, Path],
                   formats: Optional[List[str]] = None,
                   dpi: int = 300,
                   **kwargs) -> List[Path]:
        """Save figure in multiple publication-quality formats.
        
        Args:
            fig: Matplotlib figure to save
            base_path: Base path for output files (without extension)
            formats: List of formats to save (default: ['png', 'svg', 'eps'])
            dpi: Resolution for raster formats
            **kwargs: Additional arguments passed to savefig
            
        Returns:
            List of paths where files were saved
        """
        if formats is None:
            formats = ['png', 'svg', 'eps']
            
        base_path = Path(base_path)
        saved_paths = []
        
        # Ensure output directory exists
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        for fmt in formats:
            if fmt not in self.SUPPORTED_FORMATS:
                self.logger.warning(f"Unsupported format: {fmt}")
                continue
                
            # Get format-specific settings
            format_settings = self.SUPPORTED_FORMATS[fmt].copy()
            format_settings.update(kwargs)
            
            # Override DPI if specified
            if dpi != 300:
                format_settings['dpi'] = dpi
                
            # Create output path
            output_path = base_path.with_suffix(f'.{fmt}')
            
            try:
                # Save figure
                fig.savefig(output_path, 
                           format=fmt,
                           bbox_inches='tight',
                           facecolor='white',
                           edgecolor='none',
                           **format_settings)
                
                saved_paths.append(output_path)
                self.logger.info(f"Saved figure to {output_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to save figure as {fmt}: {str(e)}")
                
        return saved_paths
    
    def create_figure(self, 
                     figsize: Tuple[float, float] = (10, 8),
                     dpi: int = 300,
                     **kwargs) -> plt.Figure:
        """Create a publication-quality figure with consistent styling.
        
        Args:
            figsize: Figure size in inches
            dpi: Figure resolution
            **kwargs: Additional arguments passed to plt.figure
            
        Returns:
            Matplotlib figure object
        """
        # Set default kwargs
        figure_kwargs = {
            'figsize': figsize,
            'dpi': dpi,
            'facecolor': 'white',
            'edgecolor': 'none',
        }
        figure_kwargs.update(kwargs)
        
        # Create figure
        fig = plt.figure(**figure_kwargs)
        
        return fig
    
    def format_axes(self, 
                   ax: plt.Axes,
                   title: Optional[str] = None,
                   xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None,
                   grid: bool = True,
                   spines: Optional[List[str]] = None) -> plt.Axes:
        """Apply consistent formatting to axes.
        
        Args:
            ax: Matplotlib axes object
            title: Axes title
            xlabel: X-axis label
            ylabel: Y-axis label
            grid: Whether to show grid
            spines: List of spines to show (default: ['left', 'bottom'])
            
        Returns:
            Formatted axes object
        """
        if spines is None:
            spines = ['left', 'bottom']
            
        # Set labels and title
        if title:
            ax.set_title(title, fontweight='bold', pad=20)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
            
        # Configure spines
        for spine_name in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine_name].set_visible(spine_name in spines)
            
        # Configure grid
        if grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        else:
            ax.grid(False)
            
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        
        return ax
    
    def add_colorbar(self, 
                    mappable,
                    ax: plt.Axes,
                    label: Optional[str] = None,
                    orientation: str = 'horizontal',
                    shrink: float = 0.8,
                    aspect: int = 40,
                    pad: float = 0.05) -> plt.Axes:
        """Add a publication-quality colorbar.
        
        Args:
            mappable: Mappable object (e.g., from pcolormesh, scatter)
            ax: Axes to attach colorbar to
            label: Colorbar label
            orientation: 'horizontal' or 'vertical'
            shrink: Shrink factor for colorbar
            aspect: Aspect ratio for colorbar
            pad: Padding between axes and colorbar
            
        Returns:
            Colorbar axes object
        """
        cbar = plt.colorbar(mappable, ax=ax, 
                           orientation=orientation,
                           shrink=shrink,
                           aspect=aspect,
                           pad=pad)
        
        if label:
            cbar.set_label(label, fontsize=10)
            
        cbar.ax.tick_params(labelsize=9)
        
        return cbar
    
    def create_legend(self, 
                     ax: plt.Axes,
                     loc: str = 'best',
                     frameon: bool = True,
                     fancybox: bool = True,
                     shadow: bool = True,
                     alpha: float = 0.9,
                     **kwargs):
        """Create a publication-quality legend.
        
        Args:
            ax: Axes object
            loc: Legend location
            frameon: Whether to draw legend frame
            fancybox: Whether to use rounded corners
            shadow: Whether to draw shadow
            alpha: Legend transparency
            **kwargs: Additional legend arguments
            
        Returns:
            Legend object
        """
        legend_kwargs = {
            'loc': loc,
            'frameon': frameon,
            'fancybox': fancybox,
            'shadow': shadow,
            'framealpha': alpha,
            'fontsize': 10,
        }
        legend_kwargs.update(kwargs)
        
        legend = ax.legend(**legend_kwargs)
        
        return legend
    
    def get_color_palette(self, n_colors: int, palette: str = 'default') -> List[str]:
        """Get a publication-quality color palette.
        
        Args:
            n_colors: Number of colors needed
            palette: Palette name ('default', 'qualitative', 'sequential')
            
        Returns:
            List of color hex codes
        """
        palettes = {
            'default': [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ],
            'qualitative': [
                '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                '#ffff33', '#a65628', '#f781bf', '#999999'
            ],
            'sequential': [
                '#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c',
                '#fc4e2a', '#e31a1c', '#bd0026', '#800026'
            ]
        }
        
        colors = palettes.get(palette, palettes['default'])
        
        # Repeat colors if needed
        if n_colors > len(colors):
            colors = colors * (n_colors // len(colors) + 1)
            
        return colors[:n_colors]
    
    def create_publication_summary(self, 
                                  output_dir: Path,
                                  figures: Dict[str, plt.Figure],
                                  metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Create a summary document with all publication figures.
        
        Args:
            output_dir: Directory to save summary
            figures: Dictionary of figure names and objects
            metadata: Optional metadata to include
            
        Returns:
            Path to summary file
        """
        summary_path = output_dir / 'publication_summary.md'
        
        with open(summary_path, 'w') as f:
            f.write("# AgriRichter Analysis - Publication Summary\n\n")
            
            if metadata:
                f.write("## Analysis Metadata\n\n")
                for key, value in metadata.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            f.write("## Generated Figures\n\n")
            
            for fig_name, fig in figures.items():
                f.write(f"### {fig_name.replace('_', ' ').title()}\n\n")
                
                # List available formats
                base_path = output_dir / fig_name
                formats = []
                for fmt in ['png', 'svg', 'eps', 'pdf']:
                    if (base_path.with_suffix(f'.{fmt}')).exists():
                        formats.append(fmt)
                
                if formats:
                    f.write("Available formats: " + ", ".join(formats) + "\n\n")
                
                # Add PNG preview if available
                png_path = base_path.with_suffix('.png')
                if png_path.exists():
                    f.write(f"![{fig_name}]({png_path.name})\n\n")
        
        self.logger.info(f"Created publication summary at {summary_path}")
        return summary_path


class OutputManager:
    """Manages organized output of all AgriRichter visualizations."""
    
    def __init__(self, config: Config, base_output_dir: Optional[Path] = None):
        """Initialize the output manager.
        
        Args:
            config: Configuration object
            base_output_dir: Base directory for all outputs
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if base_output_dir is None:
            base_output_dir = Path('output')
        
        self.base_dir = Path(base_output_dir)
        self.crop_dir = self.base_dir / self.config.crop_type
        
        # Create output directory structure
        self._create_directory_structure()
        
    def _create_directory_structure(self):
        """Create organized directory structure for outputs."""
        directories = [
            self.crop_dir / 'maps',
            self.crop_dir / 'plots', 
            self.crop_dir / 'envelopes',
            self.crop_dir / 'data',
            self.crop_dir / 'publication'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"Created output directory structure at {self.base_dir}")
    
    def get_output_path(self, 
                       category: str, 
                       filename: str, 
                       extension: Optional[str] = None) -> Path:
        """Get standardized output path for a file.
        
        Args:
            category: Output category ('maps', 'plots', 'envelopes', 'data', 'publication')
            filename: Base filename
            extension: File extension (optional)
            
        Returns:
            Full output path
        """
        if category not in ['maps', 'plots', 'envelopes', 'data', 'publication']:
            raise ValueError(f"Invalid category: {category}")
            
        path = self.crop_dir / category / filename
        
        if extension and not filename.endswith(extension):
            path = path.with_suffix(extension)
            
        return path
    
    def save_publication_set(self, 
                           figures: Dict[str, plt.Figure],
                           formats: Optional[List[str]] = None) -> Dict[str, List[Path]]:
        """Save a complete set of publication-quality figures.
        
        Args:
            figures: Dictionary of figure names and objects
            formats: List of formats to save (default: ['png', 'svg', 'eps'])
            
        Returns:
            Dictionary mapping figure names to lists of saved paths
        """
        if formats is None:
            formats = ['png', 'svg', 'eps']
            
        saved_files = {}
        
        with PublicationFormatter(self.config) as formatter:
            for fig_name, fig in figures.items():
                # Determine category from figure name
                if 'map' in fig_name.lower():
                    category = 'maps'
                elif 'envelope' in fig_name.lower():
                    category = 'envelopes'
                else:
                    category = 'plots'
                
                # Get base path
                base_path = self.get_output_path(category, fig_name)
                
                # Save in multiple formats
                paths = formatter.save_figure(fig, base_path, formats)
                saved_files[fig_name] = paths
        
        # Create publication summary
        summary_path = formatter.create_publication_summary(
            self.crop_dir / 'publication',
            figures,
            metadata={
                'crop_type': self.config.crop_type,
                'analysis_date': str(Path().cwd()),
                'formats': formats
            }
        )
        
        self.logger.info(f"Saved {len(figures)} figures in {len(formats)} formats")
        return saved_files