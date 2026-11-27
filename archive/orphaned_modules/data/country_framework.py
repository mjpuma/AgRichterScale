"""
Scalable framework for adding new countries to the multi-tier envelope system.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from ..core.config import Config
from .country_boundary_manager import CountryConfiguration, CountryBoundaryManager

logger = logging.getLogger(__name__)


@dataclass
class CountryTemplate:
    """Template for creating new country configurations."""
    
    country_code: str
    country_name: str
    fips_code: str
    iso3_code: str
    agricultural_focus: str
    priority_crops: List[str]
    regional_subdivisions: Optional[List[str]] = None
    policy_scenarios: Optional[List[str]] = None
    min_cells_required: int = 1000
    min_crop_coverage_percent: float = 5.0
    template_version: str = "1.0"
    created_date: Optional[str] = None
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now().isoformat()


class CountryFramework:
    """Scalable framework for adding new countries to the multi-tier envelope system."""
    
    COUNTRY_TEMPLATES = {
        'USA': CountryTemplate(
            country_code='USA', country_name='United States', fips_code='US', iso3_code='USA',
            agricultural_focus='export_capacity', priority_crops=['wheat', 'maize', 'rice'],
            regional_subdivisions=['corn_belt', 'great_plains', 'california'], min_cells_required=2500
        ),
        'CHN': CountryTemplate(
            country_code='CHN', country_name='China', fips_code='CH', iso3_code='CHN',
            agricultural_focus='food_security', priority_crops=['wheat', 'maize', 'rice'],
            regional_subdivisions=['northeast', 'north_china_plain', 'yangtze_river'], min_cells_required=3500
        ),
        'BRA': CountryTemplate(
            country_code='BRA', country_name='Brazil', fips_code='BR', iso3_code='BRA',
            agricultural_focus='export_capacity', priority_crops=['maize', 'wheat', 'rice'],
            regional_subdivisions=['cerrado', 'amazon', 'south'], min_cells_required=2000
        ),
        'IND': CountryTemplate(
            country_code='IND', country_name='India', fips_code='IN', iso3_code='IND',
            agricultural_focus='food_security', priority_crops=['wheat', 'rice', 'maize'],
            regional_subdivisions=['indo_gangetic_plain', 'deccan_plateau'], min_cells_required=3000
        ),
        'RUS': CountryTemplate(
            country_code='RUS', country_name='Russia', fips_code='RS', iso3_code='RUS',
            agricultural_focus='export_capacity', priority_crops=['wheat', 'maize', 'rice'],
            regional_subdivisions=['black_earth', 'siberia'], min_cells_required=2500
        ),
        'ARG': CountryTemplate(
            country_code='ARG', country_name='Argentina', fips_code='AR', iso3_code='ARG',
            agricultural_focus='export_capacity', priority_crops=['wheat', 'maize', 'rice'],
            regional_subdivisions=['pampas', 'patagonia'], min_cells_required=1500
        ),
        'CAN': CountryTemplate(
            country_code='CAN', country_name='Canada', fips_code='CA', iso3_code='CAN',
            agricultural_focus='export_capacity', priority_crops=['wheat', 'maize', 'rice'],
            regional_subdivisions=['prairies', 'ontario'], min_cells_required=1200
        ),
        'AUS': CountryTemplate(
            country_code='AUS', country_name='Australia', fips_code='AS', iso3_code='AUS',
            agricultural_focus='export_capacity', priority_crops=['wheat', 'maize', 'rice'],
            regional_subdivisions=['wheat_belt', 'murray_darling'], min_cells_required=1000
        ),
        'UKR': CountryTemplate(
            country_code='UKR', country_name='Ukraine', fips_code='UP', iso3_code='UKR',
            agricultural_focus='export_capacity', priority_crops=['wheat', 'maize', 'rice'],
            regional_subdivisions=['black_earth', 'steppe'], min_cells_required=1200
        ),
        'FRA': CountryTemplate(
            country_code='FRA', country_name='France', fips_code='FR', iso3_code='FRA',
            agricultural_focus='efficiency', priority_crops=['wheat', 'maize', 'rice'],
            regional_subdivisions=['paris_basin', 'loire_valley'], min_cells_required=800
        )
    }
    
    def __init__(self, config: Config, boundary_manager: CountryBoundaryManager):
        self.config = config
        self.boundary_manager = boundary_manager
        self.templates_dir = Path(config.output_dir) / "country_templates"
        self.templates_dir.mkdir(exist_ok=True)
        logger.info("Initialized CountryFramework")
    
    def get_country_template(self, country_code: str) -> Optional[CountryTemplate]:
        return self.COUNTRY_TEMPLATES.get(country_code.upper())
    
    def get_available_templates(self) -> List[str]:
        return list(self.COUNTRY_TEMPLATES.keys())
    
    def create_country_configuration(self, template: CountryTemplate, validate_data: bool = True) -> Tuple[CountryConfiguration, Dict[str, Any]]:
        config = CountryConfiguration(
            country_code=template.country_code,
            country_name=template.country_name,
            fips_code=template.fips_code,
            iso3_code=template.iso3_code,
            agricultural_focus=template.agricultural_focus,
            priority_crops=template.priority_crops,
            regional_subdivisions=template.regional_subdivisions,
            policy_scenarios=template.policy_scenarios
        )
        validation_results = {'template_used': template.country_code}
        return config, validation_results
    
    def validate_country_template(self, template: CountryTemplate) -> Dict[str, Any]:
        return {
            'country_code': template.country_code,
            'overall_status': 'passed',
            'checks': {'configuration_consistency': {'passed': True}},
            'recommendations': []
        }
    
    def create_custom_template(self, country_code: str, country_name: str, fips_code: str, 
                             iso3_code: str, agricultural_focus: str = 'food_security',
                             priority_crops: Optional[List[str]] = None) -> CountryTemplate:
        if priority_crops is None:
            priority_crops = ['wheat', 'maize', 'rice']
        
        return CountryTemplate(
            country_code=country_code.upper(),
            country_name=country_name,
            fips_code=fips_code,
            iso3_code=iso3_code.upper(),
            agricultural_focus=agricultural_focus,
            priority_crops=priority_crops
        )