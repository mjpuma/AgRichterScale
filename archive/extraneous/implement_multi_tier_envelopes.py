#!/usr/bin/env python3
"""
Multi-Tier Envelope Bounds Implementation

This script implements the multi-tier envelope system with:
1. Productivity-based filtering (comprehensive, commercial, high-productivity, prime land)
2. National-level analysis (USA, China focus)
3. Envelope width comparison and validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class MultiTierEnvelopeCalculator:
    """Calculate envelope bounds with multiple productivity tiers."""
    
    def __init__(self):
        self.productivity_tiers = {
            'comprehensive': {
                'name': 'Comprehensive (All Lands)',
                'yield_percentile_min': 0,
                'yield_percentile_max': 100,
                'description': 'All agricultural land including marginal areas'
            },
            'commercial': {
                'name': 'Commercial Agriculture',
                'yield_percentile_min': 20,
                'yield_percentile_max': 100,
                'description': 'Economically viable agriculture (excludes bottom 20% yields)'
            },
            'high_productivity': {
                'name': 'High-Productivity Agriculture',
                'yield_percentile_min': 30,
                'yield_percentile_max': 100,
                'description': 'Intensive agriculture (excludes bottom 30% yields)'
            },
            'prime_land': {
                'name': 'Prime Agricultural Land',
                'yield_percentile_min': 50,
                'yield_percentile_max': 100,
                'description': 'Top 50% most productive land'
            }
        }
    
    def calculate_multi_tier_envelope(self, production_mt: np.ndarray, 
                                    harvest_ha: np.ndarray, 
                                    crop_name: str = "crop") -> Dict[str, Any]:
        """Calculate envelope bounds for all productivity tiers."""
        
        print(f"🌾 MULTI-TIER ENVELOPE CALCULATION - {crop_name.upper()}")
        print("=" * 60)
        
        # Step 1: Basic data preparation
        valid_mask = (production_mt > 0) & (harvest_ha > 0) & np.isfinite(production_mt) & np.isfinite(harvest_ha)
        P_base = production_mt[valid_mask]
        H_base = harvest_ha[valid_mask]
        Y_base = P_base / H_base  # MT/ha
        
        print(f"Base dataset: {len(P_base):,} valid cells")
        print(f"Yield range: {Y_base.min():.3f} - {Y_base.max():.3f} MT/ha")
        print(f"Total production: {P_base.sum()/1e6:.2f} Million MT")
        print(f"Total harvest: {H_base.sum()/1e5:.1f} Million ha")
        
        # Step 2: Calculate envelope for each tier
        tier_results = {}
        
        for tier_key, tier_config in self.productivity_tiers.items():
            print(f"\n📊 Calculating {tier_config['name']}...")
            
            # Apply productivity filtering
            yield_min = np.percentile(Y_base, tier_config['yield_percentile_min'])
            yield_max = np.percentile(Y_base, tier_config['yield_percentile_max'])
            
            tier_mask = (Y_base >= yield_min) & (Y_base <= yield_max)
            
            P_tier = P_base[tier_mask]
            H_tier = H_base[tier_mask]
            Y_tier = Y_base[tier_mask]
            
            # Calculate envelope bounds for this tier
            envelope_data = self._calculate_envelope_bounds_reference(P_tier, H_tier)
            
            # Calculate tier statistics
            tier_stats = {
                'tier_name': tier_config['name'],
                'tier_key': tier_key,
                'description': tier_config['description'],
                'yield_range_mt_ha': (yield_min, yield_max),
                'cells_included': len(P_tier),
                'cells_percentage': len(P_tier) / len(P_base) * 100,
                'total_production_mt': P_tier.sum(),
                'total_harvest_ha': H_tier.sum(),
                'total_harvest_km2': H_tier.sum() / 100,
                'avg_yield_mt_ha': P_tier.sum() / H_tier.sum(),
                'yield_std': Y_tier.std()
            }
            
            # Add tier metadata to envelope
            envelope_data['tier_info'] = tier_stats
            
            # Calculate envelope width at sample points
            sample_harvests = [100, 1000, 5000, 10000]  # km²
            envelope_widths = []
            
            for sample_h in sample_harvests:
                if sample_h <= tier_stats['total_harvest_km2']:
                    width = self._get_envelope_width_at_harvest(envelope_data, sample_h)
                    envelope_widths.append(width)
                else:
                    envelope_widths.append(None)
            
            envelope_data['sample_widths'] = {
                'harvest_points_km2': sample_harvests,
                'envelope_widths_mt': envelope_widths
            }
            
            tier_results[tier_key] = envelope_data
            
            print(f"  Cells: {len(P_tier):,} ({len(P_tier)/len(P_base)*100:.1f}%)")
            print(f"  Yield range: {yield_min:.2f} - {yield_max:.2f} MT/ha")
            print(f"  Production: {P_tier.sum()/1e6:.2f} Million MT")
            print(f"  Harvest: {H_tier.sum()/1e5:.1f} Million ha")
        
        # Step 3: Calculate width reductions between tiers
        width_reductions = self._calculate_width_reductions(tier_results)
        
        return {
            'tier_envelopes': tier_results,
            'width_reductions': width_reductions,
            'crop_name': crop_name,
            'base_stats': {
                'total_cells': len(P_base),
                'total_production_mt': P_base.sum(),
                'total_harvest_ha': H_base.sum(),
                'yield_range_mt_ha': (Y_base.min(), Y_base.max())
            }
        }
    
    def _calculate_envelope_bounds_reference(self, production_mt: np.ndarray, 
                                           harvest_ha: np.ndarray) -> Dict[str, Any]:
        """Calculate envelope bounds using validated methodology."""
        
        # Convert units
        harvest_km2 = harvest_ha / 100
        yields = production_mt / harvest_ha
        
        # Sort by yield
        lower_indices = np.argsort(yields)        # Ascending
        upper_indices = np.argsort(yields)[::-1]  # Descending
        
        # Calculate cumulative sums
        H_lower_cum = np.cumsum(harvest_km2[lower_indices])
        P_lower_cum = np.cumsum(production_mt[lower_indices])
        H_upper_cum = np.cumsum(harvest_km2[upper_indices])
        P_upper_cum = np.cumsum(production_mt[upper_indices])
        
        # Create query function
        def query_bounds(H_target_km2):
            # Lower bound
            lower_idx = np.where(H_lower_cum >= H_target_km2)[0]
            P_lower = P_lower_cum[lower_idx[0]] if len(lower_idx) > 0 else P_lower_cum[-1]
            
            # Upper bound
            upper_idx = np.where(H_upper_cum >= H_target_km2)[0]
            P_upper = P_upper_cum[upper_idx[0]] if len(upper_idx) > 0 else P_upper_cum[-1]
            
            return P_lower, P_upper
        
        return {
            'lower_bound': {'H_cum': H_lower_cum, 'P_cum': P_lower_cum},
            'upper_bound': {'H_cum': H_upper_cum, 'P_cum': P_upper_cum},
            'query_function': query_bounds,
            'total_harvest_km2': H_lower_cum[-1],
            'total_production_mt': P_lower_cum[-1]
        }
    
    def _get_envelope_width_at_harvest(self, envelope_data: Dict, H_target_km2: float) -> float:
        """Get envelope width at specific harvest area."""
        
        P_lower, P_upper = envelope_data['query_function'](H_target_km2)
        return P_upper - P_lower
    
    def _calculate_width_reductions(self, tier_results: Dict) -> Dict[str, Any]:
        """Calculate envelope width reductions between tiers."""
        
        tier_order = ['comprehensive', 'commercial', 'high_productivity', 'prime_land']
        width_reductions = {}
        
        # Sample harvest area for comparison (use 50% of smallest tier's total)
        min_harvest = min(
            tier_results[tier]['total_harvest_km2'] 
            for tier in tier_order if tier in tier_results
        )
        sample_harvest = min_harvest * 0.5
        
        print(f"\n📏 ENVELOPE WIDTH COMPARISON (at {sample_harvest:.0f} km² harvest)")
        print("-" * 60)
        
        baseline_width = None
        
        for i, tier_key in enumerate(tier_order):
            if tier_key not in tier_results:
                continue
            
            tier_data = tier_results[tier_key]
            width = self._get_envelope_width_at_harvest(tier_data, sample_harvest)
            
            if baseline_width is None:
                baseline_width = width
                reduction_pct = 0
            else:
                reduction_pct = (baseline_width - width) / baseline_width * 100
            
            width_reductions[tier_key] = {
                'envelope_width_mt': width,
                'width_reduction_pct': reduction_pct,
                'tier_name': tier_data['tier_info']['tier_name']
            }
            
            print(f"{tier_data['tier_info']['tier_name']:25}: "
                  f"{width/1e6:8.3f} Million MT "
                  f"({reduction_pct:5.1f}% reduction)")
        
        return width_reductions

class NationalEnvelopeCalculator:
    """Calculate envelope bounds for specific countries."""
    
    def __init__(self):
        self.multi_tier_calc = MultiTierEnvelopeCalculator()
        
        # Country-specific configurations
        self.country_configs = {
            'USA': {
                'name': 'United States',
                'lat_range': (25, 50),
                'lon_range': (-125, -65),
                'focus': 'high_efficiency_agriculture'
            },
            'CHN': {
                'name': 'China',
                'lat_range': (18, 54),
                'lon_range': (73, 135),
                'focus': 'food_security'
            },
            'BRA': {
                'name': 'Brazil',
                'lat_range': (-35, 5),
                'lon_range': (-75, -35),
                'focus': 'export_capacity'
            }
        }
    
    def calculate_national_envelope(self, production_df: pd.DataFrame, 
                                  harvest_df: pd.DataFrame,
                                  country_code: str,
                                  crop_col: str) -> Dict[str, Any]:
        """Calculate multi-tier envelope bounds for a specific country."""
        
        if country_code not in self.country_configs:
            raise ValueError(f"Country {country_code} not configured")
        
        country_config = self.country_configs[country_code]
        
        print(f"\n🌍 NATIONAL ENVELOPE ANALYSIS - {country_config['name'].upper()}")
        print("=" * 70)
        
        # Step 1: Filter to country boundaries (simplified lat/lon box)
        if 'lat' in production_df.columns and 'lon' in production_df.columns:
            lat_mask = ((production_df['lat'] >= country_config['lat_range'][0]) & 
                       (production_df['lat'] <= country_config['lat_range'][1]))
            lon_mask = ((production_df['lon'] >= country_config['lon_range'][0]) & 
                       (production_df['lon'] <= country_config['lon_range'][1]))
            country_mask = lat_mask & lon_mask
        else:
            print("⚠️  No lat/lon columns found, using all data as proxy")
            country_mask = np.ones(len(production_df), dtype=bool)
        
        # Extract country data
        P_country = production_df.loc[country_mask, crop_col].values
        H_country = harvest_df.loc[country_mask, crop_col].values
        
        print(f"Country cells: {np.sum(country_mask):,}")
        print(f"Valid crop cells: {np.sum((P_country > 0) & (H_country > 0)):,}")
        
        # Step 2: Calculate multi-tier envelopes for country
        national_results = self.multi_tier_calc.calculate_multi_tier_envelope(
            P_country, H_country, f"{country_code}_{crop_col}"
        )
        
        # Step 3: Add national context
        valid_mask = (P_country > 0) & (H_country > 0)
        P_valid = P_country[valid_mask]
        H_valid = H_country[valid_mask]
        
        national_context = {
            'country_code': country_code,
            'country_name': country_config['name'],
            'crop_column': crop_col,
            'total_cells': len(P_country),
            'valid_cells': len(P_valid),
            'national_production_mt': P_valid.sum(),
            'national_harvest_ha': H_valid.sum(),
            'national_harvest_km2': H_valid.sum() / 100,
            'national_avg_yield_mt_ha': P_valid.sum() / H_valid.sum() if H_valid.sum() > 0 else 0,
            'focus_area': country_config['focus']
        }
        
        national_results['national_context'] = national_context
        
        print(f"\nNational Summary:")
        print(f"  Production: {national_context['national_production_mt']/1e6:.2f} Million MT")
        print(f"  Harvest: {national_context['national_harvest_ha']/1e6:.2f} Million ha")
        print(f"  Avg Yield: {national_context['national_avg_yield_mt_ha']:.2f} MT/ha")
        
        return national_results

def demonstrate_multi_tier_system():
    """Demonstrate the multi-tier envelope system with synthetic data."""
    
    print("🚀 MULTI-TIER ENVELOPE SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Create synthetic SPAM-like data with realistic yield distribution
    np.random.seed(42)
    n_cells = 10000
    
    # Create yield distribution with marginal lands (low yields) and prime lands (high yields)
    # Realistic wheat yield distribution: 0.5-8 MT/ha with long tail of low yields
    
    # 30% marginal lands (0.5-2 MT/ha)
    marginal_yields = np.random.uniform(0.5, 2.0, int(n_cells * 0.3))
    
    # 50% commercial lands (2-5 MT/ha)  
    commercial_yields = np.random.uniform(2.0, 5.0, int(n_cells * 0.5))
    
    # 20% prime lands (4-8 MT/ha)
    prime_yields = np.random.uniform(4.0, 8.0, int(n_cells * 0.2))
    
    all_yields = np.concatenate([marginal_yields, commercial_yields, prime_yields])
    np.random.shuffle(all_yields)
    
    # Generate harvest areas (100-1000 ha per cell)
    harvest_areas = np.random.uniform(100, 1000, n_cells)
    
    # Calculate production
    production_values = harvest_areas * all_yields
    
    print(f"Synthetic dataset created:")
    print(f"  Cells: {n_cells:,}")
    print(f"  Yield range: {all_yields.min():.2f} - {all_yields.max():.2f} MT/ha")
    print(f"  Marginal lands (0.5-2 MT/ha): ~30%")
    print(f"  Commercial lands (2-5 MT/ha): ~50%")
    print(f"  Prime lands (4-8 MT/ha): ~20%")
    
    # Calculate multi-tier envelopes
    multi_tier_calc = MultiTierEnvelopeCalculator()
    results = multi_tier_calc.calculate_multi_tier_envelope(
        production_values, harvest_areas, "synthetic_wheat"
    )
    
    # Display results summary
    print(f"\n🎯 MULTI-TIER RESULTS SUMMARY")
    print("=" * 50)
    
    for tier_key, width_data in results['width_reductions'].items():
        tier_info = results['tier_envelopes'][tier_key]['tier_info']
        print(f"\n{width_data['tier_name']}:")
        print(f"  Cells included: {tier_info['cells_included']:,} ({tier_info['cells_percentage']:.1f}%)")
        print(f"  Avg yield: {tier_info['avg_yield_mt_ha']:.2f} MT/ha")
        print(f"  Envelope width: {width_data['envelope_width_mt']/1e6:.3f} Million MT")
        print(f"  Width reduction: {width_data['width_reduction_pct']:.1f}%")
    
    return results

def demonstrate_national_analysis():
    """Demonstrate national-level analysis with synthetic data."""
    
    print(f"\n🌍 NATIONAL ANALYSIS DEMONSTRATION")
    print("=" * 50)
    
    # Create synthetic global dataset with country-like regions
    np.random.seed(123)
    n_global_cells = 50000
    
    # Create lat/lon grid
    lats = np.random.uniform(-60, 70, n_global_cells)
    lons = np.random.uniform(-180, 180, n_global_cells)
    
    # Create country-specific yield patterns
    yields = np.zeros(n_global_cells)
    
    # USA region (high yields)
    usa_mask = (lats >= 25) & (lats <= 50) & (lons >= -125) & (lons <= -65)
    yields[usa_mask] = np.random.uniform(3.0, 7.0, np.sum(usa_mask))
    
    # China region (medium-high yields)
    china_mask = (lats >= 18) & (lats <= 54) & (lons >= 73) & (lons <= 135)
    yields[china_mask] = np.random.uniform(2.5, 6.0, np.sum(china_mask))
    
    # Other regions (mixed yields)
    other_mask = ~(usa_mask | china_mask)
    yields[other_mask] = np.random.uniform(0.5, 5.0, np.sum(other_mask))
    
    # Generate harvest areas and production
    harvest_areas = np.random.uniform(50, 800, n_global_cells)
    production_values = harvest_areas * yields
    
    # Create DataFrames
    production_df = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'WHEA_A': production_values
    })
    
    harvest_df = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'WHEA_A': harvest_areas
    })
    
    print(f"Global synthetic dataset:")
    print(f"  Total cells: {n_global_cells:,}")
    print(f"  USA region cells: {np.sum(usa_mask):,}")
    print(f"  China region cells: {np.sum(china_mask):,}")
    
    # Calculate national envelopes
    national_calc = NationalEnvelopeCalculator()
    
    # USA analysis
    usa_results = national_calc.calculate_national_envelope(
        production_df, harvest_df, 'USA', 'WHEA_A'
    )
    
    # China analysis
    china_results = national_calc.calculate_national_envelope(
        production_df, harvest_df, 'CHN', 'WHEA_A'
    )
    
    # Compare national results
    print(f"\n🔍 NATIONAL COMPARISON")
    print("-" * 30)
    
    for country, results in [('USA', usa_results), ('China', china_results)]:
        context = results['national_context']
        commercial_tier = results['tier_envelopes']['commercial']
        
        print(f"\n{context['country_name']}:")
        print(f"  Production: {context['national_production_mt']/1e6:.2f} Million MT")
        print(f"  Avg Yield: {context['national_avg_yield_mt_ha']:.2f} MT/ha")
        print(f"  Commercial envelope width: {results['width_reductions']['commercial']['envelope_width_mt']/1e6:.3f} Million MT")
    
    return usa_results, china_results

def main():
    """Main demonstration function."""
    
    try:
        # Demonstration 1: Multi-tier system
        multi_tier_results = demonstrate_multi_tier_system()
        
        # Demonstration 2: National analysis
        usa_results, china_results = demonstrate_national_analysis()
        
        print(f"\n🎉 MULTI-TIER ENVELOPE SYSTEM DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("Key Achievements:")
        print("✅ Multi-tier envelope bounds implemented")
        print("✅ Productivity-based filtering working")
        print("✅ Envelope width reductions demonstrated")
        print("✅ National-level analysis functional")
        print("✅ Country comparison capabilities shown")
        
        print(f"\nNext Steps:")
        print("📋 Integrate with real SPAM data")
        print("🌍 Add more countries (Brazil, India, etc.)")
        print("🔍 Implement regional subdivisions")
        print("📊 Create policy scenario framework")
        print("📈 Add visualization and reporting tools")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)