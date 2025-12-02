#!/usr/bin/env python3
"""
Multi-Tier Envelope System: Technical Integration Demo

This script demonstrates technical integration patterns for developers
integrating the multi-tier envelope system into existing applications.

Usage:
    python examples/demo_technical_integration.py

Target Audience: Software developers, system integrators, technical teams
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agririchter.analysis.multi_tier_envelope import MultiTierEnvelopeEngine, TierConfiguration
from agririchter.analysis.national_envelope_analyzer import NationalEnvelopeAnalyzer
from agririchter.core.performance import PerformanceMonitor
from agririchter.validation.spam_data_filter import SPAMDataFilter

# Configure logging for technical demonstration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiTierAPIWrapper:
    """
    Production-ready API wrapper for multi-tier envelope system.
    
    This class demonstrates how to create a robust, production-ready
    interface for the multi-tier envelope system with proper error
    handling, caching, and performance monitoring.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize API wrapper with configuration."""
        self.config = config or self._default_config()
        self.engine = MultiTierEnvelopeEngine()
        self.performance_monitor = PerformanceMonitor()
        self.cache = {}
        
        logger.info("MultiTierAPIWrapper initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration for production use."""
        return {
            'cache_enabled': True,
            'performance_monitoring': True,
            'validation_strict': True,
            'parallel_processing': True,
            'max_workers': 4,
            'timeout_seconds': 300
        }
    
    @PerformanceMonitor.monitor_performance
    def analyze_agricultural_capacity(self, 
                                    crop_data: 'CropDataset',
                                    tier: str = 'commercial',
                                    country_code: Optional[str] = None,
                                    custom_config: Optional[Dict] = None) -> Dict:
        """
        Main API method for agricultural capacity analysis.
        
        Args:
            crop_data: Crop dataset for analysis
            tier: Analysis tier ('comprehensive', 'commercial', or custom)
            country_code: Optional country code for national analysis
            custom_config: Optional custom configuration
            
        Returns:
            Dict: Analysis results with metadata
            
        Raises:
            ValueError: Invalid input parameters
            TimeoutError: Analysis timeout
            ValidationError: Results failed validation
        """
        
        # Input validation
        self._validate_inputs(crop_data, tier, country_code)
        
        # Check cache
        cache_key = self._generate_cache_key(crop_data, tier, country_code)
        if self.config['cache_enabled'] and cache_key in self.cache:
            logger.info(f"Returning cached result for {cache_key}")
            return self.cache[cache_key]
        
        try:
            # Perform analysis
            start_time = time.time()
            
            if country_code:
                # National analysis
                analyzer = NationalEnvelopeAnalyzer(country_code)
                results = analyzer.analyze_national_capacity(crop_data, tier=tier)
                analysis_type = 'national'
            else:
                # Global analysis
                results = self.engine.calculate_single_tier(crop_data, tier=tier)
                analysis_type = 'global'
            
            execution_time = time.time() - start_time
            
            # Validate results
            if self.config['validation_strict']:
                self._validate_results(results)
            
            # Format response
            response = {
                'status': 'success',
                'analysis_type': analysis_type,
                'tier': tier,
                'country_code': country_code,
                'results': results,
                'metadata': {
                    'execution_time': execution_time,
                    'data_cells': len(crop_data),
                    'validation_passed': True,
                    'cache_used': False
                }
            }
            
            # Cache results
            if self.config['cache_enabled']:
                self.cache[cache_key] = response
                logger.info(f"Cached result for {cache_key}")
            
            logger.info(f"Analysis completed in {execution_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                'status': 'error',
                'error_type': type(e).__name__,
                'error_message': str(e),
                'analysis_type': analysis_type if 'analysis_type' in locals() else 'unknown',
                'tier': tier,
                'country_code': country_code
            }
    
    def batch_analyze_countries(self, 
                               crop_data: 'CropDataset',
                               country_codes: List[str],
                               tier: str = 'commercial') -> Dict[str, Dict]:
        """
        Batch analysis for multiple countries with parallel processing.
        
        Args:
            crop_data: Crop dataset for analysis
            country_codes: List of country codes to analyze
            tier: Analysis tier to use
            
        Returns:
            Dict: Results for each country
        """
        
        logger.info(f"Starting batch analysis for {len(country_codes)} countries")
        
        if self.config['parallel_processing']:
            return self._parallel_batch_analysis(crop_data, country_codes, tier)
        else:
            return self._sequential_batch_analysis(crop_data, country_codes, tier)
    
    def _parallel_batch_analysis(self, crop_data, country_codes, tier) -> Dict[str, Dict]:
        """Parallel batch analysis implementation."""
        
        results = {}
        
        with ProcessPoolExecutor(max_workers=self.config['max_workers']) as executor:
            # Submit analysis tasks
            future_to_country = {
                executor.submit(self._analyze_single_country, crop_data, country, tier): country
                for country in country_codes
            }
            
            # Collect results
            for future in as_completed(future_to_country):
                country = future_to_country[future]
                try:
                    results[country] = future.result(timeout=self.config['timeout_seconds'])
                    logger.info(f"Completed analysis for {country}")
                except Exception as e:
                    logger.error(f"Failed analysis for {country}: {str(e)}")
                    results[country] = {
                        'status': 'error',
                        'error_message': str(e)
                    }
        
        return results
    
    def _sequential_batch_analysis(self, crop_data, country_codes, tier) -> Dict[str, Dict]:
        """Sequential batch analysis implementation."""
        
        results = {}
        
        for country in country_codes:
            try:
                results[country] = self.analyze_agricultural_capacity(
                    crop_data, tier=tier, country_code=country
                )
                logger.info(f"Completed analysis for {country}")
            except Exception as e:
                logger.error(f"Failed analysis for {country}: {str(e)}")
                results[country] = {
                    'status': 'error',
                    'error_message': str(e)
                }
        
        return results
    
    def _analyze_single_country(self, crop_data, country_code, tier):
        """Single country analysis for parallel processing."""
        return self.analyze_agricultural_capacity(
            crop_data, tier=tier, country_code=country_code
        )
    
    def _validate_inputs(self, crop_data, tier, country_code):
        """Validate input parameters."""
        
        if not hasattr(crop_data, 'production_kcal') or not hasattr(crop_data, 'harvest_km2'):
            raise ValueError("Invalid crop_data: missing required attributes")
        
        if len(crop_data) == 0:
            raise ValueError("Empty crop dataset")
        
        valid_tiers = ['comprehensive', 'commercial']
        if tier not in valid_tiers:
            raise ValueError(f"Invalid tier '{tier}'. Must be one of: {valid_tiers}")
        
        if country_code and not isinstance(country_code, str):
            raise ValueError("country_code must be a string")
    
    def _validate_results(self, results):
        """Validate analysis results."""
        
        if not hasattr(results, 'validation_results'):
            raise ValueError("Results missing validation information")
        
        if not results.validation_results.overall_status:
            raise ValueError("Results failed validation tests")
    
    def _generate_cache_key(self, crop_data, tier, country_code):
        """Generate cache key for results."""
        
        # Simple hash based on data size, tier, and country
        data_hash = hash((len(crop_data), tier, country_code))
        return f"analysis_{data_hash}"
    
    def get_performance_metrics(self) -> Dict:
        """Get performance monitoring metrics."""
        
        return self.performance_monitor.get_metrics()
    
    def clear_cache(self):
        """Clear analysis cache."""
        
        self.cache.clear()
        logger.info("Analysis cache cleared")

class CustomTierManager:
    """
    Manager for creating and managing custom tier configurations.
    
    This class demonstrates how to create custom tiers for specific
    use cases and manage tier configurations programmatically.
    """
    
    def __init__(self):
        """Initialize custom tier manager."""
        self.custom_tiers = {}
        logger.info("CustomTierManager initialized")
    
    def create_policy_tier(self, 
                          name: str,
                          yield_percentile_min: float,
                          yield_percentile_max: float,
                          policy_focus: str) -> TierConfiguration:
        """
        Create a policy-focused tier configuration.
        
        Args:
            name: Tier name
            yield_percentile_min: Minimum yield percentile (0-100)
            yield_percentile_max: Maximum yield percentile (0-100)
            policy_focus: Policy focus area
            
        Returns:
            TierConfiguration: Custom tier configuration
        """
        
        # Define policy applications based on focus
        policy_applications = {
            'food_security': ['food_security_planning', 'emergency_preparedness', 'rural_development'],
            'trade': ['export_promotion', 'trade_negotiations', 'market_development'],
            'investment': ['investment_targeting', 'development_planning', 'productivity_improvement'],
            'climate': ['climate_adaptation', 'resilience_planning', 'risk_assessment']
        }
        
        # Define target users based on focus
        target_users = {
            'food_security': ['food_security_agencies', 'emergency_planners', 'rural_development'],
            'trade': ['trade_ministries', 'export_agencies', 'agribusiness'],
            'investment': ['development_agencies', 'investors', 'agricultural_ministries'],
            'climate': ['climate_agencies', 'adaptation_planners', 'risk_managers']
        }
        
        tier_config = TierConfiguration(
            name=name,
            description=f"Custom tier for {policy_focus} policy applications",
            yield_percentile_min=yield_percentile_min,
            yield_percentile_max=yield_percentile_max,
            policy_applications=policy_applications.get(policy_focus, ['general_policy']),
            target_users=target_users.get(policy_focus, ['policy_makers'])
        )
        
        self.custom_tiers[name] = tier_config
        logger.info(f"Created custom tier: {name}")
        
        return tier_config
    
    def create_productivity_tier(self, 
                               name: str,
                               productivity_threshold: float) -> TierConfiguration:
        """
        Create a productivity-based tier configuration.
        
        Args:
            name: Tier name
            productivity_threshold: Productivity threshold (0-100 percentile)
            
        Returns:
            TierConfiguration: Custom tier configuration
        """
        
        tier_config = TierConfiguration(
            name=name,
            description=f"Productivity tier above {productivity_threshold}th percentile",
            yield_percentile_min=productivity_threshold,
            yield_percentile_max=100,
            policy_applications=['productivity_analysis', 'efficiency_assessment'],
            target_users=['agricultural_analysts', 'efficiency_experts']
        )
        
        self.custom_tiers[name] = tier_config
        logger.info(f"Created productivity tier: {name}")
        
        return tier_config
    
    def get_tier(self, name: str) -> Optional[TierConfiguration]:
        """Get tier configuration by name."""
        
        return self.custom_tiers.get(name)
    
    def list_tiers(self) -> List[str]:
        """List all custom tier names."""
        
        return list(self.custom_tiers.keys())

class IntegrationPatterns:
    """
    Demonstration of common integration patterns for the multi-tier system.
    """
    
    @staticmethod
    def web_api_integration():
        """Demonstrate web API integration pattern."""
        
        print("\n🌐 WEB API INTEGRATION PATTERN")
        print("-" * 50)
        
        # Example Flask/FastAPI integration
        integration_code = '''
from flask import Flask, request, jsonify
from agririchter.analysis import MultiTierAPIWrapper

app = Flask(__name__)
api_wrapper = MultiTierAPIWrapper()

@app.route('/api/analyze', methods=['POST'])
def analyze_agricultural_capacity():
    """API endpoint for agricultural capacity analysis."""
    
    try:
        # Parse request data
        data = request.get_json()
        crop_data = load_crop_data_from_request(data)
        tier = data.get('tier', 'commercial')
        country_code = data.get('country_code')
        
        # Perform analysis
        results = api_wrapper.analyze_agricultural_capacity(
            crop_data=crop_data,
            tier=tier,
            country_code=country_code
        )
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error_message': str(e)
        }), 400

@app.route('/api/batch_analyze', methods=['POST'])
def batch_analyze_countries():
    """API endpoint for batch country analysis."""
    
    try:
        data = request.get_json()
        crop_data = load_crop_data_from_request(data)
        country_codes = data.get('country_codes', [])
        tier = data.get('tier', 'commercial')
        
        results = api_wrapper.batch_analyze_countries(
            crop_data=crop_data,
            country_codes=country_codes,
            tier=tier
        )
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error_message': str(e)
        }), 400
'''
        
        print("✅ Web API Integration Code:")
        print(integration_code)
    
    @staticmethod
    def database_integration():
        """Demonstrate database integration pattern."""
        
        print("\n🗄️  DATABASE INTEGRATION PATTERN")
        print("-" * 50)
        
        # Example database integration
        integration_code = '''
import sqlite3
import json
from datetime import datetime
from agririchter.analysis import MultiTierAPIWrapper

class AnalysisDatabase:
    """Database integration for storing analysis results."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT UNIQUE,
                    crop_name TEXT,
                    tier TEXT,
                    country_code TEXT,
                    results TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def store_analysis(self, analysis_id: str, crop_name: str, 
                      tier: str, country_code: str, results: dict):
        """Store analysis results in database."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO analysis_results 
                (analysis_id, crop_name, tier, country_code, results, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id,
                crop_name,
                tier,
                country_code,
                json.dumps(results['results']),
                json.dumps(results['metadata'])
            ))
    
    def get_analysis(self, analysis_id: str) -> dict:
        """Retrieve analysis results from database."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT results, metadata FROM analysis_results 
                WHERE analysis_id = ?
            ''', (analysis_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'results': json.loads(row[0]),
                    'metadata': json.loads(row[1])
                }
            return None
'''
        
        print("✅ Database Integration Code:")
        print(integration_code)
    
    @staticmethod
    def microservice_integration():
        """Demonstrate microservice integration pattern."""
        
        print("\n🔧 MICROSERVICE INTEGRATION PATTERN")
        print("-" * 50)
        
        # Example microservice integration
        integration_code = '''
import asyncio
import aiohttp
from agririchter.analysis import MultiTierAPIWrapper

class MultiTierMicroservice:
    """Microservice wrapper for multi-tier analysis."""
    
    def __init__(self, config: dict):
        self.config = config
        self.api_wrapper = MultiTierAPIWrapper(config)
    
    async def health_check(self):
        """Health check endpoint."""
        
        return {
            'status': 'healthy',
            'service': 'multi-tier-envelope',
            'version': '2.0.0',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def analyze_capacity(self, request_data: dict):
        """Asynchronous capacity analysis."""
        
        # Run analysis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        results = await loop.run_in_executor(
            None,
            self.api_wrapper.analyze_agricultural_capacity,
            request_data['crop_data'],
            request_data.get('tier', 'commercial'),
            request_data.get('country_code')
        )
        
        return results
    
    async def batch_analyze(self, request_data: dict):
        """Asynchronous batch analysis."""
        
        loop = asyncio.get_event_loop()
        
        results = await loop.run_in_executor(
            None,
            self.api_wrapper.batch_analyze_countries,
            request_data['crop_data'],
            request_data['country_codes'],
            request_data.get('tier', 'commercial')
        )
        
        return results

# Example usage with aiohttp
async def create_app():
    """Create microservice application."""
    
    app = aiohttp.web.Application()
    service = MultiTierMicroservice(config={})
    
    app.router.add_get('/health', lambda r: service.health_check())
    app.router.add_post('/analyze', lambda r: service.analyze_capacity(await r.json()))
    app.router.add_post('/batch', lambda r: service.batch_analyze(await r.json()))
    
    return app
'''
        
        print("✅ Microservice Integration Code:")
        print(integration_code)

def demo_production_api_wrapper():
    """Demonstrate production-ready API wrapper."""
    
    print("\n🏭 PRODUCTION API WRAPPER DEMO")
    print("=" * 50)
    
    # Create mock crop data
    np.random.seed(42)
    n_cells = 1000
    
    class MockCropDataset:
        def __init__(self):
            yields = np.random.lognormal(mean=1.5, sigma=0.8, size=n_cells)
            areas = np.random.exponential(scale=50, size=n_cells)
            self.production_kcal = pd.DataFrame({'production': yields * areas * 2500})
            self.harvest_km2 = pd.DataFrame({'harvest_area': areas})
            self.lat = np.random.uniform(25, 50, size=n_cells)
            self.lon = np.random.uniform(-125, -70, size=n_cells)
            self.crop_name = 'demo_wheat'
        def __len__(self):
            return len(self.production_kcal)
    
    crop_data = MockCropDataset()
    
    # Initialize API wrapper
    api_wrapper = MultiTierAPIWrapper({
        'cache_enabled': True,
        'performance_monitoring': True,
        'parallel_processing': True
    })
    
    print("✅ API Wrapper initialized with production configuration")
    
    # Single analysis
    print("\n📊 Single Analysis Demo:")
    result = api_wrapper.analyze_agricultural_capacity(
        crop_data=crop_data,
        tier='commercial',
        country_code='USA'
    )
    
    print(f"   • Status: {result['status']}")
    print(f"   • Analysis Type: {result['analysis_type']}")
    print(f"   • Execution Time: {result['metadata']['execution_time']:.2f}s")
    print(f"   • Data Cells: {result['metadata']['data_cells']}")
    
    # Batch analysis
    print("\n🌍 Batch Analysis Demo:")
    batch_results = api_wrapper.batch_analyze_countries(
        crop_data=crop_data,
        country_codes=['USA', 'CHN'],
        tier='commercial'
    )
    
    for country, result in batch_results.items():
        status = result.get('status', 'unknown')
        exec_time = result.get('metadata', {}).get('execution_time', 0)
        print(f"   • {country}: {status} ({exec_time:.2f}s)")
    
    # Performance metrics
    print("\n📈 Performance Metrics:")
    metrics = api_wrapper.get_performance_metrics()
    for metric_name, value in metrics.items():
        print(f"   • {metric_name}: {value}")

def demo_custom_tier_management():
    """Demonstrate custom tier management."""
    
    print("\n⚙️  CUSTOM TIER MANAGEMENT DEMO")
    print("=" * 50)
    
    # Initialize tier manager
    tier_manager = CustomTierManager()
    
    # Create policy-focused tiers
    print("🏛️  Creating Policy-Focused Tiers:")
    
    food_security_tier = tier_manager.create_policy_tier(
        name='Food Security Tier',
        yield_percentile_min=0,
        yield_percentile_max=100,
        policy_focus='food_security'
    )
    print(f"   • {food_security_tier.name}: {food_security_tier.description}")
    
    trade_tier = tier_manager.create_policy_tier(
        name='Export Trade Tier',
        yield_percentile_min=30,
        yield_percentile_max=100,
        policy_focus='trade'
    )
    print(f"   • {trade_tier.name}: {trade_tier.description}")
    
    # Create productivity-based tiers
    print("\n📈 Creating Productivity-Based Tiers:")
    
    high_productivity_tier = tier_manager.create_productivity_tier(
        name='High Productivity Tier',
        productivity_threshold=75
    )
    print(f"   • {high_productivity_tier.name}: {high_productivity_tier.description}")
    
    # List all custom tiers
    print(f"\n📋 All Custom Tiers:")
    for tier_name in tier_manager.list_tiers():
        tier = tier_manager.get_tier(tier_name)
        print(f"   • {tier_name}: {tier.yield_percentile_min}-{tier.yield_percentile_max}th percentile")

def demo_error_handling_patterns():
    """Demonstrate error handling and resilience patterns."""
    
    print("\n🚨 ERROR HANDLING PATTERNS DEMO")
    print("=" * 50)
    
    # Example error handling patterns
    error_handling_code = '''
from agririchter.exceptions import ValidationError, DataQualityError
import logging

class RobustAnalysisService:
    """Service with comprehensive error handling."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fallback_strategies = {
            'validation_error': self._handle_validation_error,
            'data_quality_error': self._handle_data_quality_error,
            'timeout_error': self._handle_timeout_error
        }
    
    def analyze_with_fallback(self, crop_data, tier='commercial'):
        """Analysis with automatic fallback strategies."""
        
        try:
            # Primary analysis attempt
            return self._primary_analysis(crop_data, tier)
            
        except ValidationError as e:
            self.logger.warning(f"Validation error: {e}")
            return self.fallback_strategies['validation_error'](crop_data, tier)
            
        except DataQualityError as e:
            self.logger.warning(f"Data quality error: {e}")
            return self.fallback_strategies['data_quality_error'](crop_data, tier)
            
        except TimeoutError as e:
            self.logger.warning(f"Timeout error: {e}")
            return self.fallback_strategies['timeout_error'](crop_data, tier)
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return self._emergency_fallback(crop_data, tier)
    
    def _handle_validation_error(self, crop_data, tier):
        """Handle validation errors by relaxing constraints."""
        
        # Try with less strict validation
        relaxed_config = {'validation_strict': False}
        return self._retry_analysis(crop_data, tier, relaxed_config)
    
    def _handle_data_quality_error(self, crop_data, tier):
        """Handle data quality errors by cleaning data."""
        
        # Apply additional data cleaning
        cleaned_data = self._clean_data(crop_data)
        return self._retry_analysis(cleaned_data, tier)
    
    def _handle_timeout_error(self, crop_data, tier):
        """Handle timeout by using simpler analysis."""
        
        # Fall back to comprehensive tier (simpler calculation)
        if tier != 'comprehensive':
            return self._retry_analysis(crop_data, 'comprehensive')
        else:
            return self._emergency_fallback(crop_data, tier)
'''
    
    print("✅ Error Handling Pattern Code:")
    print(error_handling_code)

def main():
    """Run technical integration demonstration."""
    
    print("🔧 Multi-Tier Envelope System: Technical Integration Demo")
    print("=" * 70)
    
    # Demo 1: Production API wrapper
    demo_production_api_wrapper()
    
    # Demo 2: Custom tier management
    demo_custom_tier_management()
    
    # Demo 3: Integration patterns
    integration_patterns = IntegrationPatterns()
    integration_patterns.web_api_integration()
    integration_patterns.database_integration()
    integration_patterns.microservice_integration()
    
    # Demo 4: Error handling patterns
    demo_error_handling_patterns()
    
    # Final summary
    print(f"\n🎉 TECHNICAL INTEGRATION DEMO COMPLETE")
    print("=" * 70)
    print(f"✅ Demonstrated production-ready API wrapper")
    print(f"⚙️  Showed custom tier management capabilities")
    print(f"🌐 Provided web API, database, and microservice integration patterns")
    print(f"🚨 Illustrated comprehensive error handling strategies")
    
    print(f"\n🛠️  For Developers:")
    print(f"   • Use MultiTierAPIWrapper for production deployments")
    print(f"   • Implement proper error handling and fallback strategies")
    print(f"   • Enable caching and performance monitoring")
    print(f"   • Consider parallel processing for batch operations")
    print(f"   • Validate inputs and outputs thoroughly")
    
    print(f"\n📚 Technical Resources:")
    print(f"   • Technical Guide: docs/MULTI_TIER_TECHNICAL_GUIDE.md")
    print(f"   • API Reference: docs/MULTI_TIER_API_REFERENCE.md")
    print(f"   • Integration Examples: examples/demo_technical_integration.py")

if __name__ == '__main__':
    main()