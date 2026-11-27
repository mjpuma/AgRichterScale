"""
Envelope Calculation Caching System

Provides intelligent caching for envelope calculations to improve performance
for repeated analyses with the same data.
"""

import logging
import pickle
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


class EnvelopeCalculationCache:
    """
    Intelligent caching for envelope calculations.
    
    Provides persistent caching of envelope calculation results to avoid
    recomputation when analyzing the same data with the same parameters.
    """
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize envelope calculation cache.
        
        Args:
            cache_dir: Directory for cache files. If None, uses .cache/envelopes
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / '.cache' / 'envelopes'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()
        
        logger.info(f"EnvelopeCalculationCache initialized at {self.cache_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                import json
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        return {
            'created': time.time(),
            'cache_hits': 0,
            'cache_misses': 0,
            'entries': {}
        }
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            import json
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _generate_data_hash(self, production_df: pd.DataFrame, 
                           harvest_df: pd.DataFrame) -> str:
        """
        Generate hash for input data.
        
        Args:
            production_df: Production data
            harvest_df: Harvest area data
            
        Returns:
            Hash string representing the data
        """
        # Create hash from data shape, columns, and sample of values
        prod_info = f"{production_df.shape}_{list(production_df.columns)}"
        harv_info = f"{harvest_df.shape}_{list(harvest_df.columns)}"
        
        # Sample some values for hash (to detect data changes)
        if len(production_df) > 0:
            prod_sample = production_df.iloc[::max(1, len(production_df)//100)].values.flatten()
            prod_sample_str = str(prod_sample[:50])  # First 50 values
        else:
            prod_sample_str = "empty"
        
        if len(harvest_df) > 0:
            harv_sample = harvest_df.iloc[::max(1, len(harvest_df)//100)].values.flatten()
            harv_sample_str = str(harv_sample[:50])  # First 50 values
        else:
            harv_sample_str = "empty"
        
        combined_str = f"{prod_info}_{harv_info}_{prod_sample_str}_{harv_sample_str}"
        
        return hashlib.md5(combined_str.encode()).hexdigest()
    
    def _generate_cache_key(self, data_hash: str, tier: str, 
                           country_code: Optional[str] = None,
                           additional_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate cache key for envelope calculation.
        
        Args:
            data_hash: Hash of input data
            tier: Tier name
            country_code: Optional country code
            additional_params: Additional parameters affecting calculation
            
        Returns:
            Cache key string
        """
        key_parts = [data_hash, tier]
        
        if country_code:
            key_parts.append(f"country_{country_code}")
        
        if additional_params:
            # Sort parameters for consistent key generation
            param_str = "_".join(f"{k}_{v}" for k, v in sorted(additional_params.items()))
            key_parts.append(param_str)
        
        return "_".join(key_parts)
    
    def get_cached_result(self, production_df: pd.DataFrame, harvest_df: pd.DataFrame,
                         tier: str, country_code: Optional[str] = None,
                         additional_params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Retrieve cached envelope calculation if available.
        
        Args:
            production_df: Production data
            harvest_df: Harvest area data
            tier: Tier name
            country_code: Optional country code
            additional_params: Additional parameters
            
        Returns:
            Cached result or None if not found
        """
        try:
            data_hash = self._generate_data_hash(production_df, harvest_df)
            cache_key = self._generate_cache_key(data_hash, tier, country_code, additional_params)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                # Check if cache entry is still valid
                if self._is_cache_valid(cache_key):
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    
                    self.metadata['cache_hits'] += 1
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return result
                else:
                    # Remove invalid cache entry
                    cache_file.unlink()
                    if cache_key in self.metadata['entries']:
                        del self.metadata['entries'][cache_key]
            
            self.metadata['cache_misses'] += 1
            logger.debug(f"Cache miss for key: {cache_key}")
            return None
            
        except Exception as e:
            logger.warning(f"Error retrieving cached result: {e}")
            return None
    
    def cache_result(self, result: Any, production_df: pd.DataFrame, harvest_df: pd.DataFrame,
                    tier: str, country_code: Optional[str] = None,
                    additional_params: Optional[Dict[str, Any]] = None):
        """
        Cache envelope calculation result.
        
        Args:
            result: Calculation result to cache
            production_df: Production data
            harvest_df: Harvest area data
            tier: Tier name
            country_code: Optional country code
            additional_params: Additional parameters
        """
        try:
            data_hash = self._generate_data_hash(production_df, harvest_df)
            cache_key = self._generate_cache_key(data_hash, tier, country_code, additional_params)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            # Save result
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            # Update metadata
            self.metadata['entries'][cache_key] = {
                'created': time.time(),
                'tier': tier,
                'country_code': country_code,
                'data_hash': data_hash,
                'file_size': cache_file.stat().st_size
            }
            
            self._save_metadata()
            logger.debug(f"Cached result for key: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Error caching result: {e}")
    
    def _is_cache_valid(self, cache_key: str, max_age_hours: float = 24) -> bool:
        """
        Check if cache entry is still valid.
        
        Args:
            cache_key: Cache key to check
            max_age_hours: Maximum age in hours
            
        Returns:
            True if cache is valid
        """
        if cache_key not in self.metadata['entries']:
            return False
        
        entry = self.metadata['entries'][cache_key]
        age_hours = (time.time() - entry['created']) / 3600
        
        return age_hours < max_age_hours
    
    def clear_cache(self, older_than_hours: Optional[float] = None):
        """
        Clear cache entries.
        
        Args:
            older_than_hours: If specified, only clear entries older than this
        """
        cleared_count = 0
        
        if older_than_hours is None:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
                cleared_count += 1
            
            self.metadata['entries'].clear()
        else:
            # Clear only old entries
            current_time = time.time()
            keys_to_remove = []
            
            for cache_key, entry in self.metadata['entries'].items():
                age_hours = (current_time - entry['created']) / 3600
                if age_hours > older_than_hours:
                    cache_file = self.cache_dir / f"{cache_key}.pkl"
                    if cache_file.exists():
                        cache_file.unlink()
                    keys_to_remove.append(cache_key)
                    cleared_count += 1
            
            for key in keys_to_remove:
                del self.metadata['entries'][key]
        
        self._save_metadata()
        logger.info(f"Cleared {cleared_count} cache entries")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_size = sum(
            (self.cache_dir / f"{key}.pkl").stat().st_size 
            for key in self.metadata['entries'].keys()
            if (self.cache_dir / f"{key}.pkl").exists()
        )
        
        hit_rate = 0.0
        total_requests = self.metadata['cache_hits'] + self.metadata['cache_misses']
        if total_requests > 0:
            hit_rate = self.metadata['cache_hits'] / total_requests
        
        return {
            'cache_entries': len(self.metadata['entries']),
            'cache_hits': self.metadata['cache_hits'],
            'cache_misses': self.metadata['cache_misses'],
            'hit_rate': hit_rate,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_directory': str(self.cache_dir)
        }
    
    def __repr__(self) -> str:
        stats = self.get_cache_statistics()
        return (
            f"EnvelopeCalculationCache("
            f"entries={stats['cache_entries']}, "
            f"hit_rate={stats['hit_rate']:.2%}, "
            f"size={stats['total_size_mb']:.1f}MB)"
        )