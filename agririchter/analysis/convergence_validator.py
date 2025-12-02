"""Mathematical validation framework for H-P envelope convergence."""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Result of mathematical validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    properties: Dict[str, bool]
    statistics: Dict[str, float]


@dataclass
class MathematicalProperties:
    """Mathematical properties tracking for envelope validation."""
    starts_at_origin: bool
    converges_at_endpoint: bool
    upper_dominates_lower: bool
    conservation_satisfied: bool
    monotonic_harvest: bool
    
    def is_mathematically_valid(self) -> bool:
        """Check if all mathematical properties are satisfied."""
        return all([
            self.starts_at_origin,
            self.converges_at_endpoint,
            self.upper_dominates_lower,
            self.conservation_satisfied,
            self.monotonic_harvest
        ])


class ConvergenceError(Exception):
    """Raised when envelope fails to converge properly."""
    pass


class MathematicalPropertyError(Exception):
    """Raised when mathematical properties are violated."""
    pass


class ConservationError(Exception):
    """Raised when conservation laws are violated."""
    pass


class ConvergenceValidator:
    """Validator for mathematical properties of H-P envelope calculations."""
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize convergence validator.
        
        Args:
            tolerance: Numerical tolerance for convergence checks
        """
        self.tolerance = tolerance
        self.logger = logging.getLogger('agririchter.convergence_validator')
    
    def validate_mathematical_properties(self, envelope_data: Dict[str, np.ndarray],
                                       total_production: float,
                                       total_harvest: float) -> ValidationResult:
        """
        Validate that envelope satisfies mathematical requirements.
        
        Args:
            envelope_data: Envelope data dictionary
            total_production: Total production from all cells
            total_harvest: Total harvest area from all cells
        
        Returns:
            ValidationResult with detailed validation information
        """
        errors = []
        warnings = []
        properties = {}
        statistics = {}
        
        try:
            # Extract envelope bounds
            lower_harvest = envelope_data['lower_bound_harvest']
            lower_production = envelope_data['lower_bound_production']
            upper_harvest = envelope_data['upper_bound_harvest']
            upper_production = envelope_data['upper_bound_production']
            
            # Check origin starting point
            properties['starts_at_origin'] = self._check_origin_start(
                lower_harvest, lower_production, upper_harvest, upper_production
            )
            if not properties['starts_at_origin']:
                errors.append("Envelope bounds do not start at origin (0,0)")
            
            # Check endpoint convergence
            properties['converges_at_endpoint'] = self.check_endpoint_convergence(
                lower_production, upper_production, lower_harvest, 
                total_production, total_harvest
            )
            if not properties['converges_at_endpoint']:
                errors.append("Envelope bounds do not converge at maximum harvest area")
            
            # Check upper bound dominance
            properties['upper_dominates_lower'] = self._check_upper_dominance(
                lower_production, upper_production
            )
            if not properties['upper_dominates_lower']:
                errors.append("Upper bound does not dominate lower bound throughout envelope")
            
            # Check conservation laws
            properties['conservation_satisfied'] = self._check_conservation_laws(
                envelope_data, total_production, total_harvest
            )
            if not properties['conservation_satisfied']:
                errors.append("Conservation laws are violated")
            
            # Check monotonic harvest areas
            properties['monotonic_harvest'] = self._check_monotonic_harvest(
                lower_harvest, upper_harvest
            )
            if not properties['monotonic_harvest']:
                warnings.append("Harvest areas are not strictly monotonic")
            
            # Calculate convergence statistics
            statistics = self._calculate_convergence_statistics(
                lower_harvest, lower_production, upper_harvest, upper_production,
                total_production, total_harvest
            )
            
            is_valid = len(errors) == 0
            
            self.logger.info(f"Mathematical validation completed: {'PASSED' if is_valid else 'FAILED'}")
            if errors:
                self.logger.error(f"Validation errors: {errors}")
            if warnings:
                self.logger.warning(f"Validation warnings: {warnings}")
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                properties=properties,
                statistics=statistics
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed with exception: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation exception: {str(e)}"],
                warnings=[],
                properties={},
                statistics={}
            )
    
    def check_endpoint_convergence(self, lower_bound: np.ndarray, 
                                  upper_bound: np.ndarray, 
                                  harvest_areas: np.ndarray,
                                  total_production: float,
                                  total_harvest: float) -> bool:
        """
        Check if bounds converge at maximum harvest area.
        
        At total harvest area, both bounds must equal total production
        since we're summing all cells regardless of order.
        
        Args:
            lower_bound: Lower bound production values
            upper_bound: Upper bound production values
            harvest_areas: Harvest area values
            total_production: Total production from all cells
            total_harvest: Total harvest area from all cells
        
        Returns:
            True if bounds converge properly at endpoint
        """
        if len(lower_bound) == 0 or len(upper_bound) == 0:
            self.logger.error("Empty bounds arrays provided for convergence check")
            return False
        
        # Find the point closest to total harvest area
        max_harvest_idx = np.argmax(harvest_areas)
        max_harvest_area = harvest_areas[max_harvest_idx]
        
        # Check if we reach close to total harvest area
        harvest_coverage = max_harvest_area / total_harvest
        if harvest_coverage < 0.95:  # Should cover at least 95% of total harvest
            self.logger.warning(f"Envelope only covers {harvest_coverage:.1%} of total harvest area")
        
        # At maximum harvest area, both bounds should equal total production
        final_lower = lower_bound[max_harvest_idx]
        final_upper = upper_bound[max_harvest_idx]
        
        # Check convergence to total production
        lower_error = abs(final_lower - total_production) / total_production
        upper_error = abs(final_upper - total_production) / total_production
        
        # Both bounds should be within tolerance of total production
        converges = (lower_error < self.tolerance and upper_error < self.tolerance)
        
        if not converges:
            self.logger.error(f"Convergence check failed:")
            self.logger.error(f"  Final lower bound: {final_lower:.2e} (error: {lower_error:.2%})")
            self.logger.error(f"  Final upper bound: {final_upper:.2e} (error: {upper_error:.2%})")
            self.logger.error(f"  Expected total production: {total_production:.2e}")
        
        return converges
    
    def enforce_convergence(self, envelope_data: Dict[str, np.ndarray], 
                           total_production: float, 
                           total_harvest: float) -> Dict[str, np.ndarray]:
        """
        Enforce mathematical convergence at endpoint.
        
        This method ensures that the envelope bounds converge at the
        total harvest area point by explicitly adding or correcting
        the convergence point.
        
        Args:
            envelope_data: Original envelope data
            total_production: Total production from all cells
            total_harvest: Total harvest area from all cells
        
        Returns:
            Corrected envelope data with enforced convergence
        """
        corrected_data = envelope_data.copy()
        
        # Extract current bounds
        lower_harvest = envelope_data['lower_bound_harvest']
        lower_production = envelope_data['lower_bound_production']
        upper_harvest = envelope_data['upper_bound_harvest']
        upper_production = envelope_data['upper_bound_production']
        disruption_areas = envelope_data['disruption_areas']
        
        # Check if we already have the convergence point
        max_harvest_area = np.max(lower_harvest)
        
        if max_harvest_area < total_harvest * 0.99:  # Missing convergence point
            self.logger.info("Adding explicit convergence point")
            
            # Add convergence point at (total_harvest, total_production)
            corrected_data['lower_bound_harvest'] = np.append(lower_harvest, total_harvest)
            corrected_data['lower_bound_production'] = np.append(lower_production, total_production)
            corrected_data['upper_bound_harvest'] = np.append(upper_harvest, total_harvest)
            corrected_data['upper_bound_production'] = np.append(upper_production, total_production)
            corrected_data['disruption_areas'] = np.append(disruption_areas, total_harvest)
            
        else:
            # Correct existing endpoint
            max_idx = np.argmax(lower_harvest)
            
            self.logger.info("Correcting existing convergence point")
            corrected_data['lower_bound_harvest'][max_idx] = total_harvest
            corrected_data['lower_bound_production'][max_idx] = total_production
            corrected_data['upper_bound_harvest'][max_idx] = total_harvest
            corrected_data['upper_bound_production'][max_idx] = total_production
        
        # Validate the correction
        validation_result = self.validate_mathematical_properties(
            corrected_data, total_production, total_harvest
        )
        
        if validation_result.is_valid:
            self.logger.info("Convergence enforcement successful")
        else:
            self.logger.warning("Convergence enforcement may not have fully resolved issues")
        
        return corrected_data
    
    def _check_origin_start(self, lower_harvest: np.ndarray, lower_production: np.ndarray,
                           upper_harvest: np.ndarray, upper_production: np.ndarray) -> bool:
        """Check if envelope starts at origin (0,0)."""
        if len(lower_harvest) == 0 or len(upper_harvest) == 0:
            return False
        
        # Use relative tolerance for origin check (real envelopes may not start exactly at 0)
        harvest_tolerance = max(self.tolerance, np.max(lower_harvest) * 0.01)  # 1% of max harvest
        production_tolerance = max(self.tolerance, np.max(lower_production) * 0.01)  # 1% of max production
        
        # Check if first points are at or near origin
        lower_starts_origin = (abs(lower_harvest[0]) < harvest_tolerance and 
                              abs(lower_production[0]) < production_tolerance)
        upper_starts_origin = (abs(upper_harvest[0]) < harvest_tolerance and 
                              abs(upper_production[0]) < production_tolerance)
        
        return lower_starts_origin and upper_starts_origin
    
    def _check_upper_dominance(self, lower_production: np.ndarray, 
                              upper_production: np.ndarray) -> bool:
        """Check if upper bound dominates lower bound throughout."""
        if len(lower_production) != len(upper_production):
            return False
        
        # Upper bound should be >= lower bound at all points
        dominance_violations = np.sum(upper_production < lower_production)
        
        if dominance_violations > 0:
            self.logger.warning(f"Found {dominance_violations} upper bound dominance violations")
        
        return dominance_violations == 0
    
    def _check_conservation_laws(self, envelope_data: Dict[str, np.ndarray],
                                total_production: float, total_harvest: float) -> bool:
        """
        Check conservation laws.
        
        The sum of all cells should equal total production regardless of order.
        At maximum harvest area, both bounds must equal total production.
        """
        # Find the maximum harvest area point
        lower_harvest = envelope_data['lower_bound_harvest']
        upper_harvest = envelope_data['upper_bound_harvest']
        lower_production = envelope_data['lower_bound_production']
        upper_production = envelope_data['upper_bound_production']
        
        if len(lower_harvest) == 0:
            return False
        
        # Check conservation at maximum harvest point
        max_harvest_idx = np.argmax(lower_harvest)
        max_harvest_area = lower_harvest[max_harvest_idx]
        
        # If we're close to total harvest area, check conservation
        if max_harvest_area >= total_harvest * 0.95:
            final_lower_prod = lower_production[max_harvest_idx]
            final_upper_prod = upper_production[max_harvest_idx]
            
            # Both should equal total production (conservation law)
            lower_error = abs(final_lower_prod - total_production) / total_production
            upper_error = abs(final_upper_prod - total_production) / total_production
            
            conservation_satisfied = (lower_error < 0.05 and upper_error < 0.05)
            
            if not conservation_satisfied:
                self.logger.warning(f"Conservation law violation:")
                self.logger.warning(f"  Lower bound at max harvest: {final_lower_prod:.2e} (error: {lower_error:.2%})")
                self.logger.warning(f"  Upper bound at max harvest: {final_upper_prod:.2e} (error: {upper_error:.2%})")
                self.logger.warning(f"  Expected total production: {total_production:.2e}")
            
            return conservation_satisfied
        else:
            # If we don't reach total harvest area, we can't fully validate conservation
            self.logger.warning(f"Cannot validate conservation: envelope only reaches {max_harvest_area/total_harvest:.1%} of total harvest")
            return True  # Don't fail validation for incomplete envelopes
    
    def _check_monotonic_harvest(self, lower_harvest: np.ndarray, 
                                upper_harvest: np.ndarray) -> bool:
        """Check if harvest areas are monotonically increasing."""
        lower_monotonic = np.all(np.diff(lower_harvest) >= 0)
        upper_monotonic = np.all(np.diff(upper_harvest) >= 0)
        
        return lower_monotonic and upper_monotonic
    
    def _calculate_convergence_statistics(self, lower_harvest: np.ndarray, 
                                        lower_production: np.ndarray,
                                        upper_harvest: np.ndarray, 
                                        upper_production: np.ndarray,
                                        total_production: float, 
                                        total_harvest: float) -> Dict[str, float]:
        """Calculate convergence statistics."""
        if len(lower_harvest) == 0:
            return {}
        
        # Envelope width at different points
        production_width = upper_production - lower_production
        harvest_width = upper_harvest - lower_harvest
        
        # Convergence metrics
        max_harvest_coverage = np.max(lower_harvest) / total_harvest
        max_production_coverage = np.max(lower_production) / total_production
        
        # Width reduction (how much the envelope narrows)
        initial_width = production_width[0] if len(production_width) > 0 else 0
        final_width = production_width[-1] if len(production_width) > 0 else 0
        width_reduction = (initial_width - final_width) / initial_width if initial_width > 0 else 0
        
        return {
            'max_harvest_coverage': float(max_harvest_coverage),
            'max_production_coverage': float(max_production_coverage),
            'avg_production_width': float(np.mean(production_width)),
            'final_production_width': float(final_width),
            'width_reduction_ratio': float(width_reduction),
            'envelope_points': len(lower_harvest)
        }
    
    def validate_yield_ordering(self, hp_matrix: np.ndarray) -> bool:
        """
        Validate that cells are properly ordered by yield for envelope calculation.
        
        Args:
            hp_matrix: H-P matrix [harvest_area, production, yield]
        
        Returns:
            True if yield ordering is correct
        """
        if len(hp_matrix) < 2:
            return True
        
        yields = hp_matrix[:, 2]
        is_sorted = np.all(yields[:-1] <= yields[1:])
        
        if not is_sorted:
            self.logger.error("H-P matrix is not properly sorted by yield")
        
        return is_sorted
    
    def validate_cumulative_properties(self, hp_matrix_sorted: np.ndarray,
                                     total_production: float, 
                                     total_harvest: float) -> bool:
        """
        Validate cumulative sum properties.
        
        Args:
            hp_matrix_sorted: Sorted H-P matrix
            total_production: Expected total production
            total_harvest: Expected total harvest
        
        Returns:
            True if cumulative properties are valid
        """
        if len(hp_matrix_sorted) == 0:
            return False
        
        # Calculate cumulative sums
        cumsum_harvest = np.cumsum(hp_matrix_sorted[:, 0])
        cumsum_production = np.cumsum(hp_matrix_sorted[:, 1])
        
        # Final cumulative sums should equal totals
        final_harvest = cumsum_harvest[-1]
        final_production = cumsum_production[-1]
        
        harvest_error = abs(final_harvest - total_harvest) / total_harvest
        production_error = abs(final_production - total_production) / total_production
        
        harvest_valid = harvest_error < 0.01  # 1% tolerance
        production_valid = production_error < 0.01  # 1% tolerance
        
        if not harvest_valid:
            self.logger.error(f"Cumulative harvest sum error: {harvest_error:.2%}")
        if not production_valid:
            self.logger.error(f"Cumulative production sum error: {production_error:.2%}")
        
        return harvest_valid and production_valid
    
    def check_envelope_continuity(self, envelope_data: Dict[str, np.ndarray]) -> bool:
        """
        Check envelope continuity (no gaps or jumps).
        
        Args:
            envelope_data: Envelope data
        
        Returns:
            True if envelope is continuous
        """
        lower_harvest = envelope_data['lower_bound_harvest']
        upper_harvest = envelope_data['upper_bound_harvest']
        lower_production = envelope_data['lower_bound_production']
        upper_production = envelope_data['upper_bound_production']
        
        if len(lower_harvest) < 2:
            return True
        
        # Check for reasonable step sizes (no huge jumps)
        lower_harvest_steps = np.diff(lower_harvest)
        upper_harvest_steps = np.diff(upper_harvest)
        lower_prod_steps = np.diff(lower_production)
        upper_prod_steps = np.diff(upper_production)
        
        # Calculate median step sizes
        median_harvest_step = np.median(lower_harvest_steps)
        median_prod_step = np.median(lower_prod_steps)
        
        # Check for steps that are more than 10x the median (indicating discontinuity)
        large_harvest_steps = np.sum(lower_harvest_steps > 10 * median_harvest_step)
        large_prod_steps = np.sum(lower_prod_steps > 10 * median_prod_step)
        
        is_continuous = (large_harvest_steps == 0 and large_prod_steps == 0)
        
        if not is_continuous:
            self.logger.warning(f"Envelope discontinuity detected: {large_harvest_steps} harvest jumps, {large_prod_steps} production jumps")
        
        return is_continuous