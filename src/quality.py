"""Quality scoring and statistical analysis for paint marks.

This module provides objective quality assessment based on geometric
measurements, helping establish standards for "good" paint marks.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .config import get_config, QualityConfig


@dataclass
class QualityScore:
    """Quality assessment result for a single mark."""
    overall_score: float  # 0-100 composite score
    circularity_score: float  # 0-100
    solidity_score: float  # 0-100
    eccentricity_score: float  # 0-100
    edge_roughness_score: float  # 0-100
    grade: str  # 'excellent', 'good', 'marginal', 'poor'
    deviations: Dict[str, float]  # How many std devs from mean


@dataclass
class DistributionStats:
    """Statistical distribution for a metric."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q1: float  # 25th percentile
    q3: float  # 75th percentile
    count: int


class QualityAnalyzer:
    """Analyzes paint mark quality using statistical methods.
    
    This class provides:
    1. Composite quality scores based on multiple metrics
    2. Statistical distribution analysis (for bell curves)
    3. Sigma-based classification (excellent/good/marginal/poor)
    4. Reference baselines from collected data
    """
    
    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or get_config().quality
        
        # Cached statistics from baseline data
        self._baseline_stats: Optional[Dict[str, DistributionStats]] = None
    
    def compute_quality_score(self, 
                              circularity: float,
                              solidity: float,
                              eccentricity: float,
                              edge_roughness: float,
                              baseline_stats: Optional[Dict[str, DistributionStats]] = None) -> QualityScore:
        """
        Compute a composite quality score for a mark.
        
        Each metric is scored 0-100 based on how close it is to ideal.
        The overall score is a weighted average.
        
        Args:
            circularity: Mark circularity (0-1, higher is better)
            solidity: Mark solidity (0-1, higher is better)
            eccentricity: Mark eccentricity (0-1, lower is better)
            edge_roughness: Edge roughness (0-1, lower is better)
            baseline_stats: Optional baseline statistics for sigma calculation
            
        Returns:
            QualityScore with all component scores and grade
        """
        # Score each metric (0-100 scale)
        circ_score = self._score_higher_better(
            circularity, 
            self.config.circularity_min_acceptable,
            self.config.circularity_ideal
        )
        
        solid_score = self._score_higher_better(
            solidity,
            self.config.solidity_min_acceptable,
            self.config.solidity_ideal
        )
        
        ecc_score = self._score_lower_better(
            eccentricity,
            self.config.eccentricity_max_acceptable,
            self.config.eccentricity_ideal
        )
        
        rough_score = self._score_lower_better(
            edge_roughness,
            self.config.edge_roughness_max_acceptable,
            self.config.edge_roughness_ideal
        )
        
        # Weighted composite score
        overall = (
            circ_score * self.config.circularity_weight +
            solid_score * self.config.solidity_weight +
            ecc_score * self.config.eccentricity_weight +
            rough_score * self.config.edge_roughness_weight
        )
        
        # Calculate deviations from mean if baseline available
        deviations = {}
        if baseline_stats:
            for metric, value in [('circularity', circularity), 
                                   ('solidity', solidity),
                                   ('eccentricity', eccentricity),
                                   ('edge_roughness', edge_roughness)]:
                if metric in baseline_stats:
                    stats = baseline_stats[metric]
                    if stats.std > 0:
                        deviations[metric] = abs(value - stats.mean) / stats.std
                    else:
                        deviations[metric] = 0.0
        
        # Determine grade
        grade = self._score_to_grade(overall)
        
        return QualityScore(
            overall_score=overall,
            circularity_score=circ_score,
            solidity_score=solid_score,
            eccentricity_score=ecc_score,
            edge_roughness_score=rough_score,
            grade=grade,
            deviations=deviations
        )
    
    def _score_higher_better(self, value: float, min_acceptable: float, ideal: float) -> float:
        """Score a metric where higher values are better."""
        if value >= ideal:
            return 100.0
        elif min_acceptable <= 0 or value <= 0:
            # Handle edge case where min_acceptable is 0
            return 0.0
        elif value <= min_acceptable:
            # Linear scale from 0 at 0 to 50 at min_acceptable
            return max(0, (value / min_acceptable) * 50)
        else:
            # Linear scale from 50 at min_acceptable to 100 at ideal
            range_size = ideal - min_acceptable
            return 50 + ((value - min_acceptable) / range_size) * 50
    
    def _score_lower_better(self, value: float, max_acceptable: float, ideal: float) -> float:
        """Score a metric where lower values are better."""
        if value <= ideal:
            return 100.0
        elif value >= max_acceptable:
            # Penalty for exceeding max (50 points per unit excess, capped at 0)
            excess = value - max_acceptable
            return max(0, 50 - (excess * 50))
        else:
            # Linear scale from 100 at ideal to 50 at max_acceptable
            range_size = max_acceptable - ideal
            return 100 - ((value - ideal) / range_size) * 50
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to grade."""
        if score >= 85:
            return 'excellent'
        elif score >= 70:
            return 'good'
        elif score >= 50:
            return 'marginal'
        else:
            return 'poor'
    
    @staticmethod
    def compute_distribution_stats(values: List[float]) -> DistributionStats:
        """Compute statistical distribution for a set of values."""
        if not values:
            return DistributionStats(
                mean=0, std=0, min=0, max=0, 
                median=0, q1=0, q3=0, count=0
            )
        
        arr = np.array(values)
        return DistributionStats(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            median=float(np.median(arr)),
            q1=float(np.percentile(arr, 25)),
            q3=float(np.percentile(arr, 75)),
            count=len(arr)
        )
    
    @staticmethod
    def generate_normal_curve(mean: float, std: float, 
                              min_val: float, max_val: float,
                              num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate points for a normal distribution curve (bell curve).
        
        Args:
            mean: Distribution mean
            std: Distribution standard deviation
            min_val: Minimum x value
            max_val: Maximum x value
            num_points: Number of points to generate
            
        Returns:
            Tuple of (x_values, y_values) for plotting
        """
        if std <= 0:
            # Degenerate case - all same value, return empty arrays 
            # for the calling code to handle (skip bell curve in this case)
            x = np.array([mean])
            y = np.array([0.0])
            return x, y
        
        x = np.linspace(min_val, max_val, num_points)
        # Normal distribution PDF
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        
        return x, y
    
    @staticmethod
    def classify_by_sigma(value: float, mean: float, std: float, 
                          config: Optional[QualityConfig] = None) -> str:
        """
        Classify a value based on standard deviations from mean.
        
        Returns: 'excellent', 'good', 'marginal', or 'poor'
        """
        if config is None:
            config = get_config().quality
            
        if std <= 0:
            return 'good'  # Can't classify without variance
        
        sigma = abs(value - mean) / std
        
        if sigma <= config.sigma_excellent:
            return 'excellent'
        elif sigma <= config.sigma_good:
            return 'good'
        elif sigma <= config.sigma_marginal:
            return 'marginal'
        else:
            return 'poor'
    
    @staticmethod
    def get_sigma_thresholds(mean: float, std: float, 
                             config: Optional[QualityConfig] = None) -> Dict[str, Tuple[float, float]]:
        """
        Get the value ranges for each quality grade.
        
        Returns dict with grade -> (lower_bound, upper_bound)
        """
        if config is None:
            config = get_config().quality
        
        return {
            'excellent': (mean - config.sigma_excellent * std, 
                         mean + config.sigma_excellent * std),
            'good': (mean - config.sigma_good * std,
                    mean + config.sigma_good * std),
            'marginal': (mean - config.sigma_marginal * std,
                        mean + config.sigma_marginal * std),
        }


def compute_batch_quality_scores(marks_data: List[dict]) -> List[dict]:
    """
    Compute quality scores for a batch of marks.
    
    Args:
        marks_data: List of mark dictionaries with measurement values
        
    Returns:
        List of marks with added quality score fields
    """
    if not marks_data:
        return []
    
    analyzer = QualityAnalyzer()
    
    # Compute baseline stats from the data
    baseline_stats = {
        'circularity': analyzer.compute_distribution_stats(
            [m['circularity'] for m in marks_data if 'circularity' in m]
        ),
        'solidity': analyzer.compute_distribution_stats(
            [m['solidity'] for m in marks_data if 'solidity' in m]
        ),
        'eccentricity': analyzer.compute_distribution_stats(
            [m['eccentricity'] for m in marks_data if 'eccentricity' in m]
        ),
        'edge_roughness': analyzer.compute_distribution_stats(
            [m['edge_roughness'] for m in marks_data if 'edge_roughness' in m]
        ),
    }
    
    # Score each mark
    results = []
    for mark in marks_data:
        mark_copy = dict(mark)
        
        score = analyzer.compute_quality_score(
            circularity=mark.get('circularity', 0),
            solidity=mark.get('solidity', 0),
            eccentricity=mark.get('eccentricity', 0),
            edge_roughness=mark.get('edge_roughness', 0),
            baseline_stats=baseline_stats
        )
        
        mark_copy['quality_score'] = score.overall_score
        mark_copy['quality_grade'] = score.grade
        mark_copy['circularity_score'] = score.circularity_score
        mark_copy['solidity_score'] = score.solidity_score
        mark_copy['eccentricity_score'] = score.eccentricity_score
        mark_copy['edge_roughness_score'] = score.edge_roughness_score
        
        results.append(mark_copy)
    
    return results
