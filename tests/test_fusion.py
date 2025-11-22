"""Unit tests for fusion module"""

import pytest
import numpy as np
from amcf.core.fusion import LogitWeightedFusion


class TestLogitWeightedFusion:
    
    def test_initialization(self):
        """Test fusion initialization"""
        fusion = LogitWeightedFusion(num_detectors=3)
        assert fusion.num_detectors == 3
    
    def test_fusion_shape(self):
        """Test output shape"""
        fusion = LogitWeightedFusion(num_detectors=3)
        
        probs = np.array([[0.3, 0.5, 0.7],
                         [0.4, 0.6, 0.2],
                         [0.1, 0.9, 0.5]])
        weights = np.array([0.3, 0.4, 0.3])
        
        posterior = fusion.fuse(probs, weights)
        
        assert posterior.shape == (3,)
        assert np.all((posterior >= 0) & (posterior <= 1))
    
    def test_fusion_uniform_weights(self):
        """Test fusion with uniform weights (should average)"""
        fusion = LogitWeightedFusion(num_detectors=2)
        
        probs = np.array([[0.2, 0.8]])
        weights = np.array([0.5, 0.5])
        
        posterior = fusion.fuse(probs, weights)
        
        # Logit aggregation of [0.2, 0.8] with uniform weights
        # Should be close to 0.5 on logit scale
        assert 0.4 < posterior[0] < 0.6
    
    def test_fusion_with_bias(self):
        """Test fusion with contextual bias"""
        fusion = LogitWeightedFusion(num_detectors=2)
        
        probs = np.array([[0.5, 0.5]])
        weights = np.array([0.5, 0.5])
        
        # Without bias
        posterior_no_bias = fusion.fuse(probs, weights, bias=0.0)
        
        # With positive bias
        posterior_with_bias = fusion.fuse(probs, weights, bias=1.0)
        
        # Bias should increase posterior
        assert posterior_with_bias[0] > posterior_no_bias[0]
    
    def test_fusion_boundary_values(self):
        """Test fusion with extreme probabilities"""
        fusion = LogitWeightedFusion(num_detectors=2)
        
        # Very low probabilities
        probs_low = np.array([[0.01, 0.01]])
        weights = np.array([0.5, 0.5])
        posterior_low = fusion.fuse(probs_low, weights)
        
        assert posterior_low[0] < 0.1
        
        # Very high probabilities
        probs_high = np.array([[0.99, 0.99]])
        posterior_high = fusion.fuse(probs_high, weights)
        
        assert posterior_high[0] > 0.9