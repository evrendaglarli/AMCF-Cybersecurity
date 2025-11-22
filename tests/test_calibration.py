"""Unit tests for calibration module"""

import pytest
import numpy as np
from amcf.core.calibration import TemperatureScaler, CompetenceWeightManager


class TestTemperatureScaler:
    
    def test_initialization(self):
        """Test temperature scaler initialization"""
        scaler = TemperatureScaler('test_detector')
        assert scaler.detector_id == 'test_detector'
        assert scaler.temperature == 1.0
        assert len(scaler.sample_buffer) == 0
    
    def test_calibration(self):
        """Test temperature scaling"""
        scaler = TemperatureScaler('test')
        scores = np.array([-1.0, 0.0, 1.0])
        probs = scaler.calibrate(scores)
        
        assert probs.shape == scores.shape
        assert np.all((probs >= 0) & (probs <= 1))
        assert probs[1] == pytest.approx(0.5)  # logistic(0) = 0.5
    
    def test_update(self):
        """Test temperature update"""
        scaler = TemperatureScaler('test', update_frequency=10)
        
        # Generate 10 samples and labels
        for i in range(10):
            scores = np.array([float(i)])
            labels = np.array([i % 2])
            scaler.update(scores, labels)
        
        # Temperature should be updated
        assert scaler.temperature > 0
        assert scaler.temperature != 1.0  # Should differ from initial


class TestCompetenceWeightManager:
    
    def test_initialization(self):
        """Test weight manager initialization"""
        manager = CompetenceWeightManager(num_detectors=3)
        weights = manager.get_weights()
        
        assert len(weights) == 3
        assert np.isclose(weights.sum(), 1.0)
        assert np.allclose(weights, 1/3)  # Uniform initialization
    
    def test_update(self):
        """Test weight update with gradient"""
        manager = CompetenceWeightManager(num_detectors=3)
        
        # Generate detector probabilities
        probs = np.array([[0.1, 0.9, 0.5],
                         [0.2, 0.6, 0.5],
			 [0.2, 0.8, 0.4],
                         [0.3, 0.7, 0.6]])
        labels = np.array([0, 1, 1])
        
        initial_weights = manager.get_weights().copy()
        manager.update(probs, labels)
        updated_weights = manager.get_weights()
        
        # Weights should change
        assert not np.allclose(initial_weights, updated_weights)
        # Weights should still sum to 1
        assert np.isclose(updated_weights.sum(), 1.0)
        # All weights should be positive
        assert np.all(updated_weights > 0)
    
    def test_weight_normalization(self):
        """Test that weights are always normalized"""
        manager = CompetenceWeightManager(num_detectors=2)
        
        for _ in range(10):
            probs = np.random.rand(5, 2)
            labels = np.random.randint(0, 2, 5)
            manager.update(probs, labels)
            
            weights = manager.get_weights()
            assert np.isclose(weights.sum(), 1.0)
            assert np.all(weights >= 0.01)  # Minimum weight bound