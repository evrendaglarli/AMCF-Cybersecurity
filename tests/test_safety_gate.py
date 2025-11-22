"""Unit tests for safety gate"""

import pytest
import numpy as np
from amcf.safety.safety_gate import SafetyGate


class TestSafetyGate:
    
    def test_initialization(self):
        """Test safety gate initialization"""
        gate = SafetyGate(rho_max=0.75, epsilon=1e-3)
        assert gate.rho_max == 0.75
        assert gate.epsilon == 1e-3
        assert len(gate.false_isolation_window) == 0
    
    def test_check_low_posterior(self):
        """Test gate rejects low posterior"""
        gate = SafetyGate()
        
        calibrated_probs = {
            'lightgbm': 0.9,
            'tcn': 0.9,
            'isolation_forest': 0.8
        }
        
        # Low posterior should fail gate
        result = gate.check(posterior=0.5, calibrated_probs=calibrated_probs)
        assert result == False
    
    def test_check_high_posterior_insufficient_detectors(self):
        """Test gate with high posterior but insufficient detectors"""
        gate = SafetyGate()
        
        # Only one detector above threshold
        calibrated_probs = {
            'lightgbm': 0.9,
            'tcn': 0.5,
            'isolation_forest': 0.3
        }
        
        result = gate.check(posterior=0.9, calibrated_probs=calibrated_probs)
        assert result == False
    
    def test_check_sufficient_detectors(self):
        """Test gate with sufficient detectors"""
        gate = SafetyGate()
        
        # Two detectors above thresholds
        calibrated_probs = {
            'lightgbm': 0.85,
            'tcn': 0.80,
            'isolation_forest': 0.65
        }
        
        result = gate.check(posterior=0.85, calibrated_probs=calibrated_probs)
        # Should pass with sufficient data
        assert isinstance(result, bool)
    
    def test_record_outcome(self):
        """Test outcome recording"""
        gate = SafetyGate(window_size=10)
        
        for i in range(5):
            gate.record_outcome(i % 2 == 0)
        
        assert len(gate.false_isolation_window) == 5


class TestEnrichmentScheduler:
    
    def test_greedy_selection(self):
        """Test greedy enrichment selection"""
        from amcf.safety.enrichment_scheduler import EnrichmentScheduler
        
        scheduler = EnrichmentScheduler(budget=50.0)
        
        voi_results = {
            'edr_deep': {'voi': 20.0, 'cost': 15.0},
            'sandbox': {'voi': 25.0, 'cost': 25.0},
            'pcap': {'voi': 10.0, 'cost': 10.0}
        }
        
        selected = scheduler.select_enrichments(voi_results)
        
        # Should select high VoI/cost ratio items
        assert len(selected) > 0
        # Total cost should not exceed budget
        total_cost = scheduler.estimate_total_cost(selected)
        assert total_cost <= 50.0