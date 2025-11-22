"""Unit tests for Bayes risk module"""

import pytest
import numpy as np
from amcf.core.bayes_risk import BayesRiskCalculator, ValueOfInformationEstimator, CostModel


class TestBayesRiskCalculator:
    
    def test_initialization(self):
        """Test calculator initialization"""
        cost_model = CostModel()
        calculator = BayesRiskCalculator(cost_model)
        assert calculator.costs.C_FP == 1.0
        assert calculator.costs.C_FN_med == 15.0
    
    def test_compute_risks(self):
        """Test risk computation"""
        calculator = BayesRiskCalculator(CostModel())
        
        risks = calculator.compute_risks(
            posterior=0.8,
            severity='high',
            asset_weight=1.0
        )
        
        assert 'ignore' in risks
        assert 'automate' in risks
        assert 'escalate' in risks
        assert 'gather' in risks
        
        # All risks should be non-negative
        assert all(r >= 0 for r in risks.values())
    
    def test_select_action(self):
        """Test action selection"""
        calculator = BayesRiskCalculator(CostModel())
        
        risks = {
            'ignore': 10.0,
            'automate': 5.0,
            'escalate': 8.0,
            'gather': 0.0
        }
        
        # Should select minimum risk action
        action = calculator.select_action(risks, voi_positive=False, high_uncertainty=False)
        assert action == 'automate'
    
    def test_action_selection_with_high_uncertainty(self):
        """Test action selection under uncertainty"""
        calculator = BayesRiskCalculator(CostModel())
        
        risks = {
            'ignore': 10.0,
            'automate': 5.0,
            'escalate': 8.0,
            'gather': 0.0
        }
        
        # With high uncertainty and positive VoI, should gather
        action = calculator.select_action(risks, voi_positive=True, high_uncertainty=True)
        assert action == 'gather'


class TestValueOfInformationEstimator:
    
    def test_initialization(self):
        """Test VoI estimator initialization"""
        cost_model = CostModel()
        risk_calc = BayesRiskCalculator(cost_model)
        estimator = ValueOfInformationEstimator(risk_calc)
        
        assert estimator.N_min == 64
        assert estimator.N_max == 4096
        assert estimator.tolerance == 0.02
    
    def test_estimate_voi(self):
        """Test VoI estimation"""
        cost_model = CostModel()
        risk_calc = BayesRiskCalculator(cost_model)
        estimator = ValueOfInformationEstimator(risk_calc)
        
        voi, n_samples = estimator.estimate_voi(
            posterior_before=0.6,
            enrichment_cost=10.0,
            enrichment_type='edr_deep',
            severity='high'
        )
        
        assert isinstance(voi, float)
        assert isinstance(n_samples, int)
        assert n_samples >= estimator.N_min
        assert n_samples <= estimator.N_max