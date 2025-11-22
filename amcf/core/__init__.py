"""Core AMCF algorithms"""

from amcf.core.calibration import TemperatureScaler, CompetenceWeightManager
from amcf.core.fusion import LogitWeightedFusion
from amcf.core.bayes_risk import BayesRiskCalculator, ValueOfInformationEstimator, CostModel
from amcf.core.metacognitive_controller import MetacognitiveController, Event

__all__ = [
    'TemperatureScaler',
    'CompetenceWeightManager',
    'LogitWeightedFusion',
    'BayesRiskCalculator',
    'ValueOfInformationEstimator',
    'CostModel',
    'MetacognitiveController',
    'Event'
]