"""End-to-end integration tests"""

import pytest
import numpy as np
from amcf.core.metacognitive_controller import (
    MetacognitiveController, Event, CostModel
)


class TestMetacognitiveController:
    
    def test_initialization(self):
        """Test controller initialization"""
        controller = MetacognitiveController(
            num_detectors=3,
            detector_ids=['lightgbm', 'tcn', 'isolation_forest'],
            cost_model=CostModel(),
            config={}
        )
        
        assert controller.num_detectors == 3
        assert len(controller.detector_ids) == 3
    
    def test_process_event(self):
        """Test processing a single event"""
        controller = MetacognitiveController(
            num_detectors=3,
            detector_ids=['lightgbm', 'tcn', 'isolation_forest'],
            cost_model=CostModel(),
            config={}
        )
        
        # Create mock event
        event = Event(
            ts=1718190000000,
            src="host1",
            dst="host2",
            kind="process",
            attrs={'bytes_sent': 1000},
            detector_scores={
                'lightgbm': 0.5,
                'tcn': 0.6,
                'isolation_forest': 0.4
            },
            label=None
        )
        
        action, metadata = controller.process_event(event)
        
        assert action in ['ignore', 'gather', 'automate', 'escalate']
        assert 'posterior' in metadata
        assert 'risks' in metadata
        assert 0 <= metadata['posterior'] <= 1
    
    def test_feedback_update(self):
        """Test updating from feedback"""
        controller = MetacognitiveController(
            num_detectors=3,
            detector_ids=['lightgbm', 'tcn', 'isolation_forest'],
            cost_model=CostModel(),
            config={}
        )
        
        # Create event with label
        event = Event(
            ts=1718190000000,
            src="host1",
            dst="host2",
            kind="process",
            attrs={},
            detector_scores={
                'lightgbm': 0.8,
                'tcn': 0.7,
                'isolation_forest': 0.6
            },
            label=1
        )
        
        # Process event (will trigger feedback update)
        action, metadata = controller.process_event(event)
        
        # Weights should be updated
        assert controller.weight_manager.get_weights().sum() > 0