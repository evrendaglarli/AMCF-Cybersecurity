
"""Unit tests for memory systems"""

import pytest
import numpy as np
from amcf.memory.working_memory import WorkingMemory
from amcf.memory.episodic_memory import EpisodicMemory
from amcf.memory.procedural_memory import ProceduralMemory, Playbook


class TestWorkingMemory:
    
    def test_initialization(self):
        """Test working memory initialization"""
        wm = WorkingMemory(max_size=500, min_size=100)
        assert wm.max_size == 500
        assert wm.min_size == 100
        assert wm.current_size == 100
        assert len(wm.buffer) == 0
    
    def test_append_event(self):
        """Test appending events"""
        wm = WorkingMemory()
        
        # Create mock event
        class MockEvent:
            def __init__(self):
                self.ts = 1000
                self.src = "host1"
                self.dst = "host2"
                self.kind = "process"
                self.attrs = {}
        
        event = MockEvent()
        wm.append(event, posterior=0.7)
        
        assert len(wm.buffer) == 1
        recent = wm.get_recent_events(k=1)
        assert len(recent) == 1
    
    def test_deduplication(self):
        """Test duplicate suppression"""
        wm = WorkingMemory()
        
        class MockEvent:
            def __init__(self, ts):
                self.ts = ts
                self.src = "host1"
                self.dst = "host2"
                self.kind = "process"
                self.attrs = {}
        
        # Add same event twice in quick succession
        event1 = MockEvent(1000)
        event2 = MockEvent(1010)  # 10ms later
        
        wm.append(event1, posterior=0.7)
        wm.append(event2, posterior=0.7)
        
        # Second should be suppressed
        assert len(wm.buffer) == 1
    
    def test_adaptive_window(self):
        """Test adaptive window sizing"""
        wm = WorkingMemory(min_size=50, max_size=500)
        
        # High uncertainty should expand window
        wm._adjust_window_size(uncertainty=0.5)
        size_high_unc = wm.current_size
        
        # Low uncertainty should shrink window
        wm._adjust_window_size(uncertainty=0.1)
        size_low_unc = wm.current_size
        
        assert size_low_unc < size_high_unc


class TestEpisodicMemory:
    
    def test_initialization(self):
        """Test episodic memory initialization"""
        em = EpisodicMemory(embedding_dim=768)
        assert em.embedding_dim == 768
        assert len(em.incident_store) == 0
    
    def test_append_outcome(self):
        """Test appending incident outcome"""
        em = EpisodicMemory()
        
        class MockEvent:
            def __init__(self):
                self.ts = 1000
                self.src = "host1"
                self.dst = "host2"
                self.kind = "process"
                self.attrs = {}
        
        event = MockEvent()
        outcome = {'action': 'automate', 'status': 'success'}
        
        em.append_outcome(event, 'automate', outcome, label=1)
        
        assert len(em.incident_store) == 1
        assert em.next_id == 1


class TestProceduralMemory:
    
    def test_initialization(self):
        """Test procedural memory initialization"""
        pm = ProceduralMemory()
        assert len(pm.playbooks) > 0
    
    def test_default_playbooks(self):
        """Test default playbooks are initialized"""
        pm = ProceduralMemory()
        
        playbook_ids = list(pm.playbooks.keys())
        assert 'isolate_host_v2.3' in playbook_ids
        assert 'kill_process_v1.5' in playbook_ids
    
    def test_get_playbook(self):
        """Test retrieving playbook"""
        pm = ProceduralMemory()
        
        playbook = pm.get_playbook('isolate_host_v2.3')
        assert playbook is not None
        assert playbook.playbook_id == 'isolate_host_v2.3'