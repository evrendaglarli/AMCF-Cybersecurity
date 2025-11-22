"""
Working Memory: Short-horizon event buffer
Section 3.1 and Algorithm 1
"""

import numpy as np
from collections import deque
from typing import List, Tuple
from ..core.metacognitive_controller import Event

class WorkingMemory:
    """
    Short-horizon buffer for correlation and deduplication
    Adaptive window size based on uncertainty
    """
    
    def __init__(self, max_size: int = 5000, min_size: int = 100):
        self.max_size = max_size
        self.min_size = min_size
        self.current_size = min_size
        
        self.buffer = deque(maxlen=max_size)
        self.dedupe_index = {}  # hash -> timestamp
        
    def append(self, event: Event, posterior: float):
        """
        Add event to working memory with deduplication
        """
        # Compute semantic hash (ignore timestamps)
        event_hash = self._hash_event(event)
        
        # Deduplication: suppress duplicates within 60 seconds
        current_time = event.ts
        if event_hash in self.dedupe_index:
            last_seen = self.dedupe_index[event_hash]
            if (current_time - last_seen) < 60000:  # 60 seconds in ms
                return  # Suppress duplicate
        
        # Update deduplication index
        self.dedupe_index[event_hash] = current_time
        
        # Adaptive window sizing based on uncertainty
        uncertainty = self._compute_uncertainty(posterior)
        self._adjust_window_size(uncertainty)
        
        # Append to buffer
        self.buffer.append((event, posterior, current_time))
        
        # Trim to current adaptive size
        while len(self.buffer) > self.current_size:
            self.buffer.popleft()
    
    def get_recent_events(self, k: int = 50) -> List[Tuple[Event, float, int]]:
        """Retrieve k most recent events"""
        return list(self.buffer)[-k:]
    
    def get_all_events(self) -> List[Tuple[Event, float, int]]:
        """Retrieve all events in buffer"""
        return list(self.buffer)
    
    def clear(self):
        """Clear working memory"""
        self.buffer.clear()
        self.dedupe_index.clear()
    
    def _hash_event(self, event: Event) -> int:
        """Semantic hash ignoring timestamps"""
        hash_components = (
            event.src,
            event.dst,
            event.kind,
            frozenset(event.attrs.items()) if event.attrs else frozenset()
        )
        return hash(hash_components)
    
    def _compute_uncertainty(self, posterior: float) -> float:
        """
        Entropy-based uncertainty: H = -p*log(p) - (1-p)*log(1-p)
        """
        p = np.clip(posterior, 1e-7, 1 - 1e-7)
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        return entropy
    
    def _adjust_window_size(self, uncertainty: float):
        """
        Adaptive window: expand under high uncertainty, shrink under low
        """
        if uncertainty > 0.4:
            # High uncertainty: expand window for more context
            self.current_size = min(int(self.current_size * 1.1), self.max_size)
        else:
            # Low uncertainty: shrink window
            self.current_size = max(int(self.current_size * 0.95), self.min_size)