"""
Episodic Memory: HNSW-based incident retrieval
Section 3.1 and Algorithm 1
"""

import numpy as np
import hnswlib
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class EpisodicMemory:
    """
    Store and retrieve past incidents using hierarchical navigable small world indexing
    """
    
    def __init__(self, embedding_dim: int = 768, max_elements: int = 10_000_000):
        self.embedding_dim = embedding_dim
        
        # HNSW index for fast retrieval
        self.index = hnswlib.Index(space='cosine', dim=embedding_dim)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=200,
            M=16
        )
        
        # Incident store
        self.incident_store = {}  # incident_id -> full incident data
        self.next_id = 0
        
        # Embedding model (Sentence-BERT)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def retrieve_similar(self, recent_events: List, top_k: int = 5) -> List[Dict]:
        """
        Retrieve similar past incidents using vector similarity
        
        Args:
            recent_events: List of (event, posterior, timestamp) tuples
            top_k: Number of similar incidents to retrieve
            
        Returns:
            List of similar incident dictionaries
        """
        if len(self.incident_store) == 0:
            return []
        
        # Embed current sequence
        query_embedding = self._embed_sequence(recent_events)
        
        # kNN search
        try:
            labels, distances = self.index.knn_query(query_embedding, k=min(top_k, len(self.incident_store)))
            
            # Filter by similarity threshold (cosine similarity > 0.70)
            results = []
            for idx, dist in zip(labels[0], distances[0]):
                similarity = 1 - dist  # Convert distance to similarity
                if similarity > 0.70:
                    results.append(self.incident_store[idx])
            
            return results
        except RuntimeError:
            # Index not yet populated
            return []
    
    def append_outcome(self, event: Any, action: str, outcome: Dict, label: int):
        """
        Store incident outcome in episodic memory
        
        Args:
            event: Event object
            action: Action taken
            outcome: Execution outcome
            label: Ground truth (0/1)
        """
        # Create incident record
        incident = {
            'id': self.next_id,
            'event': event,
            'action': action,
            'outcome': outcome,
            'label': label,
            'timestamp': event.ts
        }
        
        # Embed and add to index
        embedding = self._embed_incident(incident)
        self.index.add_items(embedding, [self.next_id])
        
        # Store incident
        self.incident_store[self.next_id] = incident
        self.next_id += 1
    
    def _embed_sequence(self, recent_events: List) -> np.ndarray:
        """
        Embed sequence of events using attention pooling
        """
        if len(recent_events) == 0:
            return np.zeros((1, self.embedding_dim))
        
        # Convert events to text
        event_texts = []
        weights = []
        
        for i, (event, posterior, ts) in enumerate(recent_events[-50:]):
            text = self._event_to_text(event)
            event_texts.append(text)
            
            # Attention weight: recency * posterior confidence
            recency = np.exp(-0.1 * i)
            weights.append(posterior * recency)
        
        # Embed texts
        embeddings = self.embedding_model.encode(event_texts)
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        aggregated = np.average(embeddings, axis=0, weights=weights)
        return aggregated.reshape(1, -1)
    
    def _embed_incident(self, incident: Dict) -> np.ndarray:
        """Embed a single incident"""
        text = self._event_to_text(incident['event'])
        embedding = self.embedding_model.encode([text])
        return embedding
    
    def _event_to_text(self, event: Any) -> str:
        """Convert event to natural language description"""
        parts = [
            f"Event type: {event.kind}",
            f"Source: {event.src}",
        ]
        
        if event.dst:
            parts.append(f"Destination: {event.dst}")
        
        if hasattr(event, 'attrs') and event.attrs:
            # Include key attributes
            for key, value in list(event.attrs.items())[:5]:
                parts.append(f"{key}: {value}")
        
        return " | ".join(parts)