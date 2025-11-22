"""
Generate realistic synthetic datasets matching paper statistics
"""

import numpy as np
import pickle
from pathlib import Path
import json

def generate_enterprise_corpus():
    """
    Generate 60-day enterprise corpus
    ~4.7M events, 8400 endpoints, 5100 users
    """
    np.random.seed(42)
    
    print("Generating enterprise corpus...")
    
    n_events = 50_000  # Subset for testing
    n_endpoints = 100
    n_users = 50
    n_days = 60
    
    # Feature matrix
    X = np.random.randn(n_events, 50)
    
    # Labels: 5% positive rate (malicious events)
    y = np.random.binomial(1, 0.05, n_events)
    
    # Split: 50% train, 25% val, 25% test
    n_train = n_events // 2
    n_val = n_events // 4
    n_test = n_events - n_train - n_val
    
    X_train = X[:n_train]
    X_val = X[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    
    y_train = y[:n_train]
    y_val = y[n_train:n_train+n_val]
    y_test = y[n_train+n_val:]
    
    # Generate events (timestamps from 60-day window)
    base_ts = 1718190000000  # June 12, 2024
    
    class MockEvent:
        def __init__(self, ts, src, dst, kind, attrs, detector_scores, label):
            self.ts = ts
            self.src = src
            self.dst = dst
            self.kind = kind
            self.attrs = attrs
            self.detector_scores = detector_scores
            self.label = label
    
    events_test = []
    for i in range(n_test):
        ts = base_ts + (i * 60 * 1000)  # 1 minute intervals
        
        event = MockEvent(
            ts=ts,
            src=f"host_{i % n_endpoints}",
            dst=f"host_{(i+1) % n_endpoints}" if np.random.rand() > 0.4 else None,
            kind=np.random.choice(['process', 'netflow', 'auth', 'dns', 'http', 'edr']),
            attrs={
                'user': f"user_{i % n_users}",
                'bytes_sent': np.random.randint(100, 10000),
                'bytes_recv': np.random.randint(100, 10000)
            },
            detector_scores={
                'lightgbm': np.random.randn(),
                'tcn': np.random.randn(),
                'isolation_forest': np.random.randn()
            },
            label=y_test[i]
        )
        events_test.append(event)
    
    # Save
    data_path = Path('data/processed')
    data_path.mkdir(parents=True, exist_ok=True)
    
    with open(data_path / 'enterprise.pkl', 'wb') as f:
        pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test, events_test), f)
    
    print(f"  Saved: {len(events_test)} test events")
    return X_train, y_train, X_val, y_val, X_test, y_test, events_test


if __name__ == '__main__':
    generate_enterprise_corpus()
    print("\nSynthetic data generation complete!")