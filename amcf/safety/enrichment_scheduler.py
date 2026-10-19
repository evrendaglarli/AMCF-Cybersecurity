"""
Enrichment Scheduler: Budgeted VoI-driven selection
Algorithm 5: Sequential conditional single-query selection
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


class EnrichmentScheduler:
    """
    Select the highest positive conditional VoI candidate within the budget
    Algorithm 5: Budgeted Enrichment Selection
    """
    
    def __init__(self, budget: float = 100.0):
        """
        Args:
            budget: Λ, per-tick enrichment budget (resource units)
        """
        self.budget = budget
        self.enrichment_costs = {
            'edr_deep': 15.0,          # Deep EDR collection
            'sandbox': 25.0,            # Sandbox detonation
            'pcap': 10.0,               # Packet capture
            'memory_dump': 30.0,        # Memory dump
            'dns_resolution': 5.0,      # DNS resolution
            'whois_lookup': 8.0,        # WHOIS lookup
            'threat_intel': 12.0        # Threat intelligence lookup
        }
    
    def select_enrichments(self, voi_results: Dict[str, Dict],
                          available_budget: Optional[float] = None) -> List[str]:
        """R20 Algorithm 5: select ONE feasible enrichment with highest VoI.

        The caller must await an actual result, update the decision state,
        and recompute conditional VoI before requesting the next enrichment.
        A precomputed collection of multiple queries is not equivalent.
        """
        budget = self.budget if available_budget is None else available_budget
        if budget < 0:
            raise ValueError("Remaining budget cannot be negative")
        feasible = []
        for name, outcome in voi_results.items():
            value = float(outcome.get("voi", 0.0))
            cost = float(outcome.get("cost", self.get_enrichment_cost(name)))
            if not (np.isfinite(value) and np.isfinite(cost)):
                continue
            if value > 0 and 0 < cost <= budget:
                feasible.append((value, name))
        if not feasible:
            return []
        return [max(feasible, key=lambda candidate: candidate[0])[1]]

    def get_enrichment_cost(self, enrichment_type: str) -> float:
        """Get cost for specific enrichment type"""
        return self.enrichment_costs.get(enrichment_type, 10.0)
    
    def set_enrichment_cost(self, enrichment_type: str, cost: float):
        """Update enrichment cost"""
        self.enrichment_costs[enrichment_type] = cost
    
    def estimate_total_cost(self, enrichment_types: List[str]) -> float:
        """Estimate total cost for enrichment set"""
        return sum(self.get_enrichment_cost(et) for et in enrichment_types)
    
    def is_within_budget(self, enrichment_types: List[str]) -> bool:
        """Check if enrichment set fits within budget"""
        return self.estimate_total_cost(enrichment_types) <= self.budget