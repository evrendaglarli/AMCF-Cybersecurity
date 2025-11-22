"""
Enrichment Scheduler: Budgeted VoI-driven selection
Algorithm 5: Greedy selection by VoI/cost ratio
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


class EnrichmentScheduler:
    """
    Select enrichments under resource budget using greedy VoI/cost ratio
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
        """
        Algorithm 5: Greedy selection by VoI/cost ratio
        
        Args:
            voi_results: {enrichment_type: {'voi': float, 'cost': float, 'n_samples': int}}
            available_budget: Override default budget
            
        Returns:
            List of selected enrichment types sorted by VoI/cost ratio
        """
        budget = available_budget or self.budget
        
        # Prepare candidates with VoI/cost ratios
        candidates = []
        
        for enrich_type, results in voi_results.items():
            voi = results.get('voi', 0)
            
            # Use predefined cost if not in results
            if 'cost' in results:
                cost = results['cost']
            else:
                cost = self.enrichment_costs.get(enrich_type, 10.0)
            
            # Only consider enrichments with positive VoI
            if voi > 0 and cost > 0:
                ratio = voi / cost
                candidates.append({
                    'type': enrich_type,
                    'voi': voi,
                    'cost': cost,
                    'ratio': ratio,
                    'n_samples': results.get('n_samples', 0)
                })
        
        # Sort by VoI/cost ratio descending (best value first)
        candidates.sort(key=lambda x: x['ratio'], reverse=True)
        
        # Greedy selection under budget
        selected = []
        used_budget = 0.0
        
        for candidate in candidates:
            if used_budget + candidate['cost'] <= budget:
                selected.append(candidate['type'])
                used_budget += candidate['cost']
        
        return selected
    
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