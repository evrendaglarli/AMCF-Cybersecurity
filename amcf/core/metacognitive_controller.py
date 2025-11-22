"""
Main Metacognitive Controller
Implements Algorithms 1, 2, 3 from the paper
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from .calibration import TemperatureScaler, CompetenceWeightManager
from .fusion import LogitWeightedFusion
from .bayes_risk import BayesRiskCalculator, ValueOfInformationEstimator, CostModel
from ..memory import WorkingMemory, EpisodicMemory, ProceduralMemory
from ..safety import SafetyGate, EnrichmentScheduler

@dataclass
class Event:
    """Normalized event record (Section 4.1)"""
    ts: int  # Epoch milliseconds
    src: str
    dst: Optional[str]
    kind: str  # 'process', 'netflow', 'auth', 'dns', 'http', 'edr'
    attrs: Dict
    detector_scores: Dict[str, float]  # {detector_id: raw_score}
    label: Optional[int] = None  # Ground truth (0/1) if available


class MetacognitiveController:
    """
    Main AMCF controller implementing Sense→Reason→Act loop
    """
    
    def __init__(self, num_detectors: int, detector_ids: List[str],
                 cost_model: CostModel, config: Dict):
        self.num_detectors = num_detectors
        self.detector_ids = detector_ids
        self.config = config
        
        # Calibration components
        self.temp_scalers = {
            det_id: TemperatureScaler(det_id) for det_id in detector_ids
        }
        self.weight_manager = CompetenceWeightManager(num_detectors)
        
        # Fusion
        self.fusion = LogitWeightedFusion(num_detectors)
        
        # Risk and VoI
        self.risk_calc = BayesRiskCalculator(cost_model)
        self.voi_estimator = ValueOfInformationEstimator(self.risk_calc)
        
        # Memory systems
        self.working_memory = WorkingMemory(max_size=config.get('working_memory_size', 500))
        self.episodic_memory = EpisodicMemory(
            embedding_dim=config.get('embedding_dim', 768)
        )
        self.procedural_memory = ProceduralMemory()
        
        # Safety and enrichment
        self.safety_gate = SafetyGate(
            rho_max=config.get('rho_max', 0.75),
            epsilon=config.get('epsilon', 1e-3)
        )
        self.enrichment_scheduler = EnrichmentScheduler(
            budget=config.get('enrichment_budget', 100.0)
        )
        
        # State
        self.current_hypotheses = []
        self.hypothesis_beliefs = {}
        
    def process_event(self, event: Event) -> Tuple[str, Dict]:
        """
        Main processing loop: Sense → Reason → Act
        
        Args:
            event: Incoming normalized event
            
        Returns:
            (action, metadata) where action ∈ {'ignore', 'gather', 'automate', 'escalate'}
        """
        # ========== SENSE (Algorithm 1) ==========
        posterior, calibrated_probs = self._sense(event)
        
        # ========== REASON (Algorithm 2) ==========
        action, risks, voi_results = self._reason(event, posterior, calibrated_probs)
        
        # ========== ACT (Algorithm 3) ==========
        outcome = self._act(event, action, posterior, risks)
        
        # Update memories and weights if ground truth available
        if event.label is not None:
            self._update_from_feedback(event, calibrated_probs, action, outcome)
        
        metadata = {
            'posterior': posterior,
            'calibrated_probs': calibrated_probs,
            'risks': risks,
            'voi_results': voi_results,
            'outcome': outcome
        }
        
        return action, metadata
    
    def _sense(self, event: Event) -> Tuple[float, Dict[str, float]]:
        """
        Algorithm 1: Evidence Fusion and Memory Update
        
        Returns:
            (π_t, {detector_id: calibrated_prob})
        """
        # Step 1: Calibrate detector scores
        calibrated_probs = {}
        for det_id in self.detector_ids:
            if det_id in event.detector_scores:
                raw_score = event.detector_scores[det_id]
                calibrated_prob = self.temp_scalers[det_id].calibrate(np.array([raw_score]))[0]
                calibrated_probs[det_id] = calibrated_prob
            else:
                calibrated_probs[det_id] = 0.5  # Neutral if detector missing
        
        # Step 2: Fuse calibrated probabilities
        probs_array = np.array([calibrated_probs[det_id] for det_id in self.detector_ids])
        weights = self.weight_manager.get_weights()
        
        # Contextual bias (placeholder: could use asset criticality)
        bias = 0.0
        
        posterior = self.fusion.fuse(
            probs_array.reshape(1, -1),
            weights,
            bias
        )[0]
        
        # Step 3: Append to working memory
        self.working_memory.append(event, posterior)
        
        # Step 4-5: Query episodic memory for similar cases
        similar_cases = self.episodic_memory.retrieve_similar(
            self.working_memory.get_recent_events(k=50)
        )
        
        # Update hypotheses based on working + episodic memory
        self._update_hypotheses(event, posterior, similar_cases)
        
        return posterior, calibrated_probs
    
    def _reason(self, event: Event, posterior: float, 
                calibrated_probs: Dict) -> Tuple[str, Dict, Dict]:
        """
        Algorithm 2: Metacognitive Evaluation
        
        Returns:
            (selected_action, risks_dict, voi_results)
        """
        # Step 1: Update hypothesis coherence and beliefs
        self._update_hypothesis_beliefs(event, posterior, calibrated_probs)
        
        # Step 2: Compute Bayes risks
        severity = self._infer_severity(event)
        risks = self.risk_calc.compute_risks(
            posterior=posterior,
            severity=severity,
            asset_weight=1.0,  # Could be dynamic
            queue_length=0,  # Simplified
            expected_delay=5.0
        )
        
        # Step 3: Estimate VoI for candidate enrichments
        voi_results = {}
        candidate_enrichments = ['edr_deep', 'sandbox']
        
        for enrich_type in candidate_enrichments:
            enrich_cost = self._get_enrichment_cost(enrich_type)
            voi, n_samples = self.voi_estimator.estimate_voi(
                posterior_before=posterior,
                enrichment_cost=enrich_cost,
                enrichment_type=enrich_type,
                severity=severity
            )
            voi_results[enrich_type] = {'voi': voi, 'n_samples': n_samples}
        
        # Step 4: Monitor calibration drift (placeholder)
        # In full implementation, track ECE/PSI/KS and trigger recalibration
        
        # Step 5-6: Decision policy
        voi_positive = any(v['voi'] > 0 for v in voi_results.values())
        high_uncertainty = self._is_high_uncertainty(posterior)
        
        action = self.risk_calc.select_action(risks, voi_positive, high_uncertainty)
        
        return action, risks, voi_results
    
    def _act(self, event: Event, action: str, posterior: float, 
             risks: Dict) -> Dict:
        """
        Algorithm 3: Action Execution and Learning
        
        Returns:
            Outcome dictionary
        """
        outcome = {'action': action, 'timestamp': time.time()}
        
        if action == 'ignore':
            # Passive, retain state
            outcome['status'] = 'ignored'
            
        elif action == 'gather':
            # Select enrichments under budget (Algorithm 5)
            enrichments = self.enrichment_scheduler.select_enrichments(
                voi_results={}  # Simplified
            )
            outcome['enrichments_requested'] = enrichments
            outcome['status'] = 'gathering'
            
        elif action == 'automate':
            # Safety gate check (Algorithm 4)
            gate_passed = self.safety_gate.check(
                posterior=posterior,
                calibrated_probs=list(self.temp_scalers.keys()),  # Simplified
                asset_risk=1.0
            )
            
            if gate_passed:
                # Select and execute playbook
                playbook = self.procedural_memory.select_playbook(
                    event, posterior, self.hypothesis_beliefs
                )
                outcome['playbook_executed'] = playbook
                outcome['status'] = 'automated'
            else:
                # Downgrade to escalate
                action = 'escalate'
                outcome['status'] = 'escalated_from_failed_gate'
                
        elif action == 'escalate':
            # Package case for analyst
            case_package = self._build_case_package(event, posterior, risks)
            outcome['case_package'] = case_package
            outcome['status'] = 'escalated'
        
        return outcome
    
    def _update_from_feedback(self, event: Event, calibrated_probs: Dict,
                               action: str, outcome: Dict):
        """
        Algorithm 6: Feedback Ingestion and Online Recalibration
        """
        label = event.label
        
        # Update temperature scalers
        for det_id in self.detector_ids:
            if det_id in event.detector_scores:
                self.temp_scalers[det_id].update(
                    np.array([event.detector_scores[det_id]]),
                    np.array([label])
                )
        
        # Update competence weights
        probs_array = np.array([calibrated_probs[det_id] for det_id in self.detector_ids])
        self.weight_manager.update(
            probs_array.reshape(1, -1),
            np.array([label])
        )
        
        # Store in episodic memory
        self.episodic_memory.append_outcome(event, action, outcome, label)
    
    # ===== Helper methods =====
    
    def _update_hypotheses(self, event: Event, posterior: float, similar_cases: List):
        """Update attack-chain hypotheses from memory"""
        # Simplified: maintain up to 10 hypotheses
        # Full implementation would use attack graphs, TTP mappings, etc.
        self.current_hypotheses = []  # Placeholder
    
    def _update_hypothesis_beliefs(self, event: Event, posterior: float,
                                    calibrated_probs: Dict):
        """
        Equation (5): Update beliefs with coherence
        q_{t+1}^j ∝ σ(logit(q_t^j) + Σ w_k*logit(p_k) + κ*Γ_t^j)
        """
        # Placeholder: In full implementation,
	# Placeholder: In full implementation, compute coherence scores
        # and update beliefs for each hypothesis
        self.hypothesis_beliefs = {}  # Simplified
    
    def _infer_severity(self, event: Event) -> str:
        """Infer incident severity from event attributes"""
        # Simplified heuristic
        if event.kind in ['edr', 'process']:
            return 'high'
        elif event.kind in ['netflow', 'http']:
            return 'medium'
        return 'low'
    
    def _get_enrichment_cost(self, enrichment_type: str) -> float:
        """Get cost for enrichment type (Equation 9)"""
        costs = {
            'edr_deep': 5.0,
            'sandbox': 15.0,
            'pcap': 8.0,
            'memory_dump': 20.0
        }
        return costs.get(enrichment_type, 10.0)
    
    def _is_high_uncertainty(self, posterior: float, threshold: float = 0.3) -> bool:
        """Check if uncertainty is high (posterior near 0.5)"""
        return abs(posterior - 0.5) < threshold
    
    def _build_case_package(self, event: Event, posterior: float, 
                            risks: Dict) -> Dict:
        """Build analyst escalation package"""
        return {
            'event': event,
            'posterior': posterior,
            'risks': risks,
            'working_memory_context': self.working_memory.get_recent_events(k=10),
            'similar_incidents': self.episodic_memory.retrieve_similar(
                self.working_memory.get_recent_events(k=10)
            )[:3]
        }