"""
Main Metacognitive Controller
Implements Algorithms 1, 2, 3 from the paper
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from collections import deque
from copy import deepcopy
from .decision_ledger import DecisionLedger, digest

from .calibration import TemperatureScaler, CompetenceWeightManager
from .fusion import LogitWeightedFusion
from .hypotheses import HypothesisTracker
from .drift import DriftMonitor
from .bayes_risk import BayesRiskCalculator, ValueOfInformationEstimator, CostModel
from .monte_carlo_voi import MonteCarloVoIEstimator
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
        self.detector_score_spaces = dict(config.get("detector_score_spaces", {}))
        self.decision_ledger = DecisionLedger(config.get('decision_ledger_path', ':memory:'))
        # Time-ordered, confirmed, class-enriched decision-context window.
        self.feedback_window = deque(maxlen=200)
        self.forgetting_rho = float(config.get('feedback_forgetting_rho', 1.0))
        if not 0 < self.forgetting_rho <= 1:
            raise ValueError('feedback_forgetting_rho must be in (0,1]')
        if any(mode not in ("logit", "probability")
               for mode in self.detector_score_spaces.values()):
            raise ValueError("detector_score_spaces must use logit or probability")
        
        # Calibration components
        self.temp_scalers = {
            det_id: TemperatureScaler(det_id) for det_id in detector_ids
        }
        self.weight_manager = CompetenceWeightManager(num_detectors)
        
        # Fusion
        self.fusion = LogitWeightedFusion(num_detectors)
        
        # Risk and VoI
        self.risk_calc = BayesRiskCalculator(cost_model)
        # v45 R3 Section 3.1: adaptive conditional Monte Carlo VoI.
        # With no historically validated outcome model the estimator returns
        # zero recommendation rather than simulating posterior movement.
        voi_config = config.get("voi_estimation", {})
        self.voi_estimator = MonteCarloVoIEstimator(
            self.risk_calc,
            N_min=int(voi_config.get("N_min", 64)),
            N_max=int(voi_config.get("N_max", 4096)),
            tolerance=float(voi_config.get("tolerance", 0.02)),
            wall_clock_budget_ms=float(voi_config.get("wall_clock_budget_ms", 150)),
            seed=int(voi_config.get("seed", 42)),
        )
        
        # Memory systems
        self.working_memory = WorkingMemory(max_size=config.get('working_memory_size', 500))
        self.episodic_memory = EpisodicMemory(
            embedding_dim=config.get('embedding_dim', 384)
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
        self.hypothesis_coherence = {}
        self.drift_window = int(config.get("drift_window", 200))
        self.drift_monitor = DriftMonitor(
            min_window=self.drift_window,
            psi_threshold=config.get("psi_threshold", 0.25),
            ks_threshold=config.get("ks_threshold", 0.12),
            ece_threshold=config.get("ece_threshold", 0.03)
        )
        if config.get("validation_reference_posteriors") is not None:
            self.drift_monitor.fit_reference(config["validation_reference_posteriors"])
        self.drift_observations = deque(maxlen=self.drift_window)
        self.drift_status = {"status": "insufficient_evidence", "drift": None}
        self.hypothesis_tracker = HypothesisTracker(
            max_hypotheses=config.get("max_hypotheses", 10),
            kappa=config.get("coherence_kappa", 0.5)
        )
        self._restore_confirmed_state()
        
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
        
        # Appendix B: record a canonical snapshot and DecisionID BEFORE ACT.
        # Do not persist raw identities/payloads, only an event fingerprint and
        # the numeric decision state required for later feedback replay.
        raw_logits = {
            k: self._calibration_logit(k, event.detector_scores[k])
            for k in self.detector_ids if k in event.detector_scores
        }
        playbook_id = (self.procedural_memory.select_playbook(
            event, posterior, self.hypothesis_beliefs)
            if action == 'automate' else None)
        snapshot = {
            'event_ts_ms': int(event.ts),
            'event_hash': digest({'ts': event.ts, 'src':event.src, 'dst':event.dst,
                                  'kind':event.kind, 'attrs':event.attrs,
                                  'scores':event.detector_scores}),
            'posterior':float(posterior),
            'calibrated_probs':{k:float(v) for k,v in calibrated_probs.items()},
            'raw_logits':raw_logits,
            'hypothesis_beliefs':self.hypothesis_beliefs.copy(),
            'hypothesis_coherence':self.hypothesis_coherence.copy(),
            'risks':risks.copy(), 'voi_results':voi_results.copy(),
            'action':action, 'playbook':playbook_id,
            'weights':self.weight_manager.get_weights().tolist(),
            'versions':{
                'model':self.config.get('model_version','reference-unverified'),
                'calibration':{k:len(v.label_buffer) for k,v in self.temp_scalers.items()},
                'playbook':getattr(
                    self.procedural_memory.get_playbook(playbook_id), 'version', None)
                    if hasattr(self.procedural_memory,'get_playbook') and playbook_id else None,
            },
        }
        decision_id = self.decision_ledger.record(snapshot)
        # ========== ACT (Algorithm 3) ==========
        outcome = self._act(event, action, posterior, risks, calibrated_probs, voi_results)
        effective_action = outcome.get('effective_action', action)
        # The event.label field (including a supplied true label) must NEVER
        # silently update model state before a confirmed, available feedback.
        metadata = {
            'DecisionID': decision_id,
            'posterior': posterior,
            'calibrated_probs': calibrated_probs,
            'risks': risks,
            'voi_results': voi_results,
            'hypothesis_beliefs': self.hypothesis_beliefs.copy(),
            'hypothesis_coherence': self.hypothesis_coherence.copy(),
            'drift': self.drift_status.copy(),
            'outcome': outcome
        }
        
        return effective_action, metadata
    
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
                raw_score = self._calibration_logit(
                    det_id, event.detector_scores[det_id]
                )
                calibrated_prob = self.temp_scalers[det_id].calibrate(np.array([raw_score]))[0]
                calibrated_probs[det_id] = calibrated_prob
            else:
                calibrated_probs[det_id] = 0.5  # Neutral if detector missing
        
        # Step 2: Fuse calibrated probabilities
        probs_array = np.array([calibrated_probs[det_id] for det_id in self.detector_ids])
        weights = self.weight_manager.get_weights()
        
        # Without an independently fitted context-prior model use neutral bias.
        bias = 0.0
        
        posterior = self.fusion.fuse(
            probs_array.reshape(1, -1),
            weights,
            bias
        )[0]
        
        # Validation-frozen reference and current fused posteriors.
        # Without reference evidence, drift remains unknown.
        self.drift_observations.append(float(posterior))
        self.drift_status = self.drift_monitor.assess(
            list(self.drift_observations)
        )
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
        # Step 1: Hypotheses updated once in SENSE using deduplicated
        # stage evidence. Unknown stage tags cannot create hypotheses.
        
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
            voi_results[enrich_type] = {
                'voi': voi, 'n_samples': n_samples, 'cost': enrich_cost,
                'status': self.voi_estimator.last_result.status,
                'ci_halfwidth': self.voi_estimator.last_result.ci_halfwidth,
            }
        
        # Drift requires a validated reference stream and delayed labels.
        # No drift signal is fabricated when such evidence is unavailable.
        
        # Step 5-6: Decision policy
        # Only a positive, budget-feasible conditional query can compete.
        # Eq. (10): R_gather(e) = min_primary_risk - VoI(e), with the
        # enrichment cost already subtracted ONCE inside VoI.
        # When no validated query model is available gather stays +inf.
        budget = self.enrichment_scheduler.budget
        feasible_voi = [
            v['voi'] for v in voi_results.values()
            if v['voi'] > 0 and 0 < v['cost'] <= budget
        ]
        voi_positive = bool(feasible_voi)
        if voi_positive:
            primary_baseline = min(risks[k] for k in
                                   ("ignore", "automate", "escalate"))
            risks["gather"] = primary_baseline - max(feasible_voi)
        high_uncertainty = self._is_high_uncertainty(posterior)
        
        action = self.risk_calc.select_action(risks, voi_positive, high_uncertainty)
        if self.drift_status.get("drift") is True and action == "automate":
            action = "escalate"  # No unsafe automated control under measured shift.

        return action, risks, voi_results
    
    def _act(self, event: Event, action: str, posterior: float,
             risks: Dict, calibrated_probs: Dict[str, float],
             voi_results: Optional[Dict] = None) -> Dict:
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
            enrichments = self.enrichment_scheduler.select_enrichments(voi_results or {})
            # No installed connector: choosing an enrichment is not executing it.
            outcome['enrichments_proposed'] = enrichments
            outcome['effective_action'] = 'escalate'
            outcome['action'] = 'escalate'
            outcome['status'] = 'escalated_no_enrichment_connector'
            outcome['case_package'] = self._build_case_package(event, posterior, risks)
            
        elif action == 'automate':
            # Select the concrete playbook BEFORE testing its action class.
            # Algorithm 2's 'automate' is not itself an A_cond operation.
            playbook_id = self.procedural_memory.select_playbook(
                event, posterior, self.hypothesis_beliefs
            )
            get_playbook = getattr(self.procedural_memory, "get_playbook", None)
            playbook = (get_playbook(playbook_id)
                        if callable(get_playbook) else None)
            action_class = getattr(playbook, "action_class", "unclassified")
            outcome['playbook_selected'] = playbook_id
            outcome['playbook_action_class'] = action_class

            # v45 R3 specifies no numeric tau_risk. It must come from a
            # validation-approved configuration, never a made-up default.
            tau_risk = self.config.get("tau_risk")
            if tau_risk is None:
                allowed = False
                outcome['gate_reason'] = 'missing_validation_risk_threshold'
            else:
                allowed = self.safety_gate.check_endpoint_isolation(
                    posterior=posterior,
                    calibrated_probs=calibrated_probs,
                    risk_by_action=risks,
                    tau_risk=tau_risk,
                    playbook_id=playbook_id,
                    action_class=action_class,
                )
                outcome['gate_reason'] = ('all_reference_gates_passed'
                                          if allowed else 'safety_gate_denied')

            # Even when eligible, this package has NO governed SOAR execute,
            # rollback, or credential connector. Report a proposal only.
            outcome['effective_action'] = 'escalate'
            outcome['action'] = 'escalate'
            outcome['case_package'] = self._build_case_package(
                event, posterior, risks
            )
            outcome['status'] = ('escalated_no_actuation_connector'
                                 if allowed else 'escalated_from_failed_gate')

        elif action == 'escalate':
            # Package case for analyst
            case_package = self._build_case_package(event, posterior, risks)
            outcome['case_package'] = case_package
            outcome['status'] = 'escalated'
        
        return outcome
    
    def _restore_confirmed_state(self):
        saved = self.decision_ledger.load_state()
        if saved is None:
            return
        self.weight_manager.weights = np.asarray(saved['weights'], dtype=float)
        for key, value in saved['calibration'].items():
            if key not in self.temp_scalers:
                raise ValueError('Persisted detector configuration mismatch')
            scaler = self.temp_scalers[key]
            scaler.temperature = float(value['temperature'])
            scaler.sample_buffer = list(value['scores'])
            scaler.label_buffer = list(value['labels'])
        self.feedback_window.clear()
        self.feedback_window.extend(saved['window'])

    def _capture_confirmed_state(self):
        return {
            'weights':self.weight_manager.get_weights().tolist(),
            'calibration':{
                k:{'temperature':float(v.temperature),
                   'scores':list(v.sample_buffer),
                   'labels':list(v.label_buffer)}
                for k,v in self.temp_scalers.items()
            },
            'window':list(self.feedback_window),
        }

    def ingest_confirmed_feedback(self, decision_id: str, outcome: str, *,
                                  source: str, available_at_ms: int) -> dict:
        """Algorithm 6: delayed causally available, one-time TP/FP feedback.

        The v45 R3 outcome maps to maliciousness target TP->1, FP->0.
        A verified-label feedback stream is separate from Event.label and can
        never be inferred from a prediction or a playbook execution request.
        """
        snapshot = self.decision_ledger.read(decision_id)
        if outcome not in ('TP','FP'):
            raise ValueError('Algorithm 6 requires TP or FP')
        if source not in ('analyst_confirmed','validated_automation',
                          'red_team_confirmed','incident_closure'):
            raise ValueError('Unverified feedback source')
        if type(available_at_ms) is not int or available_at_ms < snapshot['event_ts_ms']:
            raise ValueError('Feedback must be available at/after decision time')
        if self.decision_ledger.accepted(decision_id):
            raise ValueError('Duplicate confirmed feedback')
        last = self.decision_ledger.latest_feedback_ms()
        if last is not None and available_at_ms < last:
            raise ValueError('Noncausal feedback order')
        if set(snapshot['calibrated_probs']) != set(self.detector_ids):
            raise ValueError('Snapshot detector set mismatch')
        before = deepcopy(self._capture_confirmed_state())
        label = 1 if outcome == 'TP' else 0
        try:
            self.feedback_window.append({
                'label':label,'posterior':snapshot['posterior'],
                'decision_id':decision_id
            })
            # Weight adaptation requires confirmed labels, never the test
            # event's immediate ground-truth field (Eq. 2).
            probs = np.array([[snapshot['calibrated_probs'][k]
                               for k in self.detector_ids]], dtype=float)
            self.weight_manager.update(probs, np.asarray([label]))
            # The full class-enriched 200-context window is the calibration
            # admission criterion in Section 5.5. After the first window,
            # recompute T on each new verified context using the existing
            # positive-temperature NLL criterion, not on unlabelled traffic.
            if (len(self.feedback_window) == 200 and
                len({x['label'] for x in self.feedback_window}) == 2):
                # This reference uses the 200 most recently confirmed contexts.
                # The manuscript's exponential forgetting for calibration is
                # handled by the dedicated weighted-window optimizer.
                for key in self.detector_ids:
                    records = self._confirmed_calibration_records(key)
                    if records is not None:
                        from .calibration import fit_confirmed_temperature
                        logits, labels = records
                        temp = fit_confirmed_temperature(
                            logits, labels, rho=self.forgetting_rho,
                            initial=self.temp_scalers[key].temperature)
                        scaler = self.temp_scalers[key]
                        scaler.temperature = temp
                        scaler.sample_buffer = list(logits)
                        scaler.label_buffer = list(labels)
            saved = self._capture_confirmed_state()
            self.decision_ledger.write_feedback_and_state(
                decision_id=decision_id, outcome=outcome, source=source,
                available_at_ms=available_at_ms, state=saved)
        except Exception:
            self.weight_manager.weights = np.asarray(before['weights'], dtype=float)
            for key,value in before['calibration'].items():
                scaler=self.temp_scalers[key]
                scaler.temperature=value['temperature']
                scaler.sample_buffer=list(value['scores'])
                scaler.label_buffer=list(value['labels'])
            self.feedback_window.clear()
            self.feedback_window.extend(before['window'])
            raise
        ece = None
        if len(self.feedback_window) == 200:
            scores=np.array([x['posterior'] for x in self.feedback_window])
            truths=np.array([x['label'] for x in self.feedback_window])
            ece=self.drift_monitor.calibration_error(scores, truths)
        recalibration_gate = (
            ece is not None and ece > self.drift_monitor.ece_threshold
        ) or (self.drift_status.get('drift') is True)
        # A trigger is an observable recommendation; do not claim an
        # unimplemented real-time threshold retuning actuator.
        return {'DecisionID':decision_id, 'label':label,
                'ece':ece,'recalibration_gate':bool(recalibration_gate),
                'weights':self.weight_manager.get_weights().tolist(),
                'temperature':{k:v.temperature for k,v in self.temp_scalers.items()}}

    def _confirmed_calibration_records(self, key: str):
        # All 200 replay-time-confirmed contexts must carry the relevant
        # score. Missing data may not be replaced by neutral pseudo-scores.
        scores=[]
        labels=[]
        for item in self.feedback_window:
            snapshot=self.decision_ledger.read(item['decision_id'])
            if key not in snapshot['raw_logits']:
                return None
            scores.append(float(snapshot['raw_logits'][key]))
            labels.append(int(item['label']))
        return np.asarray(scores),np.asarray(labels)

    def _calibration_logit(self, detector_id: str, score: float) -> float:
        """Explicit score-space contract for Eq. (1): sigmoid(logit/T).

        Probability-producing detectors must be declared in config instead of
        silently feeding their [0,1] outputs to a raw-logit temperature scale.
        """
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("Detector score must be finite")
        if self.detector_score_spaces.get(detector_id, "logit") == "probability":
            if not 0 <= value <= 1:
                raise ValueError("Probability score out of range")
            value = float(np.clip(value, 1e-7, 1 - 1e-7))
            return float(np.log(value / (1-value)))
        return value

    # ===== Helper methods =====
    
    def _update_hypotheses(self, event: Event, posterior: float, similar_cases: List):
        """R20 Eq. 4–5: bounded, once-only ATT&CK-stage hypotheses.

        Requires externally tagged evidence; no tactic labels are inferred
        from raw UNSW tabular records.
        """
        beliefs, coherence = self.hypothesis_tracker.update(event, posterior)
        self.hypothesis_beliefs = beliefs
        self.hypothesis_coherence = coherence
        self.current_hypotheses = list(self.hypothesis_tracker.active.values())

    def _update_hypothesis_beliefs(self, event: Event, posterior: float,
                                    calibrated_probs: Dict):
        """Compatibility no-op: evidence is processed only once in SENSE."""
        return None

    def _infer_severity(self, event: Event) -> str:
        """Infer incident severity from event attributes"""
        # Simplified heuristic
        if event.kind in ['edr', 'process']:
            return 'high'
        elif event.kind in ['netflow', 'http']:
            return 'medium'
        return 'low'
    
    def _get_enrichment_cost(self, enrichment_type: str) -> float:
        """Use the SAME validated resource-cost input for VoI and Eq. (15).

        The scheduler's current numbers are reference placeholders, not a
        demonstrated implementation of Eq. (9)'s empirical byte/CPU/latency
        coefficients or the manuscript's actual experimental query costs.
        """
        return self.enrichment_scheduler.get_enrichment_cost(enrichment_type)
    
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