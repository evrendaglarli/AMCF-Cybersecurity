"""Bounded, entity-centred attack-chain hypothesis tracking.

R20 Eq. (4)-(5) *reference subset*: accepts explicitly provided,
ATT&CK-aligned stage tags; it does not infer tactics from raw packet features.
Unknown stages do not create evidence or influence belief updates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Set, Tuple

STAGES = (
    "initial_access", "execution", "persistence", "privilege_escalation",
    "defense_evasion", "credential_access", "discovery", "lateral_movement",
    "collection", "command_and_control", "exfiltration", "impact",
)
STAGE_INDEX = {name: i for i, name in enumerate(STAGES)}


@dataclass
class Hypothesis:
    entity: str
    first_ts: int
    last_ts: int
    stages: List[str] = field(default_factory=list)
    evidence_ids: Set[str] = field(default_factory=set)
    timestamps: List[int] = field(default_factory=list)
    coherence: float = 0.0
    belief: float = 0.0
    log_evidence: float = 0.0

    def update(self, stage: str, timestamp: int, evidence_id: str,
               posterior: float, weights: Tuple[float, float, float]):
        if evidence_id in self.evidence_ids:
            return False
        self.evidence_ids.add(evidence_id)
        self.last_ts = max(self.last_ts, timestamp)
        self.timestamps.append(timestamp)
        if stage not in self.stages:
            self.stages.append(stage)
        order = [STAGE_INDEX[k] for k in self.stages]
        consistent_pairs = sum(a <= b for a, b in zip(order, order[1:]))
        path_consistency = consistent_pairs / max(1, len(order) - 1)
        stage_coverage = len(self.stages) / len(STAGES)
        gaps = [b - a for a, b in zip(self.timestamps, self.timestamps[1:])]
        temporal_plausibility = (
            sum(0 <= delta <= 24 * 3600 * 1000 for delta in gaps) / len(gaps)
            if gaps else 1.0
        )
        self.coherence = sum(w * value for w, value in zip(
            weights, (path_consistency, stage_coverage, temporal_plausibility)
        ))
        prob = min(1.0 - 1e-6, max(1e-6, float(posterior)))
        self.log_evidence += math.log(prob / (1.0 - prob))
        return True


class HypothesisTracker:
    """No inferred stage labels, bounded active state and once-only evidence."""

    def __init__(self, *, window_ms: int = 24 * 3600 * 1000,
                 max_hypotheses: int = 10, kappa: float = 0.5,
                 weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)):
        if window_ms <= 0 or max_hypotheses <= 0 or kappa < 0:
            raise ValueError("Invalid hypothesis bounds")
        if any(w < 0 for w in weights) or abs(sum(weights) - 1) > 1e-9:
            raise ValueError("Coherence weights must sum to one")
        self.window_ms = window_ms
        self.max_hypotheses = max_hypotheses
        self.kappa = kappa
        self.weights = weights
        self.active: Dict[str, Hypothesis] = {}

    def update(self, event, posterior: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        timestamp = int(event.ts)
        self.active = {
            key: hyp for key, hyp in self.active.items()
            if 0 <= timestamp - hyp.last_ts <= self.window_ms
        }
        attrs = getattr(event, "attrs", {}) or {}
        raw_stage = attrs.get("attack_stage", attrs.get("tactic", ""))
        stage = str(raw_stage).strip().lower().replace(" ", "_")
        entity = str(getattr(event, "src", "") or "")
        if entity and stage in STAGE_INDEX:
            key = entity
            identity = attrs.get("event_id")
            # Without an external event id, deduplicate identical event signatures.
            evidence_id = str(identity) if identity is not None else repr(
                (timestamp, entity, getattr(event, "dst", None),
                 getattr(event, "kind", None), stage)
            )
            hypothesis = self.active.get(key)
            if hypothesis is None:
                hypothesis = Hypothesis(entity, timestamp, timestamp)
                self.active[key] = hypothesis
            hypothesis.update(stage, timestamp, evidence_id, posterior, self.weights)

        if len(self.active) > self.max_hypotheses:
            ordered = sorted(self.active.items(),
                             key=lambda item: (item[1].last_ts, item[1].coherence),
                             reverse=True)
            self.active = dict(ordered[:self.max_hypotheses])
        if not self.active:
            return {}, {}
        # Eq. (5): sigmoid of accumulated log evidence plus coherence, then Z.
        scores = {key: 1.0 / (1.0 + math.exp(
            -max(-30.0, min(30.0, hyp.log_evidence + self.kappa * hyp.coherence))))
                  for key, hyp in self.active.items()}
        z = sum(scores.values())
        beliefs = {key: score / z for key, score in scores.items()}
        for key, hyp in self.active.items():
            hyp.belief = beliefs[key]
        return beliefs, {key: h.coherence for key, h in self.active.items()}
