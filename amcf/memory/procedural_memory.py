"""
Procedural Memory: Playbook repository
Section 3.1 and Algorithm 3
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class Playbook:
    """Parameterized SOAR playbook"""
    playbook_id: str
    name: str
    description: str
    preconditions: Dict[str, Any]
    parameters: Dict[str, Any]
    cost_model: Dict[str, float]
    rollback_procedure: Optional[str] = None
    version: str = "1.0"

class ProceduralMemory:
    """
    Store and select parameterized playbooks
    """
    
    def __init__(self, playbook_file: Optional[str] = None):
        self.playbooks = {}
        
        if playbook_file:
            self.load_from_file(playbook_file)
        else:
            self._initialize_default_playbooks()
    
    def _initialize_default_playbooks(self):
        """Initialize default playbook library"""
        
        # Playbook 1: Isolate Host
        self.playbooks['isolate_host_v2.3'] = Playbook(
            playbook_id='isolate_host_v2.3',
            name='Isolate Endpoint',
            description='Isolate compromised host from network',
            preconditions={
                'asset_type': ['endpoint', 'server'],
                'min_confidence': 0.80,
                'required_evidence': ['process_ancestry', 'network_connections']
            },
            parameters={
                'isolation_duration': '4h',
                'allow_exceptions': ['corp_dns', 'siem_collector'],
                'notification_targets': ['analyst_on_duty', 'asset_owner']
            },
            cost_model={
                'C_auto': 2.5,
                'C_SE': 1.2,
                'estimated_user_impact': 'high'
            },
            rollback_procedure='restore_network_policy_v1.1'
        )
        
        # Playbook 2: Kill Process
        self.playbooks['kill_process_v1.5'] = Playbook(
            playbook_id='kill_process_v1.5',
            name='Terminate Suspicious Process',
            description='Kill process and child processes',
            preconditions={
                'asset_type': ['endpoint', 'server'],
                'min_confidence': 0.75,
                'required_evidence': ['process_ancestry']
            },
            parameters={
                'kill_children': True,
                'backup_memory': True,
                'notification_targets': ['analyst_on_duty']
            },
            cost_model={
                'C_auto': 1.0,
                'C_SE': 0.5,
                'estimated_user_impact': 'medium'
            }
        )
        
        # Playbook 3: Reset Credentials
        self.playbooks['reset_credential_v1.0'] = Playbook(
            playbook_id='reset_credential_v1.0',
            name='Force Password Reset',
            description='Reset user credentials and revoke sessions',
            preconditions={
                'asset_type': ['user_account'],
                'min_confidence': 0.70,
                'required_evidence': ['auth']
            },
            parameters={
                'revoke_sessions': True,
                'require_mfa_reenroll': True,
                'notification_targets': ['user', 'security_team']
            },
            cost_model={
                'C_auto': 1.5,
                'C_SE': 0.8,
                'estimated_user_impact': 'medium'
            }
        )
        
        # Playbook 4: Collect EDR Deep Dive
        self.playbooks['edr_collect_v2.0'] = Playbook(
            playbook_id='edr_collect_v2.0',
            name='Deep EDR Collection',
            description='Collect detailed EDR telemetry',
            preconditions={
                'asset_type': ['endpoint'],
                'min_confidence': 0.60,
                'required_evidence': []
            },
            parameters={
                'collect_memory': False,
                'collect_registry': True,
                'collect_process_tree': True
            },
            cost_model={
                'C_auto': 0.5,
                'C_SE': 0.1,
                'estimated_user_impact': 'low'
            }
        )
    
    def select_playbook(self, event: Any, posterior: float, 
                        hypothesis_beliefs: Dict) -> str:
        """
        Select appropriate playbook based on context
        
        Args:
            event: Current event
            posterior: Fused posterior probability
            hypothesis_beliefs: Current hypothesis beliefs
            
        Returns:
            Selected playbook_id
        """
        # Filter by preconditions
        candidates = []
        
        for playbook_id, playbook in self.playbooks.items():
            if self._check_preconditions(playbook, event, posterior):
                candidates.append((playbook_id, playbook))
        
        if len(candidates) == 0:
            # No playbook matches, default to escalation
            return 'escalate'
        
        # Rank by cost-effectiveness
        # Cost-effectiveness = (risk_reduction) / (execution_cost)
        # Simplified: rank by C_auto (lower is better)
        ranked = sorted(candidates, key=lambda x: x[1].cost_model['C_auto'])
        
        return ranked[0][0]  # Return best playbook_id
    
    def _check_preconditions(self, playbook: Playbook, event: Any, 
                             posterior: float) -> bool:
        """Check if playbook preconditions are satisfied"""
        preconds = playbook.preconditions
        
        # Check minimum confidence
        if posterior < preconds.get('min_confidence', 0.5):
            return False
        
        # Check asset type
        asset_types = preconds.get('asset_type', [])
        if asset_types and event.kind not in asset_types:
            return False
        
        # Check required evidence (simplified)
        required_evidence = preconds.get('required_evidence', [])
        # In full implementation, verify evidence exists in working memory
        
        return True
    
    def get_playbook(self, playbook_id: str) -> Optional[Playbook]:
        """Retrieve playbook by ID"""
        return self.playbooks.get(playbook_id)
    
    def load_from_file(self, filepath: str):
        """Load playbooks from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for pb_dict in data:
            playbook = Playbook(**pb_dict)
            self.playbooks[playbook.playbook_id] = playbook
    
    def save_to_file(self, filepath: str):
        """Save playbooks to JSON file"""
        data = [asdict(pb) for pb in self.playbooks.values()]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)