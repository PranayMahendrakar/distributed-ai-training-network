"""
Metrics Tracker - Collects and analyzes federated learning training metrics
Tracks global model performance, node health, privacy budget, and convergence
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RoundMetrics:
    """Metrics captured for a single federated learning round"""
    round_id: int
    timestamp: float = field(default_factory=time.time)
    
    # Global model metrics
    global_loss: float = float("inf")
    global_accuracy: float = 0.0
    
    # Node metrics
    num_nodes_selected: int = 0
    num_nodes_responded: int = 0
    node_losses: Dict[str, float] = field(default_factory=dict)
    node_accuracies: Dict[str, float] = field(default_factory=dict)
    
    # Privacy metrics
    privacy_budget_used: float = 0.0
    privacy_budget_remaining: float = 0.0
    
    # Communication metrics
    total_bytes_transmitted: int = 0
    compression_ratio: float = 1.0
    
    # Timing metrics
    round_duration: float = 0.0
    avg_node_training_time: float = 0.0
    
    # Convergence metrics
    gradient_norm: float = 0.0
    model_drift: float = 0.0


class MetricsTracker:
    """
    Comprehensive metrics tracking for federated learning.
    
    Monitors:
    - Global model convergence (loss, accuracy)
    - Per-node performance and health
    - Privacy budget consumption
    - Communication efficiency
    - Training speed and throughput
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        output_dir: str = "logs"
    ):
        self.config = config or {}
        self.output_dir = output_dir
        
        self.rounds: List[RoundMetrics] = []
        self.node_health: Dict[str, Dict] = defaultdict(dict)
        self.global_history: Dict[str, List] = defaultdict(list)
        
        # Statistics
        self._start_time = time.time()
        self._total_bytes = 0
        self._privacy_events: List[Dict] = []
        
        logger.info("MetricsTracker initialized")

    def record_round(
        self,
        round_id: int,
        global_loss: float,
        global_accuracy: float,
        node_updates: List[Dict],
        privacy_info: Optional[Dict] = None,
        communication_info: Optional[Dict] = None,
        timing_info: Optional[Dict] = None
    ) -> RoundMetrics:
        """
        Record all metrics for a completed training round.
        
        Args:
            round_id: Round number
            global_loss: Aggregated global model loss
            global_accuracy: Aggregated global model accuracy
            node_updates: List of per-node update dicts
            privacy_info: Privacy budget information
            communication_info: Bytes transmitted, compression ratio
            timing_info: Duration, latency info
            
        Returns:
            RoundMetrics object for this round
        """
        metrics = RoundMetrics(round_id=round_id)
        
        # Global metrics
        metrics.global_loss = global_loss
        metrics.global_accuracy = global_accuracy
        
        # Node metrics
        metrics.num_nodes_selected = len(node_updates)
        metrics.num_nodes_responded = len([u for u in node_updates if u.get("responded", True)])
        
        for update in node_updates:
            node_id = update.get("node_id", "unknown")
            if "loss" in update.get("metadata", {}):
                metrics.node_losses[node_id] = update["metadata"]["loss"]
            if "accuracy" in update.get("metadata", {}):
                metrics.node_accuracies[node_id] = update["metadata"]["accuracy"]
            
            # Update node health
            self.node_health[node_id].update({
                "last_round": round_id,
                "last_loss": update.get("metadata", {}).get("loss", float("inf")),
                "num_samples": update.get("num_samples", 0),
                "status": "active"
            })
        
        # Privacy metrics
        if privacy_info:
            metrics.privacy_budget_used = privacy_info.get("epsilon_spent", 0.0)
            metrics.privacy_budget_remaining = privacy_info.get("remaining", 0.0)
            
            self._privacy_events.append({
                "round": round_id,
                "epsilon_spent": metrics.privacy_budget_used,
                "timestamp": metrics.timestamp
            })
        
        # Communication metrics
        if communication_info:
            metrics.total_bytes_transmitted = communication_info.get("bytes", 0)
            metrics.compression_ratio = communication_info.get("compression_ratio", 1.0)
            self._total_bytes += metrics.total_bytes_transmitted
        
        # Timing metrics
        if timing_info:
            metrics.round_duration = timing_info.get("duration", 0.0)
            metrics.avg_node_training_time = timing_info.get("avg_node_time", 0.0)
        
        # Convergence metrics
        if len(self.rounds) > 0:
            prev_loss = self.rounds[-1].global_loss
            metrics.model_drift = abs(global_loss - prev_loss)
        
        # Store round
        self.rounds.append(metrics)
        
        # Update time series
        self.global_history["loss"].append(global_loss)
        self.global_history["accuracy"].append(global_accuracy)
        self.global_history["privacy_used"].append(metrics.privacy_budget_used)
        self.global_history["round_duration"].append(metrics.round_duration)
        self.global_history["timestamps"].append(metrics.timestamp)
        
        logger.info(
            f"Round {round_id} | "
            f"Loss: {global_loss:.4f} | "
            f"Acc: {global_accuracy:.4f} | "
            f"Nodes: {metrics.num_nodes_responded}/{metrics.num_nodes_selected} | "
            f"Privacy: {metrics.privacy_budget_used:.4f}"
        )
        
        return metrics

    def get_convergence_status(self) -> Dict:
        """Analyze convergence of the federated learning process"""
        if len(self.rounds) < 3:
            return {"converged": False, "reason": "insufficient_rounds"}
        
        recent_losses = [r.global_loss for r in self.rounds[-10:]]
        
        # Check if loss is decreasing
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        # Check variance in recent rounds
        loss_variance = float(np.var(recent_losses))
        
        # Convergence criteria
        converged = loss_variance < 1e-4 and abs(loss_trend) < 1e-3
        
        return {
            "converged": converged,
            "loss_trend": float(loss_trend),
            "loss_variance": loss_variance,
            "recent_loss_mean": float(np.mean(recent_losses)),
            "best_loss": float(min(r.global_loss for r in self.rounds)),
            "best_accuracy": float(max(r.global_accuracy for r in self.rounds))
        }

    def get_node_health_report(self) -> Dict:
        """Get health status of all registered nodes"""
        return {
            "total_nodes": len(self.node_health),
            "active_nodes": len([
                n for n in self.node_health.values()
                if n.get("status") == "active"
            ]),
            "nodes": dict(self.node_health)
        }

    def get_privacy_analysis(self) -> Dict:
        """Detailed privacy budget analysis"""
        if not self._privacy_events:
            return {"total_spent": 0.0}
        
        epsilon_values = [e["epsilon_spent"] for e in self._privacy_events]
        
        return {
            "total_epsilon_spent": float(sum(epsilon_values)),
            "average_per_round": float(np.mean(epsilon_values)),
            "max_round_cost": float(max(epsilon_values)),
            "num_rounds_tracked": len(self._privacy_events),
            "budget_history": epsilon_values[-20:]  # Last 20 rounds
        }

    def get_communication_stats(self) -> Dict:
        """Communication efficiency statistics"""
        if not self.rounds:
            return {}
        
        bytes_per_round = [r.total_bytes_transmitted for r in self.rounds]
        compression_ratios = [r.compression_ratio for r in self.rounds if r.compression_ratio > 0]
        
        return {
            "total_bytes_transmitted": self._total_bytes,
            "total_mb": self._total_bytes / (1024 * 1024),
            "avg_bytes_per_round": float(np.mean(bytes_per_round)) if bytes_per_round else 0,
            "avg_compression_ratio": float(np.mean(compression_ratios)) if compression_ratios else 1.0,
            "bandwidth_savings": float(1 - np.mean(compression_ratios)) if compression_ratios else 0.0
        }

    def get_summary(self) -> Dict:
        """Get comprehensive training summary"""
        if not self.rounds:
            return {"status": "no_rounds_completed"}
        
        total_time = time.time() - self._start_time
        
        return {
            "total_rounds": len(self.rounds),
            "total_training_time": total_time,
            "total_training_time_formatted": self._format_duration(total_time),
            "best_loss": float(min(r.global_loss for r in self.rounds)),
            "best_accuracy": float(max(r.global_accuracy for r in self.rounds)),
            "final_loss": self.rounds[-1].global_loss,
            "final_accuracy": self.rounds[-1].global_accuracy,
            "convergence": self.get_convergence_status(),
            "privacy": self.get_privacy_analysis(),
            "communication": self.get_communication_stats(),
            "node_health": self.get_node_health_report(),
            "rounds_per_hour": len(self.rounds) / max(total_time / 3600, 1e-6)
        }

    def export_to_json(self, filepath: str):
        """Export all metrics to JSON file"""
        data = {
            "summary": self.get_summary(),
            "rounds": [
                {
                    "round_id": r.round_id,
                    "timestamp": r.timestamp,
                    "global_loss": r.global_loss,
                    "global_accuracy": r.global_accuracy,
                    "num_nodes": r.num_nodes_responded,
                    "privacy_budget_used": r.privacy_budget_used,
                    "round_duration": r.round_duration
                }
                for r in self.rounds
            ]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration as human-readable string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
