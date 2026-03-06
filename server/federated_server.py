"""
Federated Server - Central Coordination Node for Distributed AI Training
Manages training rounds, node registration, and global model aggregation
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a connected federated learning node"""
    node_id: str
    address: str
    num_samples: int
    status: str = "active"
    last_seen: float = field(default_factory=time.time)
    rounds_participated: int = 0
    current_loss: float = float("inf")


@dataclass
class TrainingRound:
    """Represents a single federated training round"""
    round_id: int
    selected_nodes: List[str]
    global_model_version: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    received_updates: Dict[str, Any] = field(default_factory=dict)
    aggregated_model: Optional[Dict] = None
    status: str = "in_progress"
    metrics: Dict[str, float] = field(default_factory=dict)


class FederatedServer:
    """
    Central server for coordinating federated learning across distributed nodes.
    
    Implements:
    - Node registration and management
    - Training round orchestration
    - Federated aggregation (FedAvg, FedProx, FedNova)
    - Model versioning and distribution
    - Privacy budget tracking
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes: Dict[str, NodeInfo] = {}
        self.training_rounds: List[TrainingRound] = []
        self.global_model: Optional[Dict[str, np.ndarray]] = None
        self.model_version = str(uuid.uuid4())
        self.current_round = 0
        self.total_privacy_budget = config.get("privacy", {}).get("epsilon", 10.0)
        self.privacy_spent = 0.0
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 10))
        
        # Import aggregator
        from server.aggregator import ModelAggregator
        self.aggregator = ModelAggregator(
            strategy=config.get("aggregation", {}).get("strategy", "fedavg"),
            config=config.get("aggregation", {})
        )
        
        logger.info(f"FederatedServer initialized | Strategy: {self.aggregator.strategy}")

    def register_node(self, node_id: str, address: str, num_samples: int) -> Dict:
        """Register a new federated learning node"""
        if node_id in self.nodes:
            logger.warning(f"Node {node_id} re-registering")
        
        self.nodes[node_id] = NodeInfo(
            node_id=node_id,
            address=address,
            num_samples=num_samples
        )
        
        logger.info(f"Node registered: {node_id} | Samples: {num_samples} | Total nodes: {len(self.nodes)}")
        
        return {
            "status": "registered",
            "node_id": node_id,
            "model_version": self.model_version,
            "global_model": self.global_model
        }

    def deregister_node(self, node_id: str):
        """Remove a node from the federation"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Node deregistered: {node_id}")

    def get_active_nodes(self) -> List[NodeInfo]:
        """Get list of currently active nodes"""
        current_time = time.time()
        timeout = self.config.get("node_timeout", 300)
        
        active = [
            node for node in self.nodes.values()
            if current_time - node.last_seen < timeout and node.status == "active"
        ]
        return active

    def select_nodes_for_round(self) -> List[str]:
        """
        Select nodes to participate in this training round.
        
        Strategies:
        - random: Select random subset
        - all: Use all active nodes
        - weighted: Prefer nodes with more data
        """
        active_nodes = self.get_active_nodes()
        
        if not active_nodes:
            raise ValueError("No active nodes available for training")
        
        selection_strategy = self.config.get("selection", {}).get("strategy", "random")
        fraction = self.config.get("selection", {}).get("fraction", 0.8)
        min_nodes = self.config.get("selection", {}).get("min_nodes", 2)
        
        num_to_select = max(min_nodes, int(len(active_nodes) * fraction))
        num_to_select = min(num_to_select, len(active_nodes))
        
        if selection_strategy == "random":
            selected = np.random.choice(
                [n.node_id for n in active_nodes],
                size=num_to_select,
                replace=False
            ).tolist()
        elif selection_strategy == "weighted":
            weights = np.array([n.num_samples for n in active_nodes], dtype=float)
            weights /= weights.sum()
            selected = np.random.choice(
                [n.node_id for n in active_nodes],
                size=num_to_select,
                replace=False,
                p=weights
            ).tolist()
        else:  # all
            selected = [n.node_id for n in active_nodes]
        
        logger.info(f"Round {self.current_round + 1}: Selected {len(selected)}/{len(active_nodes)} nodes")
        return selected

    def start_training_round(self) -> TrainingRound:
        """Initiate a new federated training round"""
        self.current_round += 1
        selected_nodes = self.select_nodes_for_round()
        
        training_round = TrainingRound(
            round_id=self.current_round,
            selected_nodes=selected_nodes,
            global_model_version=self.model_version
        )
        
        self.training_rounds.append(training_round)
        
        # Update node statuses
        for node_id in selected_nodes:
            if node_id in self.nodes:
                self.nodes[node_id].rounds_participated += 1
        
        logger.info(f"Training round {self.current_round} started | Nodes: {len(selected_nodes)}")
        return training_round

    def receive_model_update(
        self,
        node_id: str,
        round_id: int,
        gradients: Dict[str, np.ndarray],
        num_samples: int,
        metadata: Optional[Dict] = None
    ):
        """Receive and store model update from a node"""
        # Find the current round
        current_round = next(
            (r for r in self.training_rounds if r.round_id == round_id),
            None
        )
        
        if current_round is None:
            raise ValueError(f"Round {round_id} not found")
        
        if node_id not in current_round.selected_nodes:
            raise ValueError(f"Node {node_id} not selected for round {round_id}")
        
        # Update node info
        if node_id in self.nodes:
            self.nodes[node_id].last_seen = time.time()
            if metadata and "loss" in metadata:
                self.nodes[node_id].current_loss = metadata["loss"]
        
        current_round.received_updates[node_id] = {
            "gradients": gradients,
            "num_samples": num_samples,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        received_count = len(current_round.received_updates)
        total_selected = len(current_round.selected_nodes)
        
        logger.info(
            f"Round {round_id}: Received update from {node_id} "
            f"({received_count}/{total_selected})"
        )
        
        # Check if we have enough updates to aggregate
        min_updates = self.config.get("aggregation", {}).get("min_updates", 2)
        wait_timeout = self.config.get("aggregation", {}).get("wait_timeout", 300)
        
        if received_count >= min_updates:
            elapsed = time.time() - current_round.start_time
            if received_count == total_selected or elapsed > wait_timeout:
                self._aggregate_round(current_round)

    def _aggregate_round(self, training_round: TrainingRound):
        """Aggregate model updates and update global model"""
        logger.info(
            f"Aggregating round {training_round.round_id} | "
            f"Updates: {len(training_round.received_updates)}"
        )
        
        # Prepare updates for aggregation
        updates = []
        for node_id, update_data in training_round.received_updates.items():
            updates.append({
                "node_id": node_id,
                "gradients": update_data["gradients"],
                "num_samples": update_data["num_samples"],
                "metadata": update_data["metadata"]
            })
        
        # Aggregate using the configured strategy
        aggregated = self.aggregator.aggregate(
            updates=updates,
            global_model=self.global_model,
            round_num=training_round.round_id
        )
        
        # Update global model
        self.global_model = aggregated["model"]
        self.model_version = str(uuid.uuid4())
        
        # Track privacy budget
        if "privacy_cost" in aggregated:
            self.privacy_spent += aggregated["privacy_cost"]
        
        # Update round metrics
        training_round.aggregated_model = self.global_model
        training_round.end_time = time.time()
        training_round.status = "completed"
        training_round.metrics = {
            "duration": training_round.end_time - training_round.start_time,
            "num_updates": len(training_round.received_updates),
            "privacy_spent_total": self.privacy_spent,
            **aggregated.get("metrics", {})
        }
        
        logger.info(
            f"Round {training_round.round_id} completed | "
            f"Duration: {training_round.metrics['duration']:.2f}s | "
            f"Privacy spent: {self.privacy_spent:.4f}/{self.total_privacy_budget}"
        )

    def get_global_model(self, node_id: Optional[str] = None) -> Dict:
        """Get the current global model for distribution to nodes"""
        if node_id and node_id in self.nodes:
            self.nodes[node_id].last_seen = time.time()
        
        return {
            "model": self.global_model,
            "version": self.model_version,
            "round": self.current_round
        }

    def get_training_status(self) -> Dict:
        """Get overall training status and metrics"""
        active_nodes = self.get_active_nodes()
        
        latest_round_metrics = {}
        if self.training_rounds:
            latest = self.training_rounds[-1]
            latest_round_metrics = latest.metrics
        
        return {
            "current_round": self.current_round,
            "active_nodes": len(active_nodes),
            "total_nodes": len(self.nodes),
            "model_version": self.model_version,
            "privacy_spent": self.privacy_spent,
            "privacy_budget": self.total_privacy_budget,
            "privacy_remaining": self.total_privacy_budget - self.privacy_spent,
            "latest_round_metrics": latest_round_metrics,
            "completed_rounds": len([r for r in self.training_rounds if r.status == "completed"])
        }

    def run_training(self, num_rounds: int) -> List[Dict]:
        """
        Run full federated training for specified number of rounds.
        
        Args:
            num_rounds: Number of federated learning rounds to execute
            
        Returns:
            List of round metrics
        """
        logger.info(f"Starting federated training | Rounds: {num_rounds}")
        all_metrics = []
        
        for round_num in range(num_rounds):
            # Check privacy budget
            if self.privacy_spent >= self.total_privacy_budget:
                logger.warning(f"Privacy budget exhausted after round {round_num}")
                break
            
            # Start round
            training_round = self.start_training_round()
            
            # Wait for updates (in practice this is async/event-driven)
            # Here we simulate the wait
            timeout = self.config.get("aggregation", {}).get("wait_timeout", 300)
            start = time.time()
            
            while (
                len(training_round.received_updates) < len(training_round.selected_nodes)
                and time.time() - start < timeout
                and training_round.status == "in_progress"
            ):
                time.sleep(0.1)
            
            if training_round.status == "in_progress":
                # Force aggregation with available updates
                if len(training_round.received_updates) >= 2:
                    self._aggregate_round(training_round)
                else:
                    logger.warning(f"Round {training_round.round_id}: Insufficient updates")
                    training_round.status = "failed"
            
            all_metrics.append({
                "round": round_num + 1,
                "status": training_round.status,
                "metrics": training_round.metrics
            })
        
        logger.info(f"Training complete | Rounds completed: {self.current_round}")
        return all_metrics


def create_server_from_config(config_path: str) -> FederatedServer:
    """Create and configure a FederatedServer from a YAML config file"""
    import yaml
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return FederatedServer(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--rounds", type=int, default=10, help="Number of training rounds")
    args = parser.parse_args()
    
    server = create_server_from_config(args.config)
    logger.info(f"Federated server running on port {args.port}")
    
    # In production, this would start a gRPC/HTTP server
    # For demo, print status
    status = server.get_training_status()
    print(f"Server Status: {status}")
