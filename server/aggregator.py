"""
Model Aggregator - Implements FedAvg, FedProx, and FedNova aggregation strategies
for combining gradient updates from distributed federated learning nodes.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class ModelAggregator:
    """
    Federated model aggregation engine.
    
    Supported strategies:
    - fedavg: Federated Averaging (weighted by number of samples)
    - fedprox: FedProx (with proximal regularization term)
    - fednova: Federated Nova (normalized averaging)
    - fedadam: Federated Adam optimizer
    """

    STRATEGIES = ["fedavg", "fedprox", "fednova", "fedadam"]

    def __init__(self, strategy: str = "fedavg", config: Optional[Dict] = None):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {self.STRATEGIES}")
        
        self.strategy = strategy
        self.config = config or {}
        
        # FedProx config
        self.mu = self.config.get("mu", 0.01)
        
        # FedAdam config
        self.server_lr = self.config.get("server_lr", 0.01)
        self.beta1 = self.config.get("beta1", 0.9)
        self.beta2 = self.config.get("beta2", 0.99)
        self.epsilon = self.config.get("epsilon", 1e-8)
        self.momentum: Optional[Dict[str, np.ndarray]] = None
        self.velocity: Optional[Dict[str, np.ndarray]] = None
        self.t = 0  # FedAdam timestep
        
        logger.info(f"ModelAggregator initialized | Strategy: {strategy}")

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        global_model: Optional[Dict[str, np.ndarray]],
        round_num: int = 0
    ) -> Dict[str, Any]:
        """
        Aggregate model updates from multiple nodes.
        
        Args:
            updates: List of dicts with keys: node_id, gradients, num_samples, metadata
            global_model: Current global model weights
            round_num: Current training round number
            
        Returns:
            Dict with aggregated model, metrics, and privacy cost
        """
        if not updates:
            raise ValueError("No updates provided for aggregation")
        
        total_samples = sum(u["num_samples"] for u in updates)
        
        if self.strategy == "fedavg":
            return self._fedavg(updates, total_samples, global_model)
        elif self.strategy == "fedprox":
            return self._fedprox(updates, total_samples, global_model)
        elif self.strategy == "fednova":
            return self._fednova(updates, total_samples, global_model)
        elif self.strategy == "fedadam":
            return self._fedadam(updates, total_samples, global_model)

    def _fedavg(
        self,
        updates: List[Dict],
        total_samples: int,
        global_model: Optional[Dict[str, np.ndarray]]
    ) -> Dict:
        """
        Federated Averaging (FedAvg) - McMahan et al. 2017
        
        Computes weighted average of model updates where weights are
        proportional to the number of local training samples.
        """
        aggregated = {}
        
        # Get all parameter keys from the first update
        param_keys = list(updates[0]["gradients"].keys())
        
        for key in param_keys:
            weighted_sum = None
            
            for update in updates:
                weight = update["num_samples"] / total_samples
                grad = update["gradients"][key]
                
                if weighted_sum is None:
                    weighted_sum = weight * grad
                else:
                    weighted_sum += weight * grad
            
            # Apply gradient to global model
            if global_model and key in global_model:
                aggregated[key] = global_model[key] - weighted_sum
            else:
                aggregated[key] = weighted_sum
        
        # Compute convergence metrics
        avg_loss = np.mean([
            u["metadata"].get("loss", float("inf"))
            for u in updates
            if "loss" in u.get("metadata", {})
        ]) if any("metadata" in u for u in updates) else float("inf")
        
        logger.info(f"FedAvg aggregation complete | Params: {len(param_keys)} | Avg loss: {avg_loss:.4f}")
        
        return {
            "model": aggregated,
            "metrics": {
                "avg_loss": float(avg_loss),
                "num_updates": len(updates),
                "total_samples": total_samples,
                "strategy": "fedavg"
            },
            "privacy_cost": 0.0
        }

    def _fedprox(
        self,
        updates: List[Dict],
        total_samples: int,
        global_model: Optional[Dict[str, np.ndarray]]
    ) -> Dict:
        """
        FedProx - Li et al. 2020
        
        Adds a proximal term to limit how far local models can diverge
        from the global model, improving convergence for non-IID data.
        """
        aggregated = {}
        param_keys = list(updates[0]["gradients"].keys())
        
        for key in param_keys:
            weighted_sum = None
            
            for update in updates:
                weight = update["num_samples"] / total_samples
                grad = update["gradients"][key]
                
                # Apply proximal correction if global model available
                if global_model and key in global_model:
                    proximal_correction = self.mu * (grad - global_model[key])
                    corrected_grad = grad - proximal_correction
                else:
                    corrected_grad = grad
                
                if weighted_sum is None:
                    weighted_sum = weight * corrected_grad
                else:
                    weighted_sum += weight * corrected_grad
            
            if global_model and key in global_model:
                aggregated[key] = global_model[key] - weighted_sum
            else:
                aggregated[key] = weighted_sum
        
        avg_loss = np.mean([
            u["metadata"].get("loss", float("inf"))
            for u in updates
            if u.get("metadata", {}).get("loss") is not None
        ]) if updates else float("inf")
        
        logger.info(f"FedProx aggregation | mu={self.mu} | Params: {len(param_keys)}")
        
        return {
            "model": aggregated,
            "metrics": {
                "avg_loss": float(avg_loss),
                "mu": self.mu,
                "strategy": "fedprox"
            },
            "privacy_cost": 0.0
        }

    def _fednova(
        self,
        updates: List[Dict],
        total_samples: int,
        global_model: Optional[Dict[str, np.ndarray]]
    ) -> Dict:
        """
        FedNova - Wang et al. 2020
        
        Normalized averaging to handle heterogeneous local update steps,
        eliminating objective inconsistency in non-IID settings.
        """
        aggregated = {}
        param_keys = list(updates[0]["gradients"].keys())
        
        # Compute normalization factors (tau_eff)
        tau_values = [
            u["metadata"].get("local_steps", 1) for u in updates
        ]
        
        total_weight = sum(
            u["num_samples"] / total_samples * tau
            for u, tau in zip(updates, tau_values)
        )
        
        for key in param_keys:
            normalized_sum = None
            
            for update, tau in zip(updates, tau_values):
                weight = (update["num_samples"] / total_samples) * tau / total_weight
                grad = update["gradients"][key]
                
                if normalized_sum is None:
                    normalized_sum = weight * grad
                else:
                    normalized_sum += weight * grad
            
            if global_model and key in global_model:
                aggregated[key] = global_model[key] - normalized_sum
            else:
                aggregated[key] = normalized_sum
        
        logger.info(f"FedNova aggregation | tau_eff={total_weight:.4f} | Params: {len(param_keys)}")
        
        return {
            "model": aggregated,
            "metrics": {
                "tau_eff": float(total_weight),
                "strategy": "fednova"
            },
            "privacy_cost": 0.0
        }

    def _fedadam(
        self,
        updates: List[Dict],
        total_samples: int,
        global_model: Optional[Dict[str, np.ndarray]]
    ) -> Dict:
        """
        FedAdam - Reddi et al. 2020
        
        Server-side Adam optimizer for adaptive federated learning.
        Uses adaptive learning rates based on gradient history.
        """
        # First compute FedAvg pseudo-gradient
        pseudo_grad = {}
        param_keys = list(updates[0]["gradients"].keys())
        
        for key in param_keys:
            weighted_sum = None
            for update in updates:
                weight = update["num_samples"] / total_samples
                if weighted_sum is None:
                    weighted_sum = weight * update["gradients"][key]
                else:
                    weighted_sum += weight * update["gradients"][key]
            
            # Pseudo-gradient = old_model - aggregated_model
            if global_model and key in global_model:
                pseudo_grad[key] = global_model[key] - weighted_sum
            else:
                pseudo_grad[key] = weighted_sum
        
        # Initialize momentum and velocity if needed
        if self.momentum is None:
            self.momentum = {k: np.zeros_like(v) for k, v in pseudo_grad.items()}
            self.velocity = {k: np.zeros_like(v) for k, v in pseudo_grad.items()}
        
        self.t += 1
        aggregated = {}
        
        for key in param_keys:
            # Adam update
            self.momentum[key] = (
                self.beta1 * self.momentum[key] + (1 - self.beta1) * pseudo_grad[key]
            )
            self.velocity[key] = (
                self.beta2 * self.velocity[key] + (1 - self.beta2) * pseudo_grad[key] ** 2
            )
            
            # Bias correction
            m_hat = self.momentum[key] / (1 - self.beta1 ** self.t)
            v_hat = self.velocity[key] / (1 - self.beta2 ** self.t)
            
            if global_model and key in global_model:
                aggregated[key] = (
                    global_model[key] - self.server_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                )
            else:
                aggregated[key] = self.server_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        logger.info(f"FedAdam aggregation | lr={self.server_lr} | t={self.t}")
        
        return {
            "model": aggregated,
            "metrics": {
                "server_lr": self.server_lr,
                "timestep": self.t,
                "strategy": "fedadam"
            },
            "privacy_cost": 0.0
        }

    def reset(self):
        """Reset aggregator state (useful between training runs)"""
        self.momentum = None
        self.velocity = None
        self.t = 0
        logger.info("Aggregator state reset")
