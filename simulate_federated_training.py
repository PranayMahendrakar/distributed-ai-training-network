"""
Federated Learning Simulation - End-to-end demonstration
Simulates a complete federated training run with multiple nodes,
differential privacy, gradient compression, and metrics tracking.

Run with:
    python simulate_federated_training.py --nodes 5 --rounds 10 --epsilon 1.0
"""

import argparse
import logging
import time
import numpy as np
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("FederatedSimulation")


# ============================================================
# Inline implementations for standalone simulation
# (In production, import from respective modules)
# ============================================================

def generate_synthetic_data(num_samples: int, num_features: int = 20, num_classes: int = 3):
    """Generate synthetic classification data for a node"""
    np.random.seed(int(time.time() * 1000) % 10000)
    X = np.random.randn(num_samples, num_features)
    y = np.random.randint(0, num_classes, num_samples)
    return {"X": X, "y": y, "num_samples": num_samples}


def init_model(input_dim: int = 20, hidden_dim: int = 32, output_dim: int = 3) -> Dict:
    """Initialize a simple MLP model"""
    return {
        "layer1.weight": np.random.randn(hidden_dim, input_dim) * 0.01,
        "layer1.bias": np.zeros(hidden_dim),
        "output.weight": np.random.randn(output_dim, hidden_dim) * 0.01,
        "output.bias": np.zeros(output_dim)
    }


def forward(model: Dict, X: np.ndarray):
    """Simple forward pass"""
    h = np.maximum(0, X @ model["layer1.weight"].T + model["layer1.bias"])
    z = h @ model["output.weight"].T + model["output.bias"]
    exp_z = np.exp(z - z.max(axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True), h


def compute_loss(probs: np.ndarray, y: np.ndarray) -> float:
    """Cross-entropy loss"""
    n = len(y)
    return float(-np.mean(np.log(probs[np.arange(n), y] + 1e-10)))


def compute_accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    """Classification accuracy"""
    return float(np.mean(np.argmax(probs, axis=1) == y))


def train_local(model: Dict, data: Dict, lr: float = 0.01, epochs: int = 3) -> Dict:
    """Train model locally and return gradients"""
    local_model = {k: v.copy() for k, v in model.items()}
    X, y = data["X"], data["y"]
    n = len(X)
    
    for epoch in range(epochs):
        # Mini-batch training
        for i in range(0, n, 32):
            Xb, yb = X[i:i+32], y[i:i+32]
            probs, h = forward(local_model, Xb)
            
            # Output gradient
            delta = probs.copy()
            delta[np.arange(len(yb)), yb] -= 1
            delta /= len(yb)
            
            # Gradients
            grads = {
                "output.weight": delta.T @ h,
                "output.bias": delta.sum(axis=0),
            }
            
            h_grad = delta @ local_model["output.weight"]
            h_grad *= (h > 0)
            grads["layer1.weight"] = h_grad.T @ Xb
            grads["layer1.bias"] = h_grad.sum(axis=0)
            
            # Update
            for k in local_model:
                local_model[k] -= lr * grads.get(k, 0)
    
    # Compute gradient as model difference
    gradients = {k: model[k] - local_model[k] for k in model}
    probs, _ = forward(local_model, X)
    
    return {
        "gradients": gradients,
        "loss": compute_loss(probs, y),
        "accuracy": compute_accuracy(probs, y),
        "num_samples": n
    }


def add_dp_noise(gradients: Dict, epsilon: float, clip_norm: float = 1.0) -> Dict:
    """Add Gaussian DP noise to gradients"""
    # Compute total norm
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients.values()))
    
    # Clip
    if total_norm > clip_norm:
        scale = clip_norm / total_norm
        gradients = {k: v * scale for k, v in gradients.items()}
    
    # Gaussian noise calibrated to epsilon
    noise_scale = clip_norm * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon
    
    noisy = {}
    for k, v in gradients.items():
        noisy[k] = v + np.random.normal(0, noise_scale, v.shape)
    
    return noisy


def fedavg(updates: List[Dict], global_model: Dict) -> Dict:
    """Federated Averaging aggregation"""
    total_samples = sum(u["num_samples"] for u in updates)
    new_model = {k: np.zeros_like(v) for k, v in global_model.items()}
    
    for update in updates:
        weight = update["num_samples"] / total_samples
        for k in global_model:
            new_model[k] += weight * (global_model[k] - update["gradients"][k])
    
    return new_model


def compress_topk(gradients: Dict, ratio: float = 0.1) -> Dict:
    """Top-K gradient sparsification"""
    compressed = {}
    for k, v in gradients.items():
        flat = v.flatten()
        k_count = max(1, int(flat.size * ratio))
        top_k_idx = np.argpartition(np.abs(flat), -k_count)[-k_count:]
        result = np.zeros_like(flat)
        result[top_k_idx] = flat[top_k_idx]
        compressed[k] = result.reshape(v.shape)
    return compressed


# ============================================================
# Main Simulation
# ============================================================

class FederatedSimulation:
    """
    End-to-end federated learning simulation with:
    - Multiple distributed nodes
    - Local training with SGD
    - Differential privacy (Gaussian mechanism)
    - Gradient compression (Top-K)
    - FedAvg aggregation
    - Real-time metrics tracking
    """

    def __init__(
        self,
        num_nodes: int = 5,
        num_rounds: int = 10,
        epsilon: float = 1.0,
        local_epochs: int = 3,
        learning_rate: float = 0.01,
        compression_ratio: float = 0.5,
        min_samples_per_node: int = 100,
        max_samples_per_node: int = 500,
        use_dp: bool = True,
        use_compression: bool = True,
        non_iid: bool = True
    ):
        self.num_nodes = num_nodes
        self.num_rounds = num_rounds
        self.epsilon = epsilon
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.compression_ratio = compression_ratio
        self.use_dp = use_dp
        self.use_compression = use_compression
        self.non_iid = non_iid
        
        # Simulation state
        self.global_model = init_model()
        self.privacy_spent = 0.0
        self.metrics_history = []
        
        # Generate node data
        self.node_data = self._generate_node_data(
            num_nodes, min_samples_per_node, max_samples_per_node
        )
        
        logger.info(
            f"FederatedSimulation initialized | "
            f"Nodes: {num_nodes} | "
            f"Rounds: {num_rounds} | "
            f"Epsilon: {epsilon} | "
            f"DP: {use_dp} | "
            f"Compression: {use_compression}"
        )

    def _generate_node_data(self, num_nodes, min_samples, max_samples):
        """Generate heterogeneous data for each node (non-IID simulation)"""
        node_data = {}
        
        for i in range(num_nodes):
            n_samples = np.random.randint(min_samples, max_samples)
            
            if self.non_iid:
                # Simulate non-IID: each node has biased class distribution
                X = np.random.randn(n_samples, 20)
                # Bias towards certain classes
                class_bias = np.random.dirichlet([0.5] * 3)
                y = np.random.choice(3, n_samples, p=class_bias)
            else:
                # IID distribution
                X = np.random.randn(n_samples, 20)
                y = np.random.randint(0, 3, n_samples)
            
            node_data[f"node_{i+1}"] = {
                "X": X, "y": y, "num_samples": n_samples
            }
        
        total_samples = sum(d["num_samples"] for d in node_data.values())
        logger.info(
            f"Generated data for {num_nodes} nodes | "
            f"Total samples: {total_samples} | "
            f"Distribution: {'Non-IID' if self.non_iid else 'IID'}"
        )
        
        return node_data

    def run(self) -> Dict[str, Any]:
        """Run the complete federated training simulation"""
        logger.info("=" * 60)
        logger.info("STARTING FEDERATED TRAINING SIMULATION")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Evaluate initial model
        initial_metrics = self._evaluate_global_model()
        logger.info(
            f"Initial Global | "
            f"Loss: {initial_metrics['loss']:.4f} | "
            f"Acc: {initial_metrics['accuracy']:.4f}"
        )
        
        for round_num in range(1, self.num_rounds + 1):
            round_start = time.time()
            
            logger.info(f"\n--- Round {round_num}/{self.num_rounds} ---")
            
            # Select nodes for this round (use all in simulation)
            selected_nodes = list(self.node_data.keys())
            
            # Local training on each node
            updates = []
            node_metrics = {}
            
            for node_id in selected_nodes:
                data = self.node_data[node_id]
                
                # Local training
                result = train_local(
                    self.global_model,
                    data,
                    lr=self.learning_rate,
                    epochs=self.local_epochs
                )
                
                gradients = result["gradients"]
                
                # Apply differential privacy
                if self.use_dp:
                    gradients = add_dp_noise(
                        gradients,
                        epsilon=self.epsilon / self.num_rounds,
                        clip_norm=1.0
                    )
                    self.privacy_spent += self.epsilon / self.num_rounds
                
                # Compress gradients
                if self.use_compression:
                    gradients = compress_topk(gradients, self.compression_ratio)
                
                updates.append({
                    "node_id": node_id,
                    "gradients": gradients,
                    "num_samples": data["num_samples"],
                    "loss": result["loss"],
                    "accuracy": result["accuracy"]
                })
                
                node_metrics[node_id] = {
                    "loss": result["loss"],
                    "accuracy": result["accuracy"]
                }
                
                logger.info(
                    f"  {node_id}: loss={result['loss']:.4f} | "
                    f"acc={result['accuracy']:.4f} | "
                    f"samples={data['num_samples']}"
                )
            
            # Federated aggregation
            self.global_model = fedavg(updates, self.global_model)
            
            # Evaluate global model
            global_metrics = self._evaluate_global_model()
            round_duration = time.time() - round_start
            
            round_result = {
                "round": round_num,
                "global_loss": global_metrics["loss"],
                "global_accuracy": global_metrics["accuracy"],
                "privacy_spent": self.privacy_spent,
                "node_metrics": node_metrics,
                "duration": round_duration
            }
            
            self.metrics_history.append(round_result)
            
            logger.info(
                f"Global Model | "
                f"Loss: {global_metrics['loss']:.4f} | "
                f"Acc: {global_metrics['accuracy']:.4f} | "
                f"Privacy: {self.privacy_spent:.4f} | "
                f"Time: {round_duration:.2f}s"
            )
        
        total_time = time.time() - start_time
        
        # Final summary
        summary = self._generate_summary(total_time)
        
        logger.info("\n" + "=" * 60)
        logger.info("FEDERATED TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Final accuracy: {summary['final_accuracy']:.4f}")
        logger.info(f"Best accuracy: {summary['best_accuracy']:.4f}")
        logger.info(f"Total privacy spent: {self.privacy_spent:.4f}/{self.epsilon}")
        logger.info(f"Privacy remaining: {max(0, self.epsilon - self.privacy_spent):.4f}")
        
        return summary

    def _evaluate_global_model(self) -> Dict:
        """Evaluate global model on all node data"""
        all_X = np.vstack([d["X"] for d in self.node_data.values()])
        all_y = np.concatenate([d["y"] for d in self.node_data.values()])
        
        probs, _ = forward(self.global_model, all_X)
        
        return {
            "loss": compute_loss(probs, all_y),
            "accuracy": compute_accuracy(probs, all_y)
        }

    def _generate_summary(self, total_time: float) -> Dict:
        """Generate training summary statistics"""
        losses = [m["global_loss"] for m in self.metrics_history]
        accuracies = [m["global_accuracy"] for m in self.metrics_history]
        
        return {
            "num_nodes": self.num_nodes,
            "num_rounds": self.num_rounds,
            "total_time": total_time,
            "final_loss": losses[-1] if losses else float("inf"),
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "best_loss": min(losses) if losses else float("inf"),
            "best_accuracy": max(accuracies) if accuracies else 0.0,
            "privacy_epsilon_used": self.epsilon,
            "privacy_spent": self.privacy_spent,
            "use_dp": self.use_dp,
            "use_compression": self.use_compression,
            "metrics_history": self.metrics_history
        }


def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--nodes", type=int, default=5, help="Number of federated nodes")
    parser.add_argument("--rounds", type=int, default=10, help="Number of training rounds")
    parser.add_argument("--epsilon", type=float, default=1.0, help="DP privacy budget")
    parser.add_argument("--epochs", type=int, default=3, help="Local training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--compression", type=float, default=0.5, help="Gradient compression ratio")
    parser.add_argument("--no-dp", action="store_true", help="Disable differential privacy")
    parser.add_argument("--no-compression", action="store_true", help="Disable compression")
    parser.add_argument("--iid", action="store_true", help="Use IID data distribution")
    
    args = parser.parse_args()
    
    simulation = FederatedSimulation(
        num_nodes=args.nodes,
        num_rounds=args.rounds,
        epsilon=args.epsilon,
        local_epochs=args.epochs,
        learning_rate=args.lr,
        compression_ratio=args.compression,
        use_dp=not args.no_dp,
        use_compression=not args.no_compression,
        non_iid=not args.iid
    )
    
    results = simulation.run()
    
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Nodes:              {results['num_nodes']}")
    print(f"Rounds:             {results['num_rounds']}")
    print(f"Total Time:         {results['total_time']:.2f}s")
    print(f"Final Loss:         {results['final_loss']:.4f}")
    print(f"Final Accuracy:     {results['final_accuracy']:.4f} ({results['final_accuracy']*100:.1f}%)")
    print(f"Best Accuracy:      {results['best_accuracy']:.4f} ({results['best_accuracy']*100:.1f}%)")
    print(f"Privacy Used (eps): {results['privacy_spent']:.4f}/{results['privacy_epsilon_used']}")
    print(f"Differential Privacy: {'Enabled' if results['use_dp'] else 'Disabled'}")
    print(f"Compression: {'Enabled' if results['use_compression'] else 'Disabled'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
