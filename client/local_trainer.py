"""
Local Trainer - Executes training on local node data
Keeps raw data private while computing model updates
"""

import logging
import time
from typing import Dict, List, Optional, Callable, Any
import numpy as np

logger = logging.getLogger(__name__)


class LocalTrainer:
    """
    Executes local model training on private node data.
    
    The local training keeps raw data within the node boundary.
    Only model gradients (privatized) are shared externally.
    
    Supports:
    - Mini-batch SGD with momentum
    - Adam optimizer
    - Learning rate scheduling
    - Early stopping
    - Loss monitoring
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        optimizer: str = "sgd",
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        lr_scheduler: str = "cosine",
        warmup_rounds: int = 5
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.warmup_rounds = warmup_rounds
        
        # Optimizer state
        self._velocity: Dict[str, np.ndarray] = {}
        self._m: Dict[str, np.ndarray] = {}
        self._v: Dict[str, np.ndarray] = {}
        self._t = 0
        
        logger.info(
            f"LocalTrainer initialized | "
            f"lr={learning_rate} | optimizer={optimizer} | "
            f"batch_size={batch_size}"
        )

    def train(
        self,
        model: Dict[str, np.ndarray],
        data: Dict[str, Any],
        epochs: int = 5,
        loss_fn: Optional[Callable] = None,
        round_num: int = 0
    ) -> Dict[str, Any]:
        """
        Train model on local data for specified epochs.
        
        Args:
            model: Current global model weights
            data: Local training data (never leaves the node)
            epochs: Number of local training epochs
            loss_fn: Custom loss function (optional)
            round_num: Current federated round (for LR scheduling)
            
        Returns:
            Dict with gradients, loss history, and metrics
        """
        # Work on a local copy of the model
        local_model = {k: v.copy() for k, v in model.items()}
        initial_model = {k: v.copy() for k, v in model.items()}
        
        X = data.get("X", data.get("features"))
        y = data.get("y", data.get("labels"))
        num_samples = len(X)
        
        loss_history = []
        total_steps = 0
        
        # Adjust learning rate
        current_lr = self._get_lr(round_num)
        
        for epoch in range(epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_losses = []
            
            # Mini-batch training
            for batch_start in range(0, num_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_samples)
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Forward pass
                predictions, activations = self._forward(local_model, X_batch)
                
                # Compute loss
                if loss_fn:
                    loss = loss_fn(predictions, y_batch)
                else:
                    loss = self._cross_entropy_loss(predictions, y_batch)
                
                epoch_losses.append(loss)
                
                # Backward pass - compute gradients
                batch_grads = self._backward(
                    local_model, X_batch, y_batch, predictions, activations
                )
                
                # Apply weight decay
                if self.weight_decay > 0:
                    for key in batch_grads:
                        batch_grads[key] += self.weight_decay * local_model[key]
                
                # Update local model
                local_model = self._update_model(
                    local_model, batch_grads, current_lr
                )
                
                total_steps += 1
            
            epoch_loss = float(np.mean(epoch_losses))
            loss_history.append(epoch_loss)
            
            logger.debug(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f}")
        
        # Compute gradient as difference between updated and initial model
        gradients = {
            key: initial_model[key] - local_model[key]
            for key in local_model
        }
        
        # Compute final accuracy
        final_preds, _ = self._forward(local_model, X)
        accuracy = self._compute_accuracy(final_preds, y)
        
        logger.info(
            f"Local training complete | "
            f"Final loss: {loss_history[-1]:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Steps: {total_steps}"
        )
        
        return {
            "gradients": gradients,
            "final_loss": loss_history[-1],
            "loss_history": loss_history,
            "accuracy": accuracy,
            "total_steps": total_steps,
            "num_samples": num_samples,
            "local_model": local_model
        }

    def _forward(
        self,
        model: Dict[str, np.ndarray],
        X: np.ndarray
    ):
        """Simple 2-layer MLP forward pass"""
        activations = {}
        
        # Layer 1
        if "layer1.weight" in model:
            z1 = X @ model["layer1.weight"].T + model.get("layer1.bias", 0)
            a1 = np.maximum(0, z1)  # ReLU
            activations["z1"] = z1
            activations["a1"] = a1
            activations["input"] = X
        else:
            a1 = X
            activations["input"] = X
        
        # Layer 2
        if "layer2.weight" in model:
            z2 = a1 @ model["layer2.weight"].T + model.get("layer2.bias", 0)
            a2 = np.maximum(0, z2)  # ReLU
            activations["z2"] = z2
            activations["a2"] = a2
        else:
            a2 = a1
        
        # Output layer
        if "output.weight" in model:
            z_out = a2 @ model["output.weight"].T + model.get("output.bias", 0)
            # Softmax
            exp_z = np.exp(z_out - z_out.max(axis=1, keepdims=True))
            output = exp_z / exp_z.sum(axis=1, keepdims=True)
            activations["z_out"] = z_out
        else:
            output = a2
        
        return output, activations

    def _backward(
        self,
        model: Dict[str, np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
        predictions: np.ndarray,
        activations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute gradients via backpropagation"""
        batch_size = len(X)
        grads = {}
        
        # One-hot encode labels if needed
        if y.ndim == 1:
            num_classes = predictions.shape[1]
            y_onehot = np.zeros((batch_size, num_classes))
            y_onehot[np.arange(batch_size), y.astype(int)] = 1
        else:
            y_onehot = y
        
        # Output layer gradient
        delta_out = (predictions - y_onehot) / batch_size
        
        if "output.weight" in model:
            a2 = activations.get("a2", activations.get("a1", X))
            grads["output.weight"] = delta_out.T @ a2
            grads["output.bias"] = delta_out.sum(axis=0)
            
            # Propagate to layer 2
            delta2 = delta_out @ model["output.weight"]
            delta2 *= (a2 > 0)  # ReLU gradient
        else:
            delta2 = delta_out
        
        if "layer2.weight" in model:
            a1 = activations.get("a1", X)
            grads["layer2.weight"] = delta2.T @ a1
            grads["layer2.bias"] = delta2.sum(axis=0)
            
            # Propagate to layer 1
            delta1 = delta2 @ model["layer2.weight"]
            delta1 *= (a1 > 0)  # ReLU gradient
        else:
            delta1 = delta2
        
        if "layer1.weight" in model:
            grads["layer1.weight"] = delta1.T @ X
            grads["layer1.bias"] = delta1.sum(axis=0)
        
        return grads

    def _cross_entropy_loss(
        self,
        predictions: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Compute cross-entropy loss"""
        batch_size = len(y)
        
        if y.ndim == 1:
            # Sparse labels
            log_probs = np.log(predictions[np.arange(batch_size), y.astype(int)] + 1e-10)
        else:
            # One-hot labels
            log_probs = np.sum(y * np.log(predictions + 1e-10), axis=1)
        
        return float(-np.mean(log_probs))

    def _compute_accuracy(
        self,
        predictions: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Compute classification accuracy"""
        predicted_classes = np.argmax(predictions, axis=1)
        
        if y.ndim == 1:
            true_classes = y.astype(int)
        else:
            true_classes = np.argmax(y, axis=1)
        
        return float(np.mean(predicted_classes == true_classes))

    def _update_model(
        self,
        model: Dict[str, np.ndarray],
        gradients: Dict[str, np.ndarray],
        lr: float
    ) -> Dict[str, np.ndarray]:
        """Apply gradient update using configured optimizer"""
        self._t += 1
        updated = {}
        
        for key in model:
            if key not in gradients:
                updated[key] = model[key]
                continue
            
            grad = gradients[key]
            
            if self.optimizer == "sgd":
                # SGD with momentum
                if key not in self._velocity:
                    self._velocity[key] = np.zeros_like(grad)
                
                self._velocity[key] = (
                    self.momentum * self._velocity[key] + lr * grad
                )
                updated[key] = model[key] - self._velocity[key]
                
            elif self.optimizer == "adam":
                # Adam optimizer
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                
                if key not in self._m:
                    self._m[key] = np.zeros_like(grad)
                    self._v[key] = np.zeros_like(grad)
                
                self._m[key] = beta1 * self._m[key] + (1 - beta1) * grad
                self._v[key] = beta2 * self._v[key] + (1 - beta2) * grad ** 2
                
                m_hat = self._m[key] / (1 - beta1 ** self._t)
                v_hat = self._v[key] / (1 - beta2 ** self._t)
                
                updated[key] = model[key] - lr * m_hat / (np.sqrt(v_hat) + eps)
                
            else:
                # Vanilla gradient descent
                updated[key] = model[key] - lr * grad
        
        return updated

    def _get_lr(self, round_num: int) -> float:
        """Get current learning rate with scheduling"""
        if self.lr_scheduler == "cosine":
            # Cosine annealing
            import math
            T_max = 100  # Assumed total rounds
            lr = self.learning_rate * (
                1 + math.cos(math.pi * round_num / T_max)
            ) / 2
        elif self.lr_scheduler == "step":
            # Step decay
            decay_factor = 0.1 ** (round_num // 30)
            lr = self.learning_rate * decay_factor
        elif self.lr_scheduler == "warmup":
            # Linear warmup
            if round_num < self.warmup_rounds:
                lr = self.learning_rate * (round_num + 1) / self.warmup_rounds
            else:
                lr = self.learning_rate
        else:
            lr = self.learning_rate
        
        return max(lr, 1e-6)
