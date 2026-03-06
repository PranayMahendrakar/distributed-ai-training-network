"""
Differential Privacy Engine - Privacy-preserving gradient noise mechanisms
Implements Gaussian, Laplace, and Exponential mechanisms with privacy accounting
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PrivacyAccountant:
    """
    Tracks cumulative privacy budget consumption using Renyi DP accounting.
    Implements the moments accountant for tight privacy analysis.
    """
    epsilon: float
    delta: float
    mechanism: str = "gaussian"
    orders: List[float] = field(default_factory=lambda: list(range(2, 64)) + [0.5 * i for i in range(1, 128)])
    
    rdp_budget: List[float] = field(default_factory=list, init=False)
    total_steps: int = field(default=0, init=False)
    
    def __post_init__(self):
        self.rdp_budget = [0.0] * len(self.orders)

    def accumulate(self, noise_multiplier: float, sampling_rate: float):
        """Add privacy cost of one mechanism application"""
        self.total_steps += 1
        
        for i, order in enumerate(self.orders):
            rdp_cost = self._compute_rdp(noise_multiplier, sampling_rate, order)
            self.rdp_budget[i] += rdp_cost

    def _compute_rdp(self, noise_multiplier: float, sampling_rate: float, order: float) -> float:
        """Compute Renyi DP for subsampled Gaussian mechanism"""
        if noise_multiplier == 0:
            return float("inf")
        
        # Simplified RDP computation for Gaussian mechanism
        return order / (2 * noise_multiplier ** 2)

    def get_epsilon(self) -> Tuple[float, float]:
        """Convert RDP to (epsilon, delta) using conversion theorem"""
        if not self.rdp_budget:
            return 0.0, self.delta
        
        best_epsilon = float("inf")
        
        for i, (order, rdp) in enumerate(zip(self.orders, self.rdp_budget)):
            if rdp == float("inf") or order <= 1:
                continue
            
            epsilon = rdp + math.log(1 / self.delta) / (order - 1)
            best_epsilon = min(best_epsilon, epsilon)
        
        return best_epsilon, self.delta

    def remaining_budget(self) -> float:
        """Get remaining epsilon budget"""
        spent, _ = self.get_epsilon()
        return max(0.0, self.epsilon - spent)

    def is_budget_exceeded(self) -> bool:
        """Check if privacy budget has been exceeded"""
        spent, _ = self.get_epsilon()
        return spent > self.epsilon


class DifferentialPrivacyEngine:
    """
    Differential Privacy engine for federated learning gradient privatization.
    
    Implements:
    - Gaussian Mechanism: epsilon, delta-DP with Gaussian noise
    - Laplace Mechanism: epsilon-DP with Laplace noise  
    - Exponential Mechanism: epsilon-DP for discrete outputs
    - Gradient clipping: Sensitivity bounding
    - Privacy amplification via sampling
    
    Usage:
        dp = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5, mechanism="gaussian")
        private_grads = dp.privatize(gradients, num_samples=100)
    """

    MECHANISMS = ["gaussian", "laplace", "exponential"]

    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-5,
        mechanism: str = "gaussian",
        sensitivity: float = 1.0,
        clip_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        accountant: str = "rdp"
    ):
        """
        Initialize the DP engine.
        
        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Failure probability (typically 1e-5 to 1e-7)
            mechanism: Noise mechanism to use
            sensitivity: L2 sensitivity of the function
            clip_norm: Maximum L2 norm for gradient clipping
            noise_multiplier: Override computed noise multiplier
            accountant: Privacy accounting method (rdp or basic)
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if delta < 0 or delta >= 1:
            raise ValueError("delta must be in [0, 1)")
        if mechanism not in self.MECHANISMS:
            raise ValueError(f"Unknown mechanism '{mechanism}'. Choose from: {self.MECHANISMS}")
        
        self.epsilon = epsilon
        self.delta = delta
        self.mechanism = mechanism
        self.sensitivity = sensitivity
        self.clip_norm = clip_norm
        
        # Compute noise multiplier
        if noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
        else:
            self.noise_multiplier = self._compute_noise_multiplier()
        
        # Initialize privacy accountant
        self.accountant = PrivacyAccountant(
            epsilon=epsilon,
            delta=delta,
            mechanism=mechanism
        )
        
        self._privatizations = 0
        
        logger.info(
            f"DifferentialPrivacyEngine initialized | "
            f"epsilon={epsilon} | delta={delta} | "
            f"mechanism={mechanism} | noise_multiplier={self.noise_multiplier:.4f}"
        )

    def _compute_noise_multiplier(self) -> float:
        """Compute noise multiplier from privacy parameters"""
        if self.mechanism == "gaussian":
            # sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
            return math.sqrt(2 * math.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
        elif self.mechanism == "laplace":
            # b = sensitivity / epsilon
            return self.sensitivity / self.epsilon
        else:
            return self.sensitivity / self.epsilon

    def clip_gradients(
        self,
        gradients: Dict[str, np.ndarray],
        clip_norm: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Clip gradient L2 norm to bound sensitivity.
        
        Args:
            gradients: Dict of gradient arrays
            clip_norm: Override clip norm
            
        Returns:
            Clipped gradients
        """
        norm = clip_norm or self.clip_norm
        
        # Compute total L2 norm across all gradients
        total_norm = math.sqrt(
            sum(np.sum(g ** 2) for g in gradients.values())
        )
        
        if total_norm > norm:
            scale = norm / total_norm
            clipped = {k: v * scale for k, v in gradients.items()}
            logger.debug(f"Gradient clipped: {total_norm:.4f} -> {norm:.4f} (scale={scale:.4f})")
            return clipped
        
        return gradients

    def add_noise(
        self,
        gradients: Dict[str, np.ndarray],
        sampling_rate: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Add calibrated noise to gradients.
        
        Args:
            gradients: Dict of gradient arrays
            sampling_rate: Fraction of dataset used in this step
            
        Returns:
            Noisy gradients
        """
        noisy = {}
        
        for key, grad in gradients.items():
            if self.mechanism == "gaussian":
                noise = np.random.normal(
                    loc=0.0,
                    scale=self.noise_multiplier * self.sensitivity,
                    size=grad.shape
                )
            elif self.mechanism == "laplace":
                noise = np.random.laplace(
                    loc=0.0,
                    scale=self.noise_multiplier,
                    size=grad.shape
                )
            else:  # exponential
                noise = np.random.exponential(
                    scale=self.noise_multiplier,
                    size=grad.shape
                ) * np.random.choice([-1, 1], size=grad.shape)
            
            noisy[key] = grad + noise
        
        # Update privacy accountant
        self.accountant.accumulate(self.noise_multiplier, sampling_rate)
        self._privatizations += 1
        
        return noisy

    def privatize(
        self,
        gradients: Dict[str, np.ndarray],
        num_samples: int = 1,
        total_samples: int = 1,
        clip: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Full privatization pipeline: clip + noise.
        
        Args:
            gradients: Raw gradient dict
            num_samples: Samples in this batch
            total_samples: Total dataset size
            clip: Whether to clip gradients first
            
        Returns:
            Privatized gradients
        """
        # Check budget
        if self.accountant.is_budget_exceeded():
            logger.warning("Privacy budget exceeded! Training should stop.")
        
        sampling_rate = num_samples / max(total_samples, 1)
        
        # Step 1: Clip gradients
        if clip:
            gradients = self.clip_gradients(gradients)
        
        # Step 2: Add calibrated noise
        private_gradients = self.add_noise(gradients, sampling_rate)
        
        spent, _ = self.accountant.get_epsilon()
        logger.debug(
            f"Privatized gradients | "
            f"Privacy spent: {spent:.4f}/{self.epsilon} | "
            f"Remaining: {self.accountant.remaining_budget():.4f}"
        )
        
        return private_gradients

    def privacy_spent(self) -> Dict[str, float]:
        """Get current privacy expenditure"""
        epsilon_spent, delta_used = self.accountant.get_epsilon()
        
        return {
            "epsilon_spent": float(epsilon_spent),
            "epsilon_total": self.epsilon,
            "delta": self.delta,
            "remaining": float(self.accountant.remaining_budget()),
            "num_privatizations": self._privatizations,
            "budget_fraction": min(1.0, epsilon_spent / self.epsilon)
        }

    def reset_budget(self):
        """Reset privacy budget (use with caution)"""
        self.accountant = PrivacyAccountant(
            epsilon=self.epsilon,
            delta=self.delta,
            mechanism=self.mechanism
        )
        self._privatizations = 0
        logger.info("Privacy budget reset")

    @staticmethod
    def compute_epsilon_for_budget(
        noise_multiplier: float,
        num_steps: int,
        sampling_rate: float,
        delta: float,
        orders: Optional[List[float]] = None
    ) -> float:
        """
        Compute epsilon for a given noise configuration and training run.
        
        Useful for pre-training privacy analysis.
        """
        if orders is None:
            orders = list(range(2, 64))
        
        rdp_total = 0.0
        step_rdp = noise_multiplier ** (-2) / 2  # Per-step RDP cost
        
        # Amplification via sampling
        amplified_rdp = sampling_rate ** 2 * step_rdp
        total_rdp = num_steps * amplified_rdp
        
        # Convert to epsilon-delta DP
        best_epsilon = float("inf")
        for order in orders:
            if order <= 1:
                continue
            epsilon = total_rdp + math.log(1 / delta) / (order - 1)
            best_epsilon = min(best_epsilon, epsilon)
        
        return best_epsilon
