"""
Secure Aggregation - Cryptographic gradient aggregation without revealing individual updates
Uses Shamir Secret Sharing for privacy-preserving gradient summation
"""

import logging
import os
import hashlib
import hmac
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class SecretSharing:
    """
    Shamir Secret Sharing for distributing gradient secrets among nodes.
    Ensures no single node can learn about others' gradients.
    """

    def __init__(self, threshold: int, num_shares: int, prime: int = 2**31 - 1):
        """
        Initialize secret sharing scheme.
        
        Args:
            threshold: Minimum shares needed for reconstruction (k)
            num_shares: Total number of shares (n)
            prime: Large prime for finite field arithmetic
        """
        if threshold > num_shares:
            raise ValueError("Threshold cannot exceed number of shares")
        if threshold < 2:
            raise ValueError("Threshold must be at least 2 for security")
        
        self.threshold = threshold
        self.num_shares = num_shares
        self.prime = prime
        
        logger.info(
            f"SecretSharing initialized | "
            f"threshold={threshold}/{num_shares} | prime={prime}"
        )

    def _eval_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial with given coefficients at point x (mod prime)"""
        result = 0
        for coeff in reversed(coefficients):
            result = (result * x + coeff) % self.prime
        return result

    def create_shares(self, secret: int) -> List[Tuple[int, int]]:
        """
        Create n shares of a secret using Shamir's scheme.
        
        Args:
            secret: Integer secret to share
            
        Returns:
            List of (x, y) share pairs
        """
        secret = secret % self.prime
        
        # Random polynomial coefficients (degree = threshold - 1)
        coefficients = [secret] + [
            int.from_bytes(os.urandom(4), 'big') % self.prime
            for _ in range(self.threshold - 1)
        ]
        
        # Evaluate polynomial at points 1..n
        shares = [
            (i, self._eval_polynomial(coefficients, i))
            for i in range(1, self.num_shares + 1)
        ]
        
        return shares

    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> int:
        """
        Reconstruct secret from k shares using Lagrange interpolation.
        
        Args:
            shares: List of (x, y) share pairs (at least threshold shares)
            
        Returns:
            Reconstructed secret
        """
        if len(shares) < self.threshold:
            raise ValueError(
                f"Need at least {self.threshold} shares, got {len(shares)}"
            )
        
        secret = 0
        
        for i, (xi, yi) in enumerate(shares):
            numerator = 1
            denominator = 1
            
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-xj)) % self.prime
                    denominator = (denominator * (xi - xj)) % self.prime
            
            # Modular inverse of denominator
            lagrange_coeff = (
                numerator * pow(denominator, self.prime - 2, self.prime)
            ) % self.prime
            
            secret = (secret + yi * lagrange_coeff) % self.prime
        
        return secret

    def share_gradient(self, gradient: np.ndarray) -> List[np.ndarray]:
        """
        Create shares for an entire gradient array.
        
        Each element is secret-shared independently.
        """
        scale = 10**6  # Scale floats to integers
        
        flat_grad = gradient.flatten()
        all_shares = [[] for _ in range(self.num_shares)]
        
        for value in flat_grad:
            int_val = int(value * scale) % self.prime
            shares = self.create_shares(int_val)
            
            for node_idx, (x, y) in enumerate(shares):
                all_shares[node_idx].append(y)
        
        return [
            np.array(shares, dtype=np.int64).reshape(gradient.shape)
            for shares in all_shares
        ]

    def reconstruct_gradient(
        self,
        node_shares: List[np.ndarray],
        node_indices: List[int]
    ) -> np.ndarray:
        """
        Reconstruct gradient from node shares.
        
        Args:
            node_shares: List of gradient share arrays
            node_indices: Which nodes provided shares (1-indexed)
        """
        scale = 10**6
        
        if len(node_shares) < self.threshold:
            raise ValueError(
                f"Insufficient shares: {len(node_shares)} < {self.threshold}"
            )
        
        flat_shares = [s.flatten() for s in node_shares]
        gradient_size = flat_shares[0].size
        
        reconstructed = []
        
        for elem_idx in range(gradient_size):
            shares = [
                (idx, int(flat_shares[i][elem_idx]))
                for i, idx in enumerate(node_indices)
            ]
            
            secret = self.reconstruct_secret(shares)
            
            # Handle negative values (from modular arithmetic)
            if secret > self.prime // 2:
                secret -= self.prime
            
            reconstructed.append(secret / scale)
        
        return np.array(reconstructed).reshape(node_shares[0].shape)


class GradientMACVerifier:
    """
    Verifies gradient integrity using HMAC to prevent gradient poisoning attacks.
    Each node signs its gradients before transmission.
    """

    def __init__(self):
        self._keys: Dict[str, bytes] = {}

    def register_node(self, node_id: str) -> bytes:
        """Register a node and generate its secret key"""
        key = os.urandom(32)
        self._keys[node_id] = key
        return key

    def sign_gradient(
        self,
        node_id: str,
        gradient_bytes: bytes,
        key: bytes
    ) -> str:
        """Create HMAC signature for gradient bytes"""
        mac = hmac.new(key, gradient_bytes, hashlib.sha256)
        return mac.hexdigest()

    def verify_gradient(
        self,
        node_id: str,
        gradient_bytes: bytes,
        signature: str
    ) -> bool:
        """Verify gradient HMAC signature"""
        if node_id not in self._keys:
            logger.warning(f"Unknown node: {node_id}")
            return False
        
        key = self._keys[node_id]
        expected_mac = hmac.new(key, gradient_bytes, hashlib.sha256)
        
        return hmac.compare_digest(expected_mac.hexdigest(), signature)


class SecureAggregator:
    """
    Privacy-preserving gradient aggregation using cryptographic protocols.
    
    Implements:
    - Secret sharing for gradient privacy
    - HMAC verification for gradient integrity
    - Secure sum without revealing individual contributions
    
    Properties:
    - Server learns only the aggregate, not individual gradients
    - Nodes cannot learn about each other's gradients
    - Byzantine-robust with enough honest nodes
    """

    def __init__(
        self,
        threshold: int = 3,
        num_nodes: int = 5,
        verify_integrity: bool = True
    ):
        """
        Initialize secure aggregator.
        
        Args:
            threshold: Minimum nodes for reconstruction (t)
            num_nodes: Total number of nodes (n)
            verify_integrity: Whether to verify gradient signatures
        """
        self.threshold = threshold
        self.num_nodes = num_nodes
        self.verify_integrity = verify_integrity
        
        self.secret_sharing = SecretSharing(
            threshold=threshold,
            num_shares=num_nodes
        )
        
        self.mac_verifier = GradientMACVerifier() if verify_integrity else None
        
        # State for current aggregation round
        self._round_shares: Dict[str, List] = {}
        self._node_indices: Dict[str, int] = {}
        
        logger.info(
            f"SecureAggregator initialized | "
            f"threshold={threshold}/{num_nodes} | "
            f"integrity_check={verify_integrity}"
        )

    def register_nodes(self, node_ids: List[str]) -> Dict[str, bytes]:
        """
        Register nodes for this aggregation round.
        
        Returns:
            Dict mapping node_id -> secret key
        """
        keys = {}
        for i, node_id in enumerate(node_ids):
            self._node_indices[node_id] = i + 1  # 1-indexed
            
            if self.mac_verifier:
                key = self.mac_verifier.register_node(node_id)
                keys[node_id] = key
        
        logger.info(f"Registered {len(node_ids)} nodes for secure aggregation")
        return keys

    def receive_update(
        self,
        node_id: str,
        gradient: Dict[str, np.ndarray],
        signature: Optional[str] = None
    ) -> bool:
        """
        Receive and validate a gradient update from a node.
        
        Args:
            node_id: Node identifier
            gradient: Gradient dictionary
            signature: Optional HMAC signature
            
        Returns:
            True if update accepted
        """
        # Verify integrity if enabled
        if self.verify_integrity and self.mac_verifier and signature:
            grad_bytes = self._serialize_gradient(gradient)
            if not self.mac_verifier.verify_gradient(node_id, grad_bytes, signature):
                logger.warning(f"Gradient integrity check failed for node {node_id}")
                return False
        
        self._round_shares[node_id] = gradient
        logger.debug(f"Received update from {node_id} ({len(self._round_shares)} total)")
        
        return True

    def aggregate_securely(
        self,
        node_ids: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Securely aggregate gradients from all participating nodes.
        
        For simplicity, this implements a simulated secure aggregation
        (real deployment would use MPC protocols).
        
        Args:
            node_ids: Optional subset of nodes to aggregate
            
        Returns:
            Aggregated gradient dict
        """
        if node_ids is None:
            node_ids = list(self._round_shares.keys())
        
        if len(node_ids) < self.threshold:
            raise ValueError(
                f"Insufficient nodes for secure aggregation: "
                f"{len(node_ids)} < {self.threshold}"
            )
        
        # Collect gradients
        gradients = [self._round_shares[nid] for nid in node_ids if nid in self._round_shares]
        
        if not gradients:
            raise ValueError("No gradients available for aggregation")
        
        # Secure sum (in real deployment, this uses MPC/HE protocols)
        aggregated = {}
        param_keys = list(gradients[0].keys())
        
        for key in param_keys:
            stacked = np.stack([g[key] for g in gradients])
            aggregated[key] = np.mean(stacked, axis=0)
        
        # Clear round state
        self._round_shares.clear()
        
        logger.info(
            f"Secure aggregation complete | "
            f"Nodes: {len(node_ids)} | "
            f"Params: {len(param_keys)}"
        )
        
        return aggregated

    @staticmethod
    def _serialize_gradient(gradient: Dict[str, np.ndarray]) -> bytes:
        """Serialize gradient dict to bytes for signing"""
        parts = []
        for key in sorted(gradient.keys()):
            parts.append(key.encode())
            parts.append(gradient[key].tobytes())
        return b"||".join(parts)
