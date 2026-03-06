"""
Gradient Compressor - Reduces communication overhead in federated learning
Implements Top-K sparsification, quantization, and error feedback mechanisms
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class GradientCompressor:
    """
    Gradient compression for bandwidth-efficient federated learning.
    
    Techniques:
    - Top-K Sparsification: Keep only top-k% largest gradients
    - Quantization: Reduce bit-width of gradient values
    - Error Feedback: Accumulate compression errors for correction
    - Random Sparsification: Randomly sample gradients (unbiased estimator)
    
    Achieves 10-100x bandwidth reduction with minimal accuracy loss.
    
    Usage:
        compressor = GradientCompressor(compression_ratio=0.1)
        compressed = compressor.compress(gradients)
        restored = compressor.decompress(compressed)
    """

    METHODS = ["topk", "random", "quantization", "combined"]

    def __init__(
        self,
        compression_ratio: float = 0.1,
        method: str = "topk",
        quantize_bits: int = 8,
        use_error_feedback: bool = True
    ):
        """
        Initialize gradient compressor.
        
        Args:
            compression_ratio: Fraction of gradients to keep (0.0-1.0)
            method: Compression method (topk, random, quantization, combined)
            quantize_bits: Bit width for quantization (4, 8, or 16)
            use_error_feedback: Whether to accumulate and correct compression errors
        """
        if not 0 < compression_ratio <= 1:
            raise ValueError("compression_ratio must be in (0, 1]")
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {self.METHODS}")
        if quantize_bits not in [4, 8, 16, 32]:
            raise ValueError("quantize_bits must be 4, 8, 16, or 32")
        
        self.compression_ratio = compression_ratio
        self.method = method
        self.quantize_bits = quantize_bits
        self.use_error_feedback = use_error_feedback
        
        # Error feedback state (per parameter)
        self.error_feedback: Dict[str, np.ndarray] = {}
        
        # Compression statistics
        self.total_original_size = 0
        self.total_compressed_size = 0
        
        logger.info(
            f"GradientCompressor initialized | "
            f"Method: {method} | "
            f"Ratio: {compression_ratio} | "
            f"Bits: {quantize_bits}"
        )

    def compress(self, gradients: Dict[str, np.ndarray]) -> Dict:
        """
        Compress gradient dictionary.
        
        Args:
            gradients: Dict of layer_name -> gradient array
            
        Returns:
            Dict with compressed_gradients, indices, metadata
        """
        compressed = {}
        total_original = 0
        total_compressed = 0
        
        for key, grad in gradients.items():
            original_size = grad.size
            total_original += original_size
            
            if self.method == "topk":
                comp_grad, meta = self._topk_compress(key, grad)
            elif self.method == "random":
                comp_grad, meta = self._random_compress(grad)
            elif self.method == "quantization":
                comp_grad, meta = self._quantize(grad)
            else:  # combined
                comp_grad, meta = self._combined_compress(key, grad)
            
            compressed[key] = {
                "values": comp_grad,
                "meta": meta
            }
            
            compressed_size = comp_grad.size if hasattr(comp_grad, 'size') else len(comp_grad)
            total_compressed += compressed_size
        
        self.total_original_size += total_original
        self.total_compressed_size += total_compressed
        
        compression_achieved = 1 - (total_compressed / max(total_original, 1))
        
        logger.debug(
            f"Compression: {total_original} -> {total_compressed} params "
            f"({compression_achieved:.1%} reduction)"
        )
        
        return {
            "compressed_gradients": compressed,
            "original_size": total_original,
            "compressed_size": total_compressed,
            "compression_ratio_achieved": compression_achieved,
            "method": self.method
        }

    def decompress(self, compressed_data: Dict) -> Dict[str, np.ndarray]:
        """
        Decompress gradients back to original format.
        
        Args:
            compressed_data: Output from compress()
            
        Returns:
            Reconstructed gradient dict
        """
        if "compressed_gradients" in compressed_data:
            compressed = compressed_data["compressed_gradients"]
        else:
            compressed = compressed_data
        
        restored = {}
        
        for key, comp in compressed.items():
            values = comp["values"]
            meta = comp["meta"]
            
            if meta.get("method") == "topk":
                restored[key] = self._topk_decompress(values, meta)
            elif meta.get("method") == "random":
                restored[key] = self._random_decompress(values, meta)
            elif meta.get("method") == "quantization":
                restored[key] = self._dequantize(values, meta)
            elif meta.get("method") == "combined":
                restored[key] = self._combined_decompress(values, meta)
            else:
                restored[key] = values
        
        return restored

    def _topk_compress(
        self,
        key: str,
        grad: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Top-K sparsification: keep largest k% gradients by absolute value"""
        original_shape = grad.shape
        flat_grad = grad.flatten()
        
        # Apply error feedback
        if self.use_error_feedback:
            if key not in self.error_feedback:
                self.error_feedback[key] = np.zeros_like(flat_grad)
            flat_grad = flat_grad + self.error_feedback[key]
        
        # Determine k
        k = max(1, int(flat_grad.size * self.compression_ratio))
        
        # Find top-k by absolute value
        abs_grad = np.abs(flat_grad)
        top_k_indices = np.argpartition(abs_grad, -k)[-k:]
        top_k_values = flat_grad[top_k_indices]
        
        # Update error feedback
        if self.use_error_feedback:
            compressed_approx = np.zeros_like(flat_grad)
            compressed_approx[top_k_indices] = top_k_values
            self.error_feedback[key] = flat_grad - compressed_approx
        
        meta = {
            "method": "topk",
            "indices": top_k_indices,
            "original_shape": original_shape,
            "original_size": flat_grad.size
        }
        
        return top_k_values, meta

    def _topk_decompress(self, values: np.ndarray, meta: Dict) -> np.ndarray:
        """Reconstruct from top-k compressed gradient"""
        flat = np.zeros(meta["original_size"])
        flat[meta["indices"]] = values
        return flat.reshape(meta["original_shape"])

    def _random_compress(self, grad: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Random sparsification with unbiased estimation"""
        original_shape = grad.shape
        flat_grad = grad.flatten()
        
        k = max(1, int(flat_grad.size * self.compression_ratio))
        
        # Random sampling without replacement
        indices = np.random.choice(flat_grad.size, k, replace=False)
        values = flat_grad[indices] / self.compression_ratio  # Unbiased rescaling
        
        meta = {
            "method": "random",
            "indices": indices,
            "original_shape": original_shape,
            "original_size": flat_grad.size
        }
        
        return values, meta

    def _random_decompress(self, values: np.ndarray, meta: Dict) -> np.ndarray:
        """Reconstruct from randomly sampled gradient"""
        flat = np.zeros(meta["original_size"])
        flat[meta["indices"]] = values
        return flat.reshape(meta["original_shape"])

    def _quantize(self, grad: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Scalar quantization to reduce bit-width"""
        min_val = float(grad.min())
        max_val = float(grad.max())
        
        if min_val == max_val:
            return np.zeros_like(grad, dtype=np.float32), {
                "method": "quantization",
                "min_val": min_val,
                "max_val": max_val,
                "original_shape": grad.shape,
                "bits": self.quantize_bits
            }
        
        # Number of quantization levels
        num_levels = 2 ** self.quantize_bits - 1
        
        # Normalize to [0, 1] and quantize
        normalized = (grad - min_val) / (max_val - min_val)
        quantized = np.round(normalized * num_levels).astype(np.uint32)
        
        meta = {
            "method": "quantization",
            "min_val": min_val,
            "max_val": max_val,
            "original_shape": grad.shape,
            "bits": self.quantize_bits
        }
        
        return quantized, meta

    def _dequantize(self, values: np.ndarray, meta: Dict) -> np.ndarray:
        """Restore quantized gradients to float"""
        num_levels = 2 ** meta["bits"] - 1
        restored = values.astype(np.float32) / num_levels
        restored = restored * (meta["max_val"] - meta["min_val"]) + meta["min_val"]
        return restored.reshape(meta["original_shape"])

    def _combined_compress(
        self,
        key: str,
        grad: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Combined Top-K + quantization for maximum compression"""
        # First apply Top-K
        topk_values, topk_meta = self._topk_compress(key, grad)
        
        # Then quantize the selected values
        quantized_values, quant_meta = self._quantize(topk_values)
        
        meta = {
            "method": "combined",
            "topk_meta": topk_meta,
            "quant_meta": quant_meta
        }
        
        return quantized_values, meta

    def _combined_decompress(self, values: np.ndarray, meta: Dict) -> np.ndarray:
        """Restore combined compressed gradient"""
        # First dequantize
        dequantized = self._dequantize(values, meta["quant_meta"])
        
        # Then restore from Top-K
        return self._topk_decompress(dequantized, meta["topk_meta"])

    def get_compression_stats(self) -> Dict:
        """Get cumulative compression statistics"""
        if self.total_original_size == 0:
            return {"ratio": 0.0, "savings": 0.0}
        
        ratio = self.total_compressed_size / self.total_original_size
        savings = 1 - ratio
        
        return {
            "compression_ratio": float(ratio),
            "bandwidth_savings": float(savings),
            "total_original_params": self.total_original_size,
            "total_compressed_params": self.total_compressed_size,
            "method": self.method,
            "target_ratio": self.compression_ratio
        }

    def reset_error_feedback(self):
        """Reset accumulated compression errors"""
        self.error_feedback = {}
        logger.info("Error feedback buffers reset")
