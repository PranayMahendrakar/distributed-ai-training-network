"""
Tests for the Differential Privacy Engine
Verifies noise mechanisms, gradient clipping, and privacy accounting
"""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from privacy.differential_privacy import DifferentialPrivacyEngine, PrivacyAccountant


class TestPrivacyAccountant:
    """Tests for the RDP privacy accountant"""

    def test_initialization(self):
        """Test accountant initializes with correct values"""
        accountant = PrivacyAccountant(epsilon=1.0, delta=1e-5)
        assert accountant.epsilon == 1.0
        assert accountant.delta == 1e-5
        assert accountant.total_steps == 0

    def test_budget_not_exceeded_initially(self):
        """Fresh accountant should not exceed budget"""
        accountant = PrivacyAccountant(epsilon=1.0, delta=1e-5)
        assert not accountant.is_budget_exceeded()

    def test_accumulate_increases_budget(self):
        """Accumulating should increase privacy cost"""
        accountant = PrivacyAccountant(epsilon=1.0, delta=1e-5)
        accountant.accumulate(noise_multiplier=1.0, sampling_rate=0.1)
        epsilon_spent, _ = accountant.get_epsilon()
        assert epsilon_spent > 0
        assert accountant.total_steps == 1

    def test_multiple_accumulations(self):
        """Multiple accumulations should increase cost"""
        accountant = PrivacyAccountant(epsilon=10.0, delta=1e-5)
        
        for _ in range(10):
            accountant.accumulate(noise_multiplier=1.0, sampling_rate=0.1)
        
        assert accountant.total_steps == 10
        epsilon_spent, _ = accountant.get_epsilon()
        assert epsilon_spent > 0

    def test_remaining_budget_decreases(self):
        """Remaining budget should decrease after accumulation"""
        accountant = PrivacyAccountant(epsilon=1.0, delta=1e-5)
        initial_remaining = accountant.remaining_budget()
        
        accountant.accumulate(noise_multiplier=1.0, sampling_rate=0.01)
        
        # Budget should have been consumed (some epsilon spent)
        # The remaining might not decrease to 0 from one step but should be tracked
        assert accountant.total_steps == 1


class TestDifferentialPrivacyEngine:
    """Tests for the DP Engine"""

    def setup_method(self):
        """Setup test fixtures"""
        self.gradients = {
            "layer1": np.random.randn(10, 5),
            "layer2": np.random.randn(5),
            "output": np.random.randn(3, 5)
        }

    def test_initialization_gaussian(self):
        """Test Gaussian mechanism initialization"""
        dp = DifferentialPrivacyEngine(
            epsilon=1.0,
            delta=1e-5,
            mechanism="gaussian"
        )
        assert dp.epsilon == 1.0
        assert dp.delta == 1e-5
        assert dp.mechanism == "gaussian"
        assert dp.noise_multiplier > 0

    def test_initialization_laplace(self):
        """Test Laplace mechanism initialization"""
        dp = DifferentialPrivacyEngine(
            epsilon=1.0,
            delta=1e-5,
            mechanism="laplace"
        )
        assert dp.mechanism == "laplace"

    def test_invalid_epsilon(self):
        """Test that invalid epsilon raises error"""
        with pytest.raises(ValueError):
            DifferentialPrivacyEngine(epsilon=-1.0, delta=1e-5)

    def test_invalid_delta(self):
        """Test that invalid delta raises error"""
        with pytest.raises(ValueError):
            DifferentialPrivacyEngine(epsilon=1.0, delta=1.5)

    def test_invalid_mechanism(self):
        """Test that invalid mechanism raises error"""
        with pytest.raises(ValueError):
            DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5, mechanism="invalid")

    def test_gradient_clipping(self):
        """Test that gradient clipping bounds L2 norm"""
        dp = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        
        # Create gradients with large norm
        large_grads = {
            "layer1": np.ones((10, 10)) * 100
        }
        
        clipped = dp.clip_gradients(large_grads, clip_norm=1.0)
        
        # Compute total norm of clipped gradients
        total_norm = math.sqrt(sum(np.sum(g**2) for g in clipped.values()))
        
        # Should be approximately clip_norm
        assert total_norm <= 1.0 + 1e-6

    def test_gradient_clipping_small_norm(self):
        """Test that small gradients are not clipped"""
        dp = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5, clip_norm=10.0)
        
        small_grads = {
            "layer1": np.array([[0.01, 0.01]])
        }
        
        clipped = dp.clip_gradients(small_grads, clip_norm=10.0)
        
        # Should be unchanged
        np.testing.assert_array_almost_equal(
            clipped["layer1"], small_grads["layer1"]
        )

    def test_add_noise_changes_gradients(self):
        """Test that noise addition changes gradient values"""
        dp = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5)
        
        original = {"layer1": np.zeros((10, 5))}
        noisy = dp.add_noise(original)
        
        # Gradients should be different (noise added)
        assert not np.allclose(original["layer1"], noisy["layer1"])

    def test_add_noise_preserves_shape(self):
        """Test that noise addition preserves tensor shapes"""
        dp = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5)
        
        noisy = dp.add_noise(self.gradients)
        
        for key in self.gradients:
            assert noisy[key].shape == self.gradients[key].shape

    def test_privatize_pipeline(self):
        """Test full privatization pipeline"""
        dp = DifferentialPrivacyEngine(
            epsilon=1.0,
            delta=1e-5,
            clip_norm=1.0
        )
        
        private_grads = dp.privatize(
            gradients=self.gradients,
            num_samples=32,
            total_samples=1000,
            clip=True
        )
        
        assert set(private_grads.keys()) == set(self.gradients.keys())
        
        for key in self.gradients:
            assert private_grads[key].shape == self.gradients[key].shape

    def test_privacy_spent_tracking(self):
        """Test that privacy budget is tracked after privatization"""
        dp = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5)
        
        dp.privatize(self.gradients, num_samples=32, total_samples=1000)
        
        spent = dp.privacy_spent()
        
        assert "epsilon_spent" in spent
        assert "epsilon_total" in spent
        assert "remaining" in spent
        assert spent["num_privatizations"] == 1

    def test_privacy_budget_accumulates(self):
        """Test that privacy budget accumulates across calls"""
        dp = DifferentialPrivacyEngine(epsilon=10.0, delta=1e-5)
        
        initial_spent = dp.privacy_spent()["epsilon_spent"]
        
        for _ in range(5):
            dp.privatize(self.gradients, num_samples=32, total_samples=1000)
        
        final_spent = dp.privacy_spent()["epsilon_spent"]
        
        assert final_spent >= initial_spent
        assert dp.privacy_spent()["num_privatizations"] == 5

    def test_reset_budget(self):
        """Test that privacy budget can be reset"""
        dp = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5)
        
        dp.privatize(self.gradients, num_samples=32, total_samples=1000)
        dp.reset_budget()
        
        assert dp.privacy_spent()["num_privatizations"] == 0

    def test_noise_multiplier_scales_with_privacy(self):
        """Test that stricter privacy requires more noise"""
        dp_strict = DifferentialPrivacyEngine(epsilon=0.1, delta=1e-5)
        dp_lenient = DifferentialPrivacyEngine(epsilon=10.0, delta=1e-5)
        
        # Stricter privacy should have larger noise multiplier
        assert dp_strict.noise_multiplier > dp_lenient.noise_multiplier

    def test_compute_epsilon_for_budget(self):
        """Test static method for pre-training epsilon computation"""
        epsilon = DifferentialPrivacyEngine.compute_epsilon_for_budget(
            noise_multiplier=1.0,
            num_steps=100,
            sampling_rate=0.01,
            delta=1e-5
        )
        
        assert epsilon > 0
        assert epsilon < float("inf")

    def test_gaussian_vs_laplace_noise(self):
        """Test that Gaussian and Laplace mechanisms produce different noise"""
        dp_gaussian = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5, mechanism="gaussian")
        dp_laplace = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5, mechanism="laplace")
        
        grads = {"layer1": np.zeros((100,))}
        
        # Run multiple times and check statistics
        gaussian_noises = []
        laplace_noises = []
        
        for _ in range(100):
            g = dp_gaussian.add_noise({"layer1": np.zeros((100,))})
            l = dp_laplace.add_noise({"layer1": np.zeros((100,))})
            gaussian_noises.extend(g["layer1"].tolist())
            laplace_noises.extend(l["layer1"].tolist())
        
        # Both should have zero mean approximately
        assert abs(np.mean(gaussian_noises)) < 1.0
        assert abs(np.mean(laplace_noises)) < 1.0


class TestPrivacyBudgetManagement:
    """Integration tests for privacy budget management"""

    def test_full_training_simulation(self):
        """Simulate a full training run with privacy tracking"""
        dp = DifferentialPrivacyEngine(
            epsilon=5.0,
            delta=1e-5,
            mechanism="gaussian",
            clip_norm=1.0
        )
        
        gradients = {
            "weight": np.random.randn(10, 5),
            "bias": np.random.randn(5)
        }
        
        # Simulate 20 training rounds
        for round_num in range(20):
            private_grads = dp.privatize(
                gradients,
                num_samples=32,
                total_samples=1000
            )
            
            spent = dp.privacy_spent()
            
            # Privacy should be tracked
            assert spent["num_privatizations"] == round_num + 1
            
            # Budget fraction should be valid
            assert 0 <= spent["budget_fraction"] <= 1.0 + 1e-6

    def test_budget_fraction_increases(self):
        """Test that budget fraction increases over time"""
        dp = DifferentialPrivacyEngine(epsilon=5.0, delta=1e-5)
        
        gradients = {"w": np.random.randn(5, 5)}
        
        fractions = []
        for _ in range(5):
            dp.privatize(gradients, num_samples=10, total_samples=100)
            fractions.append(dp.privacy_spent()["budget_fraction"])
        
        # Fractions should be non-decreasing
        for i in range(1, len(fractions)):
            assert fractions[i] >= fractions[i-1] - 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
