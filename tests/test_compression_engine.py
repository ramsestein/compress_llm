import os
import sys
import unittest

import torch
import torch.nn as nn

# Ensure the project package is on the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from apply_compression import save_pretrained_with_fallback
from transformers import AutoModelForCausalLM, GPT2Config
from transformers.utils import is_safetensors_available

from create_compress.compression_engine import CompressionEngine, QuantizedLinear
from create_compress.compression_methods import LowRankApproximation


class TestCompressionEngine(unittest.TestCase):
    """Tests for the compression engine."""

    def setUp(self):
        """Set up the compression engine for each test."""
        self.engine = CompressionEngine()

    def test_calculate_quantization_params_constant_tensor(self):
        """Test quantization parameters for a constant tensor."""
        tensor = torch.zeros(10)
        scale, zero_point = self.engine._calculate_quantization_params(tensor, 8)
        self.assertEqual(scale, 1.0)
        self.assertEqual(zero_point, 0)
        q = self.engine._quantize_tensor(tensor, scale, zero_point, 8)
        self.assertEqual(q.dtype, torch.int8)
        self.assertTrue(torch.all(q == 0))

    def test_quantize_tensor_dtype_for_high_bits(self):
        """Test quantization dtype for high bit widths."""
        tensor = torch.ones(5)
        q = self.engine._quantize_tensor(tensor, 1.0, 0, 16)
        self.assertEqual(q.dtype, torch.int16)

    def test_compress_layer_unknown_method_uses_none(self):
        """Test that an unknown compression method falls back to 'none'."""
        layer = nn.Linear(4, 4)
        compressed, result = self.engine.compress_layer(layer, {"name": "unknown", "strength": 0.5})
        self.assertIsInstance(compressed, nn.Linear)
        self.assertEqual(result.method_used, "none")
        self.assertLess(abs(result.compression_ratio), 0.001)

    def test_compress_layer_int8_quantization(self):
        """Test INT8 quantization on a linear layer."""
        layer = nn.Linear(4, 4)
        compressed, result = self.engine.compress_layer(
            layer, {"name": "int8_quantization", "strength": 1.0}
        )
        self.assertIsInstance(compressed, QuantizedLinear)
        self.assertTrue(result.success)
        self.assertGreater(result.compression_ratio, 0)

    def test_randomized_svd_returns_correct_shapes(self):
        """Test randomized SVD returns correct matrix shapes."""
        method = LowRankApproximation()
        weight = torch.randn(50, 30)
        U, S, V = method._randomized_svd(weight, rank=10)
        self.assertEqual(U.shape, (50, 10))
        self.assertEqual(S.shape, (10,))
        self.assertEqual(V.shape, (30, 10))
        # Reconstruction should match the original weight shape
        recon = (U * S) @ V.t()
        self.assertEqual(recon.shape, weight.shape)

    @unittest.skipIf(not is_safetensors_available(), "safetensors not installed")
    def test_save_pretrained_with_fallback_creates_safetensors(self):
        """Test that the fallback save strategy creates a safetensors file."""
        config = GPT2Config(n_layer=1, n_head=1, n_embd=32)
        model = AutoModelForCausalLM.from_config(config)
        # Create a temporary directory for the test
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp()
        try:
            save_pretrained_with_fallback(model, None, temp_dir)
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "model.safetensors")))
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
