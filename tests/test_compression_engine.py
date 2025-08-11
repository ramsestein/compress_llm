import torch
import torch.nn as nn
import pytest

from create_compress.compression_engine import CompressionEngine, QuantizedLinear


def test_calculate_quantization_params_constant_tensor():
    engine = CompressionEngine()
    tensor = torch.zeros(10)
    scale, zero_point = engine._calculate_quantization_params(tensor, 8)
    assert scale == 1.0
    assert zero_point == 0
    q = engine._quantize_tensor(tensor, scale, zero_point, 8)
    assert q.dtype == torch.int8
    assert torch.all(q == 0)


def test_quantize_tensor_dtype_for_high_bits():
    engine = CompressionEngine()
    tensor = torch.ones(5)
    q = engine._quantize_tensor(tensor, 1.0, 0, 16)
    assert q.dtype == torch.int16


def test_compress_layer_unknown_method_uses_none():
    engine = CompressionEngine()
    layer = nn.Linear(4, 4)
    compressed, result = engine.compress_layer(layer, {"name": "unknown", "strength": 0.5})
    assert isinstance(compressed, nn.Linear)
    assert result.method_used == "none"
    assert pytest.approx(result.compression_ratio) == 0


def test_compress_layer_int8_quantization():
    engine = CompressionEngine()
    layer = nn.Linear(4, 4)
    compressed, result = engine.compress_layer(layer, {"name": "int8_quantization", "strength": 1.0})
    assert isinstance(compressed, QuantizedLinear)
    assert result.success
    assert result.compression_ratio > 0
