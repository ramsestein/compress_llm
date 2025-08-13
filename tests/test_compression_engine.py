import torch
import os
import sys
import torch.nn as nn
import pytest

# Asegurar que el paquete del proyecto esté en el path de importación
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from create_compress.compression_engine import CompressionEngine, QuantizedLinear
from create_compress.compression_methods import LowRankApproximation


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


def test_randomized_svd_returns_correct_shapes():
    method = LowRankApproximation()
    weight = torch.randn(50, 30)
    U, S, V = method._randomized_svd(weight, rank=10)
    assert U.shape == (50, 10)
    assert S.shape == (10,)
    assert V.shape == (30, 10)
    # La reconstrucción con las formas retornadas debe reproducir la forma
    # original sin errores de orientación
    recon = (U * S) @ V.t()
    assert recon.shape == weight.shape
