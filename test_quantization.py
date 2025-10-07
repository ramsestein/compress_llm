#!/usr/bin/env python3
"""
Script para probar quantization que SÍ reduce el tamaño
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from create_compress.compression_engine import CompressionEngine

def test_quantization():
    """Probar quantization que reduce el tamaño"""
    
    print("=" * 60)
    print("TESTING QUANTIZATION (REDUCE TAMAÑO)")
    print("=" * 60)
    
    # Cargar modelo original
    print("1. Cargando modelo original...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Obtener una capa específica para monitorear
    test_layer = model.transformer.h[0].mlp.c_fc
    original_weight = test_layer.weight.data.clone()
    
    print(f"   Peso original: {original_weight.shape}")
    print(f"   Dtype original: {original_weight.dtype}")
    print(f"   Elementos no cero: {torch.count_nonzero(original_weight).item()}")
    print(f"   Tamaño en bytes: {original_weight.numel() * original_weight.element_size()} bytes")
    
    # Crear motor de compresión
    print("\n2. Creando motor de compresión...")
    engine = CompressionEngine()
    
    # Aplicar quantization
    print("\n3. Aplicando quantization...")
    layer_config = {"methods": [{"name": "int8_quantization", "strength": 0.0}]}
    compressed_layer, result = engine.compress_layer(test_layer, layer_config)
    
    print(f"   Tipo comprimido: {type(compressed_layer)}")
    print(f"   Dtype comprimido: {compressed_layer.weight_int.dtype}")
    print(f"   Elementos no cero: {torch.count_nonzero(compressed_layer.weight_int).item()}")
    print(f"   Tamaño en bytes: {compressed_layer.weight_int.numel() * compressed_layer.weight_int.element_size()} bytes")
    print(f"   Ratio de compresión: {result.compression_ratio:.2%}")
    
    # Calcular reducción real
    original_size = original_weight.numel() * original_weight.element_size()
    compressed_size = compressed_layer.weight_int.numel() * compressed_layer.weight_int.element_size()
    real_reduction = 1 - (compressed_size / original_size)
    
    print(f"\n4. Reducción real:")
    print(f"   Tamaño original: {original_size} bytes")
    print(f"   Tamaño comprimido: {compressed_size} bytes")
    print(f"   Reducción real: {real_reduction:.2%}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("Quantization SÍ reduce el tamaño en bytes")
    print("Pruning solo reduce elementos no cero, no el tamaño")

if __name__ == "__main__":
    test_quantization()
