#!/usr/bin/env python3
"""
Script para debuggear por qué no cambia el tamaño en bytes
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from create_compress.compression_engine import CompressionEngine

def debug_size_issue():
    """Debug específico del problema de tamaño"""
    
    print("=" * 60)
    print("DEBUGGING TAMAÑO EN BYTES")
    print("=" * 60)
    
    # Cargar modelo original
    print("1. Cargando modelo original...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Obtener una capa específica para monitorear
    test_layer = model.transformer.h[0].mlp.c_fc
    original_weight = test_layer.weight.data.clone()
    
    print(f"   Peso original: {original_weight.shape}")
    print(f"   Elementos no cero originales: {torch.count_nonzero(original_weight).item()}")
    print(f"   Tamaño en bytes: {original_weight.numel() * original_weight.element_size()} bytes")
    
    # Crear motor de compresión
    print("\n2. Creando motor de compresión...")
    engine = CompressionEngine()
    
    # Aplicar compresión
    print("\n3. Aplicando compresión...")
    layer_config = {"methods": [{"name": "magnitude_pruning", "strength": 0.3}]}
    compressed_layer, result = engine.compress_layer(test_layer, layer_config)
    
    print(f"   Elementos no cero comprimidos: {torch.count_nonzero(compressed_layer.weight).item()}")
    print(f"   Tamaño en bytes: {compressed_layer.weight.numel() * compressed_layer.weight.element_size()} bytes")
    print(f"   Ratio de compresión: {result.compression_ratio:.2%}")
    
    # Verificar si es el mismo tensor
    print(f"\n4. Verificaciones:")
    print(f"   ¿Es el mismo tensor?: {test_layer.weight is compressed_layer.weight}")
    print(f"   ¿Son iguales?: {torch.equal(test_layer.weight, compressed_layer.weight)}")
    print(f"   ¿Tienen la misma forma?: {test_layer.weight.shape == compressed_layer.weight.shape}")
    print(f"   ¿Tienen el mismo dtype?: {test_layer.weight.dtype == compressed_layer.weight.dtype}")
    
    # Verificar si es un PrunedLinear
    print(f"\n5. Tipo de módulo:")
    print(f"   Tipo original: {type(test_layer)}")
    print(f"   Tipo comprimido: {type(compressed_layer)}")
    print(f"   ¿Es PrunedLinear?: {type(compressed_layer).__name__ == 'PrunedLinear'}")
    
    # Verificar si PrunedLinear tiene el mismo tamaño
    if hasattr(compressed_layer, 'weight'):
        print(f"   Peso de PrunedLinear: {compressed_layer.weight.shape}")
        print(f"   Elementos de PrunedLinear: {compressed_layer.weight.numel()}")
        print(f"   Tamaño en bytes de PrunedLinear: {compressed_layer.weight.numel() * compressed_layer.weight.element_size()} bytes")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("Si el tamaño en bytes no cambia, puede ser porque:")
    print("1. PrunedLinear mantiene la misma forma que el original")
    print("2. Los elementos cero siguen ocupando espacio en memoria")
    print("3. El dtype no cambia (float32 sigue siendo float32)")
    print("4. La compresión es 'sparse' pero no 'quantized'")

if __name__ == "__main__":
    debug_size_issue()
