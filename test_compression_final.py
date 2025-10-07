#!/usr/bin/env python3
"""
Test final de compresión
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from create_compress.compression_engine import CompressionEngine
import json
from pathlib import Path

def test_compression_final():
    """Test final de compresión"""
    
    print("=" * 60)
    print("TEST FINAL DE COMPRESION")
    print("=" * 60)
    
    # Cargar modelo original
    print("1. Cargando modelo original...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Obtener una capa específica para monitorear
    test_layer = model.transformer.h[0].mlp.c_fc
    original_weight = test_layer.weight.data.clone()
    print(f"   Peso original: {original_weight.shape}, elementos no cero: {torch.count_nonzero(original_weight).item()}")
    
    # Crear motor de compresión
    print("\n2. Creando motor de compresión...")
    engine = CompressionEngine()
    
    # Configuración simple
    config = {
        "layer_configs": {
            "ffn": {
                "methods": [{"name": "magnitude_pruning", "strength": 0.3}],
                "total_compression_ratio": 0.3
            }
        }
    }
    
    # Probar compresión de una capa individual
    print("\n3. Probando compresión de capa individual...")
    layer_config = config["layer_configs"]["ffn"]
    
    # Aplicar compresión manual para comparar
    print("   Aplicando compresión manual...")
    with torch.no_grad():
        weight_flat = test_layer.weight.data.abs().flatten()
        threshold = torch.quantile(weight_flat, 0.3)
        mask = test_layer.weight.data.abs() > threshold
        test_layer.weight.data *= mask
    
    manual_compressed = test_layer.weight.data.clone()
    print(f"   Manual: elementos no cero = {torch.count_nonzero(manual_compressed).item()}")
    
    # Restaurar peso original
    test_layer.weight.data = original_weight
    
    # Aplicar compresión con el motor
    print("   Aplicando compresión con motor...")
    compressed_module, result = engine.compress_layer(test_layer, layer_config)
    
    print(f"   Motor: elementos no cero = {torch.count_nonzero(compressed_module.weight).item()}")
    print(f"   Result: {result}")
    print(f"   ¿Son iguales?: {torch.equal(compressed_module.weight, manual_compressed)}")
    print(f"   ¿Es el mismo objeto?: {test_layer is compressed_module}")
    
    if torch.equal(compressed_module.weight, manual_compressed):
        print("   OK: El motor funciona igual que la compresión manual")
    else:
        print("   PROBLEMA: El motor no está aplicando la compresión correctamente")
    
    # Probar compresión completa
    print("\n4. Probando compresión completa...")
    compressed_model = engine.compress_model(model, config)
    
    final_weight = compressed_model.transformer.h[0].mlp.c_fc.weight.data
    print(f"   Final: elementos no cero = {torch.count_nonzero(final_weight).item()}")
    print(f"   ¿Igual al manual?: {torch.equal(final_weight, manual_compressed)}")
    print(f"   ¿Igual al original?: {torch.equal(final_weight, original_weight)}")
    
    if torch.equal(final_weight, manual_compressed):
        print("   OK: La compresión completa funciona")
    else:
        print("   PROBLEMA: La compresión completa no funciona")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    if torch.equal(final_weight, manual_compressed):
        print("SUCCESS: El sistema de compresión funciona correctamente")
    else:
        print("FAILURE: El sistema de compresión tiene problemas")

if __name__ == "__main__":
    test_compression_final()


