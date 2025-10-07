#!/usr/bin/env python3
"""
Script para debuggear la compresión
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from create_compress.compression_methods import QuantizationMethod, PruningMethod
import sys
from pathlib import Path

def test_compression_methods():
    """Prueba los métodos de compresión individualmente"""
    
    print("=" * 60)
    print("DEBUGGING COMPRESSION METHODS")
    print("=" * 60)
    
    # Cargar modelo
    print("Cargando modelo DistilGPT2...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Obtener una capa específica para probar
    layer = model.transformer.h[0].mlp.c_fc  # Primera capa FFN
    print(f"Capas en el modelo: {len(list(model.named_modules()))}")
    print(f"Tipo de capa: {type(layer)}")
    print(f"Peso original shape: {layer.weight.shape}")
    print(f"Peso original dtype: {layer.weight.dtype}")
    print(f"Peso original stats: min={layer.weight.min():.4f}, max={layer.weight.max():.4f}, mean={layer.weight.mean():.4f}")
    
    # Probar cuantización
    print("\n" + "=" * 40)
    print("PROBANDO CUANTIZACIÓN")
    print("=" * 40)
    
    quantizer = QuantizationMethod(bits=8)
    config = {"strength": 0.5}
    
    print("Antes de cuantización:")
    print(f"  Peso stats: min={layer.weight.min():.4f}, max={layer.weight.max():.4f}")
    
    # Aplicar cuantización
    compressed_layer = quantizer.compress(layer, config, torch.device('cpu'))
    
    print("Después de cuantización:")
    print(f"  Peso stats: min={compressed_layer.weight.min():.4f}, max={compressed_layer.weight.max():.4f}")
    print(f"  ¿Son iguales?: {torch.equal(layer.weight, compressed_layer.weight)}")
    
    # Probar pruning
    print("\n" + "=" * 40)
    print("PROBANDO PRUNING")
    print("=" * 40)
    
    pruner = PruningMethod()
    config = {"strength": 0.3}
    
    print("Antes de pruning:")
    print(f"  Peso stats: min={layer.weight.min():.4f}, max={layer.weight.max():.4f}")
    print(f"  Elementos no cero: {torch.count_nonzero(layer.weight).item()}")
    
    # Aplicar pruning
    compressed_layer = pruner.compress(layer, config, torch.device('cpu'))
    
    print("Después de pruning:")
    print(f"  Peso stats: min={compressed_layer.weight.min():.4f}, max={compressed_layer.weight.max():.4f}")
    print(f"  Elementos no cero: {torch.count_nonzero(compressed_layer.weight).item()}")
    print(f"  ¿Son iguales?: {torch.equal(layer.weight, compressed_layer.weight)}")
    
    print("\n" + "=" * 60)
    print("CONCLUSIÓN")
    print("=" * 60)
    
    if torch.equal(layer.weight, compressed_layer.weight):
        print("❌ PROBLEMA: Los métodos de compresión no están modificando los pesos")
    else:
        print("✅ Los métodos de compresión SÍ están funcionando")

if __name__ == "__main__":
    test_compression_methods()


