#!/usr/bin/env python3
"""
Script para debuggear por qué los métodos de compresión no funcionan
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from create_compress.compression_methods import QuantizationMethod, PruningMethod
import copy

def debug_methods_issue():
    """Debug específico de los métodos de compresión"""
    
    print("=" * 60)
    print("DEBUGGING COMPRESSION METHODS")
    print("=" * 60)
    
    # Cargar modelo original
    print("1. Cargando modelo original...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Obtener una capa específica para monitorear
    test_layer = model.transformer.h[0].mlp.c_fc
    original_weight = test_layer.weight.data.clone()
    print(f"   Peso original: {original_weight.shape}, elementos no cero: {torch.count_nonzero(original_weight).item()}")
    print(f"   Stats: min={original_weight.min():.4f}, max={original_weight.max():.4f}")
    
    # Probar cuantización
    print("\n2. Probando cuantización...")
    try:
        quantizer = QuantizationMethod(bits=8)
        config = {"strength": 0.5}
        
        print(f"   Config: {config}")
        print(f"   Antes: elementos no cero = {torch.count_nonzero(test_layer.weight).item()}")
        
        # Aplicar cuantización
        compressed_layer = quantizer.compress(test_layer, config, torch.device('cpu'))
        
        print(f"   Después: elementos no cero = {torch.count_nonzero(compressed_layer.weight).item()}")
        print(f"   ¿Son iguales?: {torch.equal(test_layer.weight, compressed_layer.weight)}")
        print(f"   ¿Es el mismo objeto?: {test_layer is compressed_layer}")
        
        if torch.equal(test_layer.weight, compressed_layer.weight):
            print("   PROBLEMA: La cuantización no cambió nada")
        else:
            print("   OK: La cuantización funciona")
            
    except Exception as e:
        print(f"   ERROR en cuantización: {e}")
        import traceback
        traceback.print_exc()
    
    # Probar pruning
    print("\n3. Probando pruning...")
    try:
        pruner = PruningMethod()
        config = {"strength": 0.3}
        
        print(f"   Config: {config}")
        print(f"   Antes: elementos no cero = {torch.count_nonzero(test_layer.weight).item()}")
        
        # Aplicar pruning
        compressed_layer = pruner.compress(test_layer, config, torch.device('cpu'))
        
        print(f"   Después: elementos no cero = {torch.count_nonzero(compressed_layer.weight).item()}")
        print(f"   ¿Son iguales?: {torch.equal(test_layer.weight, compressed_layer.weight)}")
        print(f"   ¿Es el mismo objeto?: {test_layer is compressed_layer}")
        
        if torch.equal(test_layer.weight, compressed_layer.weight):
            print("   PROBLEMA: El pruning no cambió nada")
        else:
            print("   OK: El pruning funciona")
            
    except Exception as e:
        print(f"   ERROR en pruning: {e}")
        import traceback
        traceback.print_exc()
    
    # Probar con copia manual
    print("\n4. Probando con copia manual...")
    try:
        # Crear copia manual
        test_layer_copy = copy.deepcopy(test_layer)
        original_weight_copy = test_layer_copy.weight.data.clone()
        
        print(f"   Copia original: elementos no cero = {torch.count_nonzero(original_weight_copy).item()}")
        
        # Aplicar pruning manual
        with torch.no_grad():
            weight_flat = test_layer_copy.weight.data.abs().flatten()
            threshold = torch.quantile(weight_flat, 0.3)
            mask = test_layer_copy.weight.data.abs() > threshold
            test_layer_copy.weight.data *= mask
        
        compressed_weight_copy = test_layer_copy.weight.data
        print(f"   Copia comprimida: elementos no cero = {torch.count_nonzero(compressed_weight_copy).item()}")
        print(f"   ¿Son diferentes?: {not torch.equal(original_weight_copy, compressed_weight_copy)}")
        
        if torch.equal(original_weight_copy, compressed_weight_copy):
            print("   PROBLEMA: La compresión manual no funciona")
        else:
            print("   OK: La compresión manual funciona")
            
    except Exception as e:
        print(f"   ERROR en compresión manual: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("Si los métodos de compresión no funcionan, el problema está en:")
    print("1. Los métodos no están modificando los pesos correctamente")
    print("2. Hay un problema con copy.deepcopy()")
    print("3. Los métodos están retornando el módulo original")

if __name__ == "__main__":
    debug_methods_issue()


