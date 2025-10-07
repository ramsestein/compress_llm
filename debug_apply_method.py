#!/usr/bin/env python3
"""
Script para debuggear el método apply_method
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from create_compress.compression_engine import CompressionEngine
from create_compress.compression_methods import QuantizationMethod, PruningMethod

def debug_apply_method():
    """Debug específico del método apply_method"""
    
    print("=" * 60)
    print("DEBUGGING APPLY_METHOD")
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
    
    # Probar apply_method directamente
    print("\n3. Probando apply_method directamente...")
    try:
        result_module = engine.apply_method(test_layer, "magnitude_pruning", 0.3, {})
        
        print(f"   Resultado: {result_module}")
        print(f"   ¿Es el mismo objeto?: {test_layer is result_module}")
        print(f"   Elementos no cero: {torch.count_nonzero(result_module.weight).item()}")
        print(f"   ¿Son iguales?: {torch.equal(original_weight, result_module.weight)}")
        
        if torch.equal(original_weight, result_module.weight):
            print("   PROBLEMA: apply_method no está aplicando compresión")
        else:
            print("   OK: apply_method funciona")
            
    except Exception as e:
        print(f"   ERROR en apply_method: {e}")
        import traceback
        traceback.print_exc()
    
    # Probar métodos de compresión directamente
    print("\n4. Probando métodos de compresión directamente...")
    try:
        pruner = PruningMethod()
        config = {"strength": 0.3}
        
        print(f"   Antes: elementos no cero = {torch.count_nonzero(test_layer.weight).item()}")
        
        # Aplicar pruning directamente
        compressed_layer = pruner.compress(test_layer, config, torch.device('cpu'))
        
        print(f"   Después: elementos no cero = {torch.count_nonzero(compressed_layer.weight).item()}")
        print(f"   ¿Son iguales?: {torch.equal(test_layer.weight, compressed_layer.weight)}")
        print(f"   ¿Es el mismo objeto?: {test_layer is compressed_layer}")
        
        if torch.equal(test_layer.weight, compressed_layer.weight):
            print("   PROBLEMA: El método de compresión no está funcionando")
        else:
            print("   OK: El método de compresión funciona")
            
    except Exception as e:
        print(f"   ERROR en método de compresión: {e}")
        import traceback
        traceback.print_exc()
    
    # Probar _apply_pruning directamente
    print("\n5. Probando _apply_pruning directamente...")
    try:
        result_module = engine._apply_pruning(test_layer, 0.3, {})
        
        print(f"   Resultado: {result_module}")
        print(f"   ¿Es el mismo objeto?: {test_layer is result_module}")
        print(f"   Elementos no cero: {torch.count_nonzero(result_module.weight).item()}")
        print(f"   ¿Son iguales?: {torch.equal(original_weight, result_module.weight)}")
        
        if torch.equal(original_weight, result_module.weight):
            print("   PROBLEMA: _apply_pruning no está aplicando compresión")
        else:
            print("   OK: _apply_pruning funciona")
            
    except Exception as e:
        print(f"   ERROR en _apply_pruning: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("Si apply_method no funciona, el problema está en:")
    print("1. _apply_pruning no está aplicando compresión")
    print("2. Los métodos de compresión no están funcionando")
    print("3. Hay un problema con la configuración")

if __name__ == "__main__":
    debug_apply_method()


