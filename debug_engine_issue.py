#!/usr/bin/env python3
"""
Script para debuggear por qué el motor de compresión no funciona
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from create_compress.compression_engine import CompressionEngine
import json
from pathlib import Path

def debug_engine_issue():
    """Debug específico del motor de compresión"""
    
    print("=" * 60)
    print("DEBUGGING COMPRESSION ENGINE")
    print("=" * 60)
    
    # Cargar modelo original
    print("1. Cargando modelo original...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Obtener una capa específica para monitorear
    test_layer = model.transformer.h[0].mlp.c_fc
    original_weight = test_layer.weight.data.clone()
    print(f"   Peso original: {original_weight.shape}, elementos no cero: {torch.count_nonzero(original_weight).item()}")
    
    # Cargar configuración
    print("\n2. Cargando configuración...")
    config_path = Path("compression_analysis/distilgpt2_compression_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    layer_configs = config.get('layer_configs', {})
    print(f"   Configuraciones disponibles: {list(layer_configs.keys())}")
    
    # Crear motor de compresión
    print("\n3. Creando motor de compresión...")
    engine = CompressionEngine()
    
    # Probar detección de tipo de capa
    print("\n4. Probando detección de tipo de capa...")
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and 'mlp.c_fc' in name:
            layer_type = engine._get_layer_type(name, module)
            print(f"   {name} -> {layer_type}")
            print(f"   ¿Está en config?: {layer_type in layer_configs}")
            if layer_type in layer_configs:
                print(f"   Config: {layer_configs[layer_type]}")
            break
    
    # Probar compresión de una capa individual
    print("\n5. Probando compresión de capa individual...")
    try:
        layer_config = layer_configs['ffn']
        print(f"   Config para ffn: {layer_config}")
        
        compressed_module, result = engine.compress_layer(test_layer, layer_config)
        print(f"   Resultado: {result}")
        
        compressed_weight = compressed_module.weight.data
        print(f"   Peso después de compresión: {compressed_weight.shape}")
        print(f"   Elementos no cero: {torch.count_nonzero(compressed_weight).item()}")
        print(f"   ¿Son diferentes?: {not torch.equal(original_weight, compressed_weight)}")
        
        if torch.equal(original_weight, compressed_weight):
            print("   PROBLEMA: La compresión de capa individual no funciona")
        else:
            print("   OK: La compresión de capa individual funciona")
            
    except Exception as e:
        print(f"   ERROR en compresión de capa: {e}")
        import traceback
        traceback.print_exc()
    
    # Probar compresión completa del modelo
    print("\n6. Probando compresión completa del modelo...")
    try:
        compressed_model = engine.compress_model(model, config)
        
        compressed_weight = compressed_model.transformer.h[0].mlp.c_fc.weight.data
        print(f"   Peso después de compresión completa: {compressed_weight.shape}")
        print(f"   Elementos no cero: {torch.count_nonzero(compressed_weight).item()}")
        print(f"   ¿Son diferentes?: {not torch.equal(original_weight, compressed_weight)}")
        
        if torch.equal(original_weight, compressed_weight):
            print("   PROBLEMA: La compresión completa no funciona")
        else:
            print("   OK: La compresión completa funciona")
            
    except Exception as e:
        print(f"   ERROR en compresión completa: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("Si la compresión de capa individual no funciona, el problema está en:")
    print("1. Los métodos de compresión no están aplicando cambios")
    print("2. La configuración no es correcta")
    print("3. El motor de compresión tiene un bug")

if __name__ == "__main__":
    debug_engine_issue()


