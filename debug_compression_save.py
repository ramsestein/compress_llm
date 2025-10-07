#!/usr/bin/env python3
"""
Script para debuggear el problema del guardado de compresión
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from create_compress.compression_engine import CompressionEngine
import json
from pathlib import Path

def debug_compression_save():
    """Debug del proceso de compresión y guardado"""
    
    print("=" * 60)
    print("DEBUGGING COMPRESSION SAVE PROCESS")
    print("=" * 60)
    
    # Cargar modelo original
    print("1. Cargando modelo original...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    # Obtener una capa específica para monitorear
    test_layer = model.transformer.h[0].mlp.c_fc
    original_weight = test_layer.weight.data.clone()
    print(f"   Peso original shape: {original_weight.shape}")
    print(f"   Peso original stats: min={original_weight.min():.4f}, max={original_weight.max():.4f}")
    print(f"   Elementos no cero: {torch.count_nonzero(original_weight).item()}")
    
    # Cargar configuración
    print("\n2. Cargando configuración de compresión...")
    config_path = Path("compression_analysis/distilgpt2_compression_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"   Configuración cargada: {config['global_settings']['name']}")
    
    # Aplicar compresión
    print("\n3. Aplicando compresión...")
    engine = CompressionEngine()
    compressed_model = engine.compress_model(model, config)
    
    # Verificar si la compresión se aplicó
    compressed_weight = compressed_model.transformer.h[0].mlp.c_fc.weight.data
    print(f"   Peso comprimido shape: {compressed_weight.shape}")
    print(f"   Peso comprimido stats: min={compressed_weight.min():.4f}, max={compressed_weight.max():.4f}")
    print(f"   Elementos no cero: {torch.count_nonzero(compressed_weight).item()}")
    print(f"   ¿Son iguales?: {torch.equal(original_weight, compressed_weight)}")
    
    # Calcular compresión real
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    compressed_size = sum(p.numel() * p.element_size() for p in compressed_model.parameters())
    compression_ratio = 1 - (compressed_size / original_size)
    
    print(f"\n4. Estadísticas de compresión:")
    print(f"   Tamaño original: {original_size / 1024 / 1024:.2f} MB")
    print(f"   Tamaño comprimido: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"   Ratio de compresión: {compression_ratio:.2%}")
    
    # Guardar modelo comprimido
    print("\n5. Guardando modelo comprimido...")
    output_dir = Path("models/distilgpt2_debug_compressed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar con save_pretrained normal
    try:
        compressed_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("   ✅ Modelo guardado con save_pretrained")
    except Exception as e:
        print(f"   ❌ Error guardando con save_pretrained: {e}")
    
    # Verificar archivo guardado
    print("\n6. Verificando archivo guardado...")
    model_file = output_dir / "pytorch_model.bin"
    if model_file.exists():
        file_size = model_file.stat().st_size
        print(f"   Archivo pytorch_model.bin: {file_size / 1024 / 1024:.2f} MB")
        
        # Cargar modelo guardado y verificar
        loaded_model = AutoModelForCausalLM.from_pretrained(str(output_dir))
        loaded_weight = loaded_model.transformer.h[0].mlp.c_fc.weight.data
        print(f"   Peso cargado stats: min={loaded_weight.min():.4f}, max={loaded_weight.max():.4f}")
        print(f"   Elementos no cero: {torch.count_nonzero(loaded_weight).item()}")
        print(f"   ¿Igual al comprimido?: {torch.equal(compressed_weight, loaded_weight)}")
        print(f"   ¿Igual al original?: {torch.equal(original_weight, loaded_weight)}")
    else:
        print("   ❌ No se encontró pytorch_model.bin")
    
    # Probar con safetensors
    print("\n7. Probando con safetensors...")
    try:
        from safetensors.torch import save_file
        from transformers import PreTrainedModel
        
        # Convertir state_dict a formato safetensors
        state_dict = compressed_model.state_dict()
        safetensors_file = output_dir / "model.safetensors"
        save_file(state_dict, safetensors_file)
        
        file_size = safetensors_file.stat().st_size
        print(f"   Archivo model.safetensors: {file_size / 1024 / 1024:.2f} MB")
        
        # Cargar desde safetensors
        from safetensors.torch import load_file
        loaded_state_dict = load_file(safetensors_file)
        loaded_model_safe = AutoModelForCausalLM.from_pretrained("distilgpt2")
        loaded_model_safe.load_state_dict(loaded_state_dict)
        
        loaded_weight_safe = loaded_model_safe.transformer.h[0].mlp.c_fc.weight.data
        print(f"   Peso safetensors stats: min={loaded_weight_safe.min():.4f}, max={loaded_weight_safe.max():.4f}")
        print(f"   Elementos no cero: {torch.count_nonzero(loaded_weight_safe).item()}")
        print(f"   ¿Igual al comprimido?: {torch.equal(compressed_weight, loaded_weight_safe)}")
        print(f"   ¿Igual al original?: {torch.equal(original_weight, loaded_weight_safe)}")
        
    except Exception as e:
        print(f"   ❌ Error con safetensors: {e}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if torch.equal(original_weight, compressed_weight):
        print("❌ PROBLEMA: La compresión no se está aplicando en memoria")
    elif torch.equal(compressed_weight, loaded_weight):
        print("✅ La compresión se aplica y se guarda correctamente")
    else:
        print("❌ PROBLEMA: La compresión se aplica pero se pierde al guardar")

if __name__ == "__main__":
    debug_compression_save()


