#!/usr/bin/env python3
"""
Script para debuggear los nombres de capas
"""
import torch
from transformers import AutoModelForCausalLM
import json
from pathlib import Path

def debug_layer_names():
    """Debug de nombres de capas vs configuración"""
    
    print("=" * 60)
    print("DEBUGGING LAYER NAMES")
    print("=" * 60)
    
    # Cargar modelo
    print("1. Cargando modelo...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Listar todas las capas
    print("\n2. Nombres de capas en el modelo:")
    layer_names = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            layer_names.append(name)
            print(f"   {name}: {type(module).__name__}")
    
    print(f"\n   Total de capas con peso: {len(layer_names)}")
    
    # Cargar configuración
    print("\n3. Cargando configuración...")
    config_path = Path("compression_analysis/distilgpt2_compression_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    layer_configs = config.get('layer_configs', {})
    print(f"   Capas en configuración: {len(layer_configs)}")
    for name, layer_config in layer_configs.items():
        print(f"   {name}: {layer_config}")
    
    # Verificar coincidencias
    print("\n4. Verificando coincidencias:")
    matches = 0
    for name in layer_names:
        if name in layer_configs:
            print(f"   ✅ {name} - ENCONTRADA")
            matches += 1
        else:
            print(f"   ❌ {name} - NO ENCONTRADA")
    
    print(f"\n   Coincidencias: {matches}/{len(layer_names)}")
    
    # Verificar si hay capas que coincidan parcialmente
    print("\n5. Verificando coincidencias parciales:")
    for name in layer_names:
        for config_name in layer_configs.keys():
            if config_name in name or name in config_name:
                print(f"   🔍 {name} <-> {config_name}")

if __name__ == "__main__":
    debug_layer_names()


