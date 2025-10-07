#!/usr/bin/env python3
"""
Script para debuggear específicamente el problema del guardado
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from create_compress.compression_engine import CompressionEngine
import json
from pathlib import Path

def debug_save_issue():
    """Debug específico del problema de guardado"""
    
    print("=" * 60)
    print("DEBUGGING SAVE ISSUE")
    print("=" * 60)
    
    # Cargar modelo original
    print("1. Cargando modelo original...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Obtener una capa específica para monitorear
    test_layer = model.transformer.h[0].mlp.c_fc
    original_weight = test_layer.weight.data.clone()
    print(f"   Peso original: {original_weight.shape}, elementos no cero: {torch.count_nonzero(original_weight).item()}")
    
    # Aplicar compresión manual simple
    print("\n2. Aplicando compresión manual...")
    with torch.no_grad():
        # Aplicar pruning simple (poner a cero el 30% de los pesos más pequeños)
        weight_flat = test_layer.weight.data.abs().flatten()
        threshold = torch.quantile(weight_flat, 0.3)
        mask = test_layer.weight.data.abs() > threshold
        test_layer.weight.data *= mask
    
    compressed_weight = test_layer.weight.data
    print(f"   Peso comprimido: {compressed_weight.shape}, elementos no cero: {torch.count_nonzero(compressed_weight).item()}")
    print(f"   ¿Son diferentes?: {not torch.equal(original_weight, compressed_weight)}")
    
    # Guardar modelo modificado
    print("\n3. Guardando modelo modificado...")
    output_dir = Path("models/distilgpt2_manual_compressed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        model.save_pretrained(output_dir)
        print("   OK: Modelo guardado con save_pretrained")
    except Exception as e:
        print(f"   ERROR: {e}")
        return
    
    # Cargar modelo guardado
    print("\n4. Cargando modelo guardado...")
    try:
        loaded_model = AutoModelForCausalLM.from_pretrained(str(output_dir))
        loaded_weight = loaded_model.transformer.h[0].mlp.c_fc.weight.data
        print(f"   Peso cargado: {loaded_weight.shape}, elementos no cero: {torch.count_nonzero(loaded_weight).item()}")
        print(f"   ¿Igual al comprimido?: {torch.equal(compressed_weight, loaded_weight)}")
        print(f"   ¿Igual al original?: {torch.equal(original_weight, loaded_weight)}")
        
        if torch.equal(compressed_weight, loaded_weight):
            print("   RESULTADO: La compresión se guardó correctamente")
        elif torch.equal(original_weight, loaded_weight):
            print("   PROBLEMA: La compresión se perdió al guardar")
        else:
            print("   PROBLEMA: Algo raro pasó")
            
    except Exception as e:
        print(f"   ERROR cargando: {e}")
    
    # Probar con safetensors directamente
    print("\n5. Probando con safetensors directamente...")
    try:
        from safetensors.torch import save_file, load_file
        
        # Guardar state_dict con safetensors
        state_dict = model.state_dict()
        safetensors_file = output_dir / "model.safetensors"
        save_file(state_dict, safetensors_file)
        print(f"   OK: State_dict guardado con safetensors")
        
        # Cargar desde safetensors
        loaded_state_dict = load_file(safetensors_file)
        loaded_model_safe = AutoModelForCausalLM.from_pretrained("distilgpt2")
        loaded_model_safe.load_state_dict(loaded_state_dict)
        
        loaded_weight_safe = loaded_model_safe.transformer.h[0].mlp.c_fc.weight.data
        print(f"   Peso safetensors: {loaded_weight_safe.shape}, elementos no cero: {torch.count_nonzero(loaded_weight_safe).item()}")
        print(f"   ¿Igual al comprimido?: {torch.equal(compressed_weight, loaded_weight_safe)}")
        print(f"   ¿Igual al original?: {torch.equal(original_weight, loaded_weight_safe)}")
        
        if torch.equal(compressed_weight, loaded_weight_safe):
            print("   RESULTADO: Safetensors funciona correctamente")
        else:
            print("   PROBLEMA: Safetensors también pierde la compresión")
            
    except Exception as e:
        print(f"   ERROR con safetensors: {e}")
    
    # Verificar archivos
    print("\n6. Verificando archivos guardados...")
    pytorch_file = output_dir / "pytorch_model.bin"
    safetensors_file = output_dir / "model.safetensors"
    
    if pytorch_file.exists():
        size_pytorch = pytorch_file.stat().st_size
        print(f"   pytorch_model.bin: {size_pytorch / 1024 / 1024:.2f} MB")
    
    if safetensors_file.exists():
        size_safetensors = safetensors_file.stat().st_size
        print(f"   model.safetensors: {size_safetensors / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("Si la compresión se pierde al guardar, el problema está en:")
    print("1. save_pretrained() no preserva modificaciones in-place")
    print("2. Necesitamos guardar el state_dict modificado directamente")

if __name__ == "__main__":
    debug_save_issue()


