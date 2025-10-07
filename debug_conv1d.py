#!/usr/bin/env python3
"""
Script para debuggear Conv1D
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

def debug_conv1d():
    """Debug específico de Conv1D"""
    
    print("=" * 60)
    print("DEBUGGING Conv1D")
    print("=" * 60)
    
    # Cargar modelo original
    print("1. Cargando modelo original...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Obtener una capa específica para monitorear
    test_layer = model.transformer.h[0].mlp.c_fc
    print(f"   Tipo de módulo: {type(test_layer)}")
    print(f"   ¿Es nn.Linear?: {isinstance(test_layer, nn.Linear)}")
    print(f"   ¿Es nn.Module?: {isinstance(test_layer, nn.Module)}")
    print(f"   ¿Tiene weight?: {hasattr(test_layer, 'weight')}")
    print(f"   ¿Tiene bias?: {hasattr(test_layer, 'bias')}")
    
    if hasattr(test_layer, 'weight'):
        print(f"   Weight shape: {test_layer.weight.shape}")
        print(f"   Weight dtype: {test_layer.weight.dtype}")
    
    if hasattr(test_layer, 'bias'):
        print(f"   Bias: {test_layer.bias}")
    
    # Verificar si es un módulo personalizado
    print(f"\n2. Información del módulo:")
    print(f"   MRO: {[cls.__name__ for cls in type(test_layer).__mro__]}")
    print(f"   __bases__: {[cls.__name__ for cls in type(test_layer).__bases__]}")
    
    # Verificar si tiene métodos de Linear
    print(f"\n3. Métodos del módulo:")
    print(f"   in_features: {getattr(test_layer, 'in_features', 'NO')}")
    print(f"   out_features: {getattr(test_layer, 'out_features', 'NO')}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("Conv1D no es nn.Linear, por lo que _apply_pruning no lo procesa")
    print("Necesitamos modificar _apply_pruning para manejar Conv1D")

if __name__ == "__main__":
    debug_conv1d()


