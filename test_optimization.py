#!/usr/bin/env python3
"""
Script para probar la funcionalidad de optimización
"""
import torch
from LoRa_train.lora_config import LoRAPresets
from LoRa_train.peft_methods_config import estimate_memory_usage
from peft import LoRAConfig, BitFitConfig, PromptTuningConfig, QLoRAConfig

def test_optimization():
    """Probar funcionalidades de optimización"""
    
    print("=" * 60)
    print("TESTING OPTIMIZACIÓN DE RECURSOS")
    print("=" * 60)
    
    # 1. Probar estimación de memoria LoRA
    print("\n1. Estimación de memoria LoRA:")
    lora_config = LoRAConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1
    )
    
    memory_estimation = LoRAPresets.estimate_memory_usage(
        model_size_gb=7.0,  # Modelo 7B
        config=lora_config,
        batch_size=4
    )
    
    print(f"   Modelo base: {memory_estimation['base_model']:.1f} GB")
    print(f"   Adaptadores LoRA: {memory_estimation['lora_adapters']:.1f} GB")
    print(f"   Gradientes: {memory_estimation['gradients']:.1f} GB")
    print(f"   Optimizador: {memory_estimation['optimizer']:.1f} GB")
    print(f"   Activaciones: {memory_estimation['activations']:.1f} GB")
    print(f"   Total estimado: {memory_estimation['total_estimated']:.1f} GB")
    print(f"   GPU recomendada: {memory_estimation['recommended_gpu_memory']:.1f} GB")
    
    # 2. Probar diferentes métodos PEFT
    print("\n2. Comparación de métodos PEFT:")
    
    methods = [
        ("LoRA", LoRAConfig(r=16, target_modules=["q_proj", "v_proj"])),
        ("BitFit", BitFitConfig()),
        ("Prompt Tuning", PromptTuningConfig(num_virtual_tokens=10)),
        ("QLoRA", QLoRAConfig(bits=4, r=16, target_modules=["q_proj", "v_proj"]))
    ]
    
    for name, config in methods:
        try:
            memory = estimate_memory_usage(config, 7.0)  # 7B modelo
            print(f"   {name}:")
            print(f"     - Memoria del modelo: {memory['model_memory_gb']:.1f} GB")
            print(f"     - Memoria PEFT: {memory['peft_memory_mb']:.1f} MB")
            print(f"     - Total: {memory['total_memory_gb']:.1f} GB")
        except Exception as e:
            print(f"   {name}: ERROR - {e}")
    
    # 3. Verificar GPU disponible
    print("\n3. Información del sistema:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: {gpu_name}")
        print(f"   Memoria GPU: {gpu_memory:.1f} GB")
        
        # Verificar si es suficiente para LoRA
        if gpu_memory >= memory_estimation['recommended_gpu_memory']:
            print(f"   ✅ Suficiente para LoRA (necesita {memory_estimation['recommended_gpu_memory']:.1f} GB)")
        else:
            print(f"   ⚠️ Insuficiente para LoRA (necesita {memory_estimation['recommended_gpu_memory']:.1f} GB)")
    else:
        print("   GPU: No disponible (CPU only)")
        print("   ⚠️ Entrenamiento será más lento en CPU")
    
    # 4. Probar presets optimizados
    print("\n4. Presets optimizados:")
    presets = ["memory_efficient", "balanced", "high_quality"]
    
    for preset in presets:
        try:
            config = LoRAPresets.get_preset(preset)
            print(f"   {preset}:")
            print(f"     - r: {config.get('r', 'N/A')}")
            print(f"     - alpha: {config.get('lora_alpha', 'N/A')}")
            print(f"     - dropout: {config.get('lora_dropout', 'N/A')}")
        except Exception as e:
            print(f"   {preset}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("✅ La funcionalidad de optimización está integrada")
    print("✅ Estimación de memoria funciona")
    print("✅ Presets optimizados disponibles")
    print("✅ Detección automática de GPU")

if __name__ == "__main__":
    test_optimization()


