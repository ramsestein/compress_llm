#!/usr/bin/env python3
"""
Script para testear la configuración de LoRA directamente
"""
import torch
from pathlib import Path
import json
from transformers import GPT2Config
from peft import LoraConfig, get_peft_model, TaskType

def test_lora_config():
    """Test simple para verificar la configuración de LoRA"""
    
    model_dir = Path("models/microsoft_DialoGPT-small_compressed")
    
    if not model_dir.exists():
        print("❌ No se encontró el modelo comprimido")
        return
    
    print("🧪 Testeando configuración de LoRA directamente...")
    
    try:
        # Cargar configuración
        config_path = model_dir / 'config.json'
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convertir diccionario a objeto de configuración
        config = GPT2Config(**config_dict)
        
        # Crear modelo desde configuración
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_config(config)
        
        print("✅ Modelo creado desde configuración")
        
        # Cargar parámetros guardados por componentes
        param_files = list(model_dir.glob('*.pt'))
        if not param_files:
            print("❌ No se encontraron archivos de parámetros .pt")
            return
        
        print(f"🔍 Encontrados {len(param_files)} archivos de parámetros")
        
        # Crear state_dict
        state_dict = {}
        for param_file in param_files:
            param_name = param_file.stem
            param_data = torch.load(param_file, map_location='cpu')
            state_dict[param_name] = param_data
        
        # Cargar parámetros en el modelo
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"📊 Parámetros cargados: {len(state_dict)}")
        print(f"⚠️ Claves faltantes: {len(missing_keys)}")
        print(f"⚠️ Claves inesperadas: {len(unexpected_keys)}")
        
        # Detectar módulos objetivo
        target_modules = []
        for name, module in model.named_modules():
            if any(keyword in name for keyword in ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']):
                target_modules.append(name)
        
        print(f"\n🎯 Módulos objetivo detectados: {len(target_modules)}")
        if target_modules:
            print("  Primeros 5 módulos:")
            for i, module in enumerate(target_modules[:5]):
                print(f"    {i+1}. {module}")
        
        # Crear configuración LoRA
        print(f"\n🔧 Creando configuración LoRA...")
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            modules_to_save=None
        )
        
        print(f"✅ Configuración LoRA creada:")
        print(f"  - Rango (r): {peft_config.r}")
        print(f"  - Alpha: {peft_config.lora_alpha}")
        print(f"  - Dropout: {peft_config.lora_dropout}")
        print(f"  - Bias: {peft_config.bias}")
        print(f"  - Task type: {peft_config.task_type}")
        print(f"  - Target modules: {len(peft_config.target_modules)}")
        
        # Aplicar LoRA
        print(f"\n🚀 Aplicando LoRA al modelo...")
        try:
            model = get_peft_model(model, peft_config)
            print("✅ LoRA aplicado exitosamente!")
            
            # Imprimir parámetros entrenables
            model.print_trainable_parameters()
            
        except Exception as e:
            print(f"❌ Error al aplicar LoRA: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n✅ Test completado!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lora_config()
