#!/usr/bin/env python3
"""
Script para testear la detección de módulos LoRA
"""
import torch
from pathlib import Path
import json
from transformers import GPT2Config

def test_lora_detection():
    """Test simple para verificar la detección de módulos LoRA"""
    
    model_dir = Path("models/microsoft_DialoGPT-small_compressed")
    
    if not model_dir.exists():
        print("❌ No se encontró el modelo comprimido")
        return
    
    print("🧪 Testeando detección de módulos LoRA...")
    
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
        
        # Simular la detección de módulos como en el trainer
        target_modules = []
        
        # Para modelos GPT-2, los módulos típicos son:
        for name, module in model.named_modules():
            if any(keyword in name for keyword in ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']):
                target_modules.append(name)
        
        print(f"🔍 Módulos detectados automáticamente: {len(target_modules)}")
        if target_modules:
            print("  Primeros 5 módulos:")
            for i, module in enumerate(target_modules[:5]):
                print(f"    {i+1}. {module}")
        
        # Si no se encontraron, usar nombres más genéricos
        if not target_modules:
            target_modules = ['c_attn', 'c_proj', 'c_fc']
            print("  Usando nombres genéricos")
        
        # Si aún no hay módulos, usar una lista hardcodeada
        if not target_modules:
            target_modules = [
                'transformer.h.0.attn.c_attn',
                'transformer.h.0.attn.c_proj',
                'transformer.h.0.mlp.c_fc',
                'transformer.h.0.mlp.c_proj'
            ]
            print("  Usando módulos hardcodeados")
        
        # Si aún no hay módulos, usar una lista más completa
        if not target_modules:
            target_modules = []
            for i in range(12):  # GPT-2 small tiene 12 capas
                target_modules.extend([
                    f'transformer.h.{i}.attn.c_attn',
                    f'transformer.h.{i}.attn.c_proj',
                    f'transformer.h.{i}.mlp.c_fc',
                    f'transformer.h.{i}.mlp.c_proj'
                ])
            print(f"  Usando módulos completos: {len(target_modules)} módulos")
        
        print(f"\n🎯 Módulos objetivo finales: {len(target_modules)}")
        if len(target_modules) <= 10:
            for i, module in enumerate(target_modules):
                print(f"  {i+1}. {module}")
        else:
            print("  Primeros 10 módulos:")
            for i, module in enumerate(target_modules[:10]):
                print(f"    {i+1}. {module}")
            print(f"  ... y {len(target_modules) - 10} más")
        
        # Verificar que los módulos existen en el modelo
        print("\n🔍 Verificando existencia de módulos:")
        existing_modules = []
        missing_modules = []
        
        for module_name in target_modules:
            try:
                # Intentar acceder al módulo
                module = model.get_submodule(module_name)
                existing_modules.append(module_name)
                print(f"  ✅ {module_name}: {type(module).__name__}")
            except Exception as e:
                missing_modules.append(module_name)
                print(f"  ❌ {module_name}: {e}")
        
        print(f"\n📊 Resumen:")
        print(f"  ✅ Módulos existentes: {len(existing_modules)}")
        print(f"  ❌ Módulos faltantes: {len(missing_modules)}")
        
        if existing_modules:
            print(f"\n🎉 ¡Éxito! Se encontraron {len(existing_modules)} módulos válidos para LoRA")
        else:
            print(f"\n💥 Error: No se encontraron módulos válidos para LoRA")
        
        print("\n✅ Test completado!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lora_detection()
