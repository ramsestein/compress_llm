#!/usr/bin/env python3
"""Script para testear la configuración de LoRA directamente."""

import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2Config


def test_lora_config():
    """Test simple para verificar la configuración de LoRA."""
    model_dir = Path("models/microsoft_DialoGPT-small_compressed")

    if not model_dir.exists():
        print("Not found el modelo comprimido")
        return

    print(" Testeando configuración de LoRA directamente...")

    try:
        # Load configuración
        config_path = model_dir / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)

        # Convertir diccionario a objeto de configuración
        config = GPT2Config(**config_dict)

        # Create modelo desde configuración
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_config(config)

        print("Modelo creado desde configuración")

        # Load parámetros guardados por componentes
        param_files = list(model_dir.glob("*.pt"))
        if not param_files:
            print("No se encontraron archivos de parámetros .pt")
            return

        print(f"Encontrados {len(param_files)} archivos de parámetros")

        # Create state_dict
        state_dict = {}
        for param_file in param_files:
            param_name = param_file.stem
            param_data = torch.load(param_file, map_location="cpu")
            state_dict[param_name] = param_data

        # Load parámetros en el modelo
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print(f"Parameters cargados: {len(state_dict)}")
        print(f"Claves faltantes: {len(missing_keys)}")
        print(f"Claves inesperadas: {len(unexpected_keys)}")

        # Detectar módulos objetivo
        target_modules = []
        for name, module in model.named_modules():
            if any(
                keyword in name
                for keyword in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
            ):
                target_modules.append(name)

        print(f"\nTarget modules detectados: {len(target_modules)}")
        if target_modules:
            print("  Primeros 5 módulos:")
            for i, module in enumerate(target_modules[:5]):
                print(f"    {i + 1}. {module}")

        # Create configuración LoRA
        print("\nCreando configuración LoRA...")
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            modules_to_save=None,
        )

        print("Configuración LoRA creada:")
        print(f"  - Rank (r): {peft_config.r}")
        print(f"  - Alpha: {peft_config.lora_alpha}")
        print(f"  - Dropout: {peft_config.lora_dropout}")
        print(f"  - Bias: {peft_config.bias}")
        print(f"  - Task type: {peft_config.task_type}")
        print(f"  - Target modules: {len(peft_config.target_modules)}")

        # Apply LoRA
        print("\nAplicando LoRA al modelo...")
        try:
            model = get_peft_model(model, peft_config)
            print("LoRA aplicado exitosamente!")

            # Imprimir parámetros entrenables
            model.print_trainable_parameters()

        except Exception as e:
            print(f"Error al aplicar LoRA: {e}")
            import traceback

            traceback.print_exc()

        print("\nTest completado!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_lora_config()
