#!/usr/bin/env python3
"""Script para testear qué módulos están disponibles en el modelo comprimido."""

import json
from pathlib import Path

import torch
from transformers import GPT2Config


def test_model_modules():
    """Test simple para ver qué módulos están disponibles."""
    model_dir = Path("models/microsoft_DialoGPT-small_compressed")

    if not model_dir.exists():
        print("Not found el modelo comprimido")
        return

    print(" Testeando módulos del modelo comprimido...")

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

        # Listar todos los módulos disponibles
        print("\nMódulos disponibles en el modelo:")
        all_modules = []
        for name, module in model.named_modules():
            all_modules.append(name)
            if len(all_modules) <= 20:  # Solo mostrar los primeros 20
                print(f"  - {name}: {type(module).__name__}")

        if len(all_modules) > 20:
            print(f"  ... y {len(all_modules) - 20} más")

        # Buscar módulos específicos para LoRA
        print("\nMódulos candidatos para LoRA:")
        lora_candidates = []
        for name in all_modules:
            if any(
                keyword in name
                for keyword in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
            ):
                lora_candidates.append(name)
                print(f"  {name}")

        if not lora_candidates:
            print("  No se encontraron módulos candidatos para LoRA")
            print("  Buscando módulos alternativos...")

            # Buscar módulos que contengan 'weight' en el state_dict
            weight_modules = []
            for name in state_dict:
                if "weight" in name and not name.endswith("_bias"):
                    weight_modules.append(name)
                    if len(weight_modules) <= 10:
                        print(f"    {name}")

            if len(weight_modules) > 10:
                print(f"    ... y {len(weight_modules) - 10} más")

        print("\nTest completado!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_model_modules()
