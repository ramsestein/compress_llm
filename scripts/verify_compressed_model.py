#!/usr/bin/env python3
"""Script para verificar modelos comprimidos guardados por componentes."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_model_from_components(model_dir: Path, device: str = "cpu"):
    """Load a model that was saved using component-based saving."""
    # Check if this was saved by components
    metadata_path = model_dir / "component_save_metadata.json"
    if not metadata_path.exists():
        raise ValueError("This directory does not contain a component-saved model")

    with open(metadata_path) as f:
        metadata = json.load(f)

    print(f"Cargando modelo guardado por componentes: {metadata['model_type']}")
    print(f"Total de parámetros: {metadata['total_parameters']}")

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("Not found config.json")

    try:
        config = AutoConfig.from_pretrained(str(model_dir))
        print("Config loaded")
    except Exception as e:
        print(f"Error cargando config: {e}")
        # Try to create a basic config
        config = AutoConfig.from_pretrained("microsoft/DialoGPT-small")
        print("Configuración básica creada")

    # Create model from config
    try:
        model = AutoModelForCausalLM.from_config(config)
        print("Modelo creado desde configuración")
    except Exception as e:
        print(f"Error creando modelo: {e}")
        raise

    # Load weights from individual parameter files
    state_dict = {}
    param_files = list(model_dir.glob("*.pt"))
    print(f"Encontrados {len(param_files)} archivos de parámetros")

    for param_file in param_files:
        try:
            # Extract parameter name from filename
            param_name = param_file.stem.replace("_", ".")

            # Load parameter
            param_tensor = torch.load(param_file, map_location=device)

            # Add to state dict
            state_dict[param_name] = param_tensor

        except Exception as e:
            print(f"Error cargando parámetro {param_file.name}: {e}")
            continue

    print(f"{len(state_dict)} parámetros cargados")

    # Load state dict into model
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Claves faltantes: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Claves inesperadas: {len(unexpected_keys)}")

        print("Estado del modelo cargado exitosamente")

    except Exception as e:
        print(f"Error cargando estado del modelo: {e}")
        raise

    # Move model to device
    model.to(device)
    model.eval()

    print(f"Modelo cargado exitosamente en {device}")
    return model


def get_model_info(model_path: Path) -> Dict[str, Any]:
    """Obtiene información detallada de un modelo."""
    info = {"path": str(model_path), "exists": model_path.exists()}

    if not info["exists"]:
        return info

    try:
        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            info["architecture"] = config.get("model_type", "unknown")
            info["hidden_size"] = config.get("hidden_size", 0)
            info["num_layers"] = config.get("num_hidden_layers", 0)

        # Size en disco
        total_size = 0
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        info["disk_size_mb"] = total_size / (1024 * 1024)

        # Contar archivos del modelo
        model_files = (
            list(model_path.glob("*.safetensors"))
            + list(model_path.glob("*.bin"))
            + list(model_path.glob("*.pt"))
        )
        info["model_files"] = len(model_files)

        # Verify metadata de compresión
        metadata_path = model_path / "compression_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                info["compression_metadata"] = json.load(f)

        return info

    except Exception as e:
        info["error"] = str(e)
        return info


def compare_outputs(
    original_model_path: Path, compressed_model_path: Path, prompt: str = "Hello, how are you?"
):
    """Compara las salidas de dos modelos con el mismo prompt."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n Comparando outputs...")
    print(f"Prompt: '{prompt}'")
    print(f" Device: {device}")

    outputs = {}

    # Load modelo original
    print(f"\n{'' * 50}")
    print(f"Cargando modelo original: {original_model_path.name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(original_model_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if device.type == "cuda":
            original_model = AutoModelForCausalLM.from_pretrained(
                str(original_model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            original_model = AutoModelForCausalLM.from_pretrained(
                str(original_model_path), torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            original_model = original_model.to(device)

        original_model.eval()

        # Generate texto
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            start_time = time.time()
            outputs_original = original_model.generate(
                inputs,
                max_length=len(inputs["input_ids"][0]) + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            generation_time = time.time() - start_time

        output_text = tokenizer.decode(outputs_original[0], skip_special_tokens=True)
        outputs["original"] = {
            "text": output_text,
            "time": generation_time,
            "length": len(outputs_original[0]),
        }

        print(f"Output original: {output_text}")
        print(f"⏱ Tiempo de generación: {generation_time:.2f}s")

        # Limpiar memoria
        del original_model
        gc.collect()

    except Exception as e:
        print(f"Error con modelo original: {e}")
        outputs["original"] = {"error": str(e)}

    # Load modelo comprimido
    print(f"\n{'' * 50}")
    print(f"Cargando modelo comprimido: {compressed_model_path.name}")

    try:
        compressed_model = load_model_from_components(compressed_model_path, str(device))

        # Generate texto
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            start_time = time.time()
            outputs_compressed = compressed_model.generate(
                inputs,
                max_length=len(inputs["input_ids"][0]) + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            generation_time = time.time() - start_time

        output_text = tokenizer.decode(outputs_compressed[0], skip_special_tokens=True)
        outputs["compressed"] = {
            "text": output_text,
            "time": generation_time,
            "length": len(outputs_compressed[0]),
        }

        print(f"Output comprimido: {output_text}")
        print(f"⏱ Tiempo de generación: {generation_time:.2f}s")

        # Limpiar memoria
        del compressed_model
        gc.collect()

    except Exception as e:
        print(f"Error con modelo comprimido: {e}")
        outputs["compressed"] = {"error": str(e)}

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Verifica modelos comprimidos guardados por componentes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Verify modelo comprimido
  python verify_compressed_model.py microsoft_DialoGPT-small

  # Especificar directorio de modelos
  python verify_compressed_model.py microsoft_DialoGPT-small --models-dir ./models
        """,
    )

    parser.add_argument("model", type=str, help="Name del modelo a verificar")

    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directorio de modelos (default: ./models)",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help='Prompt para comparar outputs (default: "Hello, how are you?")',
    )

    args = parser.parse_args()

    # Construir rutas
    models_dir = Path(args.models_dir)
    original_path = models_dir / args.model
    compressed_path = models_dir / f"{args.model}_compressed"

    print("=" * 60)
    print("VERIFICACIÓN DE MODELO COMPRIMIDO")
    print("=" * 60)

    # Verify que existen ambos modelos
    if not original_path.exists():
        print(f"Not found el modelo original: {original_path}")
        sys.exit(1)

    if not compressed_path.exists():
        print(f"Not found el modelo comprimido: {compressed_path}")
        sys.exit(1)

    # Analizar información de ambos modelos
    print("\nAnalizando modelos...")

    original_info = get_model_info(original_path)
    compressed_info = get_model_info(compressed_path)

    print(f"\nMODELO ORIGINAL: {args.model}")
    print(f"{'' * 50}")
    print(f"Ubicación: {original_info['path']}")
    print(f"   Arquitectura: {original_info.get('architecture', 'unknown')}")
    print(f"   Size en disco: {original_info.get('disk_size_mb', 0):.1f} MB")
    print(f"   Archivos del modelo: {original_info.get('model_files', 0)}")

    print(f"\nMODELO COMPRIMIDO: {args.model}_compressed")
    print(f"{'' * 50}")
    print(f"Ubicación: {compressed_info['path']}")
    print(f"   Arquitectura: {compressed_info.get('architecture', 'unknown')}")
    print(f"   Size en disco: {compressed_info.get('disk_size_mb', 0):.1f} MB")
    print(f"   Archivos del modelo: {compressed_info.get('model_files', 0)}")

    # Calculate estadísticas de compresión
    original_size = original_info.get("disk_size_mb", 0)
    compressed_size = compressed_info.get("disk_size_mb", 0)

    if original_size > 0 and compressed_size > 0:
        reduction = ((original_size - compressed_size) / original_size) * 100
        factor = original_size / compressed_size

        print("\nEstadísticas de compresión:")
        print("    Reducción lograda:")
        print(f"      Size original: {original_size:.1f} MB")
        print(f"      Size final: {compressed_size:.1f} MB")
        print(f"      Reducción: {reduction:.1f}%")
        print(f"      Factor: {factor:.1f}x más pequeño")

    # Comparar outputs
    outputs = compare_outputs(original_path, compressed_path, args.prompt)

    # Show resumen
    print(f"\n{'' * 50}")
    print("RESUMEN DE COMPARACIÓN")
    print(f"{'' * 50}")

    if "original" in outputs and "compressed" in outputs:
        if "error" not in outputs["original"] and "error" not in outputs["compressed"]:
            print("Ambos modelos generaron texto exitosamente")

            # Comparar tiempos
            original_time = outputs["original"]["time"]
            compressed_time = outputs["compressed"]["time"]
            time_ratio = compressed_time / original_time if original_time > 0 else 0

            print(f"⏱ Tiempo original: {original_time:.2f}s")
            print(f"⏱ Tiempo comprimido: {compressed_time:.2f}s")
            print(f"Ratio de tiempo: {time_ratio:.2f}x")

            if time_ratio > 1.5:
                print("El modelo comprimido es significativamente más lento")
            elif time_ratio < 0.8:
                print("El modelo comprimido es más rápido!")
            else:
                print("Rendimiento similar")
        else:
            print("Error en la generación de texto")
            if "error" in outputs["original"]:
                print(f"   Modelo original: {outputs['original']['error']}")
            if "error" in outputs["compressed"]:
                print(f"   Modelo comprimido: {outputs['compressed']['error']}")

    print("\nVerificación completada!")
    print("Next steps:")
    print(f"   1. Fine-tuning: python finetune_lora.py --model {args.model}_compressed")
    print(f"   2. Servidor Ollama: python ollama_compact_server.py --model {args.model}_compressed")


if __name__ == "__main__":
    main()
