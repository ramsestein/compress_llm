#!/usr/bin/env python3
"""Script para cargar modelos comprimidos guardados por componentes."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_from_components(model_dir: Path, device: str = "cpu") -> PreTrainedModel:
    """Load a model that was saved using component-based saving."""
    # Check if this was saved by components
    metadata_path = model_dir / "component_save_metadata.json"
    if not metadata_path.exists():
        raise ValueError("This directory does not contain a component-saved model")

    with open(metadata_path) as f:
        metadata = json.load(f)

    logger.info(f"Cargando modelo guardado por componentes: {metadata['model_type']}")
    logger.info(f"Total de parameters: {metadata['total_parameters']}")

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("Not found config.json")

    try:
        config = AutoConfig.from_pretrained(str(model_dir))
        logger.info("Config loaded")
    except Exception as e:
        logger.warning(f"Error cargando config: {e}")
        # Try to create a basic config
        config = AutoConfig.from_pretrained("microsoft/DialoGPT-small")
        logger.info("Configuración básica creada")

    # Create model from config
    try:
        model = AutoModelForCausalLM.from_config(config)
        logger.info("Modelo creado desde configuración")
    except Exception as e:
        logger.error(f"Error creando modelo: {e}")
        raise

    # Load weights from individual parameter files
    state_dict = {}
    param_files = list(model_dir.glob("*.pt"))
    logger.info(f"Encontrados {len(param_files)} archivos de parameters")

    for param_file in param_files:
        try:
            # Extract parameter name from filename
            param_name = param_file.stem.replace("_", ".")

            # Load parameter
            param_tensor = torch.load(param_file, map_location=device)

            # Add to state dict
            state_dict[param_name] = param_tensor

            logger.debug(f"Parámetro cargado: {param_name}")

        except Exception as e:
            logger.warning(f"Error cargando parámetro {param_file.name}: {e}")
            continue

    logger.info(f"{len(state_dict)} parameters cargados")

    # Load state dict into model
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Claves faltantes: {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"Claves inesperadas: {len(unexpected_keys)}")

        logger.info("Estado del modelo cargado exitosamente")

    except Exception as e:
        logger.error(f"Error cargando estado del modelo: {e}")
        raise

    # Move model to device
    model.to(device)
    model.eval()

    logger.info(f"Modelo cargado exitosamente en {device}")
    return model


def load_tokenizer(model_dir: Path) -> Optional[AutoTokenizer]:
    """Load tokenizer if available."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        logger.info("Tokenizer cargado")
        return tokenizer
    except Exception as e:
        logger.warning(f"Tokenizer not found: {e}")
        return None


def test_model_generation(
    model: PreTrainedModel,
    tokenizer: Optional[AutoTokenizer],
    prompt: str = "Hello, how are you?",
    max_length: int = 50,
) -> str:
    """Test the loaded model with a simple generation."""
    if tokenizer is None:
        return "No se puede probar sin tokenizer"

    try:
        # Encode input
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info("Generación exitosa")
        return generated_text

    except Exception as e:
        logger.error(f"Error en generación: {e}")
        return f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Carga un modelo comprimido guardado por componentes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Load modelo comprimido
  python load_compressed_model.py microsoft_DialoGPT-small_compressed

  # Especificar dispositivo
  python load_compressed_model.py microsoft_DialoGPT-small_compressed --device cuda

  # Probar generación
  python load_compressed_model.py microsoft_DialoGPT-small_compressed --test
        """,
    )

    parser.add_argument("model", type=str, help="Name del modelo comprimido a cargar")

    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directorio de modelos (default: ./models)",
    )

    parser.add_argument(
        "--device", type=str, default="cpu", help="Dispositivo para cargar el modelo (default: cpu)"
    )

    parser.add_argument(
        "--test", action="store_true", help="Test the model con generación de texto"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help='Prompt para probar el modelo (default: "Hello, how are you?")',
    )

    args = parser.parse_args()

    # Construir ruta del modelo
    models_dir = Path(args.models_dir)
    model_path = models_dir / args.model

    if not model_path.exists():
        logger.error(f"Not found el modelo: {model_path}")
        sys.exit(1)

    try:
        # Load modelo
        logger.info(f" Cargando modelo desde: {model_path}")
        model = load_model_from_components(model_path, args.device)

        # Load tokenizer
        tokenizer = load_tokenizer(model_path)

        # Show información del modelo
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total de parameters: {total_params:,}")
        logger.info(f"Size del modelo: {total_params * 4 / (1024 * 1024):.1f} MB")

        # Probar generación si se solicita
        if args.test:
            if tokenizer is None:
                logger.error("No se puede probar sin tokenizer")
                sys.exit(1)

            logger.info(f" Probando generación con prompt: '{args.prompt}'")
            result = test_model_generation(model, tokenizer, args.prompt)
            logger.info(f"Resultado: {result}")

        logger.info("Modelo cargado exitosamente")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
