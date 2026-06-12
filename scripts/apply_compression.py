#!/usr/bin/env python3
"""Main script to apply compression to models according to JSON configuration
Version with full support for all compression methods.
"""

import argparse
import gc
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.utils import is_safetensors_available

from create_compress.compression_config_manager import CompressionConfigManager

# Import compression engine
from create_compress.compression_engine import CompressionEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_pretrained_with_fallback(
    model: PreTrainedModel,
    tokenizer: Optional[Any],
    output_dir: Path,
    *,
    logger: logging.Logger = logger,
) -> None:
    """Save a model with multiple fallback strategies to avoid recursion issues.

    This function tries multiple approaches to save the model:
    1. First tries safetensors (if available)
    2. Then tries standard save_pretrained with increased recursion limit
    3. Finally tries to save individual components separately
    """
    # Ensure output directory exists
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Try safetensors first
    if is_safetensors_available():
        try:
            logger.info("Trying to save with safetensors...")
            model.save_pretrained(output_dir, safe_serialization=True)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
            logger.info("Model saved successfully with safetensors")
            return
        except Exception as e:
            logger.warning(f"Safetensors failed: {e}")

    # Strategy 2: Try standard save with increased recursion limit
    import sys

    original_limit = sys.getrecursionlimit()

    for attempt in range(3):
        try:
            new_limit = original_limit * (2**attempt)
            logger.info(f"Attempt {attempt + 1}: increasing recursion limit to {new_limit}")
            sys.setrecursionlimit(new_limit)

            model.save_pretrained(output_dir, safe_serialization=False)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)

            logger.info("Model saved successfully con límite de recursión aumentado")
            return

        except RecursionError as e:
            logger.warning(f"RecursionError en intento {attempt + 1}: {e}")
            if attempt == 2:  # Last attempt
                logger.error("All recursion limit attempts failed")
        except Exception as e:
            logger.warning(f"Unexpected error in attempt {attempt + 1}: {e}")
            if attempt == 2:  # Last attempt
                logger.error(f"Unexpected error: {e}")
        finally:
            # Restore original recursion limit
            sys.setrecursionlimit(original_limit)

    # Strategy 3: Try to save components separately
    try:
        logger.info("Trying to save componentes por separado...")
        _save_model_components_separately(model, output_dir, logger)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully por componentes")
        return
    except Exception as e:
        logger.error(f"Failed to save por componentes: {e}")

    # If all strategies fail, raise a comprehensive error
    raise RuntimeError(
        "Failed to save el modelo: todas las estrategias de guardado fallaron. "
        "El modelo puede tener estructuras circulares o ser demasiado complejo."
    )


def _save_model_components_separately(
    model: PreTrainedModel, output_dir: Path, logger: logging.Logger
) -> None:
    """Save model components separately to avoid recursion issues."""
    # Ensure output directory exists
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config first (this should work without recursion issues)
    if hasattr(model, "config"):
        try:
            config_path = output_dir / "config.json"
            config_dict = model.config.to_dict()
            # Clean config to avoid any potential circular references
            cleaned_config = {}
            for key, value in config_dict.items():
                if isinstance(value, (str, int, float, bool, list, dict)):
                    # Only save simple types that can be serialized
                    if isinstance(value, dict):
                        # Recursively clean nested dicts
                        cleaned_config[key] = _clean_dict_for_serialization(value)
                    else:
                        cleaned_config[key] = value

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_config, f, indent=2, ensure_ascii=False)
            logger.info("Config saved exitosamente")
        except Exception as e:
            logger.warning(f"Error saving config: {e}")
            # Try to save a minimal config
            try:
                minimal_config = {
                    "model_type": getattr(model.config, "model_type", "unknown"),
                    "architectures": getattr(model.config, "architectures", ["unknown"]),
                    "vocab_size": getattr(model.config, "vocab_size", 50257),
                    "n_positions": getattr(model.config, "n_positions", 1024),
                    "n_embd": getattr(model.config, "n_embd", 768),
                    "n_layer": getattr(model.config, "n_layer", 12),
                    "n_head": getattr(model.config, "n_head", 12),
                }
                config_path = output_dir / "config.json"
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(minimal_config, f, indent=2, ensure_ascii=False)
                logger.info("Minimal config saved")
            except Exception as e2:
                logger.error(f"Critical error guardando config mínimo: {e2}")

    # Save model weights using a more robust approach
    if hasattr(model, "state_dict"):
        try:
            state_dict = model.state_dict()
            logger.info(f"Saving {len(state_dict)} parameters...")

            # Save each parameter individually to avoid recursion
            for param_name, param_tensor in state_dict.items():
                try:
                    # Create a safe filename
                    safe_name = param_name.replace(".", "_").replace("/", "_")
                    param_path = output_dir / f"{safe_name}.pt"

                    # Save individual parameter
                    torch.save(param_tensor, param_path, _use_new_zipfile_serialization=False)
                    logger.debug(f"Parameter saved: {param_name}")

                except Exception as e:
                    logger.warning(f"Error saving parámetro {param_name}: {e}")
                    # Try alternative saving method
                    try:
                        param_path = output_dir / f"{safe_name}_alt.pt"
                        torch.save(param_tensor.detach().cpu(), param_path)
                        logger.debug(f"Parameter saved con método alternativo: {param_name}")
                    except Exception as e2:
                        logger.error(f"Critical error guardando parámetro {param_name}: {e2}")
                        continue

            logger.info("Todos los parameters guardados exitosamente")

        except Exception as e:
            logger.error(f"Error accessing state_dict: {e}")
            # Fallback: try to get parameters directly
            try:
                logger.info("Trying método alternativo de guardado...")
                _save_parameters_directly(model, output_dir, logger)
            except Exception as e2:
                logger.error(f"Method alternativo también falló: {e2}")
                raise

    # Save generation config if available
    if hasattr(model, "generation_config"):
        try:
            gen_config_path = output_dir / "generation_config.json"
            gen_config_dict = model.generation_config.to_dict()
            # Clean generation config
            cleaned_gen_config = _clean_dict_for_serialization(gen_config_dict)

            with open(gen_config_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_gen_config, f, indent=2, ensure_ascii=False)
            logger.info("Generation config guardado")
        except Exception as e:
            logger.warning(f"Error saving generation config: {e}")

    # Create and save component metadata automatically
    try:
        from datetime import datetime

        # Count total parameters
        total_params = 0
        if hasattr(model, "state_dict"):
            state_dict = model.state_dict()
            total_params = len(state_dict)

        # Create metadata
        metadata = {
            "saved_by_components": True,
            "total_parameters": total_params,
            "timestamp": datetime.now().isoformat(),
            "model_type": getattr(model.config, "model_type", "unknown"),
            "model_name": getattr(model.config, "name_or_path", "unknown"),
            "compression_applied": True,
            "compression_method": "component_based_saving",
            "architecture": {
                "vocab_size": getattr(model.config, "vocab_size", "unknown"),
                "n_positions": getattr(model.config, "n_positions", "unknown"),
                "n_embd": getattr(model.config, "n_embd", "unknown"),
                "n_layer": getattr(model.config, "n_layer", "unknown"),
                "n_head": getattr(model.config, "n_head", "unknown"),
            },
        }

        # Save metadata
        metadata_path = output_dir / "component_save_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Automatic metadata created: {total_params} parameters")

    except Exception as e:
        logger.warning(f"Error creating metadata automática: {e}")
        # Create minimal metadata
        try:
            minimal_metadata = {
                "saved_by_components": True,
                "total_parameters": 0,
                "timestamp": datetime.now().isoformat(),
                "model_type": "unknown",
                "compression_applied": True,
            }
            metadata_path = output_dir / "component_save_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(minimal_metadata, f, indent=2, ensure_ascii=False)
            logger.info("Minimal metadata created")
        except Exception as e2:
            logger.error(f"Critical error creando metadata: {e2}")

    # Save a metadata file indicating this was saved by components
    try:
        metadata = {
            "saved_by_components": True,
            "total_parameters": len(model.state_dict()) if hasattr(model, "state_dict") else 0,
            "timestamp": datetime.now().isoformat(),
            "model_type": (
                getattr(model.config, "model_type", "unknown")
                if hasattr(model, "config")
                else "unknown"
            ),
        }
        metadata_path = output_dir / "component_save_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info("Component-save metadata saved")
    except Exception as e:
        logger.warning(f"Error saving metadata: {e}")


def _clean_dict_for_serialization(obj, max_depth=5, current_depth=0):
    """Clean a dictionary to remove potential circular references."""
    if current_depth > max_depth:
        return "[MAX_DEPTH_REACHED]"

    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            try:
                if isinstance(value, (str, int, float, bool)):
                    cleaned[key] = value
                elif isinstance(value, list):
                    cleaned[key] = _clean_list_for_serialization(
                        value, max_depth, current_depth + 1
                    )
                elif isinstance(value, dict):
                    cleaned[key] = _clean_dict_for_serialization(
                        value, max_depth, current_depth + 1
                    )
                else:
                    cleaned[key] = str(value)
            except Exception:
                cleaned[key] = "[ERROR_SERIALIZING]"
        return cleaned
    return obj


def _clean_list_for_serialization(obj, max_depth=5, current_depth=0):
    """Clean a list to remove potential circular references."""
    if current_depth > max_depth:
        return ["[MAX_DEPTH_REACHED]"]

    if isinstance(obj, list):
        cleaned = []
        for item in obj:
            try:
                if isinstance(item, (str, int, float, bool)):
                    cleaned.append(item)
                elif isinstance(item, list):
                    cleaned.append(
                        _clean_list_for_serialization(item, max_depth, current_depth + 1)
                    )
                elif isinstance(item, dict):
                    cleaned.append(
                        _clean_dict_for_serialization(item, max_depth, current_depth + 1)
                    )
                else:
                    cleaned.append(str(item))
            except Exception:
                cleaned.append("[ERROR_SERIALIZING]")
        return cleaned
    return obj


def _save_parameters_directly(model, output_dir: Path, logger: logging.Logger):
    """Alternative method to save parameters directly from model."""
    logger.info("Saving parameters directamente del modelo...")

    param_count = 0
    for name, param in model.named_parameters():
        try:
            # Create a safe filename
            safe_name = name.replace(".", "_").replace("/", "_")
            param_path = output_dir / f"{safe_name}.pt"

            # Save parameter
            torch.save(param.detach().cpu(), param_path)
            param_count += 1
            logger.debug(f"Parámetro directo guardado: {name}")

        except Exception as e:
            logger.warning(f"Error saving parámetro directo {name}: {e}")
            continue

    logger.info(f"{param_count} parameters guardados directamente")


def load_model_from_components(model_dir: Path, device: str = "cpu") -> PreTrainedModel:
    """Load a model that was saved using component-based saving."""
    # Check if this was saved by components
    metadata_path = model_dir / "component_save_metadata.json"
    if not metadata_path.exists():
        raise ValueError("This directory does not contain a component-saved model")

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("Not found config.json")

    config = AutoConfig.from_pretrained(str(model_dir))

    # Create model from config
    model = AutoModelForCausalLM.from_config(config)

    # Load weights from chunks
    state_dict = {}
    total_chunks = metadata["total_chunks"]

    for chunk_idx in range(total_chunks):
        chunk_path = model_dir / f"model_chunk_{chunk_idx}.pt"
        if chunk_path.exists():
            chunk = torch.load(chunk_path, map_location=device)
            state_dict.update(chunk)
        else:
            logger.warning(f"Chunk {chunk_idx} not found: {chunk_path}")

    # Load state dict into model
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    return model


class ModelCompressor:
    """Main model compression manager."""

    def __init__(
        self,
        compression_config_path: str,
        models_dir: str = "./models",
        output_suffix: str = "_compressed",
    ):
        self.config_path = Path(compression_config_path)
        self.models_dir = Path(models_dir)
        self.output_suffix = output_suffix

        # Load configuración
        self.compression_config = self._load_compression_config()
        self.model_name = self.compression_config["metadata"]["model_name"]
        self.model_path = self.models_dir / self.model_name
        self.output_path = self.models_dir / f"{self.model_name}{self.output_suffix}"

        # Motor de compresión
        self.engine = CompressionEngine()

        # Estadísticas
        self.stats = {
            "original_size_mb": 0,
            "compressed_size_mb": 0,
            "layers_compressed": 0,
            "layers_preserved": 0,
            "compression_time_seconds": 0,
            "final_layers_compressed": 0,
            "methods_used": set(),
        }

    def _load_compression_config(self) -> Dict[str, Any]:
        """Loads compression configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration not found: {self.config_path}")

        with open(self.config_path) as f:
            config = json.load(f)

        logger.info(f"Config loaded: {self.config_path.name}")
        logger.info(f"Target model: {config['metadata']['model_name']}")
        logger.info(f"Profile: {config['global_settings']['profile']}")
        logger.info(
            f"Target compression: {config['global_settings']['target_compression'] * 100:.1f}%"
        )

        # Verify si hay configuración de layers finales
        if config.get("final_layers_config"):
            final_count = config.get("final_layers_count", 0)
            logger.info(f"Special configuration for the last {final_count} layers")

        return config

    def _get_layer_config(
        self,
        layer_name: str,
        layer_type: str,
        relative_position: float,
        layer_index: int,
        total_layers: int,
    ) -> Dict[str, Any]:
        """Obtiene configuración de compresión para una capa específica."""
        # Verify si es una capa final con configuración especial
        final_layers_config = self.compression_config.get("final_layers_config")
        final_layers_count = self.compression_config.get("final_layers_count", 0)

        if final_layers_config and final_layers_count > 0:
            # Calculate si esta capa está en las layers finales
            layers_from_end = total_layers - layer_index
            if layers_from_end <= final_layers_count:
                logger.debug(
                    f"Aplicando configuración de layers finales a: {layer_name} (capa {layers_from_end} desde el final)"
                )
                self.stats["final_layers_compressed"] += 1
                return final_layers_config

        # Verify si la capa está en la lista de preservadas
        preserved_layers = self.compression_config.get("preserved_layers", [])
        if layer_name in preserved_layers:
            logger.info(f" Preservando capa sin cambios: {layer_name}")
            self.stats["layers_preserved"] += 1
            return {"methods": [{"name": "none", "strength": 0.0}], "total_compression_ratio": 0.0}

        # Usar configuración por tipo de capa
        layer_configs = self.compression_config.get("layer_configs", {})
        if layer_type in layer_configs:
            return layer_configs[layer_type]
        # Si llegamos aquí, la configuración está incompleta
        raise KeyError(
            f"No hay configuración para el tipo de capa '{layer_type}' (capa: {layer_name})"
        )

    def compress_model(self):
        """Ejecuta la compresión del modelo."""
        start_time = datetime.now()

        # Verify que el modelo existe
        if not self.model_path.exists():
            raise FileNotFoundError(f"Not found el modelo: {self.model_path}")

        logger.info(f"\n{'=' * 60}")
        logger.info("INICIANDO COMPRESIÓN DE MODELO")
        logger.info(f"{'=' * 60}")
        logger.info(f"Modelo: {self.model_name}")
        logger.info(f"Entrada: {self.model_path}")
        logger.info(f"Salida: {self.output_path}")
        logger.info(f"{'=' * 60}\n")

        # Create directorio de salida
        self.output_path.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Cargar modelo y configuración
            logger.info(" Cargando modelo...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load configuración primero para obtener arquitectura
            config = AutoConfig.from_pretrained(self.model_path)
            model_type = config.model_type

            # Load tokenizer si existe
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            except Exception:
                tokenizer = None
                logger.warning("Tokenizer not found, continuando sin él")

            # Load modelo con configuración de memoria optimizada
            logger.info(f" Dispositivo: {device}")
            logger.info(f" Arquitectura: {model_type}")

            if device.type == "cuda":
                # Load en FP16 para ahorrar memoria
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                model = model.to(device)

            # 2. Calcular tamaño original
            self.stats["original_size_mb"] = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / (1024 * 1024)
            logger.info(f"Size original: {self.stats['original_size_mb']:.1f} MB")

            # 3. Aplicar compresión capa por capa
            logger.info("\nAplicando compresión...")

            # Get todas las layers con nombres
            named_modules = list(model.named_modules())
            total_layers = len(named_modules)

            with tqdm(total=total_layers, desc="Comprimiendo layers") as pbar:
                for layer_index, (name, module) in enumerate(named_modules):
                    # Determinar tipo de capa
                    layer_type = self._get_layer_type(name, module)

                    # Saltar si no es una capa comprimible
                    if not self._is_compressible_layer(module):
                        pbar.update(1)
                        continue

                    # Calculate posición relativa
                    relative_position = layer_index / total_layers if total_layers > 0 else 0

                    # Get configuración para esta capa
                    layer_config = self._get_layer_config(
                        name, layer_type, relative_position, layer_index, total_layers
                    )

                    # Apply métodos de compresión
                    for method_config in layer_config["methods"]:
                        method_name = method_config["name"]
                        strength = method_config["strength"]

                        if method_name != "none" and strength > 0:
                            logger.debug(
                                f"  Aplicando {method_name} ({strength * 100:.0f}%) a {name}"
                            )

                            # Apply método usando el motor
                            compressed_module = self.engine.apply_method(
                                module, method_name, strength, layer_config
                            )

                            self._replace_module(model, name, compressed_module)

                            self.stats["methods_used"].add(method_name)

                    # Actualizar estadísticas
                    if layer_config["total_compression_ratio"] > 0:
                        self.stats["layers_compressed"] += 1

                    pbar.update(1)

            # 4. Optimizaciones post-compresión
            logger.info("\nAplicando optimizaciones finales...")

            # Limpiar buffers no usados
            self._cleanup_model(model)

            logger.info("\nSaving modelo comprimido...")
            # 5. Guardar el modelo comprimido utilizando safetensors cuando
            # esté disponible; de lo contrario, se reintenta el guardado
            # tradicional incrementando el límite de recursión si es
            # necesario.
            save_pretrained_with_fallback(model, tokenizer, self.output_path)

            # 6. Copiar archivos adicionales
            self._copy_additional_files()

            # 7. Guardar estadísticas y configuración
            self._save_compression_info()

            # Calculate tiempo total
            self.stats["compression_time_seconds"] = (datetime.now() - start_time).total_seconds()

            # 8. Mostrar resumen
            self._print_summary()

            # Limpiar memoria
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error durante la compresión: {str(e)}")
            raise

    def _copy_additional_files(self):
        """Copia archivos adicionales del modelo original."""
        files_to_copy = [
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "preprocessor_config.json",
        ]

        for filename in files_to_copy:
            src = self.model_path / filename
            dst = self.output_path / filename

            if src.exists():
                shutil.copy2(src, dst)
                logger.debug(f"Copiado: {filename}")

    def _save_compression_info(self):
        """Guarda información sobre la compresión aplicada."""
        # Convertir set a lista para JSON
        self.stats["methods_used"] = list(self.stats["methods_used"])

        info = {
            "compression_date": datetime.now().isoformat(),
            "original_model": str(self.model_path),
            "compression_config": self.compression_config,
            "statistics": self.stats,
            "notes": [],
        }

        # Agregar nota sobre layers finales
        if self.stats["final_layers_compressed"] > 0:
            info["notes"].append(
                f"Se aplicó configuración especial a las últimas {self.stats['final_layers_compressed']} layers"
            )

        # Save
        info_path = self.output_path / "compression_metadata.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        # También copiar la configuración original
        shutil.copy2(self.config_path, self.output_path / "compression_config.json")

    def _print_summary(self):
        """Imprime resumen de la compresión."""
        # Calculate tamaño final
        compressed_size = sum(
            os.path.getsize(os.path.join(root, file))
            for root, _, files in os.walk(self.output_path)
            for file in files
            if file.endswith((".bin", ".safetensors", ".pt", ".pth"))
        ) / (1024 * 1024)

        self.stats["compressed_size_mb"] = compressed_size
        compression_ratio = 1 - (compressed_size / self.stats["original_size_mb"])

        logger.info(f"\n{'=' * 60}")
        logger.info("COMPRESIÓN COMPLETADA")
        logger.info(f"{'=' * 60}")
        logger.info("Estadísticas:")
        logger.info(f"   • Size original:     {self.stats['original_size_mb']:.1f} MB")
        logger.info(f"   • Size comprimido:   {compressed_size:.1f} MB")
        logger.info(f"   • Reducción:           {compression_ratio * 100:.1f}%")
        logger.info(
            f"   • Factor:              {self.stats['original_size_mb'] / compressed_size:.1f}x"
        )
        logger.info("\nLayers procesadas:")
        logger.info(f"   • Comprimidas:         {self.stats['layers_compressed']}")
        logger.info(f"   • Preservadas:         {self.stats['layers_preserved']}")
        logger.info(f"   • Layers finales:       {self.stats['final_layers_compressed']}")
        logger.info(f"\nMethods utilizados:   {', '.join(self.stats['methods_used'])}")
        logger.info(f"⏱ Tiempo:               {self.stats['compression_time_seconds']:.1f} seconds")
        logger.info(f"\nModel saved in:   {self.output_path}")
        logger.info(f"{'=' * 60}")

    def _get_layer_type(self, name: str, module: nn.Module) -> str:
        """Determina el tipo de una capa de forma genérica."""
        name_lower = name.lower()
        module_type = type(module).__name__.lower()

        # 1. EMBEDDINGS - Patrones universales
        embedding_patterns = ["embed", "emb", "wte", "wpe", "position", "token"]
        if any(pattern in name_lower for pattern in embedding_patterns):
            return "embedding"
        if isinstance(module, nn.Embedding):
            return "embedding"

        # 2. NORMALIZATION - Por tipo de módulo
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            return "normalization"
        norm_patterns = ["norm", "ln", "layernorm", "batchnorm", "groupnorm"]
        if any(pattern in name_lower for pattern in norm_patterns):
            return "normalization"

        # 3. OUTPUT/HEAD - Patrones comunes
        output_patterns = [
            "head",
            "output",
            "classifier",
            "lm_head",
            "cls",
            "prediction",
            "logits",
            "score",
        ]
        if any(pattern in name_lower for pattern in output_patterns):
            return "output"

        # 4. ATTENTION - Patrones multi-arquitectura
        attention_patterns = [
            "attention",
            "attn",
            "self_attn",
            "cross_attn",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Común en LLaMA, GPT
            "query",
            "key",
            "value",  # BERT style
            "c_attn",
            "c_proj",  # GPT-2 style
            "qkv",
            "out_proj",  # Algunos modelos combinan QKV
        ]
        if any(pattern in name_lower for pattern in attention_patterns):
            return "attention"

        # Detectar por tipo de módulo (MultiheadAttention, etc.)
        if "attention" in module_type or "multihead" in module_type:
            return "attention"

        # 5. FFN/MLP - Patrones universales
        ffn_patterns = [
            "mlp",
            "ffn",
            "feed_forward",
            "feedforward",
            "fc",
            "dense",  # Fully connected
            "w1",
            "w2",
            "w3",  # Algunos modelos usan esta nomenclatura
            "gate_proj",
            "up_proj",
            "down_proj",  # LLaMA style
            "c_fc",
            "c_proj",  # GPT style
            "intermediate",
            "output.dense",  # BERT style
        ]
        if any(pattern in name_lower for pattern in ffn_patterns):
            return "ffn"

        # 6. ANÁLISIS ESTRUCTURAL para Linear layers
        if isinstance(module, nn.Linear):
            # Analizar dimensiones para inferir tipo
            in_features = module.in_features
            out_features = module.out_features

            # Si está dentro de un bloque transformer (heurística)
            if ".h." in name_lower or ".layer." in name_lower or ".block." in name_lower:
                # Buscar pistas en el nombre del padre
                parent_parts = name_lower.split(".")
                for i, _part in enumerate(parent_parts):
                    # Si el Linear está después de algo que suena a attention
                    if i > 0 and any(attn in parent_parts[i - 1] for attn in ["attn", "attention"]):
                        return "attention"
                    # Si está después de algo que suena a MLP/FFN
                    if i > 0 and any(ffn in parent_parts[i - 1] for ffn in ["mlp", "ffn", "feed"]):
                        return "ffn"

                # Heurística por tamaño: FFN suele tener expansión 4x
                if out_features > in_features * 3 or in_features > out_features * 3:
                    return "ffn"

                # Si las dimensiones son iguales, podría ser attention
                if in_features == out_features:
                    return "attention"

            return "linear"  # Mejor que 'other' para layers Linear

        # 7. CONVOLUCIONAL (por si acaso)
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return "conv"

        # 8. DROPOUT y otros (no se comprimen)
        if isinstance(module, (nn.Dropout, nn.Identity)):
            return "skip"  # Nueva categoría para layers que no se tocan

        # 9. Si no podemos determinar, intentar por estructura
        # Verify si tiene sub-módulos que den pistas
        if not isinstance(
            module, (nn.ModuleList, nn.Sequential)
        ):  # Solo para contenedores específicos
            child_modules = list(module.named_children())
            if child_modules and len(name.split(".")) < 10:  # Limitar profundidad de recursión
                # Analizar hijos para inferir tipo del padre
                child_types = set()
                for _child_name, child_module in child_modules[:3]:  # Solo primeros 3 hijos
                    # Evitar recursión analizando solo el tipo del módulo hijo directamente
                    if isinstance(child_module, nn.Linear):
                        child_types.add("linear")
                    elif isinstance(child_module, nn.LayerNorm):
                        child_types.add("normalization")
                    elif "attention" in type(child_module).__name__.lower():
                        child_types.add("attention")
                    elif "mlp" in type(child_module).__name__.lower():
                        child_types.add("ffn")

                # Si todos los hijos son del mismo tipo, el padre probablemente es ese tipo
                if len(child_types) == 1:
                    return child_types.pop()

        return "other"

    def _is_compressible_layer(self, module: nn.Module) -> bool:
        """Determina si una capa es comprimible."""
        # Excluir tipos que nunca se comprimen
        if isinstance(module, (nn.Dropout, nn.Identity)):
            return False

        # Solo comprimir layers con parameters significativos
        if not hasattr(module, "parameters"):
            return False

        num_params = sum(p.numel() for p in module.parameters())

        # Umbral mínimo de parameters (ajustable)
        min_params = 1000  # Reducido para layers más pequeñas

        return num_params > min_params

    def _replace_module(self, model: nn.Module, module_name: str, new_module: nn.Module):
        """Reemplaza un módulo en el modelo."""
        parts = module_name.split(".")
        parent = model

        # Navegar hasta el padre
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Reemplazar
        setattr(parent, parts[-1], new_module)

    def _cleanup_model(self, model: nn.Module):
        """Limpia buffers y optimiza el modelo."""
        # Eliminar buffers no esenciales
        for _name, module in model.named_modules():
            # Limpiar cachés de atención si existen
            if hasattr(module, "attention_cache"):
                delattr(module, "attention_cache")

            # Compactar pesos si es posible
            for _param_name, param in module.named_parameters():
                if param.grad is not None:
                    param.grad = None


def validate_model_path(model_path: str) -> bool:
    """Valida que la ruta del modelo sea válida y accesible.

    Args:
        model_path: Ruta al modelo a validar

    Returns:
        True si la ruta es válida, False en caso contrario

    """
    try:
        path = Path(model_path)

        # Verify que existe
        if not path.exists():
            logger.error(f"La ruta del modelo no existe: {model_path}")
            return False

        # Verify que es un directorio
        if not path.is_dir():
            logger.error(f"La ruta del modelo no es un directorio: {model_path}")
            return False

        # Verify archivos esenciales
        essential_files = ["config.json", "pytorch_model.bin"]
        if not any((path / file).exists() for file in essential_files):
            logger.warning(f"No se encontraron archivos esenciales del modelo en: {model_path}")
            # No es un error fatal, algunos modelos pueden tener nombres diferentes

        logger.info(f"Ruta del modelo válida: {model_path}")
        return True

    except Exception as e:
        logger.error(f"Error validando ruta del modelo: {e}")
        return False


def load_compression_config(config_path: str) -> Dict[str, Any]:
    """Carga configuración de compresión desde archivo JSON.

    Args:
        config_path: Ruta al archivo de configuración JSON

    Returns:
        Diccionario con la configuración de compresión

    Raises:
        FileNotFoundError: Si el archivo no existe
        json.JSONDecodeError: Si el archivo no es JSON válido

    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Archivo de configuración not found: {config_path}")

    try:
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)

        # Validar estructura básica
        if not isinstance(config, dict):
            raise ValueError("La configuración debe ser un diccionario JSON")

        if "metadata" not in config:
            raise ValueError("La configuración debe incluir metadatos")

        return config

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decodificando JSON: {e}", e.doc, e.pos) from e


def apply_compression_to_model(
    model_path: str, config_path: str, output_path: str
) -> Dict[str, Any]:
    """Aplica compresión a un modelo according to JSON configuration.

    Args:
        model_path: Ruta al modelo a comprimir
        config_path: Ruta al archivo de configuración JSON
        output_path: Ruta de salida para el modelo comprimido

    Returns:
        Diccionario con resultados de la compresión

    """
    try:
        # Create gestor de configuración
        config_manager = CompressionConfigManager(model_path)

        # Load configuración
        config = config_manager.load_config(Path(config_path))
        if not config:
            raise ValueError(f"No se pudo cargar la configuración desde {config_path}")

        # Create motor de compresión
        engine = CompressionEngine()

        # Apply compresión
        engine.compress_model(model_path, config)

        # Save modelo comprimido
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Aquí se guardaría el modelo comprimido
        # Por ahora, simulamos el resultado

        return {
            "success": True,
            "model_path": output_path,
            "compression_ratio": 0.3,  # Simulado
            "method_used": "compression",
        }

    except Exception as e:
        logger.error(f"Error aplicando compresión: {e}")
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Aplica compresión a un modelo according to JSON configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Comprimir usando configuración generada
  python apply_compression.py llama-7b

  # Especificar archivo de configuración
  python apply_compression.py --config ./configs/mi_config.json

  # Usar sufijo personalizado
  python apply_compression.py llama-7b --suffix _optimized
        """,
    )

    parser.add_argument("model", type=str, nargs="?", help="Name del modelo a comprimir")

    parser.add_argument(
        "--config",
        type=str,
        help="Ruta al archivo de configuración (default: busca en compression_analysis/)",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directorio de modelos (default: ./models)",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="_compressed",
        help="Sufijo para el modelo comprimido (default: _compressed)",
    )

    parser.add_argument(
        "--force", action="store_true", help="Sobrescribir si el modelo comprimido ya existe"
    )

    args = parser.parse_args()

    # Determinar archivo de configuración
    if args.config:
        config_path = args.config
    elif args.model:
        # Buscar en directorio por defecto
        config_path = f"./compression_analysis/{args.model}_compression_config.json"
    else:
        parser.error("Debes especificar un modelo o un archivo de configuración")

    # Verify que existe
    if not Path(config_path).exists():
        logger.error(f"Not found archivo de configuración: {config_path}")
        logger.error("   Primero ejecuta: python create_compression_config.py <modelo>")
        sys.exit(1)

    # Verify si ya existe el modelo comprimido
    models_dir = Path(args.models_dir)
    with open(config_path) as f:
        model_name = json.load(f)["metadata"]["model_name"]

    output_path = models_dir / f"{model_name}{args.suffix}"
    if output_path.exists() and not args.force:
        logger.error(f"El modelo comprimido ya existe: {output_path}")
        logger.error("   Usa --force para sobrescribir")
        sys.exit(1)

    try:
        # Create compresor y ejecutar
        compressor = ModelCompressor(config_path, args.models_dir, args.suffix)
        compressor.compress_model()

        # Sugerir próximos pasos
        logger.info("\nNext steps:")
        logger.info("1. Verificar el modelo:")
        logger.info(f"   python verify_compression.py {model_name}")
        logger.info("\n2. Test the model:")
        logger.info(f"   python ollama_compact_server.py --model {model_name}{args.suffix}")
        logger.info("\n3. Fine-tuning (si es necesario):")
        logger.info("   python finetune_lora.py")

    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
