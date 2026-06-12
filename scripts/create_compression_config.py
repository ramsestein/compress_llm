#!/usr/bin/env python3
"""Creador interactivo de configuraciones de compresión - OPTIMIZADO."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from create_compress.compression_config_manager import CompressionConfigManager

# Configure logging
logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache para métodos y perfiles
_METHODS_CACHE = None
_PROFILES_CACHE = None


def get_compression_methods():
    """Lazy loading de métodos de compresión."""
    global _METHODS_CACHE
    if _METHODS_CACHE is None:
        try:
            from create_compress.compression_methods import get_available_methods
        except ImportError:
            from compression_methods import get_available_methods
        _METHODS_CACHE = get_available_methods()
    return _METHODS_CACHE


def get_compression_profiles():
    """Lazy loading de perfiles de compresión."""
    global _PROFILES_CACHE
    if _PROFILES_CACHE is None:
        try:
            from create_compress.compression_profiles import COMPRESSION_PROFILES
        except ImportError:
            from compression_profiles import COMPRESSION_PROFILES
        _PROFILES_CACHE = COMPRESSION_PROFILES
    return _PROFILES_CACHE


class OptimizedCompressionConfigCreator:
    """Creador optimizado de configuraciones de compresión."""

    def __init__(self, model_path: str, output_dir: str = "./compression_analysis"):
        self.config_manager = CompressionConfigManager(model_path, output_dir)
        self.model_name = self.config_manager.model_name
        self.model_path = Path(model_path)

        # Cache para evitar recálculos
        self._layer_types_cache = None
        self._total_layers_cache = None

    def analyze_model_layers(self) -> bool:
        """Analiza las layers del modelo desde el reporte."""
        return self.config_manager.load_from_report()

    def interactive_configuration(self):
        """Proceso interactivo optimizado de configuración."""
        print("\nCONFIGURACIÓN DE COMPRESIÓN")
        print("=" * 50)

        # 1. Selection rápida de perfil
        profile = self._quick_profile_selection()

        if profile != "custom":
            # Apply perfil predefinido
            self.config_manager.set_compression_profile(profile)
            print(f"\nProfile '{profile}' aplicado")

            # Preguntar si quiere personalizar
            if input("\n¿Deseas personalizar esta configuración? (s/N): ").strip().lower() == "s":
                self._customize_configuration()
        else:
            # Configuration personalizada completa
            self._custom_configuration()

    def _quick_profile_selection(self) -> str:
        """Selection rápida de perfil."""
        print("\n Selecciona un perfil de compresión:")
        print("\n[1]   Conservative (30% compresión, 95%+ rendimiento)")
        print("[2]   Balanced (50% compresión, 90% rendimiento)")
        print("[3] Aggressive (70% compresión, 80% rendimiento)")
        print("[4] Custom (configuración manual)")

        while True:
            choice = input("\nOpción (1-4) [2]: ").strip() or "2"

            if choice == "1":
                return "conservative"
            elif choice == "2":
                return "balanced"
            elif choice == "3":
                return "aggressive"
            elif choice == "4":
                return "custom"
            else:
                print("Invalid option")

    def _custom_configuration(self):
        """Configuración personalizada completa."""
        print("\nCONFIGURACIÓN PERSONALIZADA")
        print("=" * 40)

        # Set global_settings para custom
        self.config_manager.compression_config["global_settings"] = {
            "profile": "custom",
            "name": "Custom",
            "description": "Configuración personalizada",
            "goal": "custom",
            "target_compression": 0.5,
            "risk_level": "medium",
        }

        # Types de layers estándar que siempre deben existir
        standard_types = [
            "embedding",
            "attention",
            "ffn",
            "linear",
            "normalization",
            "output",
            "conv",
            "other",
            "skip",
        ]

        # Get tipos detectados
        detected_types = self.config_manager.get_layer_types_list()

        # Combinar tipos detectados con estándar
        all_types = list(set(detected_types + standard_types))

        for layer_type in all_types:
            layers = self.config_manager.get_layers_by_type(layer_type)

            # Si no hay layers de este tipo, configurar por defecto
            if not layers:
                # Configuration por defecto para tipos no detectados
                if layer_type in ["normalization", "skip"]:
                    methods = [{"name": "none", "strength": 0.0}]
                    ratio = 0.0
                else:
                    methods = [{"name": "none", "strength": 0.0}]
                    ratio = 0.0

                self.config_manager.set_custom_layer_config(layer_type, methods, ratio)
                continue

            # Resto del código igual...
            print(f"\nType: {layer_type.upper()}")
            print(f"   Layers: {len(layers)}")
            print(f"   Size: {sum(l['size_mb'] for l in layers):.1f} MB")

            # Show advertencia para tipos especiales
            if layer_type in ["normalization", "skip"]:
                print(f"   NOTA: Las layers de {layer_type} generalmente NO se comprimen")
                print("   (Son críticas y tienen muy pocos parameters)")
            elif layer_type == "output":
                print("   NOTA: La capa de salida requiere cuidado especial")
                print("   Se recomienda compresión mínima para mantener calidad")

            # Methods recomendados
            methods = self._get_recommended_methods(layer_type)

            # Configuration rápida o manual
            print("\n   [1] Automática (recomendada)")
            print("   [2] Manual")
            print("   [3] Sin compresión")

            choice = input("\n   Opción [1]: ").strip() or "1"

            if choice == "1":
                # Automática
                compression_methods, ratio = self._auto_configure_layer(layer_type, methods)
            elif choice == "2":
                # Manual
                compression_methods, ratio = self._manual_configure_layer(layer_type, methods)
            else:
                # Sin compresión
                compression_methods = [{"name": "none", "strength": 0.0}]
                ratio = 0.0

            # Save configuración
            self.config_manager.set_custom_layer_config(layer_type, compression_methods, ratio)

        total_original = 0
        total_compressed = 0

        for layer_type, layers in self.config_manager.layer_types.items():
            type_size = sum(l["size_mb"] for l in layers)
            total_original += type_size

            config = self.config_manager.compression_config["layer_configs"].get(layer_type, {})
            ratio = config.get("total_compression_ratio", 0)
            total_compressed += type_size * (1 - ratio)

        if total_original > 0:
            actual_compression = (total_original - total_compressed) / total_original
            self.config_manager.compression_config["global_settings"]["target_compression"] = (
                actual_compression
            )

    def _customize_configuration(self):
        """Personaliza una configuración base."""
        print("\n PERSONALIZACIÓN DE CONFIGURACIÓN")
        print("=" * 40)

        # 1. Ajustar intensidad global
        current_compression = self.config_manager.compression_config["global_settings"][
            "target_compression"
        ]
        print(f"\nCompresión actual: {current_compression * 100:.0f}%")

        new_compression = input("Nueva compresión % (Enter para mantener): ").strip()
        if new_compression:
            try:
                value = float(new_compression) / 100
                if 0 <= value <= 0.95:
                    self._scale_compression(value / current_compression)
                    print(f"Compresión ajustada a {value * 100:.0f}%")
                else:
                    print("Valor debe estar entre 0 y 95%")
            except ValueError:
                print("Valor inválido")

        # 2. Layers a preservar
        if (
            input("\n¿Hay layers específicas que NO se deben comprimir? (s/N): ").strip().lower()
            == "s"
        ):
            self._add_preserved_layers()

        # 3. Configuration for layers finales
        if (
            input("\n¿Aplicar configuración especial a las layers finales? (s/N): ").strip().lower()
            == "s"
        ):
            self._configure_final_layers()

    def _scale_compression(self, factor: float):
        """Escala toda la compresión por un factor."""
        for _layer_type, config in self.config_manager.compression_config["layer_configs"].items():
            for method in config.get("methods", []):
                method["strength"] = min(method["strength"] * factor, 0.95)
            config["total_compression_ratio"] = min(
                config["total_compression_ratio"] * factor, 0.95
            )

    def _add_preserved_layers(self):
        """Agrega layers a preservar sin compresión."""
        print("\n CAPAS A PRESERVAR")
        print("Ingresa nombres de layers (empty to finish)")
        print("Ejemplo: model.layers.0.self_attn.q_proj")

        preserved = []
        while True:
            layer = input("Layer: ").strip()
            if not layer:
                break
            preserved.append(layer)
            print(f"Agregada: {layer}")

        if preserved:
            self.config_manager.add_preserved_layers(preserved)
            print(f"\n{len(preserved)} layers marcadas para preservar")

    def _configure_final_layers(self):
        """Configura compresión especial para layers finales."""
        print("\nCONFIGURACIÓN DE CAPAS FINALES")

        # Número de layers
        num_layers = input("¿Cuántas layers finales? [3]: ").strip() or "4"
        try:
            num_layers = int(num_layers)
        except ValueError:
            num_layers = 4

        print(f"\nConfigurando las últimas {num_layers} layers...")

        # Method de compresión
        print("\n[1] Compresión suave (recomendado)")
        print("[2] Sin compresión")
        print("[3] Custom")

        choice = input("\nOpción [1]: ").strip() or "1"

        if choice == "1":
            # Suave
            methods = [{"name": "int8_quantization", "strength": 0.1}]
            ratio = 0.1
        elif choice == "2":
            # Sin compresión
            methods = [{"name": "none", "strength": 0.0}]
            ratio = 0.0
        else:
            # Custom
            methods, ratio = self._manual_configure_layer(
                "final_layers", ["int8_quantization", "pruning"]
            )

        # Save
        self.config_manager.compression_config["final_layers_config"] = {
            "methods": methods,
            "total_compression_ratio": ratio,
        }
        self.config_manager.compression_config["final_layers_count"] = num_layers

        print(f"Configuración especial aplicada a las últimas {num_layers} layers")

    def _get_recommended_methods(self, layer_type: str) -> List[str]:
        """Obtiene métodos recomendados para un tipo de capa."""
        recommendations = {
            "embedding": ["int8_quantization", "svd"],
            "attention": [
                "head_pruning",
                "attention_pruning",
                "int8_quantization",
                "low_rank_approximation",
            ],
            "ffn": [
                "magnitude_pruning",
                "structured_pruning",
                "int8_quantization",
                "int4_quantization",
            ],
            "linear": ["magnitude_pruning", "int8_quantization", "low_rank_approximation"],
            "normalization": [],  # No comprimir
            "output": ["int8_quantization", "magnitude_pruning"],
            "conv": ["structured_pruning", "int8_quantization", "magnitude_pruning"],
            "skip": [],  # No comprimir
            "other": ["int8_quantization", "magnitude_pruning"],
        }
        return recommendations.get(layer_type, ["int8_quantization"])

    def _auto_configure_layer(
        self, layer_type: str, recommended: List[str]
    ) -> Tuple[List[Dict], float]:
        """Configuración automática para un tipo de capa."""
        # Get el perfil actual para ajustar intensidad
        profile = self.config_manager.compression_config.get("global_settings", {}).get(
            "profile", "balanced"
        )

        # Multiplicadores según perfil
        strength_multipliers = {
            "conservative": 0.5,
            "balanced": 1.0,
            "aggressive": 1.5,
            "custom": 1.0,
        }
        multiplier = strength_multipliers.get(profile, 1.0)

        # Configuraciones base por tipo
        configs = {
            "embedding": [{"name": "int8_quantization", "strength": 0.3 * multiplier}],
            "attention": [
                {"name": "head_pruning", "strength": 0.3 * multiplier},
                {"name": "int8_quantization", "strength": 0.4 * multiplier},
            ],
            "ffn": [
                {"name": "magnitude_pruning", "strength": 0.4 * multiplier},
                {"name": "int8_quantization", "strength": 0.5 * multiplier},
            ],
            "linear": [
                {"name": "magnitude_pruning", "strength": 0.3 * multiplier},
                {"name": "int8_quantization", "strength": 0.4 * multiplier},
            ],
            "normalization": [{"name": "none", "strength": 0.0}],  # Nunca comprimir
            "output": [{"name": "int8_quantization", "strength": 0.2 * multiplier}],
            "conv": [
                {"name": "structured_pruning", "strength": 0.3 * multiplier},
                {"name": "int8_quantization", "strength": 0.4 * multiplier},
            ],
            "skip": [{"name": "none", "strength": 0.0}],  # Nunca comprimir
            "other": [{"name": "int8_quantization", "strength": 0.3 * multiplier}],
        }

        methods = configs.get(
            layer_type, [{"name": "int8_quantization", "strength": 0.3 * multiplier}]
        )

        # Ajustar strengths para no exceder 0.95
        for method in methods:
            method["strength"] = min(method["strength"], 0.95)

        ratio = sum(m["strength"] for m in methods)
        ratio = min(ratio, 0.95)  # Cap total en 95%

        return methods, ratio

    def _manual_configure_layer(
        self, layer_type: str, recommended: List[str]
    ) -> Tuple[List[Dict], float]:
        """Configuración manual para un tipo de capa."""
        methods = []
        total_ratio = 0.0

        # Get TODOS los métodos disponibles
        all_methods = get_compression_methods()

        print("\n   TODOS los métodos disponibles:")
        for i, (method_key, method_desc) in enumerate(all_methods.items(), 1):
            print(f"   {i:2d}. {method_key:<25} - {method_desc}")

        print(f"\n   Recomendados para {layer_type}: {', '.join(recommended)}")
        print("   Puedes agregar múltiples métodos (máximo 95% total)")

        while total_ratio < 0.95:
            if methods:
                print(f"\n   Compresión actual: {total_ratio * 100:.0f}%")

            method = input("\n   Method (empty to finish): ").strip()
            if not method:
                break

            if method not in all_methods:
                print(f"   '{method}' no es un método válido")
                # Show métodos similares
                similar = [
                    m
                    for m in all_methods
                    if method.lower() in m.lower() or m.lower() in method.lower()
                ]
                if similar:
                    print(f"   ¿Quisiste decir: {', '.join(similar)}?")
                continue

            strength = input(f"   Strength % para {method} [15]: ").strip() or "15"
            try:
                strength_val = float(strength) / 100
                if strength_val <= 0 or strength_val > 0.95:
                    print("   La intensidad debe estar entre 0 y 95%")
                    continue

                if total_ratio + strength_val > 0.95:
                    print(f"   Excedería el límite de 95% (actual: {total_ratio * 100:.0f}%)")
                    continue

                methods.append({"name": method, "strength": strength_val})
                total_ratio += strength_val
                print(f"   Added: {method} al {strength_val * 100:.0f}%")

            except ValueError:
                print("   Valor inválido")

        if not methods:
            methods = [{"name": "none", "strength": 0.0}]
            total_ratio = 0.0

        return methods, total_ratio

    def add_advanced_settings(self):
        """Agrega configuraciones avanzadas."""
        print("\n CONFIGURACIONES AVANZADAS")
        print("=" * 40)

        # 1. Modificadores por posición
        if (
            input("\n¿Aplicar modificadores por posición en el modelo? (s/N): ").strip().lower()
            == "s"
        ):
            self.config_manager.add_position_modifiers()
            print("Modificadores de posición aplicados")
            print("   • Layers iniciales: -30% compresión")
            print("   • Layers intermedias: sin cambio")
            print("   • Layers finales: +30% compresión")

        # 2. Configuration for validación
        self.config_manager.set_validation_settings()
        print("\nConfiguration for validación establecida")

        # 3. Optimizaciones adicionales
        if input("\n¿Habilitar optimizaciones experimentales? (s/N): ").strip().lower() == "s":
            self.config_manager.compression_config["experimental"] = {
                "use_mixed_precision": True,
                "gradient_checkpointing": True,
                "dynamic_compression": True,
            }
            print("Optimizaciones experimentales habilitadas")

    def calculate_summary(self):
        """Calcula y muestra resumen de la configuración."""
        summary = self.config_manager.calculate_summary()

        print("\n" + "=" * 70)
        print("RESUMEN DE CONFIGURACIÓN DE COMPRESIÓN")
        print("=" * 70)

        # Resumen general
        print("\nCompresión estimada:")
        print(f"   • Size original:    {summary['original_size_mb']:>10.1f} MB")
        print(f"   • Size comprimido:  {summary['compressed_size_mb']:>10.1f} MB")
        print(f"   • Reducción:          {summary['compression_ratio'] * 100:>10.1f}%")
        print(
            f"   • Factor:             {summary['original_size_mb'] / max(summary['compressed_size_mb'], 0.1):>10.1f}x"
        )

        # Detalles por tipo
        print("\nPor tipo de capa:")
        print(f"{'' * 60}")
        print(f"{'Type':<15} {'Layers':<8} {'Original MB':<12} {'Final MB':<12} {'Reducción':<10}")
        print(f"{'' * 60}")

        for layer_type, info in sorted(summary["by_layer_type"].items()):
            print(
                f"{layer_type:<15} {info['num_layers']:<8} "
                f"{info['original_size_mb']:<12.1f} {info['compressed_size_mb']:<12.1f} "
                f"{info['compression_ratio'] * 100:<9.1f}%"
            )

        print(f"{'' * 60}")

    def save_configuration(self):
        """Guarda la and optimized configuration."""
        output_path = self.config_manager.save_configuration()

        print("\nConfiguración guardada exitosamente")
        print(f"Ubicación: {output_path}")

        # Show próximos pasos
        print("\nNext steps:")
        print("1. Aplicar compresión:")
        print(f"   python apply_compression.py {self.model_name}")
        print("\n2. Verificar modelo comprimido:")
        print(f"   python verify_compression.py {self.model_name}")
        print("\n3. Evaluate performance:")
        print(f"   python evaluate_compressed.py {self.model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Crear configuración de compresión - OPTIMIZADO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python create_compression_config.py llama-7b
  python create_compression_config.py meta-llama/Llama-2-7b-hf --quick
  python create_compression_config.py ./models/mi-modelo --output-dir ./configs
        """,
    )

    parser.add_argument("model", help="Name del modelo o ruta")
    parser.add_argument(
        "--output-dir",
        default="./compression_analysis",
        help="Directorio de salida (default: ./compression_analysis)",
    )
    parser.add_argument(
        "--models-dir", default="./models", help="Directorio de modelos (default: ./models)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Modo rápido con valores predeterminados"
    )
    parser.add_argument(
        "--profile",
        choices=["conservative", "balanced", "aggressive"],
        help="Aplicar perfil directamente sin interacción",
    )

    args = parser.parse_args()

    # Resolver ruta del modelo
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = Path(args.models_dir) / args.model
        if not model_path.exists():
            # Intentar con _ en lugar de /
            safe_name = args.model.replace("/", "_")
            model_path = Path(args.models_dir) / safe_name

            if not model_path.exists():
                logger.error(f"Not found el modelo: {args.model}")
                logger.error(f"   Buscado en: {Path(args.models_dir).absolute()}")
                sys.exit(1)

    try:
        # Create configurador
        creator = OptimizedCompressionConfigCreator(str(model_path), args.output_dir)

        # Analizar modelo
        if not creator.analyze_model_layers():
            sys.exit(1)

        # Modo rápido o con perfil específico
        if args.profile:
            creator.config_manager.set_compression_profile(args.profile)
            creator.calculate_summary()
            creator.save_configuration()
            print(f"\nConfiguración '{args.profile}' aplicada y guardada")

        elif args.quick:
            # Modo rápido interactivo mínimo
            print("\nMODO RÁPIDO")
            profile = input("Profile (c)onservative/(b)alanced/(a)ggressive [b]: ").strip().lower()
            profile_map = {"c": "conservative", "b": "balanced", "a": "aggressive", "": "balanced"}
            selected_profile = profile_map.get(profile, "balanced")

            creator.config_manager.set_compression_profile(selected_profile)
            creator.calculate_summary()
            creator.save_configuration()

        else:
            # Modo interactivo completo
            creator.interactive_configuration()
            creator.calculate_summary()

            # Confirmar guardado
            if input("\n¿Guardar esta configuración? (S/n): ").strip().lower() != "n":
                creator.save_configuration()
            else:
                print("Configuration forscartada")

    except KeyboardInterrupt:
        print("\n\nProceso cancelado por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        if args.models_dir:
            logger.debug("Traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
