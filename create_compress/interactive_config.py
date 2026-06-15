"""Configurador interactivo para compresión de modelos
Módulo auxiliar para create_compression_config.py.
"""

import logging
from typing import Any, Dict, List

from create_compress.compression_profiles import COMPRESSION_PROFILES

logger = logging.getLogger(__name__)


class InteractiveConfig:
    """Maneja la configuración interactiva de compresión."""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.COMPRESSION_METHODS = self._load_compression_methods()
        self.COMPRESSION_PROFILES = COMPRESSION_PROFILES

    def _load_compression_methods(self) -> Dict[str, Any]:
        """Carga métodos de compresión disponibles."""
        try:
            try:
                from create_compress.compression_methods import get_available_methods
            except ImportError:
                from compression_methods import get_available_methods
            return get_available_methods()
        except ImportError:
            # Fallback con métodos básicos
            return {
                "int8_quantization": {
                    "name": "Cuantización INT8",
                    "description": "Reduce a 8 bits",
                },
                "int4_quantization": {
                    "name": "Cuantización INT4",
                    "description": "Reduce a 4 bits",
                },
                "magnitude_pruning": {
                    "name": "Poda por magnitud",
                    "description": "Elimina pesos pequeños",
                },
                "structured_pruning": {
                    "name": "Poda estructurada",
                    "description": "Elimina neuronas",
                },
                "low_rank_approximation": {
                    "name": "Aproximación bajo rango",
                    "description": "Factorización",
                },
                "none": {"name": "Sin compresión", "description": "Mantener original"},
            }

    def _load_compression_profiles(self) -> Dict[str, Any]:
        """Carga perfiles de compresión."""
        try:
            try:
                from create_compress.compression_profiles import COMPRESSION_PROFILES
            except ImportError:
                from compression_profiles import COMPRESSION_PROFILES
            return COMPRESSION_PROFILES
        except ImportError:
            # Fallback con perfiles básicos
            return {
                "conservative": {
                    "name": "Conservative",
                    "description": "Compresión mínima, máxima calidad",
                    "target_compression": 0.3,
                },
                "balanced": {
                    "name": "Balanced",
                    "description": "Balance entre tamaño y calidad",
                    "target_compression": 0.5,
                },
                "aggressive": {
                    "name": "Aggressive",
                    "description": "Máxima compresión",
                    "target_compression": 0.7,
                },
            }

    def configure_profile(self) -> str:
        """Configura el perfil de compresión."""
        print("\nSelection de Profile de Compresión")
        print("=" * 50)

        # Show perfiles disponibles
        profiles = list(self.COMPRESSION_PROFILES.items())
        for i, (_key, profile) in enumerate(profiles, 1):
            print(f"{i}. {profile['name']}: {profile['description']}")
            print(f"   Target compression: {profile['target_compression'] * 100:.0f}%")

        print(f"{len(profiles) + 1}. Custom")

        while True:
            choice = input(f"\nSelecciona perfil (1-{len(profiles) + 1}): ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(profiles):
                    return profiles[idx][0]
                elif idx == len(profiles):
                    return "custom"
            except ValueError:
                pass
            print("Selection inválida")

    def configure_layer_types(self) -> Dict[str, Any]:
        """Configura compresión por tipo de capa."""
        print("\nConfiguración por Type de Layer")
        print("=" * 50)

        layer_configs = {}

        for layer_type, layers in self.config_manager.layer_types.items():
            print(f"\n {layer_type.upper()}")
            print(f"   Layers: {len(layers)}")
            print(f"   Size: {sum(l['size_mb'] for l in layers):.1f} MB")

            # Opciones rápidas
            print("\n   1. Sin compresión")
            print("   2. Compresión baja (20-30%)")
            print("   3. Compresión media (40-60%)")
            print("   4. Compresión alta (70-80%)")
            print("   5. Personalizar métodos")

            choice = input("\n   Selection (1-5): ").strip()

            if choice == "1":
                layer_configs[layer_type] = {
                    "methods": [{"name": "none", "strength": 0}],
                    "total_compression_ratio": 0,
                }
            elif choice in ["2", "3", "4"]:
                ratios = {"2": 0.25, "3": 0.5, "4": 0.75}
                ratio = ratios[choice]
                methods = self._auto_select_methods(layer_type, ratio)
                layer_configs[layer_type] = {"methods": methods, "total_compression_ratio": ratio}
            else:
                layer_configs[layer_type] = self._custom_layer_config(layer_type)

        return layer_configs

    def _auto_select_methods(self, layer_type: str, target_ratio: float) -> List[Dict[str, Any]]:
        """Selecciona automáticamente métodos apropiados."""
        methods = []

        if layer_type == "attention":
            if target_ratio <= 0.3:
                methods.append({"name": "head_pruning", "strength": target_ratio})
            elif target_ratio <= 0.6:
                methods.append({"name": "low_rank_approximation", "strength": target_ratio * 0.8})
                methods.append({"name": "int8_quantization", "strength": 0.5})
            else:
                methods.append({"name": "attention_pruning", "strength": target_ratio * 0.9})
                methods.append({"name": "int4_quantization", "strength": 0.7})

        elif layer_type in ["ffn", "mlp", "linear"]:
            if target_ratio <= 0.3:
                methods.append({"name": "magnitude_pruning", "strength": target_ratio})
            elif target_ratio <= 0.6:
                methods.append({"name": "magnitude_pruning", "strength": target_ratio * 0.7})
                methods.append({"name": "int8_quantization", "strength": 0.6})
            else:
                methods.append({"name": "structured_pruning", "strength": target_ratio * 0.8})
                methods.append({"name": "int4_quantization", "strength": 0.8})

        elif layer_type in ["embedding", "output", "lm_head"]:
            # Layers críticas - compresión más conservadora
            if target_ratio <= 0.3:
                methods.append({"name": "int8_quantization", "strength": target_ratio * 0.5})
            else:
                methods.append({"name": "magnitude_pruning", "strength": target_ratio * 0.4})
                methods.append({"name": "int8_quantization", "strength": 0.5})

        else:
            # Type genérico
            if target_ratio <= 0.5:
                methods.append({"name": "magnitude_pruning", "strength": target_ratio})
            else:
                methods.append({"name": "int8_quantization", "strength": target_ratio})

        return methods

    def _custom_layer_config(self, layer_type: str) -> Dict[str, Any]:
        """Configuración personalizada para un tipo de capa."""
        print(f"\n   Configuración personalizada para {layer_type}")

        methods = []
        total_compression = 0

        # Show métodos disponibles
        suitable_methods = self._get_suitable_methods(layer_type)

        print("\n   Methods disponibles:")
        for i, (key, method) in enumerate(suitable_methods.items(), 1):
            print(f"   {i}. {key}: {method.get('description', '')}")

        print("\n   Agrega métodos (Enter para terminar):")

        while True:
            choice = input("   Method (número): ").strip()
            if not choice:
                break

            try:
                idx = int(choice) - 1
                method_key = list(suitable_methods.keys())[idx]

                # Get fuerza
                strength = float(input(f"   Fuerza para {method_key} (0-100%): ")) / 100

                methods.append({"name": method_key, "strength": min(1.0, max(0.0, strength))})

                # Estimar compresión acumulada (aproximada)
                total_compression = min(0.95, total_compression + strength * 0.6)

                print(f"    Added: {method_key} al {strength * 100:.0f}%")

            except (ValueError, IndexError):
                print("   Selection inválida")

        if not methods:
            methods = [{"name": "none", "strength": 0}]
            total_compression = 0

        return {"methods": methods, "total_compression_ratio": total_compression}

    def _get_suitable_methods(self, layer_type: str) -> Dict[str, Any]:
        """Obtiene métodos adecuados para un tipo de capa."""
        # Definir métodos por tipo
        method_map = {
            "attention": [
                "attention_pruning",
                "head_pruning",
                "low_rank_approximation",
                "int8_quantization",
                "int4_quantization",
            ],
            "ffn": [
                "magnitude_pruning",
                "structured_pruning",
                "low_rank_approximation",
                "int8_quantization",
                "int4_quantization",
            ],
            "embedding": ["int8_quantization", "magnitude_pruning"],
            "output": ["int8_quantization", "magnitude_pruning"],
            "normalization": ["none"],  # Generalmente no se comprimen
            "other": ["magnitude_pruning", "int8_quantization"],
        }

        suitable_keys = method_map.get(layer_type, method_map["other"])
        suitable_keys.append("none")  # Siempre incluir opción de no comprimir

        return {k: v for k, v in self.COMPRESSION_METHODS.items() if k in suitable_keys}

    def configure_advanced_settings(self):
        """Configura ajustes avanzados."""
        print("\n Configuración Avanzada")
        print("=" * 50)

        # Modificadores por posición
        if input("\n¿Aplicar modificadores por posición? (s/N): ").strip().lower() == "s":
            self.config_manager.add_position_modifiers()
            print("Modificadores de posición aplicados")

        # Layers finales
        if input("\n¿Configurar layers finales de forma especial? (s/N): ").strip().lower() == "s":
            self._configure_final_layers()

        # Layers a preservar
        if input("\n¿Hay layers que NO se deben comprimir? (s/N): ").strip().lower() == "s":
            self._configure_preserved_layers()

        # Configuration for validación
        print("\nConfiguration for validación:")
        self.config_manager.set_validation_settings()
        print("   • Umbral mínimo de rendimiento: 90%")
        print("   • Compresión máxima por capa: 80%")
        print("   • Compresión gradual requerida: Sí")

    def _configure_final_layers(self):
        """Configura las layers finales del modelo."""
        total_layers = self.config_manager.total_layers

        print(f"\nTotal de layers en el modelo: {total_layers}")

        # Sugerir número de layers finales
        suggested = min(4, max(1, total_layers // 10))
        num_str = input(f"Número de layers finales a configurar [{suggested}]: ").strip()
        num_final = int(num_str) if num_str else suggested

        print("\nConfiguración para layers finales:")
        print("1. Preservar sin cambios")
        print("2. Compresión mínima (10%)")
        print("3. Compresión reducida (25%)")

        choice = input("Selection (1-3) [2]: ").strip() or "2"

        configs = {
            "1": {"methods": [{"name": "none", "strength": 0}], "total_compression_ratio": 0},
            "2": {
                "methods": [{"name": "int8_quantization", "strength": 0.3}],
                "total_compression_ratio": 0.1,
            },
            "3": {
                "methods": [{"name": "magnitude_pruning", "strength": 0.25}],
                "total_compression_ratio": 0.25,
            },
        }

        config = configs.get(choice, configs["2"])
        self.config_manager.set_final_layers_config(num_final, config)

        print(f"Configuración aplicada a las últimas {num_final} layers")

    def _configure_preserved_layers(self):
        """Configura layers específicas a preservar."""
        preserved = []

        print("\nIngresa nombres exactos de layers (empty to finish):")
        print("Ejemplo: layers.0.self_attn.q_proj")

        # Show algunas layers como ejemplo
        example_layers = []
        for layers in list(self.config_manager.layer_types.values())[:2]:
            example_layers.extend([l["name"] for l in layers[:2]])

        if example_layers:
            print(f"Layers disponibles (examples): {', '.join(example_layers[:4])}")

        while True:
            layer = input("→ Layer: ").strip()
            if not layer:
                break

            # Verify si existe
            found = False
            for layer_list in self.config_manager.layer_types.values():
                if any(l["name"] == layer for l in layer_list):
                    found = True
                    break

            if found:
                preserved.append(layer)
                print(f"   '{layer}' agregada a la lista")
            else:
                # Buscar coincidencias parciales
                matches = []
                for layer_list in self.config_manager.layer_types.values():
                    matches.extend([l["name"] for l in layer_list if layer in l["name"]])

                if matches:
                    print(f"  Not found '{layer}' exacto. Coincidencias:")
                    for i, match in enumerate(matches[:5], 1):
                        print(f"     {i}. {match}")
                    if len(matches) > 5:
                        print(f"     ... y {len(matches) - 5} más")
                else:
                    print(f"  Not found ninguna capa con '{layer}'")

        if preserved:
            self.config_manager.add_preserved_layers(preserved)
            print(f"\n{len(preserved)} layers marcadas para preservar sin cambios")

    def show_summary(self):
        """Muestra el resumen de la configuración."""
        summary = self.config_manager.calculate_summary()

        print("\n" + "=" * 70)
        print("RESUMEN DE CONFIGURACIÓN DE COMPRESIÓN")
        print("=" * 70)

        # Resumen por tipo de capa
        print("\nConfiguración por tipo de capa:")
        print(f"{'' * 70}")
        print(
            f"{'Type':<15} {'Layers':<8} {'Original':<12} {'Comprimido':<12} {'Reducción':<12} {'Methods'}"
        )
        print(f"{'' * 70}")

        for layer_type, info in sorted(summary["by_layer_type"].items()):
            config = self.config_manager.compression_config["layer_configs"].get(layer_type, {})
            methods = [m["name"] for m in config.get("methods", [{"name": "none"}])]
            methods_str = ", ".join(methods) if methods != ["none"] else "Sin cambios"

            print(
                f"{layer_type:<15} {info['num_layers']:<8} "
                f"{info['original_size_mb']:<12.1f} {info['compressed_size_mb']:<12.1f} "
                f"{info['compression_ratio'] * 100:<11.1f}% {methods_str}"
            )

        print(f"{'' * 70}")

        # Totales
        print("\nTOTALES:")
        print(f"   • Size original:    {summary['original_size_mb']:>10.1f} MB")
        print(f"   • Size comprimido:  {summary['compressed_size_mb']:>10.1f} MB")
        print(f"   • Reducción total:    {summary['compression_ratio'] * 100:>10.1f}%")
        print(
            f"   • Factor de compresión: {summary['original_size_mb'] / max(summary['compressed_size_mb'], 0.1):>5.1f}x"
        )

        # Configuraciones especiales
        if self.config_manager.compression_config.get("preserved_layers"):
            print(
                f"\n Preserved layers: {len(self.config_manager.compression_config['preserved_layers'])}"
            )

        if self.config_manager.compression_config.get("final_layers_config"):
            num_final = self.config_manager.compression_config.get("final_layers_count", 0)
            print(f"Layers finales con config especial: {num_final}")

        if self.config_manager.compression_config.get("position_modifiers"):
            print(" Modificadores por posición: Activados")
