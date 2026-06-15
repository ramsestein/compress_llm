#!/usr/bin/env python3
"""Main script to analizar modelos locales o de HuggingFace."""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

# Import nuestros módulos
from down_report.model_analyzer import OptimizedModelAnalyzer
from down_report.report_generator import OptimizedReportGenerator


class ModelManager:
    """Gestiona modelos locales y remotos."""

    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def resolve_model_path(self, model_identifier: str) -> tuple[str, bool]:
        """Resuelve si un identificador es un modelo local o remoto.

        Returns:
            tuple: (ruta_del_modelo, es_local)

        """
        # Caso 1: Es una ruta absoluta o relativa existente
        path = Path(model_identifier)
        if path.exists() and (path / "config.json").exists():
            return str(path.resolve()), True

        # Caso 2: Es un nombre de modelo en el directorio models/
        safe_name = model_identifier.replace("/", "_")
        local_path = self.models_dir / safe_name
        if local_path.exists() and (local_path / "config.json").exists():
            return str(local_path.resolve()), True

        # Caso 3: Buscar por coincidencia parcial en models/
        if "/" not in model_identifier:  # Si no tiene formato org/model
            for model_dir in self.models_dir.iterdir():
                if model_identifier.lower() in model_dir.name.lower():
                    if (model_dir / "config.json").exists():
                        print(f"Encontrado modelo local: {model_dir.name}")
                        return str(model_dir.resolve()), True

        # Caso 4: Es un modelo de HuggingFace
        return model_identifier, False

    def get_model_info(self, model_path: str) -> dict:
        """Obtiene información de un modelo local."""
        path = Path(model_path)

        if not path.exists():
            return None

        try:
            config = AutoConfig.from_pretrained(str(path))

            # Calculate tamaño
            total_size = sum(
                os.path.getsize(os.path.join(root, file))
                for root, _, files in os.walk(path)
                for file in files
            )

            # Contar archivos de modelo
            model_files = list(path.glob("*.bin")) + list(path.glob("*.safetensors"))

            info = {
                "name": path.name,
                "path": str(path),
                "size_gb": total_size / (1024**3),
                "architecture": getattr(config, "model_type", "unknown"),
                "hidden_size": getattr(config, "hidden_size", None),
                "num_layers": getattr(config, "num_hidden_layers", None),
                "num_parameters": getattr(config, "num_parameters", None),
                "model_files": len(model_files),
                "total_files": sum(1 for _ in path.rglob("*") if _.is_file()),
            }

            # Information adicional si existe
            if (path / "download_info.txt").exists():
                info["has_download_info"] = True

            return info

        except Exception as e:
            print(f"Error leyendo información del modelo: {e}")
            return None

    def list_local_models(self, detailed: bool = False) -> List[dict]:
        """Lista todos los modelos locales disponibles."""
        models = []

        # Buscar en el directorio de modelos
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                info = self.get_model_info(str(model_dir))
                if info:
                    models.append(info)

        # Ordenar por nombre
        models.sort(key=lambda x: x["name"])

        if detailed:
            print(f"\nModelos locales en {self.models_dir}:")
            print("=" * 80)

            if not models:
                print("No models found locales")
                print(f"\nDescarga modelos con: python {sys.argv[0]} <modelo> --download-only")
            else:
                total_size = 0
                for i, model in enumerate(models, 1):
                    print(f"\n{i}. {model['name']}")
                    print(f"   Ruta: {model['path']}")
                    print(f"    Arquitectura: {model['architecture']}")
                    print(f"   Size: {model['size_gb']:.2f} GB")
                    if model["num_layers"]:
                        print(f"    Layers: {model['num_layers']}")
                    print(
                        f"   Archivos: {model['model_files']} modelo, {model['total_files']} total"
                    )
                    total_size += model["size_gb"]

                print(f"\nTotal: {len(models)} modelos, {total_size:.2f} GB")

        return models

    def download_model(self, model_name: str, force: bool = False) -> tuple[str, dict]:
        """Descarga un modelo de HuggingFace."""
        safe_name = model_name.replace("/", "_")
        local_path = self.models_dir / safe_name

        # Verify si ya existe
        if local_path.exists() and not force:
            print(f"El modelo ya existe en: {local_path}")
            print("   (usa --force-download para re-descargar)")
            return str(local_path), self.get_model_info(str(local_path))

        print(f"\n Descargando: {model_name}")
        print(f"Destino: {local_path}")

        try:
            # Descargar usando snapshot_download
            print("⏳ Descargando archivos...")

            snapshot_download(
                repo_id=model_name,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                resume_download=True,
                tqdm_class=tqdm,
            )

            # Save información de descarga
            info_file = local_path / "download_info.txt"
            with open(info_file, "w") as f:
                f.write(f"Modelo: {model_name}\n")
                f.write(f"Fecha descarga: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Fuente: HuggingFace Hub\n")

            # Get información
            model_info = self.get_model_info(str(local_path))

            print("\nDescarga completada!")
            print(f"Size: {model_info['size_gb']:.2f} GB")
            print(f" Arquitectura: {model_info['architecture']}")

            return str(local_path), model_info

        except Exception as e:
            print(f"\nError descargando modelo: {str(e)}")
            # Limpiar descarga parcial
            if local_path.exists():
                shutil.rmtree(local_path)
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Analiza modelos de lenguaje para optimización y compresión",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Analizar modelo local
  python analyze_model.py ./models/llama-7b

  # Descargar y analizar modelo de HuggingFace
  python analyze_model.py meta-llama/Llama-2-7b-hf

  # Solo descargar sin analizar
  python analyze_model.py mistralai/Mistral-7B-v0.1 --download-only

  # Listar modelos locales
  python analyze_model.py --list

  # Analysis para caso de uso específico
  python analyze_model.py modelo --use-case rag --calibration-file texts.txt
        """,
    )

    # Argumento principal
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        help="Modelo a analizar (HuggingFace ID, ruta local, o nombre parcial)",
    )

    # Opciones de gestión de modelos
    parser.add_argument(
        "--download-only", action="store_true", help="Solo descargar el modelo sin analizar"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="Listar modelos locales disponibles"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directorio para modelos (default: ./models)",
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Forzar re-descarga aunque exista"
    )

    # Opciones de análisis
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./compression_analysis",
        help="Directorio para resultados (default: ./compression_analysis)",
    )
    parser.add_argument(
        "--calibration-file", "-c", type=str, help="Archivo con textos de calibración"
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Dispositivo para análisis (default: cuda)",
    )
    parser.add_argument(
        "--use-case",
        "-u",
        type=str,
        choices=["rag", "ner", "chatbot", "agent", "all"],
        default="all",
        help="Caso de uso específico para optimizar (default: all)",
    )

    args = parser.parse_args()

    # Create gestor de modelos
    manager = ModelManager(args.models_dir)

    # Opción: listar modelos
    if args.list:
        manager.list_local_models(detailed=True)
        return

    # Verify que se proporcionó un modelo
    if not args.model:
        # Modo interactivo si no se especifica modelo
        models = manager.list_local_models(detailed=True)

        if models:
            print("\n¿Qué modelo quieres analizar?")
            print("Ingresa el número o el nombre del modelo (o 'q' para salir): ")

            choice = input("\n> ").strip()

            if choice.lower() == "q":
                return

            try:
                # Intentar como número
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    args.model = models[idx]["path"]
                else:
                    print("Número inválido")
                    return
            except ValueError:
                # Usar como nombre
                args.model = choice
        else:
            parser.error("No models found locales. Especifica un modelo para descargar.")

    # Resolver modelo (local o remoto)
    model_path, is_local = manager.resolve_model_path(args.model)

    # Si es remoto y se pidió download-only
    if not is_local and args.download_only:
        try:
            local_path, info = manager.download_model(model_path, args.force_download)
            print("\nPara analizar este modelo, usa:")
            print(f"   python {sys.argv[0]} {local_path}")
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
        return

    # Si es remoto y no se pidió download-only, preguntar
    if not is_local:
        print(f"\n Modelo remoto detectado: {model_path}")
        print("¿Qué deseas hacer?")
        print("1. Descargar y analizar")
        print("2. Solo descargar")
        print("3. Cancelar")

        choice = input("\nOpción (1/2/3): ").strip()

        if choice == "2":
            try:
                local_path, info = manager.download_model(model_path, args.force_download)
                print("\nPara analizar este modelo, usa:")
                print(f"   python {sys.argv[0]} {local_path}")
            except Exception as e:
                print(f"Error: {str(e)}")
                sys.exit(1)
            return
        elif choice == "3":
            return
        elif choice == "1":
            # Descargar primero
            try:
                print("\n Descargando modelo antes de analizar...")
                model_path, info = manager.download_model(model_path, args.force_download)
                is_local = True
            except Exception as e:
                print(f"Error descargando: {str(e)}")
                sys.exit(1)
        else:
            print("Invalid option")
            return

    # Show información del modelo
    if is_local:
        info = manager.get_model_info(model_path)
        if info:
            print("\nInformación del modelo:")
            print(f"   Ruta: {info['path']}")
            print(f"    Arquitectura: {info['architecture']}")
            print(f"   Size: {info['size_gb']:.2f} GB")
            if info["num_layers"]:
                print(f"    Layers: {info['num_layers']}")

    # Preparar análisis
    os.makedirs(args.output_dir, exist_ok=True)

    # Load textos de calibración
    calibration_texts = None
    if args.calibration_file and os.path.exists(args.calibration_file):
        with open(args.calibration_file) as f:
            calibration_texts = [line.strip() for line in f if line.strip()]
        print(f"\nLoaded {len(calibration_texts)} textos de calibración")

    # Run análisis
    try:
        print("\nIniciando análisis...")

        # Load el modelo primero
        print(" Cargando modelo en memoria...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            device_map="auto" if args.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        # Mover a dispositivo si es necesario
        if args.device == "cpu" and hasattr(model, "to"):
            model = model.to("cpu")

        # Get estadísticas del modelo
        model_stats = {
            "model_path": model_path,
            "device": args.device,
            "num_layers": len([n for n, _ in model.named_modules() if "layer" in n]),
            "architecture": model.config.model_type if hasattr(model, "config") else "unknown",
        }

        # Create analizador con el modelo cargado
        analyzer = OptimizedModelAnalyzer(model, model_stats)

        # Analizar
        strategies = analyzer.analyze(calibration_texts, quick_mode=True, use_case=args.use_case)

        # Generate reporte
        model_name_safe = Path(model_path).name if is_local else args.model.replace("/", "_")

        # Create generador de reportes
        report_generator = OptimizedReportGenerator(args.output_dir)

        # Preparar datos para el reporte
        analysis_data = {
            "model_info": {
                "name": model_name_safe,
                "analysis_date": datetime.now().isoformat(),
                "path": model_path,
            },
            "model_stats": model_stats,
            "compression_recommendations": strategies,
            "layer_summary": _get_layer_summary(analyzer.layer_profiles),
            "complete_layer_analysis": {
                "all_layers": {
                    name: vars(profile) for name, profile in analyzer.layer_profiles.items()
                }
            },
        }

        # Generate reporte HTML
        report_path = report_generator.generate_analysis_report(
            model_name_safe, analysis_data, format="html"
        )

        report_generator.generate_analysis_report(model_name_safe, analysis_data, format="json")

        complete_report_path = args.output_dir + "/" + f"{model_name_safe}_report_complete.json"
        with open(complete_report_path, "w") as f:
            json.dump(analysis_data, f, indent=2, default=str)
        print(f"Reporte JSON completo guardado en: {complete_report_path}")

        # Show resumen
        print(_generate_summary(strategies, analyzer.model_stats))

        # Si se especificó un caso de uso, mostrar detalles
        if args.use_case != "all":
            if args.use_case in strategies:
                strategy = strategies[args.use_case]
                print(f"\nOptimización para {args.use_case.upper()}:")
                print(f"   Compresión total: {strategy.get('expected_compression', 0) * 100:.1f}%")
                print(f"   Size final: {strategy.get('final_size_mb', 0):.1f} MB")
                print(
                    f"   Performance esperado: {strategy.get('expected_performance', 0) * 100:.0f}%"
                )
            else:
                print(f"\nEstrategia '{args.use_case}' no disponible.")
                print(f"   Estrategias disponibles: {', '.join(strategies.keys())}")

        print("\nAnálisis completado!")
        print(f"Informe guardado en: {report_path}")

        # Liberar memoria del modelo
        del model
        torch.cuda.empty_cache() if args.device == "cuda" else None

    except KeyboardInterrupt:
        print("\nAnálisis interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\nError durante el análisis: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _get_layer_summary(layer_profiles):
    """Genera resumen de layers por tipo."""
    summary = {}
    total_size = 0

    for _name, profile in layer_profiles.items():
        layer_type = profile.type
        if layer_type not in summary:
            summary[layer_type] = {"count": 0, "total_size_mb": 0, "layers": []}

        summary[layer_type]["count"] += 1
        summary[layer_type]["total_size_mb"] += profile.size_mb
        total_size += profile.size_mb

    # Calculate porcentajes
    for layer_type, info in summary.items():
        info["percentage_of_model"] = (
            (info["total_size_mb"] / total_size * 100) if total_size > 0 else 0
        )
        info["average_size_mb"] = info["total_size_mb"] / info["count"] if info["count"] > 0 else 0

    return summary


def _generate_summary(strategies, model_stats):
    """Genera resumen en texto."""
    lines = [
        "\n" + "=" * 80,
        "RESUMEN DE ANÁLISIS",
        "=" * 80,
        f"Size del modelo: {model_stats.get('total_size_mb', 0):.1f} MB",
        f"Estrategia recomendada: {strategies.get('recommended', 'balanced')}",
        "",
    ]

    # Show opciones de compresión
    for strategy_name in ["conservative", "balanced", "aggressive"]:
        if strategy_name in strategies:
            strategy = strategies[strategy_name]
            compression = strategy.get("estimated_compression", 0)
            final_size = model_stats.get("total_size_mb", 0) * (1 - compression)
            lines.append(
                f"• {strategy_name.capitalize()}: "
                f"{compression * 100:.0f}% compresión → {final_size:.1f} MB"
            )

    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
