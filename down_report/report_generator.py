"""Generador optimizado de reportes para análisis y compresión de modelos."""

import base64
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class LayerStats:
    """Estadísticas de una capa."""

    name: str
    type: str
    size_mb: float
    parameters: int
    compression_ratio: float = 0.0
    methods_applied: List[str] = None
    performance_impact: float = 0.0

    def __post_init__(self):
        if self.methods_applied is None:
            self.methods_applied = []


class OptimizedReportGenerator:
    """Generador optimizado de reportes con visualizaciones."""

    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache para datos procesados
        self._cache = {}

        # Configure estilo de gráficos
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def generate_analysis_report(
        self, model_name: str, analysis_data: Dict[str, Any], format: str = "html"
    ) -> Path:
        """Genera reporte completo de análisis de modelo."""
        logger.info(f"Generando reporte de análisis para {model_name}...")

        # Process datos
        processed_data = self._process_analysis_data(analysis_data)

        # Generate visualizaciones
        visualizations = self._generate_analysis_visualizations(processed_data)

        # Create reporte según formato
        if format == "html":
            report_path = self._generate_html_report(
                model_name, processed_data, visualizations, "analysis"
            )
        elif format == "json":
            report_path = self._generate_json_report(model_name, processed_data, "analysis")
        elif format == "markdown":
            report_path = self._generate_markdown_report(model_name, processed_data, "analysis")
        else:
            raise ValueError(f"Format no soportado: {format}")

        logger.info(f"Reporte generado: {report_path}")
        return report_path

    def generate_compression_report(
        self,
        model_name: str,
        compression_data: Dict[str, Any],
        verification_data: Optional[Dict[str, Any]] = None,
        format: str = "html",
    ) -> Path:
        """Generates compression report con resultados."""
        logger.info(f"Generando reporte de compresión para {model_name}...")

        # Combinar datos si hay verificación
        if verification_data:
            compression_data["verification"] = verification_data

        # Process datos
        processed_data = self._process_compression_data(compression_data)

        # Generate visualizaciones
        visualizations = self._generate_compression_visualizations(processed_data)

        # Create reporte
        if format == "html":
            report_path = self._generate_html_report(
                model_name, processed_data, visualizations, "compression"
            )
        elif format == "json":
            report_path = self._generate_json_report(model_name, processed_data, "compression")
        elif format == "markdown":
            report_path = self._generate_markdown_report(model_name, processed_data, "compression")
        else:
            raise ValueError(f"Format no soportado: {format}")

        logger.info(f"Reporte generado: {report_path}")
        return report_path

    def _process_analysis_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa datos de análisis para el reporte."""
        processed = {
            "metadata": {
                "model_name": data.get("model_info", {}).get("name", "Unknown"),
                "analysis_date": data.get("model_info", {}).get(
                    "analysis_date", datetime.now().isoformat()
                ),
                "analysis_time": data.get("model_stats", {}).get("analysis_time", 0),
            },
            "model_stats": data.get("model_stats", {}),
            "layer_summary": self._process_layer_summary(data.get("layer_summary", {})),
            "recommendations": data.get("compression_recommendations", {}),
            "layer_details": self._process_layer_details(data.get("complete_layer_analysis", {})),
        }

        # Calculate estadísticas adicionales
        processed["statistics"] = self._calculate_statistics(processed)

        return processed

    def _process_compression_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa datos de compresión para el reporte."""
        processed = {
            "metadata": {
                "compression_date": data.get("compression_date", datetime.now().isoformat()),
                "original_model": data.get("original_model", "Unknown"),
                "compressed_model": data.get("compressed_model", "Unknown"),
            },
            "compression_config": data.get("compression_config", {}),
            "statistics": data.get("statistics", {}),
            "layer_changes": self._process_layer_changes(data),
            "performance": self._process_performance_data(data.get("verification", {})),
        }

        # Calculate estadísticas adicionales
        processed["statistics"] = self._calculate_statistics(processed)

        # AGREGAR: Incluir información original del modelo
        processed["model_info"] = data.get("model_info", {})

        return processed

    def _process_layer_summary(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Procesa resumen de layers para visualización."""
        processed = []

        for layer_type, info in summary.items():
            processed.append(
                {
                    "type": layer_type,
                    "count": info.get("count", 0),
                    "size_mb": info.get("total_size_mb", 0),
                    "percentage": info.get("percentage_of_model", 0),
                    "avg_size_mb": info.get("average_size_mb", 0),
                }
            )

        # Ordenar por tamaño
        processed.sort(key=lambda x: x["size_mb"], reverse=True)
        return processed

    def _process_layer_details(self, layer_data: Dict[str, Any]) -> pd.DataFrame:
        """Procesa detalles de layers en DataFrame."""
        all_layers = layer_data.get("all_layers", {})

        if not all_layers:
            return pd.DataFrame()

        # Convertir a lista de diccionarios
        layer_list = []
        for name, info in all_layers.items():
            layer_info = {
                "name": name,
                "type": info.get("type", "unknown"),
                "size_mb": info.get("size_mb", 0),
                "parameters": info.get("parameters", 0),
                "position": info.get("relative_position", 0),
                "compression_potential": info.get("metrics", {}).get("compression_potential", 0),
            }
            layer_list.append(layer_info)

        return pd.DataFrame(layer_list)

    def _process_layer_changes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa cambios en las layers por compresión."""
        stats = data.get("statistics", {})
        data.get("compression_config", {})

        changes = {
            "total_layers": stats.get("layers_compressed", 0) + stats.get("layers_preserved", 0),
            "compressed_layers": stats.get("layers_compressed", 0),
            "preserved_layers": stats.get("layers_preserved", 0),
            "methods_used": stats.get("methods_used", []),
            "size_reduction": {
                "original_mb": stats.get("original_size_mb", 0),
                "compressed_mb": stats.get("compressed_size_mb", 0),
                "reduction_mb": stats.get("original_size_mb", 0)
                - stats.get("compressed_size_mb", 0),
                "reduction_percent": stats.get("compression_ratio", 0) * 100,
            },
        }

        return changes

    def _process_performance_data(self, verification: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa datos de rendimiento de la verificación."""
        if not verification:
            return {}

        results = verification.get("results", {})
        perf_metrics = results.get("performance_metrics", {})

        return {
            "inference_speedup": perf_metrics.get("speedup", 1.0),
            "output_difference": perf_metrics.get("avg_output_difference", 0),
            "quality_assessment": results.get("compression_quality", {}).get("assessment", "N/A"),
            "original_time": perf_metrics.get("original_inference_time", 0),
            "compressed_time": perf_metrics.get("compressed_inference_time", 0),
        }

    def _calculate_statistics(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula estadísticas adicionales."""
        stats = {}

        # Estadísticas del modelo
        model_stats = processed_data.get("model_stats", {})
        stats["total_parameters"] = model_stats.get("total_parameters", 0)
        stats["total_size_gb"] = model_stats.get("total_size_mb", 0) / 1024

        # Distribución por tipo de capa
        layer_summary = processed_data.get("layer_summary", [])
        if layer_summary:
            stats["layer_distribution"] = {
                item["type"]: item["percentage"] for item in layer_summary
            }

        return stats

    def _generate_analysis_visualizations(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Genera visualizaciones para análisis."""
        vizs = {}

        # 1. Distribución de tamaños por tipo de capa
        layer_summary = data.get("layer_summary", [])
        if layer_summary:
            fig, ax = plt.subplots(figsize=(10, 6))

            types = [item["type"] for item in layer_summary]
            sizes = [item["size_mb"] for item in layer_summary]

            ax.bar(types, sizes)
            ax.set_xlabel("Type de Layer")
            ax.set_ylabel("Size (MB)")
            ax.set_title("Distribución de Sizes por Type de Layer")
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save como base64
            import base64
            from io import BytesIO

            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            vizs["layer_distribution"] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        # 2. Mapa de calor de potencial de compresión
        layer_df = data.get("layer_details", pd.DataFrame())
        if not layer_df.empty and "compression_potential" in layer_df.columns:
            # Create matriz para mapa de calor
            fig, ax = plt.subplots(figsize=(12, 8))

            # Agrupar por tipo y posición
            pivot = layer_df.pivot_table(
                values="compression_potential",
                index="type",
                columns=pd.cut(layer_df["position"], bins=10),
                aggfunc="mean",
            )

            sns.heatmap(pivot, cmap="YlOrRd", cbar_kws={"label": "Potencial de Compresión"})
            ax.set_xlabel("Posición Relativa en el Modelo")
            ax.set_ylabel("Type de Layer")
            ax.set_title("Mapa de Potencial de Compresión")
            plt.tight_layout()

            # Save
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            vizs["compression_heatmap"] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        return vizs

    def _generate_compression_visualizations(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Genera visualizaciones para compresión."""
        vizs = {}

        # 1. Comparación de tamaños antes/después
        changes = data.get("layer_changes", {})
        size_data = changes.get("size_reduction", {})

        if size_data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Gráfico de barras
            sizes = [size_data["original_mb"], size_data["compressed_mb"]]
            labels = ["Original", "Comprimido"]
            colors = ["#3498db", "#e74c3c"]

            ax1.bar(labels, sizes, color=colors)
            ax1.set_ylabel("Size (MB)")
            ax1.set_title("Comparación de Sizes")

            # Añadir texto con porcentaje
            reduction = size_data["reduction_percent"]
            ax1.text(
                0.5,
                max(sizes) * 0.5,
                f"-{reduction:.1f}%",
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
            )

            # Gráfico circular de métodos usados
            methods = changes.get("methods_used", [])
            if methods:
                method_counts = pd.Series(methods).value_counts()
                ax2.pie(method_counts.values, labels=method_counts.index, autopct="%1.1f%%")
                ax2.set_title("Methods de Compresión Utilizados")

            plt.tight_layout()

            # Save
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            vizs["compression_summary"] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        # 2. Métricas de rendimiento
        perf = data.get("performance", {})
        if perf and perf.get("inference_speedup"):
            fig, ax = plt.subplots(figsize=(8, 6))

            metrics = {
                "Aceleración": perf["inference_speedup"],
                "Calidad": 1 - perf.get("output_difference", 0),
                "Compresión": changes["size_reduction"]["reduction_percent"] / 100,
            }

            # Gráfico de radar
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            values = list(metrics.values())
            values += values[:1]  # Cerrar el polígono
            angles += angles[:1]

            ax = plt.subplot(111, projection="polar")
            ax.plot(angles, values, "o-", linewidth=2)
            ax.fill(angles, values, alpha=0.3)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics.keys())
            ax.set_ylim(0, max(values) * 1.2)
            ax.set_title("Métricas de Rendimiento")

            plt.tight_layout()

            # Save
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            vizs["performance_metrics"] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        return vizs

    def _generate_html_report(
        self,
        model_name: str,
        data: Dict[str, Any],
        visualizations: Dict[str, str],
        report_type: str,
    ) -> Path:
        """Genera reporte HTML interactivo."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{model_name}_{report_type}_report_{timestamp}.html"
        report_path = self.output_dir / report_name

        # Template HTML
        html_template = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .section {{
            background: white;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 1rem;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 5px;
            text-align: center;
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .viz-container {{
            text-align: center;
            margin: 2rem 0;
        }}
        .viz-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .danger {{ color: #dc3545; }}
        .info {{ color: #17a2b8; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Generado el {date}</p>
    </div>

    {content}

    <div class="section">
        <p style="text-align: center; color: #666;">
            Reporte generado por Model Compression Toolkit<br>
            © 2024 - Optimizado para rendimiento
        </p>
    </div>
</body>
</html>
        """

        # Generate contenido según tipo
        if report_type == "analysis":
            content = self._generate_analysis_html_content(data, visualizations)
            title = f"Análisis de Modelo: {model_name}"
        else:
            content = self._generate_compression_html_content(data, visualizations)
            title = f"Reporte de Compresión: {model_name}"

        # Renderizar HTML
        html = html_template.format(
            title=title, date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), content=content
        )

        # Save
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        return report_path

    def _generate_analysis_html_content(
        self, data: Dict[str, Any], visualizations: Dict[str, str]
    ) -> str:
        """Genera contenido HTML para reporte de análisis."""
        sections = []

        # Resumen ejecutivo
        stats = data.get("statistics", {})
        sections.append(f"""
        <div class="section">
            <h2>Resumen Ejecutivo</h2>
            <div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
                <div class="metric">
                    <div class="metric-value">{stats.get("total_parameters", 0):,}</div>
                    <div class="metric-label">Parameters Totales</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stats.get("total_size_gb", 0):.2f} GB</div>
                    <div class="metric-label">Size del Modelo</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{data.get("metadata", {}).get("analysis_time", 0):.1f}s</div>
                    <div class="metric-label">Tiempo de Análisis</div>
                </div>
            </div>
        </div>
        """)

        # Distribución de layers
        if "layer_distribution" in visualizations:
            sections.append(f"""
            <div class="section">
                <h2>Distribución de Layers</h2>
                <div class="viz-container">
                    <img src="data:image/png;base64,{visualizations["layer_distribution"]}"
                         alt="Distribución de layers">
                </div>
            </div>
            """)

        # Recommendations
        recs = data.get("recommendations", {})
        if recs:
            sections.append(f"""
            <div class="section">
                <h2>Recommendations de Compresión</h2>
                <p><strong>Profile sugerido:</strong>
                   <span class="info">{recs.get("suggested_profile", "balanced")}</span></p>
                <p><strong>Compresión estimada posible:</strong>
                   <span class="success">{recs.get("estimated_compression", 0):.1f} MB</span></p>
                <p><strong>Layers compresibles:</strong> {len(recs.get("compressible_layers", []))}</p>
                <p><strong>Layers a preservar:</strong> {len(recs.get("preserve_layers", []))}</p>
            </div>
            """)

        return "\n".join(sections)

    def _generate_compression_html_content(
        self, data: Dict[str, Any], visualizations: Dict[str, str]
    ) -> str:
        """Genera contenido HTML para reporte de compresión."""
        sections = []

        # Compression results
        changes = data.get("layer_changes", {})
        size_red = changes.get("size_reduction", {})

        sections.append(f"""
        <div class="section">
            <h2>Resultados de Compresión</h2>
            <div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
                <div class="metric">
                    <div class="metric-value">{size_red.get("reduction_percent", 0):.1f}%</div>
                    <div class="metric-label">Reducción Total</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{size_red.get("original_mb", 0):.1f} MB</div>
                    <div class="metric-label">Size Original</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{size_red.get("compressed_mb", 0):.1f} MB</div>
                    <div class="metric-label">Size Comprimido</div>
                </div>
            </div>
        </div>
        """)

        # Visualizaciones
        if "compression_summary" in visualizations:
            sections.append(f"""
            <div class="section">
                <h2>Resumen Visual</h2>
                <div class="viz-container">
                    <img src="data:image/png;base64,{visualizations["compression_summary"]}"
                         alt="Resumen de compresión">
                </div>
            </div>
            """)

        # Métricas de rendimiento
        perf = data.get("performance", {})
        if perf:
            quality_class = "success" if perf.get("output_difference", 1) < 0.05 else "warning"
            sections.append(f"""
            <div class="section">
                <h2>Rendimiento</h2>
                <div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
                    <div class="metric">
                        <div class="metric-value">{perf.get("inference_speedup", 1):.2f}x</div>
                        <div class="metric-label">Aceleración</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value {quality_class}">{perf.get("quality_assessment", "N/A")}</div>
                        <div class="metric-label">Calidad</div>
                    </div>
                </div>
            </div>
            """)

        return "\n".join(sections)

    def _generate_json_report(
        self, model_name: str, data: Dict[str, Any], report_type: str
    ) -> Path:
        """Genera reporte en formato JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{model_name}_{report_type}_report_{timestamp}.json"
        report_path = self.output_dir / report_name

        # Asegurar que todos los datos necesarios estén presentes
        if report_type == "analysis":
            # Enriquecer datos para compression_config
            data = self._enrich_analysis_data_for_compression(data)

        # Convertir DataFrames a diccionarios
        if "layer_details" in data and hasattr(data["layer_details"], "to_dict"):
            data["layer_details"] = data["layer_details"].to_dict("records")

        # Save JSON
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        return report_path

    def _enrich_analysis_data_for_compression(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enriquece los datos de análisis para ser usados por compression_config."""
        # Asegurar que complete_layer_analysis existe y tiene el formato correcto
        if "complete_layer_analysis" not in data:
            data["complete_layer_analysis"] = {"all_layers": {}}

        # Si tenemos layer_details como DataFrame, convertirlo
        if "layer_details" in data and hasattr(data["layer_details"], "iterrows"):
            all_layers = {}
            for _, row in data["layer_details"].iterrows():
                layer_name = row["name"]
                all_layers[layer_name] = {
                    "name": layer_name,
                    "type": row.get("type", "unknown"),
                    "size_mb": row.get("size_mb", 0),
                    "parameters": row.get("parameters", 0),
                    "relative_position": row.get("position", 0),
                    "compression_potential": row.get("compression_potential", 0),
                    "layer_index": int(row.get("position", 0) * 100),  # Aproximación
                }
            data["complete_layer_analysis"]["all_layers"] = all_layers

        return data

    def _generate_markdown_report(
        self, model_name: str, data: Dict[str, Any], report_type: str
    ) -> Path:
        """Genera reporte en formato Markdown."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{model_name}_{report_type}_report_{timestamp}.md"
        report_path = self.output_dir / report_name

        # Generate contenido Markdown
        md_lines = [
            f"# Report de {report_type.title()}: {model_name}",
            f"\n**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n---\n",
        ]

        if report_type == "analysis":
            md_lines.extend(self._generate_analysis_markdown(data))
        else:
            md_lines.extend(self._generate_compression_markdown(data))

        # Save
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        return report_path

    def _generate_analysis_markdown(self, data: Dict[str, Any]) -> List[str]:
        """Genera contenido Markdown para análisis."""
        lines = []

        # Estadísticas
        stats = data.get("statistics", {})
        lines.extend(
            [
                "## Estadísticas del Modelo",
                f"- **Parameters totales:** {stats.get('total_parameters', 0):,}",
                f"- **Size:** {stats.get('total_size_gb', 0):.2f} GB",
                f"- **Tiempo de análisis:** {data.get('metadata', {}).get('analysis_time', 0):.1f}s",
                "",
            ]
        )

        # Distribución de layers
        lines.append("## Distribución por Type de Layer\n")
        lines.append("| Type | Layers | Size (MB) | % del Modelo |")
        lines.append("|------|-------|-------------|--------------|")

        for item in data.get("layer_summary", []):
            lines.append(
                f"| {item['type']} | {item['count']} | "
                f"{item['size_mb']:.1f} | {item['percentage']:.1f}% |"
            )

        lines.append("")

        # Recommendations
        recs = data.get("recommendations", {})
        if recs:
            lines.extend(
                [
                    "## Recommendations",
                    f"- **Profile sugerido:** {recs.get('suggested_profile', 'balanced')}",
                    f"- **Compresión estimada:** {recs.get('estimated_compression', 0):.1f} MB",
                    f"- **Layers compresibles:** {len(recs.get('compressible_layers', []))}",
                    f"- **Layers a preservar:** {len(recs.get('preserve_layers', []))}",
                    "",
                ]
            )

        return lines

    def _generate_compression_markdown(self, data: Dict[str, Any]) -> List[str]:
        """Genera contenido Markdown para compresión."""
        lines = []

        # Results
        changes = data.get("layer_changes", {})
        size_red = changes.get("size_reduction", {})

        lines.extend(
            [
                "## Results de Compresión",
                f"- **Reducción total:** {size_red.get('reduction_percent', 0):.1f}%",
                f"- **Size original:** {size_red.get('original_mb', 0):.1f} MB",
                f"- **Size comprimido:** {size_red.get('compressed_mb', 0):.1f} MB",
                f"- **Compressed layers:** {changes.get('compressed_layers', 0)}",
                f"- **Preserved layers:** {changes.get('preserved_layers', 0)}",
                "",
            ]
        )

        # Methods utilizados
        methods = changes.get("methods_used", [])
        if methods:
            lines.extend(
                [
                    "## Methods de Compresión Utilizados",
                    *[f"- {method}" for method in set(methods)],
                    "",
                ]
            )

        # Rendimiento
        perf = data.get("performance", {})
        if perf:
            lines.extend(
                [
                    "## Métricas de Rendimiento",
                    f"- **Aceleración:** {perf.get('inference_speedup', 1):.2f}x",
                    f"- **Calidad:** {perf.get('quality_assessment', 'N/A')}",
                    f"- **Diferencia en outputs:** {perf.get('output_difference', 0):.6f}",
                    "",
                ]
            )

        return lines

    def generate_comparison_report(
        self, models: List[Dict[str, Any]], output_name: str = "model_comparison"
    ) -> Path:
        """Genera reporte comparativo de múltiples modelos."""
        logger.info(f"Generando reporte comparativo de {len(models)} modelos...")

        # Create DataFrame comparativo
        comparison_df = pd.DataFrame(models)

        # Generate visualizaciones comparativas
        vizs = self._generate_comparison_visualizations(comparison_df)

        # Create reporte HTML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{output_name}_{timestamp}.html"
        report_path = self.output_dir / report_name

        # Generate contenido
        self._generate_comparison_html_content(comparison_df, vizs)

        # Template HTML (reusar el anterior)
        self._generate_html_report(
            "Comparación de Modelos",
            {"comparison_data": comparison_df.to_dict("records")},
            vizs,
            "comparison",
        )

        logger.info(f"Reporte comparativo generado: {report_path}")
        return report_path

    def _generate_comparison_visualizations(self, df: pd.DataFrame) -> Dict[str, str]:
        """Genera visualizaciones para comparación de modelos."""
        vizs = {}

        # Implementar visualizaciones comparativas
        # (Similar a las anteriores pero comparando múltiples modelos)

        return vizs


# Funciones de utilidad
def create_report_generator(output_dir: str = "./reports") -> OptimizedReportGenerator:
    """Crea una instancia del generador de reportes."""
    return OptimizedReportGenerator(output_dir)


def generate_quick_summary(data: Dict[str, Any]) -> str:
    """Genera un resumen rápido en texto plano."""
    summary_lines = [
        "=== RESUMEN RÁPIDO ===",
        f"Modelo: {data.get('model_name', 'Unknown')}",
        f"Size: {data.get('size_mb', 0):.1f} MB",
        f"Compresión lograda: {data.get('compression_ratio', 0) * 100:.1f}%",
        "====================",
    ]
    return "\n".join(summary_lines)
