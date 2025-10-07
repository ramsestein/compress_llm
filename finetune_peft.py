#!/usr/bin/env python3
"""
Script interactivo para fine-tuning con múltiples métodos PEFT
Soporta: LoRA, MoLoRA, GaLore, DoRA, BitFit, IA³, Prompt Tuning, Adapter, QLoRA
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
from datetime import datetime
import logging
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.columns import Columns
from rich import print as rprint

# Importar configuraciones PEFT
from LoRa_train.peft_methods_config import (
    PEFTMethod, BasePEFTConfig, LoRAConfig, MoLoRAConfig, GaLoreConfig,
    DoRAConfig, BitFitConfig, IA3Config, PromptTuningConfig,
    AdapterConfig, QLoRAConfig, PEFTPresets, get_config_by_name
)

# Importar componentes existentes
from LoRa_train.dataset_manager import OptimizedDatasetManager, DatasetConfig
from LoRa_train.lora_trainer import LoRATrainer  # Adaptaremos esto

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

class PEFTFineTuneWizard:
    """Asistente interactivo para fine-tuning con métodos PEFT"""
    
    # Descripciones de métodos
    METHOD_DESCRIPTIONS = {
        PEFTMethod.LORA: "🎯 LoRA - Low-Rank Adaptation\n• Eficiente y probado\n• 0.1-1% parámetros\n• Ideal para: Uso general",
        PEFTMethod.MOLORA: "🎭 MoLoRA - Mixture of LoRAs\n• Múltiples expertos\n• Multi-tarea/dominio\n• Ideal para: Modelos versátiles",
        PEFTMethod.GALORE: "🚀 GaLore - Gradient Low-Rank\n• Proyección de gradientes\n• Memoria ultra-baja\n• Ideal para: GPUs limitadas",
        PEFTMethod.DORA: "🎯 DoRA - Decomposed LoRA\n• Magnitud + Dirección\n• Mejor que LoRA\n• Ideal para: Máxima calidad",
        PEFTMethod.BITFIT: "💡 BitFit - Bias Tuning\n• Solo bias (~0.1%)\n• Súper eficiente\n• Ideal para: Ajustes sutiles",
        PEFTMethod.IA3: "⚡ IA³ - Infused Adapter\n• Vectores de escala\n• 10x menos que LoRA\n• Ideal para: Máxima eficiencia",
        PEFTMethod.PROMPT_TUNING: "📝 Prompt Tuning\n• Tokens virtuales\n• < 0.01% parámetros\n• Ideal para: Modelos enormes",
        PEFTMethod.ADAPTER: "🧩 Adapter Tuning\n• Módulos bottleneck\n• Más expresivo\n• Ideal para: Cambios grandes",
        PEFTMethod.QLORA: "🔥 QLoRA - Quantized LoRA\n• LoRA + 4-bit\n• 10x menos memoria\n• Ideal para: Modelos 65B+",
        PEFTMethod.COMPACTER: "🔧 Compacter - Compressed Adapters\n• Adapters comprimidos\n• Menos parámetros\n• Ideal para: Eficiencia extrema",
        PEFTMethod.KRONA: "🔺 KronA - Kronecker Adapters\n• Descomposición Kronecker\n• Muy eficiente\n• Ideal para: Modelos grandes",
        PEFTMethod.S4: "🔄 S4 - Structured State Space\n• Estado estructurado\n• Secuencias largas\n• Ideal para: Procesamiento secuencial",
        PEFTMethod.HOULSBY: "🏗️ Houlsby - Adapter Layers\n• Capas adaptadoras\n• Arquitectura estándar\n• Ideal para: Casos generales"
    }
    
    def __init__(self, models_dir: str = "./models", datasets_dir: str = "./datasets",
                 output_dir: str = "./finetuned_models"):
        self.models_dir = Path(models_dir)
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_manager = OptimizedDatasetManager(datasets_dir)
        self.selected_datasets = []
        self.model_name = None
        self.peft_method = None
        self.peft_config = None
    
    def run(self):
        """Ejecuta el asistente interactivo"""
        console.clear()
        self._show_welcome()
        
        try:
            # Paso 1: Seleccionar método PEFT
            self.peft_method = self._select_peft_method()
            
            # Paso 2: Seleccionar modelo
            self.model_name = self._select_model()
            
            # Paso 3: Seleccionar tipo de aprendizaje
            self.learning_type = self._select_learning_type()
            
            # Paso 4: Seleccionar datasets
            self.selected_datasets = self._select_and_configure_datasets()
            
            # Paso 5: Configurar método PEFT
            self.peft_config = self._configure_peft_method()
            
            # Paso 5: Revisar y confirmar
            if self._review_configuration():
                # Paso 6: Ejecutar entrenamiento
                self._run_training()
            else:
                console.print("[yellow]Entrenamiento cancelado[/yellow]")
                
        except KeyboardInterrupt:
            console.print("\n[red]Proceso interrumpido[/red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]")
            logger.exception("Error en el asistente")
            sys.exit(1)
    
    def _show_welcome(self):
        """Muestra pantalla de bienvenida"""
        welcome_text = """
        [bold cyan]🚀 Asistente Universal de Fine-Tuning PEFT[/bold cyan]
        
        Este asistente soporta múltiples métodos de fine-tuning eficiente:
        • LoRA, QLoRA, DoRA, MoLoRA
        • BitFit, IA³, Prompt Tuning, Adapter Tuning
        • GaLore (gradient projection)
        
        Te guiará paso a paso para configurar y entrenar tu modelo.
        """
        
        panel = Panel(welcome_text, title="Bienvenida", border_style="cyan")
        console.print(panel)
        console.print()
    
    def _select_peft_method(self) -> PEFTMethod:
        """Selecciona el método PEFT a utilizar"""
        console.print("[bold]🎯 Selecciona el método de fine-tuning:[/bold]\n")
        
        # Crear tabla de métodos
        table = Table(title="Métodos PEFT Disponibles", show_lines=True)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Método", style="green", width=20)
        table.add_column("Descripción", style="white")
        table.add_column("Params", style="yellow", width=10)
        
        methods = list(PEFTMethod)
        for i, method in enumerate(methods, 1):
            desc_lines = self.METHOD_DESCRIPTIONS[method].split('\n')
            if len(desc_lines) > 0 and ' - ' in desc_lines[0]:
                main_desc = desc_lines[0].split(' - ')[1]
            else:
                main_desc = method.value.upper()
            
            if len(desc_lines) > 1:
                params = desc_lines[1].replace('• ', '')
            else:
                params = "N/A"
            
            table.add_row(
                str(i),
                method.value.upper(),
                main_desc,
                params
            )
        
        console.print(table)
        
        # Recomendaciones
        console.print("\n[dim]Recomendaciones:[/dim]")
        console.print("[dim]• Nuevo en PEFT? → LoRA (1)[/dim]")
        console.print("[dim]• Memoria limitada? → BitFit (6) o QLoRA (10)[/dim]")
        console.print("[dim]• Máxima calidad? → DoRA (4) o MoLoRA (2)[/dim]")
        
        while True:
            choice = IntPrompt.ask("\nSelecciona método (1-10)", default=1)
            if 1 <= choice <= len(methods):
                return methods[choice - 1]
            console.print("[red]Opción inválida[/red]")
    
    def _select_model(self) -> str:
        """Selecciona el modelo base"""
        console.print("\n[bold]📦 Selecciona el modelo base:[/bold]\n")
        
        # Listar modelos disponibles
        models = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                size_gb = sum(
                    f.stat().st_size for f in model_dir.rglob('*') if f.is_file()
                ) / (1024**3)
                
                models.append({
                    'name': model_dir.name,
                    'path': model_dir,
                    'size_gb': size_gb
                })
        
        if not models:
            raise ValueError("No se encontraron modelos en el directorio")
        
        # Mostrar tabla
        table = Table(title="Modelos Disponibles")
        table.add_column("#", style="cyan")
        table.add_column("Nombre", style="green")
        table.add_column("Tamaño", justify="right")
        
        for i, model in enumerate(models, 1):
            table.add_row(str(i), model['name'], f"{model['size_gb']:.1f} GB")
        
        console.print(table)
        
        # Verificar compatibilidad con método seleccionado
        if self.peft_method == PEFTMethod.QLORA:
            console.print("\n[yellow]⚠️ QLoRA requiere GPU con soporte INT4[/yellow]")
        elif self.peft_method == PEFTMethod.GALORE:
            console.print("\n[yellow]⚠️ GaLore requiere modificaciones en el optimizer[/yellow]")
        
        while True:
            choice = IntPrompt.ask("\nSelecciona modelo", default=1)
            if 1 <= choice <= len(models):
                return models[choice - 1]['name']
            console.print("[red]Opción inválida[/red]")
    
    def _select_learning_type(self) -> str:
        """Selecciona el tipo de aprendizaje"""
        console.print("\n[bold]🎯 Tipo de aprendizaje:[/bold]")
        console.print("  [1] Supervisado (input → output)")
        console.print("  [2] Instrucción (instruction → response)")
        console.print("  [3] Traducción (source → target)")
        console.print("  [4] Personalizado")
        
        while True:
            choice = IntPrompt.ask("Tipo de aprendizaje", default=1)
            if choice == 1:
                return "supervised"
            elif choice == 2:
                return "instruction"
            elif choice == 3:
                return "translation"
            elif choice == 4:
                return "custom"
            else:
                console.print("[red]Opción inválida[/red]")
    
    def _select_and_configure_datasets(self) -> List[DatasetConfig]:
        """Selecciona y configura datasets"""
        console.print("\n[bold]📊 Configuración de datasets:[/bold]\n")
        
        available = self.dataset_manager.scan_datasets(use_cache=False)
        
        if not available:
            raise ValueError("No se encontraron datasets")
        
        # Mostrar tabla
        table = Table(title="Datasets Disponibles")
        table.add_column("#", style="cyan")
        table.add_column("Nombre", style="green")
        table.add_column("Formato")
        table.add_column("Registros", justify="right")
        
        for i, dataset in enumerate(available, 1):
            # Get size from various possible fields
            size = dataset.get('size', dataset.get('estimated_rows', dataset.get('num_rows', 'N/A')))
            table.add_row(
                str(i),
                dataset['name'],
                dataset['format'].upper(),
                str(size)
            )
        
        console.print(table)
        
        # Seleccionar
        selected = []
        console.print("\n[dim]Selecciona datasets (vacío para terminar)[/dim]")
        
        while True:
            choice = Prompt.ask("Dataset #", default="")
            if not choice:
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    if self.learning_type == "supervised":
                        config = self._configure_dataset_with_columns(available[idx])
                    else:
                        config = self._configure_dataset_simple(available[idx])
                    
                    if config:
                        selected.append(config)
                        console.print(f"[green]✓ Agregado[/green]")
            except:
                console.print("[red]Entrada inválida[/red]")
        
        if not selected:
            raise ValueError("Debes seleccionar al menos un dataset")
        
        return selected
    
    def _configure_dataset_simple(self, dataset_info: Dict[str, Any]) -> Optional[DatasetConfig]:
        """Configura dataset para aprendizaje no supervisado"""
        console.print(f"\n[bold]Configurando: {dataset_info['name']}[/bold]")
        
        # Crear configuración simple
        from LoRa_train.dataset_manager import DatasetConfig
        from pathlib import Path
        
        config = DatasetConfig(
            file_path=Path(dataset_info['file_path']),
            format=dataset_info['format'],
            columns={},  # Sin mapeo de columnas específico
            name=dataset_info.get('name', 'dataset'),
            size=dataset_info.get('size', 0),
            instruction_template="{text}",  # Template simple
            dataset_type=self.learning_type,
            delimiter=dataset_info.get('delimiter', ',')
        )
        
        console.print(f"[green]✓ Configurado para {self.learning_type}[/green]")
        return config
    
    def _configure_dataset_with_columns(self, dataset_info: Dict[str, Any]) -> Optional[DatasetConfig]:
        """Configura dataset con entrada explícita de columnas"""
        console.print(f"\n[bold]📋 Configurando: {dataset_info['name']}[/bold]")
        
        # Detectar formato automáticamente
        try:
            import pandas as pd
            
            # Analizar el archivo para detectar formato
            df, detected_format = self._auto_detect_format(dataset_info['file_path'])
            
            if df is not None:
                console.print(f"[green]✓ Formato detectado automáticamente: {detected_format}[/green]")
                console.print(f"[green]✓ Columnas detectadas: {list(df.columns)}[/green]")
                console.print(f"[dim]Muestra de datos (primeras 2 filas):[/dim]")
                console.print(df.head(2).to_string())
            else:
                console.print(f"[yellow]No se pudo detectar el formato del dataset[/yellow]")
                return None
                
        except Exception as e:
            console.print(f"[yellow]Error analizando dataset: {e}[/yellow]")
            return None
        
        # Detectar automáticamente las columnas de entrada y salida
        console.print("\n[bold]🤖 Detección automática de columnas:[/bold]")
        
        # Buscar columnas comunes para traducción
        input_col = None
        output_col = None
        
        # Buscar columnas que contengan "catalan", "spanish", "source", "input"
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['catalan', 'spanish', 'source', 'input', 'text']):
                input_col = col
                break
        
        # Buscar columnas que contengan "chino", "chinese", "target", "output"
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['chino', 'chinese', 'target', 'output', 'translation']):
                output_col = col
                break
        
        # Si no se encontraron, usar las primeras dos columnas (excluyendo 'linea')
        if not input_col or not output_col:
            data_columns = [col for col in df.columns if col.lower() != 'linea']
            if len(data_columns) >= 2:
                input_col = data_columns[0]
                output_col = data_columns[1]
            else:
                console.print(f"[red]No se pudieron detectar columnas válidas[/red]")
                return None
        
        console.print(f"[green]✓ Columna de entrada detectada: {input_col}[/green]")
        console.print(f"[green]✓ Columna de salida detectada: {output_col}[/green]")
        
        # Detectar automáticamente el tipo de template basado en las columnas
        if 'catalan' in input_col.lower() and 'chino' in output_col.lower():
            # Template de traducción
            instruction_template = "Translate from Catalan to Chinese:\n{input}\n{output}"
            console.print(f"[green]✓ Template detectado: Traducción (Catalan → Chinese)[/green]")
        else:
            # Template simple
            instruction_template = "{input}\n{output}"
            console.print(f"[green]✓ Template detectado: Simple[/green]")
        
        # Limpiar nombres de columnas (remover sufijos como ;;;)
        input_col = input_col.strip().strip('"').rstrip(';;;')
        output_col = output_col.strip().strip('"').rstrip(';;;')
        
        console.print(f"[green]✓ Configuración final: {input_col} → {output_col}[/green]")
        
        # Crear mapeo de columnas
        columns_map = {input_col: "input", output_col: "output"}
        
        # Crear configuración
        from LoRa_train.dataset_manager import DatasetConfig
        from pathlib import Path
        
        # Determinar el separador detectado
        detected_sep = ','
        if ';;;' in detected_format:
            detected_sep = ';;;'
        elif 'Tab-separated' in detected_format:
            detected_sep = '\t'
        elif 'Pipe-separated' in detected_format:
            detected_sep = '|'
        
        config = DatasetConfig(
            file_path=Path(dataset_info['file_path']),
            format='csv',  # Siempre CSV, pero con separador personalizado
            columns=columns_map,
            name=dataset_info.get('name', 'dataset'),
            size=dataset_info.get('size', 0),
            instruction_template=instruction_template,
            dataset_type="supervised",
            delimiter=detected_sep  # Usar el separador detectado
        )
        
        # Añadir información de columnas para el procesamiento
        config.input_column = input_col
        config.output_column = output_col
        config.detected_format = detected_format
        
        console.print(f"[green]✓ Configurado: {input_col} → {output_col}[/green]")
        return config
    
    def _auto_detect_format(self, file_path: str) -> tuple:
        """Detecta automáticamente el formato del dataset"""
        import pandas as pd
        import re
        
        # Leer las primeras líneas para análisis
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_lines = [f.readline().strip() for _ in range(5)]
        
        # Analizar la primera línea (header)
        header = first_lines[0]
        console.print(f"[blue]Header detectado: {repr(header)}[/blue]")
        
        # Detectar separadores comunes
        separators = [',', ';', '\t', '|', ';;;', '|||']
        best_sep = None
        best_score = 0
        
        for sep in separators:
            parts = header.split(sep)
            if len(parts) > 1:
                # Puntuar basado en número de columnas y consistencia
                score = len(parts)
                
                # Verificar consistencia en las siguientes líneas (datos)
                consistent_lines = 0
                for line in first_lines[1:]:
                    if line and len(line.split(sep)) == len(parts):
                        consistent_lines += 1
                
                score += consistent_lines * 0.5
                
                if score > best_score:
                    best_score = score
                    best_sep = sep
        
        if not best_sep:
            return None, "No se pudo detectar separador"
        
        console.print(f"[blue]Mejor separador detectado: {repr(best_sep)}[/blue]")
        
        # Intentar cargar con el mejor separador
        try:
            if best_sep in [';;;', '|||']:
                # Para separadores personalizados, necesitamos un enfoque especial
                # Leer el archivo línea por línea y procesar manualmente
                df = self._parse_custom_csv(file_path, best_sep, nrows=5)
            else:
                # Separadores estándar
                df = pd.read_csv(file_path, sep=best_sep, nrows=5, on_bad_lines='skip')
            
            # Limpiar nombres de columnas
            df.columns = [col.strip().strip('"').rstrip(';;;') for col in df.columns]
            console.print(f"[blue]Columnas detectadas: {list(df.columns)}[/blue]")
            
            # Verificar que tenemos columnas válidas (no solo números de línea)
            valid_columns = []
            for col in df.columns:
                # Si la columna no es solo números, es probablemente una columna de datos
                if not str(col).strip().isdigit() and str(col).strip() != 'linea':
                    valid_columns.append(col)
            
            console.print(f"[blue]Columnas válidas para datos: {valid_columns}[/blue]")
            
            if len(valid_columns) < 2:
                # Si no tenemos suficientes columnas válidas, intentar con diferentes configuraciones
                console.print(f"[yellow]Advertencia: Solo se detectaron {len(valid_columns)} columnas válidas[/yellow]")
            
            # Detectar tipo de formato
            format_type = "CSV estándar"
            if best_sep == ';;;':
                format_type = "CSV con separador personalizado (;;;)"
            elif best_sep == '\t':
                format_type = "TSV (Tab-separated)"
            elif best_sep == '|':
                format_type = "Pipe-separated"
            
            return df, format_type
            
        except Exception as e:
            # Fallback: intentar con diferentes configuraciones
            try:
                df = pd.read_csv(file_path, sep=best_sep, quotechar='"', nrows=5, on_bad_lines='skip')
                df.columns = [col.strip().strip('"').rstrip(';;;') for col in df.columns]
                return df, f"CSV con comillas ({best_sep})"
            except:
                return None, f"Error: {str(e)}"
    
    def _parse_custom_csv(self, file_path: str, separator: str, nrows: int = None):
        """Parsea CSV con separador personalizado como ;;; - maneja formatos mixtos"""
        import pandas as pd
        import csv
        import io
        import re
        
        rows = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if nrows and i >= nrows:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                
                # Dividir por el separador personalizado
                parts = line.split(separator)
                if len(parts) < 2:
                    continue
                
                # La primera parte contiene los datos CSV
                csv_data = parts[0]
                
                # Para la primera línea (header), usar división simple
                if i == 0:
                    # Header: linea,catalan,chino
                    row = csv_data.split(',')
                else:
                    # Detectar el formato de la línea de datos
                    if csv_data.startswith('"') and csv_data.endswith('"'):
                        # Formato con comillas: "558,""text1"",""text2"""
                        row = self._parse_quoted_line(csv_data)
                    else:
                        # Formato simple: 31524,text1,text2
                        row = csv_data.split(',')
                
                if row:
                    rows.append(row)
        
        if not rows:
            return pd.DataFrame()
        
        # Asegurar que todas las filas tengan el mismo número de columnas
        header = rows[0]
        num_cols = len(header)
        
        # Procesar las filas de datos
        data_rows = []
        for row in rows[1:]:
            if len(row) == num_cols:
                data_rows.append(row)
            elif len(row) > num_cols:
                # Si hay más columnas, tomar solo las primeras
                data_rows.append(row[:num_cols])
            else:
                # Si hay menos columnas, rellenar con valores vacíos
                padded_row = row + [''] * (num_cols - len(row))
                data_rows.append(padded_row)
        
        # Crear DataFrame
        df = pd.DataFrame(data_rows, columns=header)
        
        # Limpiar nombres de columnas - remover sufijos como ;;; y comillas
        df.columns = [col.strip().strip('"').rstrip(';;;') for col in df.columns]
        
        return df
    
    def _parse_quoted_line(self, csv_data: str):
        """Parsea una línea con formato de comillas complejo usando regex mejorado"""
        import re
        
        # Remover las comillas exteriores
        if csv_data.startswith('"') and csv_data.endswith('"'):
            csv_data = csv_data[1:-1]
        
        # Patrón 1: número,""texto_con_comillas"",texto_chino
        pattern1 = r'^(\d+),""([^"]*(?:""[^"]*)*)"",(.+)$'
        match1 = re.match(pattern1, csv_data)
        
        if match1:
            number = match1.group(1)
            quoted_text = match1.group(2).replace('""', '"')
            rest = match1.group(3)
            return [number, quoted_text, rest]
        
        # Patrón 2: número,"texto_con_comillas",texto_chino
        pattern2 = r'^(\d+),"([^"]*(?:\\.[^"]*)*)",(.+)$'
        match2 = re.match(pattern2, csv_data)
        
        if match2:
            number = match2.group(1)
            quoted_text = match2.group(2)
            rest = match2.group(3)
            return [number, quoted_text, rest]
        
        # Patrón 3: número,texto_sin_comillas,texto_chino
        pattern3 = r'^(\d+),([^,]+),(.+)$'
        match3 = re.match(pattern3, csv_data)
        
        if match3:
            number = match3.group(1)
            text = match3.group(2)
            rest = match3.group(3)
            return [number, text, rest]
        
        # Fallback a división simple por comas
        return csv_data.split(',')
    
    def _load_dataset_with_detected_format(self, dataset_config):
        """Carga el dataset usando el formato detectado"""
        import pandas as pd
        
        # Usar el separador detectado
        delimiter = getattr(dataset_config, 'delimiter', ',')
        
        try:
            if delimiter in [';;;', '|||']:
                # Separadores personalizados - usar parser personalizado
                df = self._parse_custom_csv(dataset_config.file_path, delimiter)
            else:
                # Separadores estándar
                df = pd.read_csv(dataset_config.file_path, sep=delimiter, on_bad_lines='skip')
            
            # Limpiar nombres de columnas
            df.columns = [col.strip().strip('"') for col in df.columns]
            
            return df
            
        except Exception as e:
            console.print(f"[red]Error cargando dataset: {e}[/red]")
            # Fallback: intentar con diferentes configuraciones
            try:
                df = pd.read_csv(dataset_config.file_path, sep=delimiter, quotechar='"', on_bad_lines='skip')
                df.columns = [col.strip().strip('"') for col in df.columns]
                return df
            except:
                raise e
    
    def _configure_peft_method(self) -> BasePEFTConfig:
        """Configura el método PEFT seleccionado"""
        console.print(f"\n[bold]⚙️ Configuración de {self.peft_method.value.upper()}:[/bold]\n")
        
        # Configuración base común
        base_kwargs = {
            'method': self.peft_method,
            'learning_rate': FloatPrompt.ask("Learning rate", default=2e-4),
            'num_train_epochs': IntPrompt.ask("Épocas", default=3),
            'per_device_train_batch_size': IntPrompt.ask("Batch size", default=4)
        }
        
        # Configuración específica por método
        if self.peft_method == PEFTMethod.LORA:
            console.print("\n[cyan]Parámetros LoRA:[/cyan]")
            config = LoRAConfig(
                **base_kwargs,
                r=IntPrompt.ask("Rango (r)", default=16),
                lora_alpha=IntPrompt.ask("Alpha", default=32),
                lora_dropout=FloatPrompt.ask("Dropout", default=0.1)
            )
            
        elif self.peft_method == PEFTMethod.MOLORA:
            console.print("\n[cyan]Parámetros MoLoRA:[/cyan]")
            config = MoLoRAConfig(
                **base_kwargs,
                num_experts=IntPrompt.ask("Número de expertos", default=4),
                expert_r=[IntPrompt.ask(f"Rango experto {i+1}", default=8) 
                         for i in range(4)]
            )
            
        elif self.peft_method == PEFTMethod.GALORE:
            console.print("\n[cyan]Parámetros GaLore:[/cyan]")
            config = GaLoreConfig(
                **base_kwargs,
                rank=IntPrompt.ask("Rango gradiente", default=128),
                scale=FloatPrompt.ask("Factor de escala", default=0.25)
            )
            
        elif self.peft_method == PEFTMethod.DORA:
            console.print("\n[cyan]Parámetros DoRA:[/cyan]")
            config = DoRAConfig(
                **base_kwargs,
                r=IntPrompt.ask("Rango (r)", default=16),
                lora_alpha=IntPrompt.ask("Alpha", default=32),
                magnitude_lr_scale=FloatPrompt.ask("Escala LR magnitud", default=0.1)
            )
            

            
        elif self.peft_method == PEFTMethod.BITFIT:
            console.print("\n[cyan]Parámetros BitFit:[/cyan]")
            config = BitFitConfig(
                **base_kwargs,
                train_embeddings=Confirm.ask("¿Entrenar embeddings?", default=False),
                train_layer_norms=Confirm.ask("¿Entrenar layer norms?", default=True)
            )
            
        elif self.peft_method == PEFTMethod.IA3:
            console.print("\n[cyan]Parámetros IA³:[/cyan]")
            config = IA3Config(
                **base_kwargs,
                init_ia3_weights=Prompt.ask("Inicialización", choices=["ones", "zeros"], default="ones")
            )
            
        elif self.peft_method == PEFTMethod.PROMPT_TUNING:
            console.print("\n[cyan]Parámetros Prompt Tuning:[/cyan]")
            config = PromptTuningConfig(
                **base_kwargs,
                num_virtual_tokens=IntPrompt.ask("Tokens virtuales", default=20),
                prompt_tuning_init=Prompt.ask("Inicialización", choices=["random", "text"], default="random")
            )
            
        elif self.peft_method == PEFTMethod.ADAPTER:
            console.print("\n[cyan]Parámetros Adapter:[/cyan]")
            config = AdapterConfig(
                **base_kwargs,
                adapter_size=IntPrompt.ask("Tamaño adapter", default=64),
                adapter_type=Prompt.ask("Tipo", choices=["pfeiffer", "houlsby"], default="pfeiffer")
            )
            
        elif self.peft_method == PEFTMethod.QLORA:
            console.print("\n[cyan]Parámetros QLoRA:[/cyan]")
            config = QLoRAConfig(
                **base_kwargs,
                r=IntPrompt.ask("Rango (r)", default=16),
                lora_alpha=IntPrompt.ask("Alpha", default=32),
                bits=IntPrompt.ask("Bits cuantización", choices=[4, 8], default=4)
            )
        
        # Módulos objetivo (si aplica)
        if hasattr(config, 'target_modules') and self.peft_method not in [
            PEFTMethod.BITFIT, PEFTMethod.PROMPT_TUNING
        ]:
            config.target_modules = self._select_target_modules()
        
        return config
    
    def _select_target_modules(self) -> List[str]:
        """Selecciona módulos objetivo"""
        console.print("\n[cyan]Módulos objetivo:[/cyan]")
        
        # Módulos comunes por arquitectura
        common_modules = {
            "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "gpt": ["c_attn", "c_proj", "c_fc"],
            "bert": ["query", "key", "value", "dense"]
        }
        
        console.print("Opciones:")
        console.print("  [1] Todos los módulos de atención")
        console.print("  [2] Atención + FFN/MLP")
        console.print("  [3] Solo proyecciones QV")
        console.print("  [4] Personalizado")
        
        choice = Prompt.ask("Selección", choices=["1", "2", "3", "4"], default="2")
        
        # Detectar nombres de capas según el modelo
        if "distilgpt2" in self.model_name.lower() or "gpt2" in self.model_name.lower():
            # GPT-2/DistilGPT-2 usa estos nombres
            attention_modules = ["c_attn", "c_proj"]
            ffn_modules = ["c_fc"]
        elif "tinyllama" in self.model_name.lower() or "llama" in self.model_name.lower():
            # TinyLlama/Llama usa estos nombres
            attention_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            ffn_modules = ["gate_proj", "up_proj", "down_proj"]
        else:
            # Por defecto, usar nombres estándar
            attention_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            ffn_modules = ["gate_proj", "up_proj", "down_proj"]
        
        if choice == "1":
            return attention_modules
        elif choice == "2":
            return attention_modules + ffn_modules
        elif choice == "3":
            if "tinyllama" in self.model_name.lower() or "llama" in self.model_name.lower():
                return ["q_proj", "v_proj"]
            else:
                return ["c_attn"]
        else:
            # Personalizado
            modules = Prompt.ask("Módulos (separados por comas)")
            return [m.strip() for m in modules.split(",")]
    
    def _review_configuration(self) -> bool:
        """Revisa la configuración antes de entrenar"""
        console.print("\n[bold]📋 Resumen de configuración:[/bold]\n")
        
        # Crear paneles de información
        method_info = Panel(
            f"[cyan]Método:[/cyan] {self.peft_method.value.upper()}\n"
            f"[cyan]Descripción:[/cyan] {self.METHOD_DESCRIPTIONS[self.peft_method].split(chr(10))[0]}",
            title="Método PEFT",
            border_style="blue"
        )
        
        model_info = Panel(
            f"[cyan]Modelo:[/cyan] {self.model_name}\n"
            f"[cyan]Datasets:[/cyan] {len(self.selected_datasets)} archivos",
            title="Datos",
            border_style="green"
        )
        
        # Parámetros específicos
        params_text = self._format_config_params()
        params_info = Panel(
            params_text,
            title="Parámetros",
            border_style="yellow"
        )
        
        # Mostrar paneles
        console.print(Columns([method_info, model_info]))
        console.print(params_info)
        
        # Estimación de recursos
        self._estimate_resources()
        
        return Confirm.ask("\n¿Proceder con el entrenamiento?", default=True)
    
    def _format_config_params(self) -> str:
        """Formatea los parámetros de configuración"""
        lines = []
        
        # Parámetros comunes
        lines.append(f"[cyan]Learning rate:[/cyan] {self.peft_config.learning_rate}")
        lines.append(f"[cyan]Épocas:[/cyan] {self.peft_config.num_train_epochs}")
        lines.append(f"[cyan]Batch size:[/cyan] {self.peft_config.per_device_train_batch_size}")
        
        # Parámetros específicos
        if hasattr(self.peft_config, 'r'):
            lines.append(f"[cyan]Rango:[/cyan] {self.peft_config.r}")
        if hasattr(self.peft_config, 'lora_alpha'):
            lines.append(f"[cyan]Alpha:[/cyan] {self.peft_config.lora_alpha}")
        if hasattr(self.peft_config, 'num_experts'):
            lines.append(f"[cyan]Expertos:[/cyan] {self.peft_config.num_experts}")
        if hasattr(self.peft_config, 'adapter_size'):
            lines.append(f"[cyan]Tamaño adapter:[/cyan] {self.peft_config.adapter_size}")
        if hasattr(self.peft_config, 'num_virtual_tokens'):
            lines.append(f"[cyan]Tokens virtuales:[/cyan] {self.peft_config.num_virtual_tokens}")
        
        return "\n".join(lines)
    
    def _estimate_resources(self):
        """Estima recursos necesarios"""
        console.print("\n[dim]Estimación de recursos:[/dim]")
        
        # Calcular parámetros entrenables aproximados
        trainable_params = 0
        
        if self.peft_method == PEFTMethod.LORA:
            # r * d * 2 * num_modules
            trainable_params = self.peft_config.r * 4096 * 2 * 7  # Aproximado
        elif self.peft_method == PEFTMethod.BITFIT:
            # Solo bias ~ 0.1% del modelo
            trainable_params = 70_000_000 * 0.001  # Para 7B modelo
        elif self.peft_method == PEFTMethod.PROMPT_TUNING:
            # num_tokens * embedding_dim
            trainable_params = self.peft_config.num_virtual_tokens * 4096
        
        trainable_mb = (trainable_params * 4) / (1024 * 1024)  # FP32
        
        console.print(f"  • Parámetros entrenables: ~{trainable_params/1e6:.1f}M")
        console.print(f"  • Memoria para parámetros: ~{trainable_mb:.1f} MB")
        
        # Verificar compatibilidad GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            console.print(f"  • GPU disponible: {torch.cuda.get_device_name(0)}")
            console.print(f"  • Memoria GPU: {gpu_memory:.1f} GB")
            
            if self.peft_method == PEFTMethod.QLORA and gpu_memory < 24:
                console.print("[yellow]  ⚠️ QLoRA en 4-bit requiere ~24GB para modelos 7B[/yellow]")
    
    def _run_training(self):
        """Ejecuta el entrenamiento"""
        console.print("\n[bold green]🚀 Iniciando entrenamiento...[/bold green]\n")
        
        # Crear directorio de salida
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{self.model_name}_{self.peft_method.value}_{timestamp}"
        output_path = self.output_dir / output_name
        
        # Preparar datos
        all_data = []
        for dataset_config in self.selected_datasets:
            # Usar el formato detectado para cargar los datos
            data = self._load_dataset_with_detected_format(dataset_config)
            # Convertir DataFrame a lista de diccionarios
            if hasattr(data, 'to_dict'):
                data_list = data.to_dict('records')
            else:
                data_list = data
            all_data.extend(data_list)
        
        console.print(f"[green]✓[/green] Cargados {len(all_data)} ejemplos")
        
        # Importar trainer universal
        from LoRa_train.peft_universal_trainer import PEFTUniversalTrainer
        
        # Crear trainer
        trainer = PEFTUniversalTrainer(
            model_name=self.model_name,
            model_path=self.models_dir / self.model_name,
            output_dir=output_path,
            peft_config=self.peft_config
        )
        
        # Pasar las configuraciones de dataset al trainer
        trainer.dataset_configs = self.selected_datasets
        
        # Entrenar
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Entrenando...", total=None)
                
                results = trainer.train(all_data)
                
                progress.stop()
            
            # Mostrar resultados
            self._show_results(results, output_path)
            
        except Exception as e:
            console.print(f"\n[red]Error durante el entrenamiento: {str(e)}[/red]")
            raise
    
    def _show_results(self, results: Dict[str, Any], output_path: Path):
        """Muestra los resultados del entrenamiento"""
        console.print("\n[bold green]✅ Entrenamiento completado![/bold green]\n")
        
        # Métricas
        if 'metrics' in results:
            console.print("[cyan]Métricas finales:[/cyan]")
            for key, value in results['metrics'].items():
                if isinstance(value, float):
                    console.print(f"  • {key}: {value:.4f}")
                else:
                    console.print(f"  • {key}: {value}")
        
        # Archivos guardados
        console.print(f"\n[cyan]Modelo guardado en:[/cyan]")
        console.print(f"  📁 {output_path}")
        
        # Información específica del método
        if self.peft_method in [PEFTMethod.LORA, PEFTMethod.QLORA, PEFTMethod.DORA]:
            console.print(f"\n[cyan]Información del método:[/cyan]")
            console.print(f"  • Rango (r): {getattr(self.peft_config, 'r', 'N/A')}")
            console.print(f"  • Alpha: {getattr(self.peft_config, 'lora_alpha', 'N/A')}")
        
        # Próximos pasos
        console.print("\n[cyan]Próximos pasos:[/cyan]")
        
        if self.peft_method in [PEFTMethod.LORA, PEFTMethod.QLORA, PEFTMethod.DORA]:
            console.print(f"  1. Fusionar con modelo base:")
            console.print(f"     python merge_lora.py {output_path}")
        
        console.print(f"\n  2. Probar el modelo:")
        console.print(f"     python test_model.py {output_path}")
        
        console.print(f"\n  3. Evaluar rendimiento:")
        console.print(f"     python evaluate_model.py {output_path} --method {self.peft_method.value}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning universal con métodos PEFT"
    )
    
    parser.add_argument('--models-dir', default='./models', help='Directorio de modelos')
    parser.add_argument('--datasets-dir', default='./datasets', help='Directorio de datasets')
    parser.add_argument('--output-dir', default='./finetuned_models', help='Directorio de salida')
    parser.add_argument('--method', type=str, help='Método PEFT directo (omite selección interactiva)')
    
    args = parser.parse_args()
    
    # Crear wizard
    wizard = PEFTFineTuneWizard(
        models_dir=args.models_dir,
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir
    )
    
    # Si se especificó método, usarlo directamente
    if args.method:
        try:
            wizard.peft_method = PEFTMethod(args.method.lower())
        except ValueError:
            console.print(f"[red]Método inválido: {args.method}[/red]")
            console.print(f"Métodos válidos: {[m.value for m in PEFTMethod]}")
            sys.exit(1)
    
    # Ejecutar
    try:
        wizard.run()
    except Exception as e:
        console.print(f"\n[red]Error fatal: {str(e)}[/red]")
        logger.exception("Error en PEFT wizard")
        sys.exit(1)


if __name__ == "__main__":
    main()