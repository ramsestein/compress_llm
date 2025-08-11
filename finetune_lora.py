#!/usr/bin/env python3
"""
Script mejorado para fine-tuning/healing de modelos con LoRA
Incluye configuración avanzada por tipo de capa y compresión personalizada
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
from rich import print as rprint
from dataclasses import dataclass, field
from enum import Enum

# Importar módulos locales
from LoRa_train.dataset_manager import DatasetManager, DatasetConfig
from LoRa_train.lora_config import LoRAConfig, TrainingConfig, DataConfig, LoRAPresets
from LoRa_train.lora_trainer import LoRATrainer

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

class TrainingMethod(Enum):
    """Métodos de entrenamiento/compresión disponibles"""
    LORA_STANDARD = "lora_standard"
    LORA_INT8 = "lora_int8"
    LORA_INT4 = "lora_int4"
    LORA_PRUNED = "lora_pruned"
    LORA_TUCKER = "lora_tucker"
    LORA_MPO = "lora_mpo"
    FULL_FREEZE = "full_freeze"
    FULL_TRAIN = "full_train"

@dataclass
class LayerTrainingConfig:
    """Configuración de entrenamiento para un tipo de capa"""
    layer_type: str
    training_method: TrainingMethod
    compression_ratio: float = 0.0
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    dropout: float = 0.1
    learning_rate_multiplier: float = 1.0
    quantization_bits: Optional[int] = None
    pruning_ratio: Optional[float] = None
    
@dataclass
class AdvancedLoRAConfig(LoRAConfig):
    """Configuración LoRA extendida con soporte para múltiples métodos por capa"""
    layer_configs: Dict[str, LayerTrainingConfig] = field(default_factory=dict)
    final_layers_config: Optional[LayerTrainingConfig] = None
    final_layers_count: int = 3  # Número de capas finales con config especial
    
    # Nuevos parámetros globales
    enable_gradient_checkpointing: bool = True
    mixed_precision_training: bool = True
    compression_aware_training: bool = False
    
    def get_layer_config(self, layer_name: str, layer_position: int, total_layers: int) -> LayerTrainingConfig:
        """Obtiene la configuración para una capa específica"""
        # Verificar si es una capa final
        if self.final_layers_config and (total_layers - layer_position) <= self.final_layers_count:
            return self.final_layers_config
        
        # Buscar por tipo de capa
        layer_type = self._get_layer_type(layer_name)
        if layer_type in self.layer_configs:
            return self.layer_configs[layer_type]
        
        # Default
        return LayerTrainingConfig(
            layer_type=layer_type,
            training_method=TrainingMethod.LORA_STANDARD,
            lora_rank=self.r,
            lora_alpha=self.lora_alpha,
            dropout=self.lora_dropout
        )
    
    def _get_layer_type(self, layer_name: str) -> str:
        """Determina el tipo de capa basado en el nombre"""
        name_lower = layer_name.lower()
        
        if any(x in name_lower for x in ['embed', 'embedding']):
            return 'embedding'
        elif any(x in name_lower for x in ['attention', 'attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
            return 'attention'
        elif any(x in name_lower for x in ['mlp', 'ffn', 'fc1', 'fc2', 'gate_proj', 'up_proj', 'down_proj']):
            return 'ffn'
        elif any(x in name_lower for x in ['norm', 'layernorm', 'ln']):
            return 'normalization'
        elif any(x in name_lower for x in ['lm_head', 'output', 'classifier']):
            return 'output'
        else:
            return 'other'

class FineTuneWizard:
    """Asistente interactivo para fine-tuning con LoRA"""
    
    def __init__(self, models_dir: str = "./models", datasets_dir: str = "./datasets",
                 output_dir: str = "./finetuned_models"):
        self.models_dir = Path(models_dir)
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_manager = DatasetManager(datasets_dir)
        self.selected_datasets = []
        self.model_name = None
        self.model_path = None
        self.training_mode = None
    
    def run(self):
        """Ejecuta el asistente interactivo"""
        console.clear()
        self._show_welcome()
        
        try:
            # Paso 1: Seleccionar modo de entrenamiento
            self.training_mode = self._select_training_mode()
            
            # Paso 2: Seleccionar modelo
            self.model_name, self.model_path = self._select_model()
            
            # Paso 3: Seleccionar y configurar datasets
            self.selected_datasets = self._select_and_configure_datasets()
            
            # Paso 4: Configurar LoRA según el modo
            if self.training_mode == "healing":
                lora_config, training_config = self._configure_healing()
            elif self.training_mode == "advanced":
                lora_config, training_config = self._configure_advanced()
            else:
                lora_config, training_config = self._configure_standard()
            
            # Paso 5: Configurar datos
            data_config = self._configure_data()
            
            # Paso 6: Revisar y confirmar
            if self._review_configuration(lora_config, training_config, data_config):
                # Paso 7: Ejecutar entrenamiento
                self._run_training(lora_config, training_config, data_config)
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
        welcome_panel = Panel(
            """[bold cyan]🚀 Asistente de Fine-Tuning con LoRA[/bold cyan]
            
Este asistente te guiará para:
- Fine-tuning estándar con LoRA
- Healing de modelos comprimidos
- Configuración avanzada por capas

[dim]Presiona Ctrl+C para salir en cualquier momento[/dim]""",
            title="Bienvenida",
            border_style="cyan"
        )
        console.print(welcome_panel)
        console.print()
    
    def _select_training_mode(self) -> str:
        """Selecciona el modo de entrenamiento"""
        console.print("[bold]🎯 Selecciona el modo de entrenamiento:[/bold]\n")
        
        modes = {
            "1": ("standard", "Fine-tuning estándar", "Entrenamiento normal con LoRA"),
            "2": ("healing", "Healing de modelo comprimido", "Recuperar calidad después de compresión"),
            "3": ("advanced", "Configuración avanzada", "Control total por tipo de capa")
        }
        
        table = Table(show_header=False, box=None)
        for key, (mode, title, desc) in modes.items():
            table.add_row(f"[cyan]{key}[/cyan]", f"[bold]{title}[/bold]", f"[dim]{desc}[/dim]")
        
        console.print(table)
        
        while True:
            choice = Prompt.ask("\nOpción", choices=["1", "2", "3"], default="1")
            return modes[choice][0]
    
    def _select_model(self) -> Tuple[str, Path]:
        """Selecciona el modelo a entrenar"""
        console.print("\n[bold]📦 Modelos disponibles:[/bold]\n")
        
        # Listar modelos
        models = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                # Calcular tamaño
                size_gb = sum(
                    f.stat().st_size for f in model_dir.rglob('*') if f.is_file()
                ) / (1024**3)
                
                # Verificar si es comprimido
                is_compressed = (model_dir / "compression_metadata.json").exists()
                
                models.append({
                    'name': model_dir.name,
                    'path': model_dir,
                    'size_gb': size_gb,
                    'compressed': is_compressed
                })
        
        if not models:
            raise ValueError("No se encontraron modelos en el directorio")
        
        # Mostrar tabla
        table = Table(title="Modelos Locales")
        table.add_column("#", style="cyan")
        table.add_column("Nombre", style="green")
        table.add_column("Tamaño", justify="right")
        table.add_column("Estado", justify="center")
        
        for i, model in enumerate(models, 1):
            status = "🗜️ Comprimido" if model['compressed'] else "📦 Original"
            table.add_row(
                str(i),
                model['name'],
                f"{model['size_gb']:.1f} GB",
                status
            )
        
        console.print(table)
        
        # Seleccionar
        while True:
            choice = IntPrompt.ask("\nSelecciona un modelo", default=1)
            if 1 <= choice <= len(models):
                selected = models[choice - 1]
                return selected['name'], selected['path']
            console.print("[red]Opción inválida[/red]")
    
    def _select_and_configure_datasets(self) -> List[DatasetConfig]:
        """Selecciona y configura datasets"""
        console.print("\n[bold]📊 Configuración de datasets:[/bold]\n")
        
        # Escanear datasets disponibles
        available_datasets = self.dataset_manager.scan_datasets()
        
        if not available_datasets:
            console.print("[yellow]No se encontraron datasets[/yellow]")
            console.print("Coloca archivos CSV o JSONL en:", self.datasets_dir)
            raise ValueError("No hay datasets disponibles")
        
        # Mostrar tabla
        table = Table(title="Datasets Disponibles")
        table.add_column("#", style="cyan")
        table.add_column("Nombre", style="green")
        table.add_column("Formato")
        table.add_column("Registros", justify="right")
        table.add_column("Columnas detectadas")
        
        for i, dataset in enumerate(available_datasets, 1):
            cols = ", ".join(dataset.get('detected_columns', [])[:3])
            if len(dataset.get('detected_columns', [])) > 3:
                cols += "..."
            
            table.add_row(
                str(i),
                dataset['name'],
                dataset['format'].upper(),
                str(dataset['size']),
                cols
            )
        
        console.print(table)
        
        # Seleccionar datasets
        selected_configs = []
        
        console.print("\n[dim]Puedes seleccionar múltiples datasets (vacío para terminar)[/dim]")
        
        while True:
            choice = Prompt.ask("\nDataset #", default="")
            if not choice:
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available_datasets):
                    dataset_info = available_datasets[idx]
                    
                    # Configurar columnas
                    console.print(f"\n[cyan]Configurando: {dataset_info['name']}[/cyan]")
                    config = self.dataset_manager.configure_dataset_interactive(dataset_info)
                    
                    if config:
                        selected_configs.append(config)
                        console.print(f"[green]✓ {dataset_info['name']} configurado[/green]")
                else:
                    console.print("[red]Índice inválido[/red]")
            except ValueError:
                console.print("[red]Entrada inválida[/red]")
        
        if not selected_configs:
            raise ValueError("Debes seleccionar al menos un dataset")
        
        return selected_configs
    
    def _configure_standard(self) -> Tuple[LoRAConfig, TrainingConfig]:
        """Configuración estándar de LoRA"""
        console.print("\n[bold]⚙️ Configuración de LoRA:[/bold]\n")
        
        # Presets disponibles
        presets = {
            "1": ("conservative", "Conservador (r=8, epochs=1)"),
            "2": ("balanced", "Balanceado (r=16, epochs=3)"),
            "3": ("aggressive", "Agresivo (r=32, epochs=5)"),
            "4": ("custom", "Personalizado")
        }
        
        console.print("Presets disponibles:")
        for key, (name, desc) in presets.items():
            console.print(f"  [{key}] {desc}")
        
        choice = Prompt.ask("\nSelecciona preset", choices=list(presets.keys()), default="2")
        
        if choice == "4":
            # Configuración personalizada
            lora_config = LoRAConfig(
                r=IntPrompt.ask("Rango (r)", default=16, min_value=1, max_value=256),
                lora_alpha=IntPrompt.ask("Alpha", default=32),
                lora_dropout=FloatPrompt.ask("Dropout", default=0.1),
                target_modules=self._select_target_modules()
            )
            
            training_config = TrainingConfig(
                num_train_epochs=IntPrompt.ask("Épocas", default=3),
                learning_rate=FloatPrompt.ask("Learning rate", default=2e-4),
                per_device_train_batch_size=IntPrompt.ask("Batch size", default=4)
            )
        else:
            # Usar preset
            preset_name = presets[choice][0]
            configs = LoRAPresets.get_preset(preset_name)
            lora_config = configs["lora"]
            training_config = configs["training"]
        
        return lora_config, training_config
    
    def _configure_healing(self) -> Tuple[LoRAConfig, TrainingConfig]:
        """Configuración para healing de modelos comprimidos"""
        console.print("\n[bold]🔧 Configuración de Healing:[/bold]\n")
        console.print("[dim]El healing usa configuración optimizada para recuperar calidad[/dim]\n")
        
        # Detectar nivel de compresión
        compression_level = self._detect_compression_level()
        
        if compression_level == "high":
            console.print("[yellow]⚠️ Compresión alta detectada - se recomienda configuración agresiva[/yellow]")
            lora_config = LoRAConfig(
                r=64,
                lora_alpha=128,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head"
                ]
            )
            training_config = TrainingConfig(
                num_train_epochs=5,
                learning_rate=1e-4,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=200
            )
        else:
            console.print("[green]✓ Compresión moderada - configuración balanceada[/green]")
            configs = LoRAPresets.get_preset("healing")
            lora_config = configs["lora"]
            training_config = configs["training"]
        
        # Ajustes específicos para healing
        training_config.fp16 = True
        training_config.gradient_checkpointing = True
        training_config.save_total_limit = 5  # Guardar más checkpoints
        
        return lora_config, training_config
    
    def _configure_advanced(self) -> Tuple[AdvancedLoRAConfig, TrainingConfig]:
        """Configuración avanzada por tipo de capa"""
        console.print("\n[bold]🎛️ Configuración Avanzada por Capas:[/bold]\n")
        
        # Configuración base
        base_config = AdvancedLoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        
        # Configurar cada tipo de capa
        layer_types = ['embedding', 'attention', 'ffn', 'normalization', 'output']
        
        for layer_type in layer_types:
            console.print(f"\n[cyan]Configurando capas {layer_type.upper()}:[/cyan]")
            
            # Método de entrenamiento
            console.print("Métodos disponibles:")
            console.print("  [1] LoRA estándar")
            console.print("  [2] LoRA + INT8")
            console.print("  [3] LoRA + Pruning")
            console.print("  [4] Sin entrenamiento")
            
            method_choice = Prompt.ask("Método", choices=["1", "2", "3", "4"], default="1")
            
            if method_choice == "4":
                continue
            
            # Crear configuración para este tipo
            layer_config = LayerTrainingConfig(
                layer_type=layer_type,
                training_method=TrainingMethod.LORA_STANDARD if method_choice == "1" else
                               TrainingMethod.LORA_INT8 if method_choice == "2" else
                               TrainingMethod.LORA_PRUNED
            )
            
            # Parámetros específicos
            if method_choice != "4":
                layer_config.lora_rank = IntPrompt.ask(f"  Rango para {layer_type}", default=16)
                layer_config.learning_rate_multiplier = FloatPrompt.ask(
                    f"  Multiplicador LR", default=1.0
                )
            
            base_config.layer_configs[layer_type] = layer_config
        
        # Configuración de capas finales
        if Confirm.ask("\n¿Configuración especial para capas finales?", default=True):
            num_final = IntPrompt.ask("Número de capas finales", default=3)
            base_config.final_layers_count = num_final
            base_config.final_layers_config = LayerTrainingConfig(
                layer_type="final",
                training_method=TrainingMethod.LORA_STANDARD,
                lora_rank=8,
                learning_rate_multiplier=0.5
            )
        
        # Training config
        training_config = TrainingConfig(
            num_train_epochs=IntPrompt.ask("\nÉpocas totales", default=3),
            learning_rate=FloatPrompt.ask("Learning rate base", default=2e-4),
            per_device_train_batch_size=IntPrompt.ask("Batch size", default=4)
        )
        
        return base_config, training_config
    
    def _configure_data(self) -> DataConfig:
        """Configura parámetros de datos"""
        console.print("\n[bold]📝 Configuración de datos:[/bold]\n")
        
        data_config = DataConfig()
        
        # Longitud máxima
        data_config.max_length = IntPrompt.ask(
            "Longitud máxima de secuencia", 
            default=512,
            min_value=128,
            max_value=4096
        )
        
        # Template
        console.print("\nTemplates disponibles:")
        console.print("  [1] Instruction/Response")
        console.print("  [2] Chat (usuario/asistente)")
        console.print("  [3] Texto plano")
        console.print("  [4] Personalizado")
        
        template_choice = Prompt.ask("Template", choices=["1", "2", "3", "4"], default="1")
        
        if template_choice == "1":
            data_config.instruction_template = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
        elif template_choice == "2":
            data_config.chat_template = "<|user|>\n{instruction}\n<|assistant|>\n{response}"
        elif template_choice == "4":
            data_config.instruction_template = Prompt.ask("Template personalizado")
        
        # Split de evaluación
        data_config.eval_split_ratio = FloatPrompt.ask(
            "Proporción para evaluación", 
            default=0.1,
            min_value=0.0,
            max_value=0.5
        )
        
        return data_config
    
    def _review_configuration(self, lora_config, training_config, data_config) -> bool:
        """Revisa y confirma la configuración"""
        console.print("\n[bold]📋 Resumen de configuración:[/bold]\n")
        
        # Modelo
        console.print(f"[cyan]Modelo:[/cyan] {self.model_name}")
        console.print(f"[cyan]Modo:[/cyan] {self.training_mode}")
        
        # LoRA
        console.print(f"\n[cyan]LoRA:[/cyan]")
        console.print(f"  • Rango: {lora_config.r}")
        console.print(f"  • Alpha: {lora_config.lora_alpha}")
        console.print(f"  • Dropout: {lora_config.lora_dropout}")
        console.print(f"  • Módulos: {len(lora_config.target_modules)}")
        
        # Training
        console.print(f"\n[cyan]Entrenamiento:[/cyan]")
        console.print(f"  • Épocas: {training_config.num_train_epochs}")
        console.print(f"  • Learning rate: {training_config.learning_rate}")
        console.print(f"  • Batch size: {training_config.per_device_train_batch_size}")
        
        # Datasets
        console.print(f"\n[cyan]Datasets:[/cyan]")
        total_samples = sum(d.size for d in self.selected_datasets)
        console.print(f"  • Archivos: {len(self.selected_datasets)}")
        console.print(f"  • Muestras totales: {total_samples}")
        
        # Estimación de tiempo y memoria
        self._estimate_resources(lora_config, training_config, total_samples)
        
        return Confirm.ask("\n¿Proceder con el entrenamiento?", default=True)
    
    def _run_training(self, lora_config, training_config, data_config):
        """Ejecuta el entrenamiento"""
        console.print("\n[bold green]🚀 Iniciando entrenamiento...[/bold green]\n")
        
        # Crear nombre para el output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{self.model_name}_lora_{self.training_mode}_{timestamp}"
        output_path = self.output_dir / output_name
        
        # Preparar datos de entrenamiento
        all_data = []
        for dataset_config in self.selected_datasets:
            data = self.dataset_manager.load_dataset(dataset_config)
            all_data.extend(data)
        
        console.print(f"[green]✓[/green] Cargados {len(all_data)} ejemplos")
        
        # Crear trainer
        trainer = LoRATrainer(
            model_name=self.model_name,
            model_path=self.model_path,
            output_dir=output_path
        )
        
        # Entrenar
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Entrenando...", total=None)
                
                results = trainer.train(
                    training_data=all_data,
                    lora_config=lora_config,
                    training_config=training_config,
                    data_config=data_config
                )
                
                progress.stop()
            
            # Mostrar resultados
            self._show_results(results, output_path)
            
        except Exception as e:
            console.print(f"\n[red]Error durante el entrenamiento: {str(e)}[/red]")
            raise
    
    def _detect_compression_level(self) -> str:
        """Detecta el nivel de compresión del modelo"""
        metadata_path = self.model_path / "compression_metadata.json"
        
        if not metadata_path.exists():
            return "none"
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            compression_ratio = metadata.get('statistics', {}).get('compression_ratio', 0)
            
            if compression_ratio > 0.5:
                return "high"
            elif compression_ratio > 0.3:
                return "medium"
            else:
                return "low"
        except:
            return "unknown"
    
    def _select_target_modules(self) -> List[str]:
        """Selecciona módulos objetivo para LoRA"""
        console.print("\n[cyan]Selecciona módulos objetivo:[/cyan]")
        
        all_modules = [
            ("q_proj", "Query projection"),
            ("v_proj", "Value projection"),
            ("k_proj", "Key projection"),
            ("o_proj", "Output projection"),
            ("gate_proj", "Gate projection (MLP)"),
            ("up_proj", "Up projection (MLP)"),
            ("down_proj", "Down projection (MLP)"),
            ("embed_tokens", "Token embeddings"),
            ("lm_head", "Language model head")
        ]
        
        selected = []
        
        console.print("\nMódulos disponibles:")
        for i, (module, desc) in enumerate(all_modules, 1):
            console.print(f"  [{i}] {module:<15} - {desc}")
        
        console.print("\n[dim]Ingresa números separados por comas (ej: 1,2,3)[/dim]")
        console.print("[dim]Vacío para seleccionar attention + MLP (recomendado)[/dim]")
        
        selection = Prompt.ask("\nSelección", default="1,2,3,4,5,6,7")
        
        if selection:
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            selected = [all_modules[i][0] for i in indices if 0 <= i < len(all_modules)]
        else:
            # Default: attention + MLP
            selected = ["q_proj", "v_proj", "k_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"]
        
        return selected
    
    def _estimate_resources(self, lora_config, training_config, num_samples):
        """Estima recursos necesarios"""
        console.print("\n[dim]Estimación de recursos:[/dim]")
        
        # Estimar memoria (simplificado)
        model_size_gb = sum(
            f.stat().st_size for f in self.model_path.rglob('*.safetensors')
        ) / (1024**3)
        
        lora_memory_gb = (lora_config.r * 2 * 1024) / (1024**3)  # Aproximado
        gradient_memory_gb = lora_memory_gb * 2
        total_memory_gb = model_size_gb + lora_memory_gb + gradient_memory_gb + 2  # +2GB overhead
        
        console.print(f"  • Memoria GPU estimada: {total_memory_gb:.1f} GB")
        
        # Estimar tiempo
        steps_per_epoch = num_samples // training_config.per_device_train_batch_size
        total_steps = steps_per_epoch * training_config.num_train_epochs
        time_per_step = 0.5  # segundos, aproximado
        total_time = total_steps * time_per_step
        
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        console.print(f"  • Tiempo estimado: {hours}h {minutes}m")
        console.print(f"  • Steps totales: {total_steps}")
    
    def _show_results(self, results: Dict[str, Any], output_path: Path):
        """Muestra los resultados del entrenamiento"""
        console.print("\n[bold green]✅ Entrenamiento completado![/bold green]\n")
        
        # Métricas finales
        if 'final_metrics' in results:
            metrics = results['final_metrics']
            console.print("[cyan]Métricas finales:[/cyan]")
            console.print(f"  • Loss: {metrics.get('loss', 'N/A'):.4f}")
            console.print(f"  • Learning rate: {metrics.get('learning_rate', 'N/A'):.2e}")
            if 'eval_loss' in metrics:
                console.print(f"  • Eval loss: {metrics['eval_loss']:.4f}")
        
        # Información del modelo
        console.print(f"\n[cyan]Modelo guardado en:[/cyan]")
        console.print(f"  📁 {output_path}")
        
        # Archivos guardados
        saved_files = list(output_path.glob("*"))
        console.print(f"\n[cyan]Archivos guardados:[/cyan]")
        for file in saved_files[:5]:  # Mostrar máximo 5
            console.print(f"  • {file.name}")
        if len(saved_files) > 5:
            console.print(f"  • ... y {len(saved_files) - 5} más")
        
        # Próximos pasos
        console.print("\n[cyan]Próximos pasos:[/cyan]")
        console.print(f"  1. Probar el modelo:")
        console.print(f"     python test_lora_model.py {output_path}")
        console.print(f"\n  2. Fusionar con el modelo base:")
        console.print(f"     python merge_lora.py {output_path}")
        console.print(f"\n  3. Evaluar calidad:")
        console.print(f"     python evaluate_model.py {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning interactivo con LoRA - Incluye healing y configuración avanzada"
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./models',
        help='Directorio de modelos (default: ./models)'
    )
    
    parser.add_argument(
        '--datasets-dir',
        type=str,
        default='./datasets',
        help='Directorio de datasets (default: ./datasets)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./finetuned_models',
        help='Directorio de salida (default: ./finetuned_models)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Modo rápido con configuración por defecto'
    )
    
    args = parser.parse_args()
    
    # Verificar directorios
    for dir_path in [args.models_dir, args.datasets_dir]:
        if not Path(dir_path).exists():
            console.print(f"[red]Error: No existe el directorio {dir_path}[/red]")
            sys.exit(1)
    
    # Ejecutar asistente
    wizard = FineTuneWizard(
        models_dir=args.models_dir,
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir
    )
    
    try:
        wizard.run()
    except Exception as e:
        console.print(f"\n[red]Error fatal: {str(e)}[/red]")
        logger.exception("Error en fine-tuning")
        sys.exit(1)


if __name__ == "__main__":
    main()