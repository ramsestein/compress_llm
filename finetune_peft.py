#!/usr/bin/env python3
"""
Script interactivo para fine-tuning con múltiples métodos PEFT
Soporta: LoRA, MoLoRA, GaLore, DoRA, AdaLoRA, BitFit, IA³, Prompt Tuning, Adapter, QLoRA
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
    DoRAConfig, AdaLoRAConfig, BitFitConfig, IA3Config, PromptTuningConfig,
    AdapterConfig, QLoRAConfig, PEFTPresets, get_config_by_name
)

# Importar componentes existentes
from LoRa_train.dataset_manager import DatasetManager, DatasetConfig
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
        PEFTMethod.ADALORA: "🔄 AdaLoRA - Adaptive LoRA\n• Rangos dinámicos\n• Auto-optimización\n• Ideal para: Sin tuning manual",
        PEFTMethod.BITFIT: "💡 BitFit - Bias Tuning\n• Solo bias (~0.1%)\n• Súper eficiente\n• Ideal para: Ajustes sutiles",
        PEFTMethod.IA3: "⚡ IA³ - Infused Adapter\n• Vectores de escala\n• 10x menos que LoRA\n• Ideal para: Máxima eficiencia",
        PEFTMethod.PROMPT_TUNING: "📝 Prompt Tuning\n• Tokens virtuales\n• < 0.01% parámetros\n• Ideal para: Modelos enormes",
        PEFTMethod.ADAPTER: "🧩 Adapter Tuning\n• Módulos bottleneck\n• Más expresivo\n• Ideal para: Cambios grandes",
        PEFTMethod.QLORA: "🔥 QLoRA - Quantized LoRA\n• LoRA + 4-bit\n• 10x menos memoria\n• Ideal para: Modelos 65B+"
    }
    
    def __init__(self, models_dir: str = "./models", datasets_dir: str = "./datasets",
                 output_dir: str = "./finetuned_models"):
        self.models_dir = Path(models_dir)
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_manager = DatasetManager(datasets_dir)
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
            
            # Paso 3: Seleccionar datasets
            self.selected_datasets = self._select_and_configure_datasets()
            
            # Paso 4: Configurar método PEFT
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
        • LoRA, QLoRA, DoRA, AdaLoRA, MoLoRA
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
            main_desc = desc_lines[0].split(' - ')[1]
            params = desc_lines[1].replace('• ', '')
            
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
        console.print("[dim]• Máxima calidad? → DoRA (4) o AdaLoRA (5)[/dim]")
        
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
    
    def _select_and_configure_datasets(self) -> List[DatasetConfig]:
        """Selecciona y configura datasets"""
        console.print("\n[bold]📊 Configuración de datasets:[/bold]\n")
        
        available = self.dataset_manager.scan_datasets()
        
        if not available:
            raise ValueError("No se encontraron datasets")
        
        # Mostrar tabla
        table = Table(title="Datasets Disponibles")
        table.add_column("#", style="cyan")
        table.add_column("Nombre", style="green")
        table.add_column("Formato")
        table.add_column("Registros", justify="right")
        
        for i, dataset in enumerate(available, 1):
            table.add_row(
                str(i),
                dataset['name'],
                dataset['format'].upper(),
                str(dataset['size'])
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
                    config = self.dataset_manager.configure_dataset_interactive(
                        available[idx]
                    )
                    if config:
                        selected.append(config)
                        console.print(f"[green]✓ Agregado[/green]")
            except:
                console.print("[red]Entrada inválida[/red]")
        
        if not selected:
            raise ValueError("Debes seleccionar al menos un dataset")
        
        return selected
    
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
            
        elif self.peft_method == PEFTMethod.ADALORA:
            console.print("\n[cyan]Parámetros AdaLoRA:[/cyan]")
            config = AdaLoRAConfig(
                **base_kwargs,
                init_r=IntPrompt.ask("Rango inicial", default=64),
                target_r=IntPrompt.ask("Rango objetivo", default=16),
                tinit=IntPrompt.ask("Pasos init", default=200),
                tfinal=IntPrompt.ask("Pasos final", default=1000)
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
        
        if choice == "1":
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif choice == "2":
            return ["q_proj", "v_proj", "k_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]
        elif choice == "3":
            return ["q_proj", "v_proj"]
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
            data = self.dataset_manager.load_dataset(dataset_config)
            all_data.extend(data)
        
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
        if self.peft_method == PEFTMethod.ADALORA:
            console.print(f"\n[cyan]Evolución de rangos:[/cyan]")
            console.print(f"  • Rango inicial: {self.peft_config.init_r}")
            console.print(f"  • Rango final: {results.get('final_rank', 'N/A')}")
        
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