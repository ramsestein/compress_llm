#!/usr/bin/env python3
"""Interactive script for fine-tuning with multiple PEFT methods.
Supports: LoRA, MoLoRA, GaLore, DoRA, BitFit, IA3, Prompt Tuning, Adapter, QLoRA.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on path so LoRa_train/ can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

# Import existing components
from LoRa_train.dataset_manager import DatasetConfig, OptimizedDatasetManager

# Import PEFT configurations
from LoRa_train.peft_methods_config import (
    AdapterConfig,
    BasePEFTConfig,
    BitFitConfig,
    DoRAConfig,
    GaLoreConfig,
    IA3Config,
    LoRAConfig,
    MoLoRAConfig,
    PEFTMethod,
    PromptTuningConfig,
    QLoRAConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()


class PEFTFineTuneWizard:
    """Interactive assistant for fine-tuning with PEFT methods."""

    # Method descriptions
    METHOD_DESCRIPTIONS = {
        PEFTMethod.LORA: "LoRA - Low-Rank Adaptation\n• Efficient and well-tested\n• 0.1-1% parameters\n• Best for: General use",
        PEFTMethod.MOLORA: "MoLoRA - Mixture of LoRAs\n• Multiple experts\n• Multi-task/domain\n• Best for: Versatile models",
        PEFTMethod.GALORE: "GaLore - Gradient Low-Rank\n• Gradient projection\n• Ultra-low memory\n• Best for: Limited GPUs",
        PEFTMethod.DORA: "DoRA - Decomposed LoRA\n• Magnitude + Direction\n• Better than LoRA\n• Best for: Maximum quality",
        PEFTMethod.BITFIT: "BitFit - Bias Tuning\n• Biases only (~0.1%)\n• Super efficient\n• Best for: Subtle adjustments",
        PEFTMethod.IA3: "IA3 - Infused Adapter\n• Scale vectors\n• 10x fewer than LoRA\n• Best for: Maximum efficiency",
        PEFTMethod.PROMPT_TUNING: "Prompt Tuning\n• Virtual tokens\n• < 0.01% parameters\n• Best for: Huge models",
        PEFTMethod.ADAPTER: "Adapter Tuning\n• Bottleneck modules\n• More expressive\n• Best for: Large changes",
        PEFTMethod.QLORA: "QLoRA - Quantized LoRA\n• LoRA + 4-bit\n• 10x less memory\n• Best for: 65B+ models",
    }

    def __init__(
        self,
        models_dir: str = "./models",
        datasets_dir: str = "./datasets",
        output_dir: str = "./finetuned_models",
    ):
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
        """Runs the interactive assistant."""
        console.clear()
        self._show_welcome()

        try:
            # Step 1: Select PEFT method
            self.peft_method = self._select_peft_method()

            # Step 2: Select model
            self.model_name = self._select_model()

            # Step 3: Select datasets
            self.selected_datasets = self._select_and_configure_datasets()

            # Step 4: Configure PEFT method
            self.peft_config = self._configure_peft_method()

            # Step 5: Review and confirm
            if self._review_configuration():
                # Step 6: Run training
                self._run_training()
            else:
                console.print("[yellow]Training cancelled[/yellow]")

        except KeyboardInterrupt:
            console.print("\n[red]Process interrupted[/red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]")
            logger.exception("Error in assistant")
            sys.exit(1)

    def _show_welcome(self):
        """Shows the welcome screen."""
        welcome_text = """
        [bold cyan]Universal PEFT Fine-Tuning Assistant[/bold cyan]

        This assistant supports multiple parameter-efficient fine-tuning methods:
        • LoRA, QLoRA, DoRA, MoLoRA
        • BitFit, IA3, Prompt Tuning, Adapter Tuning
        • GaLore (gradient projection)

        It will guide you step-by-step to configure and train your model.
        """

        panel = Panel(welcome_text, title="Welcome", border_style="cyan")
        console.print(panel)
        console.print()

    def _select_peft_method(self) -> PEFTMethod:
        """Selects the PEFT method to use."""
        console.print("[bold]Select the fine-tuning method:[/bold]\n")

        # Create methods table
        table = Table(title="Available PEFT Methods", show_lines=True)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Method", style="green", width=20)
        table.add_column("Description", style="white")
        table.add_column("Params", style="yellow", width=10)

        methods = list(PEFTMethod)
        for i, method in enumerate(methods, 1):
            desc_lines = self.METHOD_DESCRIPTIONS[method].split("\n")
            if len(desc_lines) > 0 and " - " in desc_lines[0]:
                main_desc = desc_lines[0].split(" - ")[1]
            else:
                main_desc = method.value.upper()

            params = desc_lines[1].replace("• ", "") if len(desc_lines) > 1 else "N/A"

            table.add_row(str(i), method.value.upper(), main_desc, params)

        console.print(table)

        # Recommendations
        console.print("\n[dim]Recommendations:[/dim]")
        console.print("[dim]• New to PEFT? → LoRA (1)[/dim]")
        console.print("[dim]• Limited memory? → BitFit (6) or QLoRA (10)[/dim]")
        console.print("[dim]• Maximum quality? → DoRA (4) or MoLoRA (2)[/dim]")

        while True:
            choice = IntPrompt.ask("\nSelect method (1-10)", default=1)
            if 1 <= choice <= len(methods):
                return methods[choice - 1]
            console.print("[red]Invalid option[/red]")

    def _select_model(self) -> str:
        """Selects the base model."""
        console.print("\n[bold]Select the base model:[/bold]\n")

        # List available models
        models = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                size_gb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / (
                    1024**3
                )

                models.append({"name": model_dir.name, "path": model_dir, "size_gb": size_gb})

        if not models:
            raise ValueError("No models found in the directory")

        # Show table
        table = Table(title="Available Models")
        table.add_column("#", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Size", justify="right")

        for i, model in enumerate(models, 1):
            table.add_row(str(i), model["name"], f"{model['size_gb']:.1f} GB")

        console.print(table)

        # Check compatibility with selected method
        if self.peft_method == PEFTMethod.QLORA:
            console.print("\n[yellow]QLoRA requires a GPU with INT4 support[/yellow]")
        elif self.peft_method == PEFTMethod.GALORE:
            console.print("\n[yellow]GaLore requires optimizer modifications[/yellow]")

        while True:
            choice = IntPrompt.ask("\nSelect model", default=1)
            if 1 <= choice <= len(models):
                return models[choice - 1]["name"]
            console.print("[red]Invalid option[/red]")

    def _select_and_configure_datasets(self) -> List[DatasetConfig]:
        """Selects and configures datasets."""
        console.print("\n[bold]Dataset configuration:[/bold]\n")

        available = self.dataset_manager.scan_datasets()

        if not available:
            raise ValueError("No datasets found")

        # Show table
        table = Table(title="Available Datasets")
        table.add_column("#", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Format")
        table.add_column("Records", justify="right")

        for i, dataset in enumerate(available, 1):
            # Get size from various possible fields
            size = dataset.get(
                "size", dataset.get("estimated_rows", dataset.get("num_rows", "N/A"))
            )
            table.add_row(str(i), dataset["name"], dataset["format"].upper(), str(size))

        console.print(table)

        # Select
        selected = []
        console.print("\n[dim]Select datasets (empty to finish)[/dim]")

        while True:
            choice = Prompt.ask("Dataset #", default="")
            if not choice:
                break

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    config = self.dataset_manager.configure_dataset_interactive(available[idx])
                    if config:
                        selected.append(config)
                        console.print("[green] Added[/green]")
            except Exception:
                console.print("[red]Invalid input[/red]")

        if not selected:
            raise ValueError("You must select at least one dataset")

        return selected

    def _configure_peft_method(self) -> BasePEFTConfig:
        """Configures the selected PEFT method."""
        console.print(f"\n[bold] Configuration for {self.peft_method.value.upper()}:[/bold]\n")

        # Configuration base común
        base_kwargs = {
            "method": self.peft_method,
            "learning_rate": FloatPrompt.ask("Learning rate", default=2e-4),
            "num_train_epochs": IntPrompt.ask("Épocas", default=3),
            "per_device_train_batch_size": IntPrompt.ask("Batch size", default=4),
        }

        # Configuration específica por método
        if self.peft_method == PEFTMethod.LORA:
            console.print("\n[cyan]Parameters LoRA:[/cyan]")
            config = LoRAConfig(
                **base_kwargs,
                r=IntPrompt.ask("Rank (r)", default=16),
                lora_alpha=IntPrompt.ask("Alpha", default=32),
                lora_dropout=FloatPrompt.ask("Dropout", default=0.1),
            )

        elif self.peft_method == PEFTMethod.MOLORA:
            console.print("\n[cyan]Parameters MoLoRA:[/cyan]")
            config = MoLoRAConfig(
                **base_kwargs,
                num_experts=IntPrompt.ask("Number of experts", default=4),
                expert_r=[IntPrompt.ask(f"Rank experto {i + 1}", default=8) for i in range(4)],
            )

        elif self.peft_method == PEFTMethod.GALORE:
            console.print("\n[cyan]Parameters GaLore:[/cyan]")
            config = GaLoreConfig(
                **base_kwargs,
                rank=IntPrompt.ask("Rank gradiente", default=128),
                scale=FloatPrompt.ask("Scale factor", default=0.25),
            )

        elif self.peft_method == PEFTMethod.DORA:
            console.print("\n[cyan]Parameters DoRA:[/cyan]")
            config = DoRAConfig(
                **base_kwargs,
                r=IntPrompt.ask("Rank (r)", default=16),
                lora_alpha=IntPrompt.ask("Alpha", default=32),
                magnitude_lr_scale=FloatPrompt.ask("Magnitude LR scale", default=0.1),
            )

        elif self.peft_method == PEFTMethod.BITFIT:
            console.print("\n[cyan]Parameters BitFit:[/cyan]")
            config = BitFitConfig(
                **base_kwargs,
                train_embeddings=Confirm.ask("¿Train embeddings?", default=False),
                train_layer_norms=Confirm.ask("¿Train layer norms?", default=True),
            )

        elif self.peft_method == PEFTMethod.IA3:
            console.print("\n[cyan]Parameters IA³:[/cyan]")
            config = IA3Config(
                **base_kwargs,
                init_ia3_weights=Prompt.ask(
                    "Initialization", choices=["ones", "zeros"], default="ones"
                ),
            )

        elif self.peft_method == PEFTMethod.PROMPT_TUNING:
            console.print("\n[cyan]Parameters Prompt Tuning:[/cyan]")
            config = PromptTuningConfig(
                **base_kwargs,
                num_virtual_tokens=IntPrompt.ask("Virtual tokens", default=20),
                prompt_tuning_init=Prompt.ask(
                    "Initialization", choices=["random", "text"], default="random"
                ),
            )

        elif self.peft_method == PEFTMethod.ADAPTER:
            console.print("\n[cyan]Parameters Adapter:[/cyan]")
            config = AdapterConfig(
                **base_kwargs,
                adapter_size=IntPrompt.ask("Size adapter", default=64),
                adapter_type=Prompt.ask(
                    "Type", choices=["pfeiffer", "houlsby"], default="pfeiffer"
                ),
            )

        elif self.peft_method == PEFTMethod.QLORA:
            console.print("\n[cyan]Parameters QLoRA:[/cyan]")
            config = QLoRAConfig(
                **base_kwargs,
                r=IntPrompt.ask("Rank (r)", default=16),
                lora_alpha=IntPrompt.ask("Alpha", default=32),
                bits=IntPrompt.ask("Quantization bits", choices=[4, 8], default=4),
            )

        # Target modules (si aplica)
        if hasattr(config, "target_modules") and self.peft_method not in [
            PEFTMethod.BITFIT,
            PEFTMethod.PROMPT_TUNING,
        ]:
            config.target_modules = self._select_target_modules()

        return config

    def _select_target_modules(self) -> List[str]:
        """Selects target modules."""
        console.print("\n[cyan]Target modules:[/cyan]")

        # Common modules by architecture

        console.print("Options:")
        console.print("  [1] All attention modules")
        console.print("  [2] Attention + FFN/MLP")
        console.print("  [3] QV projections only")
        console.print("  [4] Custom")

        choice = Prompt.ask("Selection", choices=["1", "2", "3", "4"], default="2")

        if choice == "1":
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif choice == "2":
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif choice == "3":
            return ["q_proj", "v_proj"]
        else:
            # Custom
            modules = Prompt.ask("Modules (comma-separated)")
            return [m.strip() for m in modules.split(",")]

    def _review_configuration(self) -> bool:
        """Reviews configuration before training."""
        console.print("\n[bold]Configuration summary:[/bold]\n")

        # Create info panels
        method_info = Panel(
            f"[cyan]Method:[/cyan] {self.peft_method.value.upper()}\n"
            f"[cyan]Description:[/cyan] {self.METHOD_DESCRIPTIONS[self.peft_method].split(chr(10))[0]}",
            title="Method PEFT",
            border_style="blue",
        )

        model_info = Panel(
            f"[cyan]Modelo:[/cyan] {self.model_name}\n"
            f"[cyan]Datasets:[/cyan] {len(self.selected_datasets)} archivos",
            title="Data",
            border_style="green",
        )

        # Parameters específicos
        params_text = self._format_config_params()
        params_info = Panel(params_text, title="Parameters", border_style="yellow")

        # Show panels
        console.print(Columns([method_info, model_info]))
        console.print(params_info)

        # Estimación de recursos
        self._estimate_resources()

        return Confirm.ask("\n¿Proceder con el entrenamiento?", default=True)

    def _format_config_params(self) -> str:
        """Formats configuration parameters."""
        lines = []

        # Parameters comunes
        lines.append(f"[cyan]Learning rate:[/cyan] {self.peft_config.learning_rate}")
        lines.append(f"[cyan]Épocas:[/cyan] {self.peft_config.num_train_epochs}")
        lines.append(f"[cyan]Batch size:[/cyan] {self.peft_config.per_device_train_batch_size}")

        # Parameters específicos
        if hasattr(self.peft_config, "r"):
            lines.append(f"[cyan]Rank:[/cyan] {self.peft_config.r}")
        if hasattr(self.peft_config, "lora_alpha"):
            lines.append(f"[cyan]Alpha:[/cyan] {self.peft_config.lora_alpha}")
        if hasattr(self.peft_config, "num_experts"):
            lines.append(f"[cyan]Expertos:[/cyan] {self.peft_config.num_experts}")
        if hasattr(self.peft_config, "adapter_size"):
            lines.append(f"[cyan]Size adapter:[/cyan] {self.peft_config.adapter_size}")
        if hasattr(self.peft_config, "num_virtual_tokens"):
            lines.append(f"[cyan]Virtual tokens:[/cyan] {self.peft_config.num_virtual_tokens}")

        return "\n".join(lines)

    def _estimate_resources(self):
        """Estimates required resources."""
        console.print("\n[dim]Estimación de recursos:[/dim]")

        # Calculate parameters entrenables aproximados
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

        console.print(f"  • Parameters entrenables: ~{trainable_params / 1e6:.1f}M")
        console.print(f"  • Memoria para parameters: ~{trainable_mb:.1f} MB")

        # Verify compatibilidad GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            console.print(f"  • GPU disponible: {torch.cuda.get_device_name(0)}")
            console.print(f"  • Memoria GPU: {gpu_memory:.1f} GB")

            if self.peft_method == PEFTMethod.QLORA and gpu_memory < 24:
                console.print("[yellow]  QLoRA en 4-bit requiere ~24GB para modelos 7B[/yellow]")

    def _run_training(self):
        """Runs training."""
        console.print("\n[bold green]Starting training...[/bold green]\n")

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{self.model_name}_{self.peft_method.value}_{timestamp}"
        output_path = self.output_dir / output_name

        # Prepare data
        all_data = []
        for dataset_config in self.selected_datasets:
            data = self.dataset_manager.load_dataset(dataset_config)
            all_data.extend(data)

        console.print(f"[green][/green] Loaded {len(all_data)} examples")

        # Import universal trainer
        from LoRa_train.peft_universal_trainer import PEFTUniversalTrainer

        # Create trainer
        trainer = PEFTUniversalTrainer(
            model_name=self.model_name,
            model_path=self.models_dir / self.model_name,
            output_dir=output_path,
            peft_config=self.peft_config,
        )

        # Entrenar
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Training...", total=None)

                results = trainer.train(all_data)

                progress.stop()

            # Show resultados
            self._show_results(results, output_path)

        except Exception as e:
            console.print(f"\n[red]Error during training: {str(e)}[/red]")
            raise

    def _show_results(self, results: Dict[str, Any], output_path: Path):
        """Shows training results."""
        console.print("\n[bold green]Training completed![/bold green]\n")

        # Métricas
        if "metrics" in results:
            console.print("[cyan]Final metrics:[/cyan]")
            for key, value in results["metrics"].items():
                if isinstance(value, float):
                    console.print(f"  • {key}: {value:.4f}")
                else:
                    console.print(f"  • {key}: {value}")

        # Archivos guardados
        console.print("\n[cyan]Model saved in:[/cyan]")
        console.print(f"  {output_path}")

        # Information específica del método
        if self.peft_method in [PEFTMethod.LORA, PEFTMethod.QLORA, PEFTMethod.DORA]:
            console.print("\n[cyan]Method information:[/cyan]")
            console.print(f"  • Rank (r): {getattr(self.peft_config, 'r', 'N/A')}")
            console.print(f"  • Alpha: {getattr(self.peft_config, 'lora_alpha', 'N/A')}")

        # Next steps
        console.print("\n[cyan]Next steps:[/cyan]")

        if self.peft_method in [PEFTMethod.LORA, PEFTMethod.QLORA, PEFTMethod.DORA]:
            console.print("  1. Merge with base model:")
            console.print(f"     python merge_lora.py {output_path}")

        console.print("\n  2. Test the model:")
        console.print(f"     python test_model.py {output_path}")

        console.print("\n  3. Evaluate performance:")
        console.print(
            f"     python evaluate_model.py {output_path} --method {self.peft_method.value}"
        )


def main():
    parser = argparse.ArgumentParser(description="Universal fine-tuning with PEFT methods")

    parser.add_argument("--models-dir", default="./models", help="Directorio de modelos")
    parser.add_argument("--datasets-dir", default="./datasets", help="Directorio de datasets")
    parser.add_argument("--output-dir", default="./finetuned_models", help="Directorio de salida")
    parser.add_argument(
        "--method", type=str, help="Method PEFT directo (omite selección interactiva)"
    )

    args = parser.parse_args()

    # Create wizard
    wizard = PEFTFineTuneWizard(
        models_dir=args.models_dir, datasets_dir=args.datasets_dir, output_dir=args.output_dir
    )

    # Si se especificó método, usarlo directamente
    if args.method:
        try:
            wizard.peft_method = PEFTMethod(args.method.lower())
        except ValueError:
            console.print(f"[red]Method inválido: {args.method}[/red]")
            console.print(f"Methods válidos: {[m.value for m in PEFTMethod]}")
            sys.exit(1)

    # Run
    try:
        wizard.run()
    except Exception as e:
        console.print(f"\n[red]Fatal error: {str(e)}[/red]")
        logger.exception("Error en PEFT wizard")
        sys.exit(1)


if __name__ == "__main__":
    main()
