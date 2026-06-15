#!/usr/bin/env python3
"""Interactive zero-code CLI for knowledge distillation.

Allows non-technical users to specialize a small local model on any topic
using a large teacher model with just a few prompts.
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt

from compress_llm.distillation import KnowledgeDistiller
from LoRa_train.peft_methods_config import (
    BitFitConfig,
    DoRAConfig,
    IA3Config,
    LoRAConfig,
    PEFTMethod,
    QLoRAConfig,
)

console = Console()


def _show_welcome() -> None:
    welcome_text = """
    [bold cyan]Knowledge Distillation Assistant[/bold cyan]

    This wizard lets you create a specialized small model on any topic
    using a large teacher model. No programming required.

    Steps:
    1. Choose a topic (e.g., "medical diagnosis", "legal contracts")
    2. Pick a teacher model (large LLM) and a student model (small LLM)
    3. Select a fine-tuning method
    4. The teacher generates training data; the student learns from it
    """
    panel = Panel(welcome_text, title="Welcome", border_style="cyan")
    console.print(panel)
    console.print()


def _select_method():
    console.print("[bold]Select a fine-tuning method for the student:[/bold]\n")
    console.print("  1. LoRA       - Balanced speed and quality")
    console.print("  2. QLoRA      - Very low memory (recommended for large students)")
    console.print("  3. DoRA       - Maximum quality")
    console.print("  4. BitFit     - Ultra-fast, minimal memory")
    console.print("  5. IA3        - Efficient scale tuning")
    choice = IntPrompt.ask("Method", default=1)
    methods = {
        1: (PEFTMethod.LORA, LoRAConfig),
        2: (PEFTMethod.QLORA, QLoRAConfig),
        3: (PEFTMethod.DORA, DoRAConfig),
        4: (PEFTMethod.BITFIT, BitFitConfig),
        5: (PEFTMethod.IA3, IA3Config),
    }
    return methods.get(choice, methods[1])


def _build_config(method_cls):
    lr = float(Prompt.ask("Learning rate", default="2e-4"))
    epochs = IntPrompt.ask("Epochs", default=3)
    batch = IntPrompt.ask("Batch size", default=4)
    kwargs = {
        "method": None,
        "learning_rate": lr,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch,
    }
    if method_cls is LoRAConfig:
        kwargs["method"] = PEFTMethod.LORA
        kwargs["r"] = IntPrompt.ask("Rank (r)", default=16)
        kwargs["lora_alpha"] = IntPrompt.ask("Alpha", default=32)
        kwargs["lora_dropout"] = float(Prompt.ask("Dropout", default="0.1"))
    elif method_cls is QLoRAConfig:
        kwargs["method"] = PEFTMethod.QLORA
        kwargs["r"] = IntPrompt.ask("Rank (r)", default=16)
        kwargs["lora_alpha"] = IntPrompt.ask("Alpha", default=32)
        kwargs["bits"] = IntPrompt.ask("Quantization bits", choices=[4, 8], default=4)
    elif method_cls is DoRAConfig:
        kwargs["method"] = PEFTMethod.DORA
        kwargs["r"] = IntPrompt.ask("Rank (r)", default=16)
        kwargs["lora_alpha"] = IntPrompt.ask("Alpha", default=32)
        kwargs["magnitude_lr_scale"] = float(Prompt.ask("Magnitude LR scale", default="0.1"))
    elif method_cls is BitFitConfig:
        kwargs["method"] = PEFTMethod.BITFIT
        kwargs["train_embeddings"] = Confirm.ask("Train embeddings?", default=False)
        kwargs["train_layer_norms"] = Confirm.ask("Train layer norms?", default=True)
    elif method_cls is IA3Config:
        kwargs["method"] = PEFTMethod.IA3
        kwargs["init_ia3_weights"] = Prompt.ask(
            "Initialization", choices=["ones", "zeros"], default="ones"
        )
    return method_cls(**kwargs)


def main() -> None:
    """Run the interactive distillation wizard."""
    _show_welcome()

    try:
        topic = Prompt.ask("Topic to specialize on", default="general knowledge")
        num_samples = IntPrompt.ask("Number of training examples to generate", default=100)
        teacher = Prompt.ask("Teacher model (HuggingFace ID)", default="microsoft/DialoGPT-medium")
        student = Prompt.ask(
            "Student model (HuggingFace ID or local path)", default="microsoft/DialoGPT-small"
        )

        method_enum, method_cls = _select_method()
        peft_config = _build_config(method_cls)

        output_dir = Path(Prompt.ask("Output directory", default="./distilled_models"))

        if not Confirm.ask("Proceed with distillation?", default=True):
            console.print("[yellow]Cancelled.[/yellow]")
            sys.exit(0)

        console.print("[bold green]Starting knowledge distillation...[/bold green]")
        distiller = KnowledgeDistiller(teacher_model_name=teacher, student_model_name=student)
        result = distiller.distill(
            topic=topic,
            num_samples=num_samples,
            output_dir=output_dir,
            peft_config=peft_config,
        )

        console.print("[bold green]Distillation complete![/bold green]")
        console.print(f"[cyan]Dataset:[/cyan] {result['dataset_path']}")
        console.print(f"[cyan]Model:[/cyan] {result['output_dir']}")
        if result.get("metrics"):
            console.print("[cyan]Metrics:[/cyan]")
            for k, v in result["metrics"].items():
                console.print(f"  {k}: {v}")

    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
