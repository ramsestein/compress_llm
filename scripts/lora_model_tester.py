#!/usr/bin/env python3
"""Script para probar modelos fine-tuneados con LoRA."""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()


class LoRAModelTester:
    """Probador interactivo de modelos LoRA."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Verify que es un modelo LoRA
        if not (self.model_path / "adapter_config.json").exists():
            raise ValueError(f"Not found adapter_config.json en {model_path}")

        # Load info del entrenamiento
        self.training_info = self._load_training_info()

        # Componentes del modelo
        self.base_model = None
        self.model = None
        self.tokenizer = None

    def _load_training_info(self) -> dict:
        """Carga información del entrenamiento."""
        info_path = self.model_path / "training_info.json"
        if info_path.exists():
            with open(info_path) as f:
                return json.load(f)

        # Si no existe, intentar leer adapter_config
        adapter_config_path = self.model_path / "adapter_config.json"
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)

        return {"base_model_path": adapter_config.get("base_model_name_or_path", "unknown")}

    def load_model(self):
        """Carga el modelo con adaptadores LoRA."""
        console.print("[cyan]Cargando modelo...[/cyan]")

        # Determinar ruta del modelo base
        base_model_path = self.training_info.get("base_model_path", "")
        if not Path(base_model_path).exists():
            # Intentar en directorio models/
            base_name = Path(base_model_path).name
            base_model_path = Path("models") / base_name

            if not base_model_path.exists():
                raise ValueError(f"Not found el modelo base en: {base_model_path}")

        console.print(f"Modelo base: {base_model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load modelo base
        console.print("⏳ Cargando modelo base...")
        if self.device.type == "cuda":
            self.base_model = AutoModelForCausalLM.from_pretrained(
                str(base_model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                str(base_model_path),
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        # Load adaptadores LoRA
        console.print("Cargando adaptadores LoRA...")
        self.model = PeftModel.from_pretrained(self.base_model, str(self.model_path))

        # Modo evaluación
        self.model.eval()

        console.print(f"Modelo cargado en: {self.device}")

        # Show info del modelo
        adapter_config_path = self.model_path / "adapter_config.json"
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)

        console.print("\n[bold]Información del modelo:[/bold]")
        console.print(f"  • LoRA rank: {adapter_config.get('r', 'N/A')}")
        console.print(f"  • Alpha: {adapter_config.get('lora_alpha', 'N/A')}")
        console.print(f"  • Target modules: {', '.join(adapter_config.get('target_modules', []))}")

    def generate(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Genera texto con el modelo."""
        # Tokenizar
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decodificar
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remover el prompt de la respuesta
        if response.startswith(prompt):
            response = response[len(prompt) :].strip()

        return response

    def interactive_test(self):
        """Modo de prueba interactiva."""
        console.clear()
        console.print(
            Panel(
                f"[bold cyan]Probador de Modelo LoRA[/bold cyan]\n\n"
                f"Modelo: {self.model_path.name}\n"
                f"Device: {self.device}\n\n"
                f"[dim]Escribe 'salir' para terminar[/dim]",
                title=" Chat Interactivo",
                border_style="cyan",
            )
        )

        # Load modelo
        self.load_model()

        console.print("\n[green]¡Modelo listo! Puedes empezar a chatear.[/green]\n")

        # Loop interactivo
        while True:
            # Get prompt
            prompt = Prompt.ask("\n[bold blue]Tú[/bold blue]")

            if prompt.lower() in ["salir", "exit", "quit"]:
                console.print("\n[yellow]¡Hasta luego![/yellow]")
                break

            # Generate respuesta
            console.print("\n[bold green]Modelo[/bold green]: ", end="")

            try:
                response = self.generate(prompt)
                console.print(response)
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")

    def batch_test(self, test_prompts: list):
        """Prueba el modelo con múltiples prompts."""
        console.print("[cyan]Ejecutando pruebas batch...[/cyan]\n")

        # Load modelo
        self.load_model()

        results = []

        for i, prompt in enumerate(test_prompts, 1):
            console.print(f"\n[bold]Test {i}/{len(test_prompts)}:[/bold]")
            console.print(f"[blue]Prompt:[/blue] {prompt}")

            try:
                response = self.generate(prompt)
                console.print(f"[green]Respuesta:[/green] {response}")

                results.append({"prompt": prompt, "response": response, "success": True})
            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")
                results.append({"prompt": prompt, "error": str(e), "success": False})

        # Resumen
        successful = sum(1 for r in results if r["success"])
        console.print("\n[bold]Resumen:[/bold]")
        console.print(f"  • Tests exitosos: {successful}/{len(test_prompts)}")

        return results

    def compare_with_base(self, prompts: list):
        """Compara respuestas con el modelo base."""
        console.print("[cyan]Comparando con modelo base...[/cyan]\n")

        # Load modelo
        self.load_model()

        comparisons = []

        for prompt in prompts:
            console.print(f"\n[bold]Prompt:[/bold] {prompt}")

            # Generate con LoRA
            console.print("\n[green]Con LoRA:[/green]")
            lora_response = self.generate(prompt)
            console.print(f"  {lora_response}")

            # Generate sin LoRA (desactivar adaptadores)
            self.model.disable_adapter_layers()

            console.print("\n[yellow]Sin LoRA (base):[/yellow]")
            base_response = self.generate(prompt)
            console.print(f"  {base_response}")

            # Reactivar adaptadores
            self.model.enable_adapter_layers()

            comparisons.append(
                {"prompt": prompt, "lora_response": lora_response, "base_response": base_response}
            )

        return comparisons


def main():
    parser = argparse.ArgumentParser(description="Prueba modelos fine-tuneados con LoRA")

    parser.add_argument("model_path", type=str, help="Ruta al modelo LoRA")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "batch", "compare"],
        default="interactive",
        help="Modo de prueba (default: interactive)",
    )

    parser.add_argument("--prompts-file", type=str, help="Archivo con prompts para modo batch")

    parser.add_argument(
        "--max-length", type=int, default=200, help="Longitud máxima de generación (default: 200)"
    )

    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperatura de generación (default: 0.7)"
    )

    args = parser.parse_args()

    # Verify que el modelo existe
    if not Path(args.model_path).exists():
        console.print(f"[red]Error: Not found el modelo en {args.model_path}[/red]")
        sys.exit(1)

    # Create tester
    tester = LoRAModelTester(args.model_path)

    try:
        if args.mode == "interactive":
            # Modo interactivo
            tester.interactive_test()

        elif args.mode == "batch":
            # Modo batch
            if not args.prompts_file:
                # Prompts de ejemplo
                test_prompts = [
                    "Explica qué es el machine learning en términos simples",
                    "¿Cuáles son los beneficios de hacer ejercicio?",
                    "Escribe un haiku sobre la naturaleza",
                    "¿Cómo funciona un motor de combustión?",
                    "Dame 3 consejos para aprender a programar",
                ]
            else:
                # Load prompts del archivo
                with open(args.prompts_file) as f:
                    test_prompts = [line.strip() for line in f if line.strip()]

            results = tester.batch_test(test_prompts)

            # Save resultados
            output_file = Path(args.model_path) / "test_results.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            console.print(f"\nResultados guardados en: {output_file}")

        elif args.mode == "compare":
            # Modo comparación
            compare_prompts = [
                "¿Qué es la inteligencia artificial?",
                "Escribe un párrafo sobre el cambio climático",
                "¿Cómo se hace una tortilla española?",
            ]

            if args.prompts_file:
                with open(args.prompts_file) as f:
                    compare_prompts = [line.strip() for line in f if line.strip()][:5]

            comparisons = tester.compare_with_base(compare_prompts)

            # Analizar diferencias
            console.print("\n[bold]Análisis de diferencias:[/bold]")

            total_chars_base = sum(len(c["base_response"]) for c in comparisons)
            total_chars_lora = sum(len(c["lora_response"]) for c in comparisons)

            console.print(
                f"  • Longitud promedio base: {total_chars_base / len(comparisons):.0f} chars"
            )
            console.print(
                f"  • Longitud promedio LoRA: {total_chars_lora / len(comparisons):.0f} chars"
            )

            # Calculate similitud simple
            similar_words = 0
            total_words = 0

            for comp in comparisons:
                base_words = set(comp["base_response"].lower().split())
                lora_words = set(comp["lora_response"].lower().split())
                similar_words += len(base_words.intersection(lora_words))
                total_words += len(base_words.union(lora_words))

            similarity = (similar_words / total_words * 100) if total_words > 0 else 0
            console.print(f"  • Similitud aproximada: {similarity:.1f}%")

            # Save comparaciones
            output_file = Path(args.model_path) / "comparison_results.json"
            with open(output_file, "w") as f:
                json.dump(comparisons, f, indent=2, ensure_ascii=False)

            console.print(f"\nComparaciones guardadas en: {output_file}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Prueba interrumpida[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
