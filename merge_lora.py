#!/usr/bin/env python3
"""
Script para fusionar adaptadores LoRA con el modelo base
Crea un modelo completo sin dependencia de PEFT
"""
import os
import sys
import torch
from pathlib import Path
import argparse
import json
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Confirm
import gc

console = Console()

def merge_lora_adapter(lora_model_path: str, output_path: str, 
                      push_to_hub: bool = False, hub_model_id: str = None):
    """Fusiona adaptadores LoRA con el modelo base"""
    
    lora_path = Path(lora_model_path)
    output_path = Path(output_path)
    
    # Verificar que es un modelo LoRA
    if not (lora_path / "adapter_config.json").exists():
        console.print(f"[red]Error: No se encontró adapter_config.json en {lora_path}[/red]")
        sys.exit(1)
    
    # Crear directorio de salida
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold cyan]🔀 Fusionando modelo LoRA[/bold cyan]")
    console.print(f"📁 LoRA: {lora_path}")
    console.print(f"📁 Salida: {output_path}\n")
    
    try:
        # Leer configuración del adaptador
        with open(lora_path / "adapter_config.json", 'r') as f:
            adapter_config = json.load(f)
        
        base_model_path = adapter_config.get('base_model_name_or_path', '')
        
        # Encontrar modelo base
        if not Path(base_model_path).exists():
            # Buscar en directorio models/
            base_name = Path(base_model_path).name
            base_model_path = Path("models") / base_name
            
            if not base_model_path.exists():
                console.print(f"[red]Error: No se encontró el modelo base en: {base_model_path}[/red]")
                console.print("\n💡 Opciones:")
                console.print("1. Coloca el modelo base en el directorio 'models/'")
                console.print("2. Especifica la ruta correcta con --base-model")
                sys.exit(1)
        
        console.print(f"📦 Modelo base: {base_model_path}")
        
        # Detectar dispositivo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"🖥️ Device: {device}\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            # Cargar tokenizer
            task1 = progress.add_task("Cargando tokenizer...", total=1)
            tokenizer = AutoTokenizer.from_pretrained(str(lora_path))
            progress.update(task1, advance=1)
            
            # Cargar modelo base
            task2 = progress.add_task("Cargando modelo base...", total=1)
            
            if device.type == "cuda":
                # Cargar en FP16 para ahorrar memoria
                model = AutoModelForCausalLM.from_pretrained(
                    str(base_model_path),
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    str(base_model_path),
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            
            progress.update(task2, advance=1)
            
            # Cargar adaptadores LoRA
            task3 = progress.add_task("Cargando adaptadores LoRA...", total=1)
            model = PeftModel.from_pretrained(model, str(lora_path))
            progress.update(task3, advance=1)
            
            # Fusionar adaptadores
            task4 = progress.add_task("Fusionando adaptadores...", total=1)
            model = model.merge_and_unload()
            progress.update(task4, advance=1)
            
            # Guardar modelo fusionado
            task5 = progress.add_task("Guardando modelo fusionado...", total=1)
            model.save_pretrained(
                str(output_path),
                safe_serialization=True,
                max_shard_size="5GB"
            )
            tokenizer.save_pretrained(str(output_path))
            progress.update(task5, advance=1)
        
        # Copiar archivos adicionales del modelo base
        console.print("\n📄 Copiando archivos adicionales...")
        files_to_copy = [
            'generation_config.json',
            'special_tokens_map.json',
            'tokenizer.model',  # Para modelos Llama
            'vocab.json',
            'merges.txt',
            'added_tokens.json'
        ]
        
        for filename in files_to_copy:
            src = Path(base_model_path) / filename
            if src.exists():
                dst = output_path / filename
                shutil.copy2(src, dst)
                console.print(f"  ✓ {filename}")
        
        # Crear metadata del merge
        merge_info = {
            'merge_date': torch.tensor(0).numpy().item(),  # Timestamp
            'base_model': str(base_model_path),
            'lora_model': str(lora_path),
            'adapter_config': adapter_config,
            'merge_method': 'merge_and_unload',
            'device_used': str(device)
        }
        
        with open(output_path / "merge_info.json", 'w') as f:
            json.dump(merge_info, f, indent=2)
        
        # Calcular tamaño final
        total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
        console.print(f"\n✅ [bold green]¡Fusión completada![/bold green]")
        console.print(f"📦 Tamaño del modelo: {total_size / (1024**3):.2f} GB")
        console.print(f"📁 Guardado en: {output_path}")
        
        # Subir a HuggingFace Hub si se solicita
        if push_to_hub and hub_model_id:
            console.print(f"\n📤 Subiendo a HuggingFace Hub...")
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                
                # Crear repositorio si no existe
                api.create_repo(hub_model_id, exist_ok=True)
                
                # Subir archivos
                api.upload_folder(
                    folder_path=str(output_path),
                    repo_id=hub_model_id,
                    commit_message="Upload merged LoRA model"
                )
                
                console.print(f"✅ Modelo subido a: https://huggingface.co/{hub_model_id}")
            except Exception as e:
                console.print(f"[red]Error subiendo a Hub: {str(e)}[/red]")
        
        # Limpiar memoria
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
    except Exception as e:
        console.print(f"[red]Error durante la fusión: {str(e)}[/red]")
        raise

def verify_merged_model(model_path: Path):
    """Verifica que el modelo fusionado funcione correctamente"""
    console.print("\n🔍 Verificando modelo fusionado...")
    
    try:
        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        # Cargar modelo (solo config para verificación rápida)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(str(model_path))
        
        console.print("✅ Configuración válida")
        console.print(f"   Arquitectura: {config.model_type}")
        console.print(f"   Capas: {getattr(config, 'num_hidden_layers', 'N/A')}")
        
        # Verificar archivos del modelo
        model_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
        console.print(f"   Archivos del modelo: {len(model_files)}")
        
        # Test rápido de generación (opcional)
        if Confirm.ask("\n¿Realizar test de generación?", default=False):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            console.print("\n⏳ Cargando modelo para test...")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device_map="auto" if device.type == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if device.type == "cpu":
                model = model.to(device)
            
            # Generar texto de prueba
            prompt = "Hello, my name is"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            console.print(f"\n📝 Prompt: '{prompt}'")
            console.print("🤖 Generando...")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            console.print(f"✅ Output: '{generated}'")
            
            # Limpiar
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        console.print(f"[red]Error en verificación: {str(e)}[/red]")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Fusiona adaptadores LoRA con el modelo base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Fusionar LoRA local
  python merge_lora.py ./finetuned_models/mi_lora ./models/mi_modelo_completo
  
  # Especificar modelo base diferente
  python merge_lora.py ./mi_lora ./output --base-model ./models/llama-7b
  
  # Fusionar y subir a HuggingFace
  python merge_lora.py ./mi_lora ./output --push-to-hub --hub-model-id usuario/modelo
        """
    )
    
    parser.add_argument(
        'lora_model',
        type=str,
        help='Ruta al modelo LoRA (con adapter_config.json)'
    )
    
    parser.add_argument(
        'output_path',
        type=str,
        help='Ruta de salida para el modelo fusionado'
    )
    
    parser.add_argument(
        '--base-model',
        type=str,
        help='Ruta al modelo base (si no se detecta automáticamente)'
    )
    
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Subir el modelo fusionado a HuggingFace Hub'
    )
    
    parser.add_argument(
        '--hub-model-id',
        type=str,
        help='ID del modelo en HuggingFace Hub (usuario/modelo)'
    )
    
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Omitir verificación del modelo fusionado'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Sobrescribir si el directorio de salida existe'
    )
    
    args = parser.parse_args()
    
    # Validar argumentos
    lora_path = Path(args.lora_model)
    output_path = Path(args.output_path)
    
    if not lora_path.exists():
        console.print(f"[red]Error: No se encontró el modelo LoRA en {lora_path}[/red]")
        sys.exit(1)
    
    if output_path.exists() and not args.force:
        console.print(f"[red]Error: El directorio de salida ya existe: {output_path}[/red]")
        console.print("Usa --force para sobrescribir")
        sys.exit(1)
    
    if args.push_to_hub and not args.hub_model_id:
        console.print("[red]Error: --hub-model-id es requerido cuando se usa --push-to-hub[/red]")
        sys.exit(1)
    
    # Si se especificó modelo base, actualizar el adapter_config
    if args.base_model:
        adapter_config_path = lora_path / "adapter_config.json"
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        adapter_config['base_model_name_or_path'] = args.base_model
        
        # Guardar actualización temporal
        with open(adapter_config_path, 'w') as f:
            json.dump(adapter_config, f, indent=2)
    
    try:
        # Ejecutar fusión
        merge_lora_adapter(
            args.lora_model,
            args.output_path,
            args.push_to_hub,
            args.hub_model_id
        )
        
        # Verificar si se solicita
        if not args.no_verify:
            if verify_merged_model(output_path):
                console.print("\n✅ [bold green]Modelo fusionado y verificado correctamente[/bold green]")
            else:
                console.print("\n⚠️  [yellow]Advertencia: La verificación encontró problemas[/yellow]")
        
        # Sugerencias finales
        console.print("\n📝 Próximos pasos:")
        console.print(f"1. Probar el modelo:")
        console.print(f"   python test_model.py {output_path}")
        console.print(f"\n2. Usar con Ollama-compatible server:")
        console.print(f"   python ollama_compact_server.py --model {output_path}")
        
        if not args.push_to_hub:
            console.print(f"\n3. Subir a HuggingFace Hub:")
            console.print(f"   huggingface-cli upload {output_path} usuario/nombre-modelo")
        
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()