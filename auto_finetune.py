#!/usr/bin/env python3
"""
Script automático para fine-tuning con configuración predefinida
"""

import sys
import os
from pathlib import Path

# Agregar el directorio del proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from LoRa_train.peft_universal_trainer import PEFTUniversalTrainer
from LoRa_train.dataset_manager import DatasetManager
from LoRa_train.lora_config import LoRAConfig, PEFTMethod
from rich.console import Console

console = Console()

def main():
    """Función principal del script automático"""
    console.print("🚀 [bold blue]Script Automático de Fine-Tuning[/bold blue]")
    console.print("=" * 50)
    
    try:
        # Configuración automática
        config = setup_automatic_config()
        
        # Inicializar el trainer
        trainer = PEFTUniversalTrainer(config)
        
        # Cargar y preparar datos
        console.print("\n📊 [bold]Preparando datos...[/bold]")
        dataset_manager = DatasetManager()
        
        # Usar el dataset simple
        dataset_path = "datasets/simple_traducciones.csv"
        if not os.path.exists(dataset_path):
            console.print(f"❌ [red]Error: No se encontró el dataset {dataset_path}[/red]")
            return
        
        # Configurar el dataset automáticamente
        dataset_config = {
            'file_path': dataset_path,
            'format': 'CSV',
            'name': 'simple_traducciones',
            'size': 10
        }
        
        # Crear configuración del dataset
        from LoRa_train.dataset_manager import DatasetConfig
        dataset = DatasetConfig(
            file_path=Path(dataset_path),
            format='CSV',
            columns={'instruction': 'text', 'response': 'text'},
            name='simple_traducciones',
            size=10,
            dataset_type='supervised',
            instruction_template='{instruction}\n{response}',
            max_length=512,
            eval_split_ratio=0.2
        )
        
        console.print("✅ [green]Dataset configurado correctamente[/green]")
        
        # Ejecutar entrenamiento
        console.print("\n🚀 [bold]Iniciando entrenamiento...[/bold]")
        results = trainer.train([dataset])
        
        console.print("\n✅ [bold green]¡Entrenamiento completado exitosamente![/bold green]")
        console.print(f"📁 [cyan]Modelo guardado en: {config.output_dir}[/cyan]")
        
    except Exception as e:
        console.print(f"\n❌ [red]Error durante el entrenamiento: {e}[/red]")
        import traceback
        traceback.print_exc()

def setup_automatic_config():
    """Configuración automática del entrenamiento"""
    console.print("\n⚙️ [bold]Configurando parámetros automáticamente...[/bold]")
    
    # Configuración básica
    config = LoRAConfig()
    
    # Método PEFT
    config.peft_method = PEFTMethod.LORA
    console.print("✅ Método: LoRA")
    
    # Modelo
    config.model_name = "distilgpt2"
    console.print("✅ Modelo: distilgpt2")
    
    # Parámetros de entrenamiento
    config.learning_rate = 0.0002
    config.num_epochs = 3
    config.batch_size = 2  # Reducido para el dataset pequeño
    console.print("✅ Parámetros de entrenamiento configurados")
    
    # Parámetros LoRA
    config.lora_r = 16
    config.lora_alpha = 32
    config.lora_dropout = 0.1
    config.target_modules = ["c_attn"]  # Para distilgpt2
    console.print("✅ Parámetros LoRA configurados")
    
    # Directorio de salida
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir = f"finetuned_models/distilgpt2_lora_auto_{timestamp}"
    console.print(f"✅ Directorio de salida: {config.output_dir}")
    
    # Configuración del dataset
    config.max_length = 512
    config.eval_split_ratio = 0.2
    config.instruction_template = "{instruction}\n{response}"
    console.print("✅ Configuración del dataset completada")
    
    return config

if __name__ == "__main__":
    main()


