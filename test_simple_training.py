#!/usr/bin/env python3
"""
Script simple para probar el entrenamiento PEFT sin input interactivo
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LoRa_train.peft_universal_trainer import PEFTUniversalTrainer
# from LoRa_train.dataset_manager import DatasetManager
from LoRa_train.peft_methods_config import PEFTMethod
import pandas as pd

def test_training():
    print("🧪 Iniciando prueba de entrenamiento...")
    
    # Configuración básica
    model_name = "distilgpt2"
    method = PEFTMethod.LORA
    
    # Cargar dataset
    print("📊 Cargando dataset...")
    df = pd.read_csv('datasets/test_traducciones.csv')
    print(f"✅ Dataset cargado: {df.shape[0]} filas")
    print(f"📋 Columnas: {list(df.columns)}")
    
    # Convertir a formato esperado
    training_data = []
    for _, row in df.iterrows():
        training_data.append({
            'catalan': row['catalan'],
            'chino': row['chino']
        })
    
    print(f"✅ Datos preparados: {len(training_data)} muestras")
    
    # Crear configuración PEFT
    print("⚙️ Configurando trainer...")
    from LoRa_train.peft_methods_config import LoRAConfig
    from pathlib import Path
    
    peft_config = LoRAConfig(
        method=PEFTMethod.LORA,
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        learning_rate=0.0002,
        num_train_epochs=1,
        per_device_train_batch_size=2
    )
    
    trainer = PEFTUniversalTrainer(
        model_name=model_name,
        model_path=Path("models/distilgpt2"),
        output_dir=Path("output"),
        peft_config=peft_config
    )
    
    # Entrenar
    print("🚀 Iniciando entrenamiento...")
    try:
        results = trainer.train(training_data)
        print("✅ ¡Entrenamiento completado!")
        print(f"📊 Resultados: {results}")
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training()
