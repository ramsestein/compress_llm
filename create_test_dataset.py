#!/usr/bin/env python3
"""
Script para crear un dataset de prueba simple
"""
import pandas as pd
from pathlib import Path

def create_test_dataset():
    """Crea un dataset de prueba simple para fine-tuning"""
    
    print("🧪 Creando dataset de prueba...")
    
    # Crear datos de prueba
    test_data = [
        {
            "linea": 1,
            "catalan": "Hola, com estàs?",
            "chino": "你好，你好吗？"
        },
        {
            "linea": 2,
            "catalan": "Bon dia, què tal?",
            "chino": "早上好，怎么样？"
        },
        {
            "linea": 3,
            "catalan": "Gràcies per la teva ajuda.",
            "chino": "谢谢你的帮助。"
        },
        {
            "linea": 4,
            "catalan": "De res, és un plaer ajudar-te.",
            "chino": "不客气，很高兴能帮助你。"
        },
        {
            "linea": 5,
            "catalan": "On vius?",
            "chino": "你住在哪里？"
        },
        {
            "linea": 6,
            "catalan": "Visc a Barcelona.",
            "chino": "我住在巴塞罗那。"
        },
        {
            "linea": 7,
            "catalan": "Què fas per feina?",
            "chino": "你做什么工作？"
        },
        {
            "linea": 8,
            "catalan": "Sóc programador.",
            "chino": "我是程序员。"
        },
        {
            "linea": 9,
            "catalan": "M'agrada molt la teva ciutat.",
            "chino": "我很喜欢你的城市。"
        },
        {
            "linea": 10,
            "catalan": "Gràcies, és molt bonica.",
            "chino": "谢谢，它很漂亮。"
        }
    ]
    
    # Crear DataFrame
    df = pd.DataFrame(test_data)
    
    # Guardar dataset
    output_path = Path("datasets/test_dataset.csv")
    df.to_csv(output_path, index=False)
    
    print(f"💾 Dataset de prueba guardado en: {output_path}")
    print(f"📊 Total de muestras: {len(df)}")
    print(f"📋 Columnas: {list(df.columns)}")
    
    # Mostrar primeras filas
    print("\n🔍 Dataset de prueba:")
    print(df.to_string())
    
    print("\n✅ Dataset de prueba creado exitosamente!")

if __name__ == "__main__":
    create_test_dataset()
