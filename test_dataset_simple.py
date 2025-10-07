#!/usr/bin/env python3
"""
Script simple para probar el dataset de traducciones
"""

import pandas as pd
from transformers import AutoTokenizer

def test_dataset():
    print("🧪 Probando dataset de traducciones...")
    
    # Cargar dataset
    df = pd.read_csv('datasets/test_traducciones.csv')
    print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"📋 Columnas: {list(df.columns)}")
    print(f"📄 Primeras 3 filas:")
    print(df.head(3))
    
    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer cargado: {tokenizer.name_or_path}")
    
    # Probar tokenización
    print("\n🔤 Probando tokenización...")
    for i, row in df.iterrows():
        text = f"{row['catalan']}\n{row['chino']}"
        print(f"  Texto {i+1}: {text[:50]}...")
        
        try:
            encoded = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors=None
            )
            print(f"    ✅ Tokenizado: {len(encoded['input_ids'])} tokens")
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    print("\n🎉 Prueba completada!")

if __name__ == "__main__":
    test_dataset()

