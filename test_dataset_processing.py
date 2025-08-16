#!/usr/bin/env python3
"""
Script para testear el procesamiento del dataset
"""
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset
import json

def test_dataset_processing():
    """Test simple para verificar el procesamiento del dataset"""
    
    print("🧪 Testeando procesamiento del dataset...")
    
    # Cargar el dataset CSV
    dataset_path = Path("datasets/muestra_traducciones_10000.csv")
    
    if not dataset_path.exists():
        print("❌ No se encontró el dataset")
        return
    
    print(f"📁 Dataset encontrado: {dataset_path}")
    
    try:
        # Cargar CSV
        df = pd.read_csv(dataset_path)
        print(f"📊 Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")
        print(f"📋 Columnas: {list(df.columns)}")
        
        # Mostrar primeras filas
        print("\n🔍 Primeras 3 filas:")
        print(df.head(3).to_string())
        
        # Verificar si hay datos
        if len(df) == 0:
            print("❌ Dataset vacío")
            return
        
        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained("models/microsoft_DialoGPT-small")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"\n🔧 Tokenizer cargado: {tokenizer.name_or_path}")
        
        # Procesar una fila de ejemplo
        print("\n🔧 Procesando fila de ejemplo...")
        
        # Tomar la primera fila
        example_row = df.iloc[0]
        print(f"📝 Fila de ejemplo: {example_row.to_dict()}")
        
        # Crear texto de entrada (asumiendo que hay columnas de texto)
        text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'input', 'source', 'target', 'translation'])]
        
        if text_columns:
            print(f"📝 Columnas de texto encontradas: {text_columns}")
            
            # Usar la primera columna de texto
            text_col = text_columns[0]
            input_text = str(example_row[text_col])
            
            print(f"📝 Texto de entrada: {input_text}")
            
            # Tokenizar
            tokens = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            
            print(f"🔤 Tokens generados:")
            print(f"  - input_ids shape: {tokens['input_ids'].shape}")
            print(f"  - attention_mask shape: {tokens['attention_mask'].shape}")
            print(f"  - Texto decodificado: {tokenizer.decode(tokens['input_ids'][0])}")
            
            # Crear dataset de Hugging Face
            print(f"\n🔧 Creando dataset de Hugging Face...")
            
            # Crear lista de textos
            texts = df[text_col].astype(str).tolist()
            print(f"📝 Total de textos: {len(texts)}")
            
            # Crear dataset
            dataset = Dataset.from_dict({"text": texts})
            print(f"✅ Dataset creado: {len(dataset)} muestras")
            
            # Tokenizar todo el dataset
            print(f"\n🔧 Tokenizando dataset completo...")
            
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors=None
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            print(f"✅ Dataset tokenizado: {len(tokenized_dataset)} muestras")
            
            # Verificar estructura
            print(f"\n🔍 Estructura del dataset tokenizado:")
            print(f"  - Columnas: {tokenized_dataset.column_names}")
            if len(tokenized_dataset) > 0:
                print(f"  - Primera muestra: {tokenized_dataset[0]}")
            
        else:
            print("❌ No se encontraron columnas de texto")
            print(f"📋 Columnas disponibles: {list(df.columns)}")
        
        print("\n✅ Test completado!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_processing()
