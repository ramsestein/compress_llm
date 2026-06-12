#!/usr/bin/env python3
"""Script para testear el procesamiento del dataset."""

from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


def test_dataset_processing():
    """Test simple para verificar el procesamiento del dataset."""
    print(" Testeando procesamiento del dataset...")

    # Load el dataset CSV
    dataset_path = Path("datasets/muestra_traducciones_10000.csv")

    if not dataset_path.exists():
        print("Not found el dataset")
        return

    print(f"Dataset encontrado: {dataset_path}")

    try:
        # Load CSV
        df = pd.read_csv(dataset_path)
        print(f"Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")
        print(f"Columnas: {list(df.columns)}")

        # Show primeras filas
        print("\nPrimeras 3 filas:")
        print(df.head(3).to_string())

        # Verify si hay datos
        if len(df) == 0:
            print("Dataset vacío")
            return

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("models/microsoft_DialoGPT-small")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"\nTokenizer cargado: {tokenizer.name_or_path}")

        # Process una fila de ejemplo
        print("\nProcesando fila de ejemplo...")

        # Tomar la primera fila
        example_row = df.iloc[0]
        print(f"Fila de ejemplo: {example_row.to_dict()}")

        # Create texto de entrada (asumiendo que hay columnas de texto)
        text_columns = [
            col
            for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in ["text", "input", "source", "target", "translation"]
            )
        ]

        if text_columns:
            print(f"Columnas de texto encontradas: {text_columns}")

            # Usar la primera columna de texto
            text_col = text_columns[0]
            input_text = str(example_row[text_col])

            print(f"Texto de entrada: {input_text}")

            # Tokenizar
            tokens = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

            print(" Tokens generados:")
            print(f"  - input_ids shape: {tokens['input_ids'].shape}")
            print(f"  - attention_mask shape: {tokens['attention_mask'].shape}")
            print(f"  - Texto decodificado: {tokenizer.decode(tokens['input_ids'][0])}")

            # Create dataset de Hugging Face
            print("\nCreando dataset de Hugging Face...")

            # Create lista de textos
            texts = df[text_col].astype(str).tolist()
            print(f"Total de textos: {len(texts)}")

            # Create dataset
            dataset = Dataset.from_dict({"text": texts})
            print(f"Dataset creado: {len(dataset)} muestras")

            # Tokenizar todo el dataset
            print("\nTokenizando dataset completo...")

            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors=None,
                )

            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            print(f"Dataset tokenizado: {len(tokenized_dataset)} muestras")

            # Verify estructura
            print("\nEstructura del dataset tokenizado:")
            print(f"  - Columnas: {tokenized_dataset.column_names}")
            if len(tokenized_dataset) > 0:
                print(f"  - Primera muestra: {tokenized_dataset[0]}")

        else:
            print("No se encontraron columnas de texto")
            print(f"Columnas disponibles: {list(df.columns)}")

        print("\nTest completado!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_dataset_processing()
