#!/usr/bin/env python3
"""Script para crear un dataset de prueba simple."""

from pathlib import Path

import pandas as pd


def create_test_dataset():
    """Crea un dataset de prueba simple para fine-tuning."""
    print(" Creando dataset de prueba...")

    # Create datos de prueba
    test_data = [
        {"linea": 1, "catalan": "Hola, com estàs?", "chino": ""},
        {"linea": 2, "catalan": "Bon dia, què tal?", "chino": ""},
        {"linea": 3, "catalan": "Gràcies per la teva ajuda.", "chino": ""},
        {
            "linea": 4,
            "catalan": "De res, és un plaer ajudar-te.",
            "chino": "",
        },
        {"linea": 5, "catalan": "On vius?", "chino": ""},
        {"linea": 6, "catalan": "Visc a Barcelona.", "chino": ""},
        {"linea": 7, "catalan": "Què fas per feina?", "chino": ""},
        {"linea": 8, "catalan": "Sóc programador.", "chino": ""},
        {"linea": 9, "catalan": "M'agrada molt la teva ciutat.", "chino": ""},
        {"linea": 10, "catalan": "Gràcies, és molt bonica.", "chino": ""},
    ]

    # Create DataFrame
    df = pd.DataFrame(test_data)

    # Save dataset
    output_path = Path("datasets/test_dataset.csv")
    df.to_csv(output_path, index=False)

    print(f"Dataset de prueba guardado en: {output_path}")
    print(f"Total de muestras: {len(df)}")
    print(f"Columnas: {list(df.columns)}")

    # Show primeras filas
    print("\nDataset de prueba:")
    print(df.to_string())

    print("\nDataset de prueba creado exitosamente!")


if __name__ == "__main__":
    create_test_dataset()
