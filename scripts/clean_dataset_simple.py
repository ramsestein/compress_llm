#!/usr/bin/env python3
"""Script simple para limpiar el dataset CSV."""

import csv
from pathlib import Path

import pandas as pd


def clean_dataset_simple():
    """Limpia el dataset CSV de forma simple."""
    print(" Limpiando dataset CSV (método simple)...")

    input_path = Path("datasets/muestra_traducciones_10000.csv")
    output_path = Path("datasets/muestra_traducciones_10000_clean.csv")

    if not input_path.exists():
        print("Not found el dataset original")
        return

    try:
        # Leer el archivo como texto
        with open(input_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        print(f"Size del archivo: {len(content)} caracteres")

        # Dividir en líneas
        lines = content.split("\n")
        print(f"Total de líneas: {len(lines)}")

        # Encontrar la línea de encabezados
        header_line = None
        for i, line in enumerate(lines):
            if "linea" in line.lower() and "catalan" in line.lower():
                header_line = i
                break

        if header_line is None:
            print("Not found la línea de encabezados")
            return

        print(f"Encabezados encontrados en línea {header_line + 1}")

        # Process líneas de datos
        clean_data = []
        clean_data.append(["linea", "catalan", "chino"])

        data_lines = 0
        for i, line in enumerate(lines[header_line + 1 :], header_line + 2):
            line = line.strip()
            if not line:
                continue

            # Limpiar la línea
            cleaned_row = clean_line_simple(line)
            if cleaned_row:
                clean_data.append(cleaned_row)
                data_lines += 1

        print(f"Líneas de datos limpiadas: {data_lines}")

        # Escribir archivo limpio
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(clean_data)

        print(f"Dataset limpio guardado en: {output_path}")

        # Verify que se puede leer correctamente
        print("\nVerificando archivo limpio...")
        df = pd.read_csv(output_path)
        print(f"Dataset limpio: {len(df)} filas, {len(df.columns)} columnas")
        print(f"Columnas: {list(df.columns)}")

        # Show primeras filas
        if len(df) > 0:
            print("\nPrimeras 3 filas del dataset limpio:")
            print(df.head(3).to_string())
        else:
            print("\nNo se generaron datos limpios")

        print("\nLimpieza completada!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


def clean_line_simple(line):
    """Limpia una línea de forma simple."""
    try:
        # Eliminar comillas extra al inicio y final
        line = line.strip('"')

        # Buscar el primer número (ID de línea)
        parts = line.split(",")
        if len(parts) < 3:
            return None

        # El primer campo debe ser un número
        try:
            linea = int(parts[0])
        except ValueError:
            return None

        # Buscar el texto en catalán (entre comillas)
        catalan_start = line.find('"', 1)
        if catalan_start == -1:
            return None

        catalan_end = line.find('"', catalan_start + 1)
        if catalan_end == -1:
            return None

        catalan = line[catalan_start + 1 : catalan_end]

        # Buscar el texto en chino (después del catalán)
        remaining = line[catalan_end + 1 :]
        if not remaining.startswith(","):
            return None

        # El texto chino está después de la coma
        chino = remaining[1:].split(",")[0].strip('"')

        # Verify que no estén vacíos y tengan longitud mínima
        if (
            catalan
            and chino
            and len(catalan) > 10
            and len(chino) > 10
            and not catalan.startswith(";;")
            and not chino.startswith(";;")
        ):
            return [linea, catalan, chino]

        return None

    except Exception:
        return None


if __name__ == "__main__":
    clean_dataset_simple()
