#!/usr/bin/env python3
"""
Script para limpiar y reformatear el dataset CSV problemático
"""
import pandas as pd
import csv
from pathlib import Path
import re

def clean_dataset():
    """Limpia y reformatea el dataset CSV"""
    
    print("🧹 Limpiando dataset CSV...")
    
    input_path = Path("datasets/muestra_traducciones_10000.csv")
    output_path = Path("datasets/muestra_traducciones_10000_clean.csv")
    
    if not input_path.exists():
        print("❌ No se encontró el dataset original")
        return
    
    try:
        # Leer el archivo línea por línea para entender su estructura
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        print(f"📊 Total de líneas: {len(lines)}")
        
        # Analizar las primeras líneas para entender la estructura
        print("\n🔍 Analizando estructura...")
        for i, line in enumerate(lines[:10]):
            print(f"Línea {i+1}: {repr(line.strip())}")
        
        # Encontrar la línea de encabezados
        header_line = None
        for i, line in enumerate(lines):
            if 'linea' in line.lower() and 'catalan' in line.lower():
                header_line = i
                break
        
        if header_line is None:
            print("❌ No se encontró la línea de encabezados")
            return
        
        print(f"\n📋 Encabezados encontrados en línea {header_line + 1}")
        
        # Leer desde la línea de encabezados
        clean_lines = []
        
        # Agregar encabezados
        clean_lines.append("linea,catalan,chino\n")
        
        # Procesar líneas de datos
        data_lines = 0
        for i, line in enumerate(lines[header_line + 1:], header_line + 2):
            line = line.strip()
            if not line:
                continue
            
            # Limpiar la línea
            cleaned_line = clean_csv_line(line)
            if cleaned_line:
                clean_lines.append(cleaned_line + "\n")
                data_lines += 1
        
        print(f"✅ Líneas de datos limpiadas: {data_lines}")
        
        # Escribir archivo limpio
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(clean_lines)
        
        print(f"💾 Dataset limpio guardado en: {output_path}")
        
        # Verificar que se puede leer correctamente
        print("\n🔍 Verificando archivo limpio...")
        df = pd.read_csv(output_path)
        print(f"📊 Dataset limpio: {len(df)} filas, {len(df.columns)} columnas")
        print(f"📋 Columnas: {list(df.columns)}")
        
        # Mostrar primeras filas
        print("\n🔍 Primeras 3 filas del dataset limpio:")
        print(df.head(3).to_string())
        
        print("\n✅ Limpieza completada exitosamente!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def clean_csv_line(line):
    """Limpia una línea del CSV"""
    
    # Eliminar comillas extra al inicio y final
    line = line.strip('"')
    
    # Buscar el patrón: número, texto en catalán, texto en chino
    # Patrón: número, "texto catalán", "texto chino"
    pattern = r'^(\d+),(".*?"),(".*?").*$'
    match = re.match(pattern, line)
    
    if match:
        linea = match.group(1)
        catalan = match.group(2).strip('"')
        chino = match.group(3).strip('"')
        
        # Limpiar texto
        catalan = clean_text(catalan)
        chino = clean_text(chino)
        
        # Verificar que no estén vacíos
        if catalan and chino and len(catalan) > 10 and len(chino) > 10:
            return f'{linea},"{catalan}","{chino}"'
    
    return None

def clean_text(text):
    """Limpia el texto eliminando caracteres problemáticos"""
    
    # Eliminar caracteres de control
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminar caracteres extraños
    text = re.sub(r'[^\w\s\.,!?;:()"\'-]', '', text)
    
    return text.strip()

if __name__ == "__main__":
    clean_dataset()
