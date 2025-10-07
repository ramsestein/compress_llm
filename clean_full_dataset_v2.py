#!/usr/bin/env python3
"""
Script MEJORADO para limpiar COMPLETAMENTE el dataset de traducciones malformado
"""

import pandas as pd
import re
from pathlib import Path

def clean_full_dataset_v2():
    print("🧹 Limpiando dataset COMPLETO v2.0...")
    
    input_path = Path('datasets/muestra_traducciones_10000.csv')
    output_path = Path('datasets/clean_full_traducciones_v2.csv')
    
    if not input_path.exists():
        print(f"❌ No se encontró el archivo: {input_path}")
        return
    
    print(f"📂 Leyendo archivo: {input_path}")
    
    # Leer línea por línea para parsear el formato malformado
    parsed_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📊 Total de líneas: {len(lines)}")
    
    # Procesar cada línea
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Eliminar el ';;;' final si existe
        line = line.replace(';;;', '')
        
        # Saltar header
        if i == 0 and 'linea,catalan,chino' in line:
            continue
        
        # Patrón 1: "linea,""catalan",""chino"" (comillas escapadas simples)
        match_quoted_simple = re.match(r'^"(\d+),""(.+)"",""(.+)""$', line)
        if match_quoted_simple:
            line_num, catalan, chino = match_quoted_simple.groups()
            catalan = catalan.replace('""', '"')
            chino = chino.replace('""', '"')
            parsed_data.append({
                'catalan': catalan,
                'chino': chino
            })
            continue
        
        # Patrón 2: "linea,""""""catalan"""""","""chino"""" (comillas múltiples)
        match_quoted_multiple = re.match(r'^"(\d+),""""""(.+)"""""","""(.+)""""$', line)
        if match_quoted_multiple:
            line_num, catalan, chino = match_quoted_multiple.groups()
            catalan = catalan.replace('""', '"')
            chino = chino.replace('""', '"')
            parsed_data.append({
                'catalan': catalan,
                'chino': chino
            })
            continue
        
        # Patrón 3: linea,catalan,chino (separado por comas simple)
        parts = line.split(',', 2)  # Dividir solo en 3 partes
        if len(parts) == 3:
            # Verificar si la primera parte es un número (linea)
            if parts[0].isdigit():
                catalan = parts[1].strip()
                chino = parts[2].strip()
                parsed_data.append({
                    'catalan': catalan,
                    'chino': chino
                })
                continue
        
        # Patrón 4: "linea,"catalan","chino"" (comillas normales)
        match_normal_quotes = re.match(r'^"(\d+),"(.+)","(.+)""$', line)
        if match_normal_quotes:
            line_num, catalan, chino = match_normal_quotes.groups()
            parsed_data.append({
                'catalan': catalan,
                'chino': chino
            })
            continue
        
        # Si no coincide con ningún patrón, mostrar advertencia solo para las primeras 20
        if i < 20:
            print(f"⚠️ Línea {i+1} no parseada: {line[:100]}...")
    
    # Crear DataFrame
    df = pd.DataFrame(parsed_data)
    
    print(f"\n✅ Datos parseados: {len(df)} traducciones")
    
    if not df.empty:
        print("\n📄 Primeras 5 traducciones:")
        for i, row in df.head(5).iterrows():
            print(f"  {i+1}. {row['catalan'][:50]}... → {row['chino'][:50]}...")
        
        # Limpiar datos
        print("\n🧹 Limpiando datos...")
        
        # Eliminar filas con valores nulos
        df = df.dropna(subset=['catalan', 'chino'])
        
        # Eliminar duplicados
        df = df.drop_duplicates(subset=['catalan', 'chino'])
        
        # Limpiar espacios en blanco
        df['catalan'] = df['catalan'].str.strip()
        df['chino'] = df['chino'].str.strip()
        
        # Eliminar filas vacías
        df = df[(df['catalan'] != '') & (df['chino'] != '')]
        
        print(f"✅ Datos limpios: {len(df)} traducciones")
        
        # Guardar CSV limpio con comillas escapadas
        print(f"\n💾 Guardando en: {output_path}")
        df.to_csv(output_path, index=False, encoding='utf-8', quoting=1)  # quoting=1 para escapar comas
        
        print(f"🎉 ¡Dataset limpio v2.0 guardado!")
        print(f"📊 Total de traducciones: {len(df)}")
        print(f"📁 Archivo: {output_path}")
        
        # Mostrar estadísticas
        print(f"\n📈 Estadísticas:")
        print(f"  - Longitud promedio catalán: {df['catalan'].str.len().mean():.1f} caracteres")
        print(f"  - Longitud promedio chino: {df['chino'].str.len().mean():.1f} caracteres")
        print(f"  - Traducciones con comas: {df['catalan'].str.contains(',').sum() + df['chino'].str.contains(',').sum()}")
        
    else:
        print("❌ No se pudieron parsear datos del archivo")

if __name__ == "__main__":
    clean_full_dataset_v2()



