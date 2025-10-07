#!/usr/bin/env python3
"""
Parser robusto para el CSV de traducciones mal formateado
"""

import re
import csv

def parse_malformed_csv():
    print("🔧 Parseando CSV mal formateado...")
    
    clean_data = []
    
    with open('datasets/muestra_traducciones_10000.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📊 Total de líneas: {len(lines)}")
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Saltar header
        if i == 0:
            continue
            
        # Saltar líneas vacías
        if not line:
            continue
        
        # Patrón 1: Líneas con comillas (formato: "id,""catalán"",""chino""")
        if line.startswith('"') and '","' in line:
            # Extraer usando regex
            match = re.match(r'"(\d+),""([^"]+)"",""([^"]+)""', line)
            if match:
                id_num, catalan, chino = match.groups()
                clean_data.append({
                    'catalan': catalan,
                    'chino': chino
                })
                continue
        
        # Patrón 2: Líneas sin comillas (formato: id,catalán,chino)
        if ',' in line and not line.startswith('"'):
            parts = line.split(',')
            if len(parts) >= 3:
                # El último elemento puede tener ;;; al final
                catalan = parts[1].strip()
                chino = parts[2].strip()
                if chino.endswith(';;;'):
                    chino = chino[:-3]
                
                if catalan and chino:
                    clean_data.append({
                        'catalan': catalan,
                        'chino': chino
                    })
                    continue
        
        # Si no coincide con ningún patrón, mostrar para debug
        if i < 10:  # Solo mostrar las primeras 10 líneas problemáticas
            print(f"⚠️  Línea {i+1} no parseada: {line[:100]}...")
    
    print(f"\n✅ Datos parseados: {len(clean_data)} traducciones")
    
    # Mostrar algunas muestras
    print("\n📄 Primeras 5 traducciones:")
    for i, item in enumerate(clean_data[:5]):
        print(f"  {i+1}. {item['catalan']} → {item['chino']}")
    
    # Guardar CSV limpio
    import pandas as pd
    df = pd.DataFrame(clean_data)
    df.to_csv('datasets/clean_traducciones.csv', index=False)
    print(f"\n💾 Guardado en: datasets/clean_traducciones.csv")
    
    return clean_data

if __name__ == "__main__":
    parse_malformed_csv()



