#!/usr/bin/env python3
"""
Script para debuggear la línea problemática
"""

import re

def debug_line():
    # Línea 2 del dataset original
    line = '"558,""Si no hagués estat per la intervenció de Déu, moltes persones haurien patit un càstig terrible."",要不是天主的干预很多人都会遭遇了可怕的惩罚。";;;'
    
    print("🔍 Analizando línea problemática:")
    print(f"Línea original: {repr(line)}")
    print()
    
    # Sin ;;; al final
    line_clean = line.replace(';;;', '')
    print(f"Sin ;;;: {repr(line_clean)}")
    print()
    
    # Probar diferentes patrones
    patterns = [
        (r'^"(\d+),""(.+)"",""(.+)""$', "Patrón 1: comillas escapadas"),
        (r'^"(\d+),""(.+)"",""(.+)"";;;$', "Patrón 2: con ;;; al final"),
        (r'^"(\d+),""(.+)"",""(.+)""$', "Patrón 3: sin ;;;"),
    ]
    
    for pattern, description in patterns:
        match = re.match(pattern, line)
        print(f"{description}: {match is not None}")
        if match:
            print(f"  Grupos: {match.groups()}")
        print()
    
    # Analizar la estructura manualmente
    print("🔍 Análisis manual:")
    print(f"  - Empieza con comilla: {line.startswith('\"')}")
    print(f"  - Termina con ;;;: {line.endswith(';;;')}")
    print(f"  - Contiene comillas dobles: {'\"\"' in line}")
    print(f"  - Número de comillas: {line.count('\"')}")

if __name__ == "__main__":
    debug_line()



