#!/usr/bin/env python3
"""
Script para probar la detección de datasets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LoRa_train.dataset_manager import OptimizedDatasetManager

def test_datasets():
    print("🔍 Probando detección de datasets...")
    
    # Crear manager
    manager = OptimizedDatasetManager()
    
    # Escanear sin cache
    print("📊 Escaneando datasets (sin cache)...")
    datasets = manager.scan_datasets(use_cache=False)
    
    print(f"\n✅ Encontrados {len(datasets)} datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset['name']} ({dataset['format']}) - {dataset['size']} registros")
        print(f"     Archivo: {dataset['file_path']}")
    
    # Verificar si está el dataset limpio
    clean_found = any('clean_traducciones' in d['name'] for d in datasets)
    print(f"\n🎯 Dataset limpio encontrado: {'✅ SÍ' if clean_found else '❌ NO'}")

if __name__ == "__main__":
    test_datasets()
