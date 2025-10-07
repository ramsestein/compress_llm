#!/usr/bin/env python3
"""
Script para actualizar el cache de datasets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LoRa_train.dataset_manager import OptimizedDatasetManager

def update_cache():
    print("🔄 Actualizando cache de datasets...")
    
    # Crear manager
    manager = OptimizedDatasetManager()
    
    # Escanear sin cache para forzar actualización
    print("📊 Escaneando datasets (sin cache)...")
    datasets = manager.scan_datasets(use_cache=False)
    
    print(f"\n✅ Encontrados {len(datasets)} datasets:")
    for i, ds in enumerate(datasets):
        print(f"  {i+1}. {ds['name']} ({ds['format']}) - {ds['size']} registros")
        print(f"     Archivo: {ds['file_path']}")
    
    # Verificar si el dataset limpio está presente
    clean_dataset_found = any(ds['name'] == 'clean_full_traducciones' for ds in datasets)
    print(f"\n🎯 Dataset limpio encontrado: {'✅ SÍ' if clean_dataset_found else '❌ NO'}")
    
    if clean_dataset_found:
        print("🎉 ¡Cache actualizado correctamente!")
    else:
        print("⚠️ El dataset limpio no se detectó. Verificando archivo...")
        import os
        if os.path.exists('datasets/clean_full_traducciones.csv'):
            print("✅ El archivo existe, pero no se detectó. Revisando formato...")
        else:
            print("❌ El archivo no existe.")

if __name__ == "__main__":
    update_cache()



