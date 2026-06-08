#!/usr/bin/env python3
"""
Ejecutor principal de todos los tests comprehensivos
Ejecuta tests del sistema de compresión, LoRA, scripts principales y utilidades
"""
import unittest
import sys
import os
from pathlib import Path
import importlib.util
import time
import traceback

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def discover_test_files():
    """Descubre todos los archivos de test en el directorio tests/"""
    test_dir = Path(__file__).parent
    test_files = []
    
    # Definir categorías de tests
    test_categories = {
                'compression': [
                    'test_compression_system_comprehensive.py',
                    'test_compression_methods_specific.py',
                    'test_compression_engine.py',
                    'test_compression_verification.py'
                ],
                'lora': [
                    'test_lora_system_comprehensive.py',
                    'test_peft_methods_specific.py',
                    'test_lora_trainer.py',
                    'test_lora_model.py',
                    'test_peft_methods_config.py',
                    'test_peft_universal_trainer.py',
                    'test_dataset_manager.py',
                    'test_dataset_manager_comprehensive.py',
                    'test_training_execution.py'
                ],
                'scripts': [
                    'test_main_scripts_comprehensive.py',
                    'test_merge_lora.py',
                    'test_ollama_server.py'
                ],
                'utilities': [
                    'test_utilities_comprehensive.py'
                ]
            }
    
    # Recopilar todos los archivos de test
    for category, files in test_categories.items():
        for file_name in files:
            file_path = test_dir / file_name
            if file_path.exists():
                test_files.append((category, file_path))
    
    return test_files

def run_tests_by_category():
    """Ejecuta tests organizados por categoría"""
    print("🔍 Descubriendo archivos de test...")
    
    test_files = discover_test_files()
    
    if not test_files:
        print("❌ No se encontraron archivos de test")
        return False
    
    print(f"📁 Encontrados {len(test_files)} archivos de test:")
    
    # Organizar por categoría
    categories = {}
    for category, file_path in test_files:
        if category not in categories:
            categories[category] = []
        categories[category].append(file_path)
        print(f"   - {category}: {file_path.name}")
    
    print("\n🚀 Ejecutando tests por categoría...")
    
    # Ejecutar tests por categoría
    results = {}
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for category, files in categories.items():
        print(f"\n{'='*60}")
        print(f"📋 Ejecutando tests de {category.upper()}")
        print(f"{'='*60}")
        
        category_results = run_category_tests(category, files)
        results[category] = category_results
        
        total_tests += category_results['total_tests']
        total_failures += category_results['failures']
        total_errors += category_results['errors']
        
        # Mostrar resumen de categoría
        print(f"\n📊 Resumen {category}:")
        print(f"   ✅ Tests pasados: {category_results['total_tests'] - category_results['failures'] - category_results['errors']}")
        print(f"   ❌ Fallos: {category_results['failures']}")
        print(f"   ⚠️  Errores: {category_results['errors']}")
        print(f"   📈 Tiempo: {category_results['execution_time']:.2f}s")
    
    # Resumen general
    print(f"\n{'='*60}")
    print("📊 RESUMEN GENERAL")
    print(f"{'='*60}")
    print(f"   📁 Categorías ejecutadas: {len(categories)}")
    print(f"   🧪 Tests totales: {total_tests}")
    print(f"   ✅ Tests pasados: {total_tests - total_failures - total_errors}")
    print(f"   ❌ Fallos totales: {total_failures}")
    print(f"   ⚠️  Errores totales: {total_errors}")
    print(f"   📈 Tiempo total: {sum(r['execution_time'] for r in results.values()):.2f}s")
    
    success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
    print(f"   🎯 Tasa de éxito: {success_rate:.1f}%")
    
    return total_failures == 0 and total_errors == 0

def run_category_tests(category, test_files):
    """Ejecuta tests de una categoría específica"""
    start_time = time.time()
    
    # Usar unittest
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    total_tests = 0
    failures = 0
    errors = 0
    
    for test_file in test_files:
        try:
            # Importar el módulo de test
            module_name = test_file.stem
            spec = importlib.util.spec_from_file_location(module_name, test_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Cargar tests del módulo
            tests = loader.loadTestsFromModule(module)
            suite.addTests(tests)
            
            # Contar tests
            test_count = tests.countTestCases()
            total_tests += test_count
            
        except Exception as e:
            print(f"⚠️  Error cargando {test_file.name}: {e}")
            errors += 1
            continue
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
    result = runner.run(suite)
    
    execution_time = time.time() - start_time
    
    return {
        'total_tests': total_tests,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'execution_time': execution_time,
        'result': result
    }

def run_specific_test(test_file_path):
    """Ejecuta un test específico"""
    print(f"🧪 Ejecutando test específico: {test_file_path.name}")
    
    try:
        # Importar el módulo de test
        module_name = test_file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, test_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Ejecutar test
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Error ejecutando {test_file_path.name}: {e}")
        traceback.print_exc()
        return False

def run_quick_tests():
    """Ejecuta tests rápidos para verificación básica"""
    print("⚡ Ejecutando tests rápidos...")
    
    quick_tests = [
        'test_compression_engine.py',
        'test_lora_model.py',
        'test_dataset_manager.py'
    ]
    
    test_dir = Path(__file__).parent
    success_count = 0
    
    for test_file in quick_tests:
        test_path = test_dir / test_file
        if test_path.exists():
            if run_specific_test(test_path):
                success_count += 1
                print(f"✅ {test_file} - PASÓ")
            else:
                print(f"❌ {test_file} - FALLÓ")
        else:
            print(f"⚠️  {test_file} - NO ENCONTRADO")
    
    print(f"\n📊 Tests rápidos: {success_count}/{len(quick_tests)} pasaron")
    return success_count == len(quick_tests)

def run_comprehensive_tests():
    """Ejecuta todos los tests comprehensivos"""
    print("🔬 Ejecutando tests comprehensivos...")
    
    comprehensive_tests = [
        'test_compression_system_comprehensive.py',
        'test_lora_system_comprehensive.py',
        'test_main_scripts_comprehensive.py',
        'test_utilities_comprehensive.py'
    ]
    
    test_dir = Path(__file__).parent
    success_count = 0
    
    for test_file in comprehensive_tests:
        test_path = test_dir / test_file
        if test_path.exists():
            print(f"\n{'='*50}")
            print(f"🧪 Ejecutando: {test_file}")
            print(f"{'='*50}")
            
            if run_specific_test(test_path):
                success_count += 1
                print(f"✅ {test_file} - PASÓ")
            else:
                print(f"❌ {test_file} - FALLÓ")
        else:
            print(f"⚠️  {test_file} - NO ENCONTRADO")
    
    print(f"\n📊 Tests comprehensivos: {success_count}/{len(comprehensive_tests)} pasaron")
    return success_count == len(comprehensive_tests)

def main():
    """Función principal"""
    print("🚀 Iniciando ejecución de tests comprehensivos...")
    print("=" * 60)
    
    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'quick':
            print("⚡ Modo: Tests rápidos")
            success = run_quick_tests()
        elif mode == 'comprehensive':
            print("🔬 Modo: Tests comprehensivos")
            success = run_comprehensive_tests()
        elif mode == 'category':
            print("📁 Modo: Tests por categoría")
            success = run_tests_by_category()
        elif mode == 'specific' and len(sys.argv) > 2:
            test_file = sys.argv[2]
            test_path = Path(__file__).parent / test_file
            if test_path.exists():
                print(f"🎯 Modo: Test específico - {test_file}")
                success = run_specific_test(test_path)
            else:
                print(f"❌ Test file no encontrado: {test_file}")
                success = False
        else:
            print("❌ Modo no reconocido. Usar: quick, comprehensive, category, o specific <file>")
            success = False
    else:
        # Modo por defecto: tests por categoría
        print("📁 Modo por defecto: Tests por categoría")
        success = run_tests_by_category()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Todos los tests pasaron exitosamente!")
        exit_code = 0
    else:
        print("❌ Algunos tests fallaron. Revisa los errores arriba.")
        exit_code = 1
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
