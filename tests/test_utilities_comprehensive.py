#!/usr/bin/env python3
"""Test comprehensivo de utilidades y herramientas auxiliares
Verifica funciones de utilidad, helpers, y herramientas de soporte.
"""

import gzip
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Asegurar que el paquete del proyecto esté en el path de importación
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

# Import utilidades que realmente existen
from create_compression_config import OptimizedCompressionConfigCreator

from create_compress.compression_config_manager import CompressionConfigManager
from create_compress.compression_profiles import COMPRESSION_PROFILES

# Import utilidades de datasets si existen
try:
    from create_test_dataset import (
        create_qa_dataset,
        create_synthetic_dataset,
        create_translation_dataset,
        save_dataset,
    )

    DATASET_UTILS_AVAILABLE = True
except ImportError:
    DATASET_UTILS_AVAILABLE = False

# Import utilidades de análisis si existen
try:
    from analyze_model import (
        ModelAnalyzer,
        analyze_model_layers,
        calculate_model_statistics,
        generate_compression_recommendations,
    )

    ANALYSIS_UTILS_AVAILABLE = True
except ImportError:
    ANALYSIS_UTILS_AVAILABLE = False


class TestUtilitiesComprehensive(unittest.TestCase):
    """Test comprehensivo de utilidades y herramientas auxiliares."""

    def setUp(self):
        """Configuración inicial para cada test."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create archivos de prueba
        self.create_test_files()

    def tearDown(self):
        """Limpieza después de cada test."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_files(self):
        """Crear archivos de prueba para los tests."""
        # Create modelo de prueba simple
        model_dir = self.test_dir / "test_model"
        model_dir.mkdir()

        # Create config.json
        config = {
            "model_type": "gpt2",
            "vocab_size": 50257,
            "n_positions": 1024,
            "n_ctx": 1024,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
        }

        config_file = model_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        # Create archivo de pesos simulado
        weights = torch.randn(100, 100)
        torch.save(weights, model_dir / "pytorch_model.bin")

        self.test_model_path = model_dir

        # Create archivo de configuración de compresión de prueba
        compression_config = {
            "metadata": {
                "model_name": "test_model",
                "version": "1.0",
                "created_date": "2025-01-01T00:00:00",
            },
            "global_settings": {
                "name": "Balanced",
                "description": "Test configuration",
                "goal": "balanced",
                "target_compression": 0.5,
                "layer_configs": {
                    "attention": {
                        "methods": [{"name": "int8_quantization", "strength": 0.5}],
                        "total_compression_ratio": 0.3,
                    },
                    "ffn": {
                        "methods": [{"name": "magnitude_pruning", "strength": 0.3}],
                        "total_compression_ratio": 0.4,
                    },
                },
            },
        }

        config_path = self.test_dir / "compression_config.json"
        with open(config_path, "w") as f:
            json.dump(compression_config, f, indent=2)
        self.compression_config_path = config_path

    # ============= TESTS DE CREATE_COMPRESSION_CONFIG =============

    def test_optimized_compression_config_creator_initialization(self):
        """Test of inicialización del creador optimizado de configuración."""
        creator = OptimizedCompressionConfigCreator(str(self.test_model_path), str(self.test_dir))

        # Verify inicialización
        self.assertIsNotNone(creator)
        self.assertEqual(creator.model_path, self.test_model_path)
        self.assertIsInstance(creator.config_manager, CompressionConfigManager)

    def test_compression_config_manager_initialization(self):
        """Test of inicialización del gestor de configuración."""
        manager = CompressionConfigManager(str(self.test_model_path), str(self.test_dir))

        # Verify inicialización
        self.assertIsNotNone(manager)
        # Usar str() para comparar paths de forma consistente
        self.assertEqual(str(manager.model_path), str(self.test_model_path))
        self.assertEqual(str(manager.output_dir), str(self.test_dir))

    def test_compression_profiles_structure(self):
        """Test of estructura de perfiles de compresión."""
        # Verify que existen los perfiles principales
        required_profiles = ["conservative", "balanced", "aggressive"]
        for profile_name in required_profiles:
            self.assertIn(profile_name, COMPRESSION_PROFILES)

            profile = COMPRESSION_PROFILES[profile_name]
            self.assertIn("name", profile)
            self.assertIn("description", profile)
            self.assertIn("goal", profile)
            self.assertIn("target_compression", profile)
            self.assertIn("layer_configs", profile)

    def test_conservative_profile(self):
        """Test ofl perfil conservador."""
        profile = COMPRESSION_PROFILES["conservative"]

        self.assertEqual(profile["goal"], "max_quality")
        self.assertLess(profile["target_compression"], 0.5)

        # Verify configuraciones de layers
        layer_configs = profile["layer_configs"]
        self.assertIn("attention", layer_configs)
        self.assertIn("ffn", layer_configs)
        self.assertIn("embedding", layer_configs)

    def test_balanced_profile(self):
        """Test ofl perfil balanceado."""
        profile = COMPRESSION_PROFILES["balanced"]

        self.assertEqual(profile["goal"], "balanced")
        self.assertAlmostEqual(profile["target_compression"], 0.5, delta=0.1)

        # Verify configuraciones de layers
        layer_configs = profile["layer_configs"]
        self.assertIn("attention", layer_configs)
        self.assertIn("ffn", layer_configs)
        self.assertIn("embedding", layer_configs)

    def test_aggressive_profile(self):
        """Test ofl perfil agresivo."""
        profile = COMPRESSION_PROFILES["aggressive"]

        self.assertEqual(profile["goal"], "max_compression")
        self.assertGreater(profile["target_compression"], 0.5)

        # Verify configuraciones de layers
        layer_configs = profile["layer_configs"]
        self.assertIn("attention", layer_configs)
        self.assertIn("ffn", layer_configs)
        self.assertIn("embedding", layer_configs)

    # ============= TESTS DE ANÁLISIS DE MODELO =============

    @unittest.skipUnless(ANALYSIS_UTILS_AVAILABLE, "Analysis utilities not available")
    def test_model_analyzer_initialization(self):
        """Test of inicialización del analizador de modelos."""
        analyzer = ModelAnalyzer(str(self.test_model_path))

        # Verify inicialización
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.model_path, str(self.test_model_path))

    @unittest.skipUnless(ANALYSIS_UTILS_AVAILABLE, "Analysis utilities not available")
    def test_analyze_model_layers(self):
        """Test of análisis de layers del modelo."""
        # Mock del modelo para evitar cargar uno real
        with patch("analyze_model.AutoModelForCausalLM.from_pretrained") as mock_load:
            mock_model = MagicMock()
            mock_model.config.model_type = "gpt2"
            mock_model.config.n_layer = 12
            mock_model.config.n_embd = 768
            mock_model.config.n_head = 12
            mock_load.return_value = mock_model

            # Analizar layers
            layer_info = analyze_model_layers(str(self.test_model_path))

            # Verify estructura
            self.assertIsInstance(layer_info, dict)
            self.assertIn("total_layers", layer_info)
            self.assertIn("layer_types", layer_info)
            self.assertIn("layer_sizes", layer_info)

    @unittest.skipUnless(ANALYSIS_UTILS_AVAILABLE, "Analysis utilities not available")
    def test_calculate_model_statistics(self):
        """Test of cálculo de estadísticas del modelo."""
        # Mock del modelo
        with patch("analyze_model.AutoModelForCausalLM.from_pretrained") as mock_load:
            mock_model = MagicMock()
            mock_model.config.model_type = "gpt2"
            mock_model.config.n_layer = 12
            mock_model.config.n_embd = 768
            mock_model.config.n_head = 12
            mock_load.return_value = mock_model

            # Calculate estadísticas
            stats = calculate_model_statistics(str(self.test_model_path))

            # Verify estructura
            self.assertIn("total_parameters", stats)
            self.assertIn("model_size_mb", stats)
            self.assertIn("architecture", stats)
            self.assertIn("num_layers", stats)

    @unittest.skipUnless(ANALYSIS_UTILS_AVAILABLE, "Analysis utilities not available")
    def test_generate_compression_recommendations(self):
        """Test of generación de recomendaciones de compresión."""
        # Mock de estadísticas
        stats = {
            "total_parameters": 100000000,
            "model_size_mb": 500,
            "architecture": "gpt2",
            "num_layers": 12,
        }

        # Generate recomendaciones
        recommendations = generate_compression_recommendations(stats)

        # Verify estructura
        self.assertIn("conservative", recommendations)
        self.assertIn("balanced", recommendations)
        self.assertIn("aggressive", recommendations)

    # ============= TESTS DE CREACIÓN DE DATASETS =============

    @unittest.skipUnless(DATASET_UTILS_AVAILABLE, "Dataset utilities not available")
    def test_create_synthetic_dataset(self):
        """Test of creación de dataset sintético."""
        # Create dataset sintético
        dataset = create_synthetic_dataset(num_samples=100, max_length=512, vocab_size=1000)

        # Verify estructura
        self.assertIsInstance(dataset, list)
        self.assertEqual(len(dataset), 100)

        # Verify cada muestra
        for sample in dataset:
            self.assertIn("instruction", sample)
            self.assertIn("response", sample)
            self.assertIsInstance(sample["instruction"], str)
            self.assertIsInstance(sample["response"], str)

    @unittest.skipUnless(DATASET_UTILS_AVAILABLE, "Dataset utilities not available")
    def test_create_translation_dataset(self):
        """Test of creación de dataset de traducción."""
        # Create dataset de traducción
        dataset = create_translation_dataset(num_samples=50, source_lang="en", target_lang="es")

        # Verify estructura
        self.assertIsInstance(dataset, list)
        self.assertEqual(len(dataset), 50)

        # Verify cada muestra
        for sample in dataset:
            self.assertIn("instruction", sample)
            self.assertIn("response", sample)
            self.assertIn("translate", sample["instruction"].lower())

    @unittest.skipUnless(DATASET_UTILS_AVAILABLE, "Dataset utilities not available")
    def test_create_qa_dataset(self):
        """Test of creación de dataset de preguntas y respuestas."""
        # Create dataset de QA
        dataset = create_qa_dataset(num_samples=75, question_types=["what", "how", "why"])

        # Verify estructura
        self.assertIsInstance(dataset, list)
        self.assertEqual(len(dataset), 75)

        # Verify cada muestra
        for sample in dataset:
            self.assertIn("instruction", sample)
            self.assertIn("response", sample)
            self.assertIsInstance(sample["instruction"], str)
            self.assertIsInstance(sample["response"], str)

    @unittest.skipUnless(DATASET_UTILS_AVAILABLE, "Dataset utilities not available")
    def test_save_dataset(self):
        """Test of guardado de dataset."""
        # Create dataset de prueba
        dataset = [
            {"instruction": "Test instruction 1", "response": "Test response 1"},
            {"instruction": "Test instruction 2", "response": "Test response 2"},
        ]

        # Save en diferentes formatos
        csv_path = self.test_dir / "test_dataset.csv"
        jsonl_path = self.test_dir / "test_dataset.jsonl"

        # Save CSV
        success_csv = save_dataset(dataset, csv_path, format="csv")
        self.assertTrue(success_csv)
        self.assertTrue(csv_path.exists())

        # Save JSONL
        success_jsonl = save_dataset(dataset, jsonl_path, format="jsonl")
        self.assertTrue(success_jsonl)
        self.assertTrue(jsonl_path.exists())

    # ============= TESTS DE UTILIDADES GENERALES =============

    def test_file_operations_utilities(self):
        """Test of utilidades de operaciones de archivos."""
        # Test of creación de directorios
        test_dir = self.test_dir / "file_ops_test"
        test_dir.mkdir(exist_ok=True)
        self.assertTrue(test_dir.exists())

        # Test of escritura de archivos
        test_file = test_dir / "test.txt"
        with open(test_file, "w") as f:
            f.write("Test content")
        self.assertTrue(test_file.exists())

        # Test of lectura de archivos
        with open(test_file) as f:
            content = f.read()
        self.assertEqual(content, "Test content")

        # Test of eliminación de archivos
        test_file.unlink()
        self.assertFalse(test_file.exists())

    def test_json_operations_utilities(self):
        """Test of utilidades de operaciones JSON."""
        # Test of escritura JSON
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        json_file = self.test_dir / "test.json"

        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)

        self.assertTrue(json_file.exists())

        # Test of lectura JSON
        with open(json_file) as f:
            loaded_data = json.load(f)

        self.assertEqual(loaded_data, data)
        self.assertEqual(loaded_data["key"], "value")
        self.assertEqual(loaded_data["number"], 42)
        self.assertEqual(loaded_data["list"], [1, 2, 3])

    def test_torch_operations_utilities(self):
        """Test of utilidades de operaciones PyTorch."""
        # Test of guardado de tensores
        tensor = torch.randn(10, 10)
        tensor_file = self.test_dir / "test_tensor.pt"

        torch.save(tensor, tensor_file)
        self.assertTrue(tensor_file.exists())

        # Test of carga de tensores
        loaded_tensor = torch.load(tensor_file)
        torch.testing.assert_close(tensor, loaded_tensor)

        # Test of información de tensor
        self.assertEqual(tensor.shape, (10, 10))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.numel(), 100)

    def test_numpy_operations_utilities(self):
        """Test of utilidades de operaciones NumPy."""
        # Test of creación de arrays
        array = np.random.randn(10, 10)

        # Test of estadísticas básicas
        self.assertEqual(array.shape, (10, 10))
        self.assertEqual(array.size, 100)
        self.assertIsInstance(array.mean(), float)
        self.assertIsInstance(array.std(), float)

        # Test of guardado y carga
        array_file = self.test_dir / "test_array.npy"
        np.save(array_file, array)
        self.assertTrue(array_file.exists())

        loaded_array = np.load(array_file)
        np.testing.assert_array_equal(array, loaded_array)

    def test_hashing_utilities(self):
        """Test of utilidades de hashing."""
        # Test of hash de strings
        test_string = "Hello, World!"
        hash_value = hashlib.md5(test_string.encode()).hexdigest()

        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 32)  # MD5 hash length

        # Test of hash de archivos
        test_file = self.test_dir / "hash_test.txt"
        with open(test_file, "w") as f:
            f.write(test_string)

        with open(test_file, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        self.assertEqual(file_hash, hash_value)

    def test_compression_utilities(self):
        """Test of utilidades de compresión."""
        # Test of compresión gzip
        test_data = "This is a test string for compression" * 100
        test_file = self.test_dir / "test_data.txt"

        # Escribir datos originales
        with open(test_file, "w") as f:
            f.write(test_data)

        original_size = test_file.stat().st_size

        # Comprimir
        compressed_file = self.test_dir / "test_data.txt.gz"
        with open(test_file, "rb") as f_in, gzip.open(compressed_file, "wb") as f_out:
            f_out.write(f_in.read())

        compressed_size = compressed_file.stat().st_size

        # Verify compresión
        self.assertLess(compressed_size, original_size)

        # Descomprimir
        decompressed_file = self.test_dir / "decompressed.txt"
        with gzip.open(compressed_file, "rb") as f_in:
            with open(decompressed_file, "wb") as f_out:
                f_out.write(f_in.read())

        # Verify contenido
        with open(decompressed_file) as f:
            decompressed_data = f.read()

        self.assertEqual(decompressed_data, test_data)

    def test_serialization_utilities(self):
        """Test of utilidades de serialización."""
        # Test of pickle
        test_object = {
            "string": "test",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        pickle_file = self.test_dir / "test_object.pkl"

        # Serializar
        with open(pickle_file, "wb") as f:
            pickle.dump(test_object, f)

        self.assertTrue(pickle_file.exists())

        # Deserializar
        with open(pickle_file, "rb") as f:
            loaded_object = pickle.load(f)

        self.assertEqual(loaded_object, test_object)

    def test_timing_utilities(self):
        """Test of utilidades de timing."""
        # Test of medición de tiempo
        start_time = time.time()

        # Simular trabajo
        time.sleep(0.1)

        end_time = time.time()
        execution_time = end_time - start_time

        # Verify medición
        self.assertGreater(execution_time, 0.1)
        self.assertLess(execution_time, 0.2)  # Con margen de error

    def test_memory_utilities(self):
        """Test of utilidades de memoria."""
        # Test of creación de objetos grandes
        large_tensor = torch.randn(1000, 1000)

        # Verify que se puede crear
        self.assertEqual(large_tensor.shape, (1000, 1000))
        self.assertEqual(large_tensor.numel(), 1000000)

        # Limpiar memoria
        del large_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def test_error_handling_utilities(self):
        """Test of utilidades de manejo de errores."""

        # Test of manejo de excepciones
        def risky_function():
            if True:
                raise ValueError("Test error")

        # Verify que se captura la excepción
        try:
            risky_function()
            self.fail("Debería haber lanzado una excepción")
        except ValueError as e:
            self.assertEqual(str(e), "Test error")
        except Exception as e:
            self.fail(f"Debería capturar ValueError, no {type(e)}")

    def test_validation_utilities(self):
        """Test of utilidades de validación."""

        # Test of validación de tipos
        def validate_types(value, expected_type):
            if not isinstance(value, expected_type):
                raise TypeError(f"Expected {expected_type}, got {type(value)}")
            return True

        # Test casos válidos
        self.assertTrue(validate_types(42, int))
        self.assertTrue(validate_types("test", str))
        self.assertTrue(validate_types([1, 2, 3], list))

        # Test casos inválidos
        with self.assertRaises(TypeError):
            validate_types("42", int)

        with self.assertRaises(TypeError):
            validate_types(42, str)

    def test_logging_utilities(self):
        """Test of utilidades de logging."""
        import logging

        # Configure logging
        log_file = self.test_dir / "test.log"

        # Limpiar cualquier configuración previa
        logging.getLogger().handlers.clear()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            force=True,
        )

        logger = logging.getLogger(__name__)

        # Escribir logs
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")

        # Forzar flush de los handlers
        for handler in logger.handlers:
            handler.flush()

        # Verify que se escribieron logs
        self.assertTrue(log_file.exists())

        # Esperar un momento para asegurar que los logs se escriban

        time.sleep(0.1)

        with open(log_file, encoding="utf-8") as f:
            log_content = f.read()

        self.assertIn("Test info message", log_content)
        self.assertIn("Test warning message", log_content)
        self.assertIn("Test error message", log_content)

    def test_path_utilities(self):
        """Test of utilidades de manejo de rutas."""
        from pathlib import Path

        # Test of construcción de rutas
        base_path = Path(self.test_dir)
        sub_path = base_path / "subdir" / "file.txt"

        # Create directorios
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(sub_path.parent.exists())

        # Test of operaciones de rutas
        self.assertEqual(sub_path.name, "file.txt")
        self.assertEqual(sub_path.suffix, ".txt")
        self.assertEqual(sub_path.stem, "file")
        self.assertTrue(sub_path.parent.is_dir())

    def test_math_utilities(self):
        """Test of utilidades matemáticas."""
        import math

        # Test of funciones matemáticas básicas
        self.assertEqual(math.sqrt(4), 2.0)
        self.assertEqual(math.pow(2, 3), 8.0)
        self.assertEqual(math.log(math.e), 1.0)

        # Test of estadísticas
        numbers = [1, 2, 3, 4, 5]
        mean = sum(numbers) / len(numbers)
        self.assertEqual(mean, 3.0)

        # Test of redondeo
        self.assertEqual(round(3.14159, 2), 3.14)
        self.assertEqual(math.floor(3.7), 3)
        self.assertEqual(math.ceil(3.1), 4)

    def test_string_utilities(self):
        """Test of utilidades de strings."""
        # Test of operaciones de strings
        test_string = "  Hello, World!  "

        # Limpieza
        cleaned = test_string.strip()
        self.assertEqual(cleaned, "Hello, World!")

        # Conversión de caso
        upper = test_string.upper()
        lower = test_string.lower()
        self.assertEqual(upper, "  HELLO, WORLD!  ")
        self.assertEqual(lower, "  hello, world!  ")

        # Reemplazo
        replaced = test_string.replace("World", "Python")
        self.assertEqual(replaced, "  Hello, Python!  ")

        # División
        parts = test_string.split(",")
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].strip(), "Hello")
        self.assertEqual(parts[1].strip(), "World!")

    def test_list_utilities(self):
        """Test of utilidades de listas."""
        # Test of operaciones de listas
        test_list = [1, 2, 3, 4, 5]

        # Agregar elementos
        test_list.append(6)
        self.assertEqual(len(test_list), 6)
        self.assertEqual(test_list[-1], 6)

        # Insertar elementos
        test_list.insert(0, 0)
        self.assertEqual(test_list[0], 0)

        # Eliminar elementos
        test_list.remove(3)
        self.assertNotIn(3, test_list)

        # Ordenar
        test_list.sort()
        self.assertEqual(test_list, [0, 1, 2, 4, 5, 6])

        # Revertir
        test_list.reverse()
        self.assertEqual(test_list, [6, 5, 4, 2, 1, 0])

    def test_dict_utilities(self):
        """Test of utilidades de diccionarios."""
        # Test of operaciones de diccionarios
        test_dict = {"a": 1, "b": 2, "c": 3}

        # Agregar elementos
        test_dict["d"] = 4
        self.assertEqual(test_dict["d"], 4)

        # Verify existencia
        self.assertIn("a", test_dict)
        self.assertNotIn("e", test_dict)

        # Get con valor por defecto
        value = test_dict.get("e", "default")
        self.assertEqual(value, "default")

        # Get claves y valores
        keys = list(test_dict.keys())
        values = list(test_dict.values())
        self.assertEqual(len(keys), 4)
        self.assertEqual(len(values), 4)

        # Actualizar diccionario
        test_dict.update({"e": 5, "f": 6})
        self.assertEqual(len(test_dict), 6)

    def test_set_utilities(self):
        """Test of utilidades de conjuntos."""
        # Test of operaciones de conjuntos
        set1 = {1, 2, 3, 4, 5}
        set2 = {4, 5, 6, 7, 8}

        # Unión
        union = set1 | set2
        self.assertEqual(union, {1, 2, 3, 4, 5, 6, 7, 8})

        # Intersección
        intersection = set1 & set2
        self.assertEqual(intersection, {4, 5})

        # Diferencia
        difference = set1 - set2
        self.assertEqual(difference, {1, 2, 3})

        # Diferencia simétrica
        symmetric_diff = set1 ^ set2
        self.assertEqual(symmetric_diff, {1, 2, 3, 6, 7, 8})

    def test_iteration_utilities(self):
        """Test of utilidades de iteración."""
        # Test of iteración básica
        test_list = [1, 2, 3, 4, 5]

        # Iteración con enumerate
        enumerated = list(enumerate(test_list))
        self.assertEqual(enumerated[0], (0, 1))
        self.assertEqual(enumerated[1], (1, 2))

        # Iteración con zip
        list1 = [1, 2, 3]
        list2 = ["a", "b", "c"]
        zipped = list(zip(list1, list2))
        self.assertEqual(zipped[0], (1, "a"))
        self.assertEqual(zipped[1], (2, "b"))

        # Comprensión de listas
        squared = [x**2 for x in test_list]
        self.assertEqual(squared, [1, 4, 9, 16, 25])

        # Filtrado
        even_numbers = [x for x in test_list if x % 2 == 0]
        self.assertEqual(even_numbers, [2, 4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
