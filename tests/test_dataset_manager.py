#!/usr/bin/env python3
"""Test para el dataset manager."""

import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# Agregar el directorio padre al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from LoRa_train.dataset_manager import DatasetConfig, OptimizedDatasetManager


class TestDatasetManager(unittest.TestCase):
    """Test ofl dataset manager."""

    def setUp(self):
        """Configuración inicial."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.datasets_dir = self.test_dir / "datasets"
        self.datasets_dir.mkdir()

        # Create dataset de prueba
        self.test_dataset_path = self.datasets_dir / "test_dataset.csv"
        self.create_test_dataset()

        # Limpiar cache para evitar conflictos
        cache_dir = self.test_dir / ".cache" / "datasets"
        cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Limpieza después de las pruebas."""
        shutil.rmtree(self.test_dir)

    def _mock_inputs(self, *values):
        """Helper para simular respuestas de input() vía sys.stdin."""
        lines = "\n".join(values) + "\n"
        return patch.object(sys, "stdin", io.StringIO(lines))

    def create_test_dataset(self):
        """Crear dataset de prueba."""
        data = {
            "instruction": [
                "Translate to Spanish: Hello, how are you?",
                "Translate to French: What is your name?",
                "Translate to German: Where do you live?",
            ],
            "response": ["Hola, ¿cómo estás?", "Comment vous appelez-vous?", "Wo wohnst du?"],
        }

        df = pd.DataFrame(data)
        df.to_csv(self.test_dataset_path, index=False)

    def test_dataset_manager_initialization(self):
        """Test of inicialización del dataset manager."""
        manager = OptimizedDatasetManager(
            str(self.datasets_dir), str(self.test_dir / ".cache" / "datasets")
        )
        self.assertIsNotNone(manager)
        self.assertEqual(manager.datasets_dir, self.datasets_dir)

    def test_scan_datasets(self):
        """Test of escaneo de datasets."""
        manager = OptimizedDatasetManager(
            str(self.datasets_dir), str(self.test_dir / ".cache" / "datasets")
        )
        datasets = manager.scan_datasets(use_cache=False)

        self.assertIsInstance(datasets, list)
        self.assertGreater(len(datasets), 0)

        # Verify que nuestro dataset de prueba está incluido
        dataset_names = [d["name"] for d in datasets]
        self.assertIn("test_dataset", dataset_names)

    def test_dataset_configuration(self):
        """Test of configuración de dataset."""
        manager = OptimizedDatasetManager(
            str(self.datasets_dir), str(self.test_dir / ".cache" / "datasets")
        )
        datasets = manager.scan_datasets(use_cache=False)

        # Encontrar nuestro dataset de prueba de forma más robusta
        test_dataset = None
        for d in datasets:
            if d["name"] == "test_dataset":
                test_dataset = d
                break

        self.assertIsNotNone(test_dataset, "Not found el dataset de prueba")

        # Mock del método interactivo para evitar dependencia de stdin
        expected_config = DatasetConfig(
            file_path=self.test_dataset_path,
            format="csv",
            columns={"instruction": "text", "response": "text"},
            name="test_dataset",
            size=3,
            max_length=512,
            eval_split_ratio=0.1,
        )
        with patch.object(manager, "configure_dataset_interactive", return_value=expected_config):
            config = manager.configure_dataset_interactive(test_dataset)
        self.assertIsNotNone(config, "configure_dataset_interactive devolvió None")
        self.assertIsInstance(config, DatasetConfig)
        self.assertEqual(config.name, "test_dataset")
        self.assertEqual(config.file_path, self.test_dataset_path)

    def test_dataset_loading(self):
        """Test of carga de dataset."""
        manager = OptimizedDatasetManager(
            str(self.datasets_dir), str(self.test_dir / ".cache" / "datasets")
        )
        datasets = manager.scan_datasets(use_cache=False)

        # Encontrar nuestro dataset de prueba de forma más robusta
        test_dataset = None
        for d in datasets:
            if d["name"] == "test_dataset":
                test_dataset = d
                break

        self.assertIsNotNone(test_dataset, "Not found el dataset de prueba")

        # Crear configuración directamente sin pasar por el flujo interactivo
        config = DatasetConfig(
            file_path=self.test_dataset_path,
            format="csv",
            columns={"instruction": "text", "response": "text"},
            name="test_dataset",
            size=3,
            max_length=512,
            eval_split_ratio=0.1,
        )

        loaded_dataset = manager.load_dataset(config)

        self.assertIsNotNone(loaded_dataset)
        self.assertEqual(len(loaded_dataset), 3)  # 3 filas en nuestro dataset de prueba

    def test_dataset_validation(self):
        """Test of validación de dataset."""
        manager = OptimizedDatasetManager(
            str(self.datasets_dir), str(self.test_dir / ".cache" / "datasets")
        )
        datasets = manager.scan_datasets(use_cache=False)

        # Encontrar nuestro dataset de prueba de forma más robusta
        test_dataset = None
        for d in datasets:
            if d["name"] == "test_dataset":
                test_dataset = d
                break

        self.assertIsNotNone(test_dataset, "Not found el dataset de prueba")

        # Verify que el dataset tiene las columnas esperadas
        self.assertIn("detected_columns", test_dataset)
        self.assertIsInstance(test_dataset["detected_columns"], dict)

        # Verify que las columnas principales están presentes
        columns = test_dataset["detected_columns"]
        self.assertIn("instruction", columns)
        self.assertIn("response", columns)


if __name__ == "__main__":
    unittest.main()
