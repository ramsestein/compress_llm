import json
import sys
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lora_model_tester import LoRAModelTester


class TestLoRAModel(unittest.TestCase):
    """Tests para el modelo LoRA"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        # Crear directorio temporal
        self.temp_dir = tempfile.mkdtemp()
        self.dummy_model_dir = Path(self.temp_dir)
        
        # Crear archivo de configuración del adaptador
        adapter_config = {"base_model_name_or_path": "dummy-base"}
        (self.dummy_model_dir / "adapter_config.json").write_text(json.dumps(adapter_config))
    
    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_batch_test_returns_expected_keys(self):
        """Test que batch_test retorna las claves esperadas"""
        tester = LoRAModelTester(str(self.dummy_model_dir))
        
        with patch.object(LoRAModelTester, "load_model", return_value=None), \
             patch.object(LoRAModelTester, "generate", 
                         side_effect=lambda prompt, max_length=200, temperature=0.7: f"echo: {prompt}"):
            
            prompts = ["hi", "there"]
            results = tester.batch_test(prompts)

            self.assertEqual(len(results), len(prompts))
            for res, prompt in zip(results, prompts):
                self.assertEqual(res["prompt"], prompt)
                self.assertTrue(set(["prompt", "response", "success"]).issubset(res.keys()))
                self.assertTrue(res["success"])
    
    def test_batch_test_handles_generation_errors(self):
        """Test que batch_test maneja errores de generación"""
        tester = LoRAModelTester(str(self.dummy_model_dir))
        
        with patch.object(LoRAModelTester, "load_model", return_value=None):
            
            def faulty_generate(prompt, max_length=200, temperature=0.7):
                raise RuntimeError("generation failed")

            with patch.object(LoRAModelTester, "generate", side_effect=faulty_generate):
                prompts = ["fail"]
                results = tester.batch_test(prompts)

                self.assertEqual(len(results), 1)
                entry = results[0]
                self.assertEqual(entry["prompt"], "fail")
                self.assertFalse(entry["success"])
                self.assertIn("error", entry)


if __name__ == "__main__":
    unittest.main()
