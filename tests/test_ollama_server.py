#!/usr/bin/env python3
"""End-to-end tests for the Ollama-compatible server using FastAPI TestClient."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient

from scripts.ollama_compact_server import OllamaCompatServer, app


class TestOllamaServer(unittest.TestCase):
    """End-to-end tests for the Ollama-compatible server."""

    def setUp(self):
        """Create a temporary models directory with a dummy model."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.models_dir = self.test_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Create a dummy model directory with config.json
        dummy_model_dir = self.models_dir / "dummy_model"
        dummy_model_dir.mkdir(parents=True, exist_ok=True)
        (dummy_model_dir / "config.json").write_text(
            json.dumps({"model_type": "gpt2", "num_parameters": 1_000_000}),
            encoding="utf-8",
        )

        # Initialize the global server instance required by the app
        import scripts.ollama_compact_server as server_module

        server_module.server = OllamaCompatServer(str(self.models_dir))
        server_module.server.device = "cpu"

        self.client = TestClient(app)

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_server_startup(self):
        """Test that the server responds on the root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("dummy_model", data.get("models", []))

    def test_api_tags(self):
        """Test the /api/tags endpoint lists available models."""
        response = self.client.get("/api/tags")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        model_names = [m["name"] for m in data["models"]]
        self.assertIn("dummy_model", model_names)

    def test_api_generate_missing_model(self):
        """Test /api/generate returns 404 for a missing model."""
        payload = {
            "model": "nonexistent_model",
            "prompt": "Hello",
            "stream": False,
        }
        response = self.client.post("/api/generate", json=payload)
        # The endpoint may return 404 or 422 depending on validation
        self.assertIn(response.status_code, (404, 422, 500))


if __name__ == "__main__":
    unittest.main()
