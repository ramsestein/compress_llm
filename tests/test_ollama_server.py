#!/usr/bin/env python3
"""
Test para el servidor Ollama
"""
import unittest
import requests
import time
import subprocess
import sys
from pathlib import Path

class TestOllamaServer(unittest.TestCase):
    """Test del servidor Ollama"""
    
    def setUp(self):
        """Configuración inicial"""
        self.base_url = "http://localhost:11435"
        self.server_process = None
        
    def tearDown(self):
        """Limpieza después de las pruebas"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
    
    def test_server_startup(self):
        """Test de inicio del servidor"""
        try:
            # Iniciar servidor en background
            self.server_process = subprocess.Popen([
                sys.executable, "ollama_compact_server.py",
                "--port", "11435",
                "--host", "127.0.0.1"
            ])
            
            # Esperar a que el servidor esté listo
            time.sleep(5)
            
            # Verificar que el servidor responde
            response = requests.get(f"{self.base_url}/api/tags")
            self.assertEqual(response.status_code, 200)
            
        except Exception as e:
            self.fail(f"Error iniciando servidor: {e}")
    
    def test_api_endpoints(self):
        """Test de endpoints de la API"""
        try:
            # Iniciar servidor
            self.server_process = subprocess.Popen([
                sys.executable, "ollama_compact_server.py",
                "--port", "11435",
                "--host", "127.0.0.1"
            ])
            
            time.sleep(5)
            
            # Test de endpoint de tags
            response = requests.get(f"{self.base_url}/api/tags")
            self.assertEqual(response.status_code, 200)
            
            # Test de endpoint de modelos
            response = requests.get(f"{self.base_url}/api/tags")
            self.assertEqual(response.status_code, 200)
            
        except Exception as e:
            self.fail(f"Error en endpoints: {e}")

if __name__ == "__main__":
    unittest.main()
