#!/usr/bin/env python3
"""
Script de instalación para Ubuntu/WSL con detección de GPU y configuración optimizada
"""
import subprocess
import sys
import os
import platform
import shutil
from pathlib import Path
import importlib.util
import re

def get_platform():
    """Detecta el sistema operativo"""
    return platform.system().lower()

def get_python_version():
    """Obtiene la versión de Python"""
    return sys.version_info

def is_wsl():
    """Detecta si estamos en WSL"""
    if get_platform() != 'linux':
        return False
    
    # Verificar si existe /proc/version
    try:
        with open('/proc/version', 'r') as f:
            version_string = f.read().lower()
            return 'microsoft' in version_string or 'wsl' in version_string
    except:
        return False

def get_wsl_version():
    """Obtiene la versión de WSL (1 o 2)"""
    if not is_wsl():
        return None
    
    try:
        result = subprocess.run(['wsl.exe', '--status'], capture_output=True, text=True)
        if 'WSL 2' in result.stdout or 'WSL version: 2' in result.stdout:
            return 2
        else:
            return 1
    except:
        # Si no podemos ejecutar wsl.exe, probablemente es WSL1
        return 1

def check_wsl_gpu_support():
    """Verifica si WSL2 tiene soporte de GPU"""
    if not is_wsl() or get_wsl_version() != 2:
        return False, "No es WSL2"
    
    # Verificar si existe /dev/dxg (indicador de GPU en WSL2)
    if os.path.exists('/dev/dxg'):
        return True, "GPU support detectado en WSL2"
    
    # Verificar nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True)
        if result.returncode == 0:
            return True, "NVIDIA GPU detectada en WSL2"
    except:
        pass
    
    # Verificar si hay runtime de NVIDIA
    try:
        result = subprocess.run(['nvidia-container-cli', 'info'], capture_output=True)
        if result.returncode == 0:
            return True, "NVIDIA runtime detectado en WSL2"
    except:
        pass
    
    return False, "GPU no disponible en WSL (verifica drivers de Windows)"

def has_cuda():
    """Verifica si CUDA está disponible"""
    if is_wsl():
        gpu_available, message = check_wsl_gpu_support()
        if not gpu_available:
            print(f"⚠️  {message}")
            return False
    
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        return True
    except:
        return False

def install_system_dependencies():
    """Instala dependencias del sistema para Ubuntu"""
    print("\n📦 Verificando dependencias del sistema...")
    
    deps_to_install = []
    
    # Verificar dependencias
    required_deps = {
        'python3-dev': 'cabeceras de Python',
        'python3-pip': 'pip',
        'build-essential': 'herramientas de compilación',
        'libssl-dev': 'SSL',
        'libffi-dev': 'FFI',
        'python3-venv': 'entornos virtuales'
    }
    
    for dep, desc in required_deps.items():
        try:
            result = subprocess.run(
                ['dpkg', '-l', dep], 
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                deps_to_install.append(dep)
                print(f"  ❌ {desc} no instalado")
            else:
                print(f"  ✅ {desc} instalado")
        except:
            deps_to_install.append(dep)
    
    if deps_to_install:
        print("\n📦 Instalando dependencias del sistema...")
        print("   (puede requerir contraseña de sudo)")
        
        # Actualizar apt primero
        subprocess.run(['sudo', 'apt', 'update'], check=True)
        
        # Instalar dependencias
        cmd = ['sudo', 'apt', 'install', '-y'] + deps_to_install
        subprocess.run(cmd, check=True)
        print("✅ Dependencias del sistema instaladas")

def install_pytorch():
    """Instala PyTorch con la configuración correcta para WSL/Ubuntu"""
    system = get_platform()
    python_version = get_python_version()
    cuda_available = has_cuda()
    
    print(f"\n🖥️  Sistema: {system}")
    print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if is_wsl():
        print("🐧 WSL detectado")
        wsl_version = get_wsl_version()
        if wsl_version == 2:
            print("✅ WSL2 - Soporte completo")
        else:
            print("⚠️  WSL1 - Funcionalidad limitada")
        
        if cuda_available:
            print("🎮 GPU disponible en WSL2")
        else:
            print("💻 Modo CPU (GPU no disponible en WSL)")
            print("\n💡 Para habilitar GPU en WSL2:")
            print("   1. Asegúrate de tener Windows 11 o Windows 10 21H2+")
            print("   2. Instala drivers NVIDIA para WSL desde:")
            print("      https://developer.nvidia.com/cuda/wsl")
            print("   3. NO instales drivers CUDA dentro de WSL")
    
    # Comando base
    cmd = [sys.executable, "-m", "pip", "install"]
    
    # Para Ubuntu/WSL, usar las versiones estables
    if cuda_available and not is_wsl():
        # CUDA nativo en Linux
        cmd.extend([
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
    elif cuda_available and is_wsl():
        # CUDA en WSL2
        cmd.extend([
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
    else:
        # CPU only
        cmd.extend([
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
    
    print(f"\n📦 Instalando PyTorch...")
    print(f"   Comando: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ PyTorch instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando PyTorch: {e}")
        
        # Intento alternativo para WSL
        if is_wsl():
            print("\n🔄 Intentando instalación alternativa para WSL...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio"
                ], check=True)
                print("✅ PyTorch instalado (versión por defecto)")
                return True
            except:
                pass
        
        return False

def install_requirements():
    """Instala los requirements apropiados para Ubuntu/WSL"""
    python_version = get_python_version()
    
    # Instalar dependencias del sistema primero
    if shutil.which('apt'):
        install_system_dependencies()
    
    # Actualizar pip
    print("\n📦 Actualizando pip, setuptools y wheel...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "--upgrade", "pip", "setuptools", "wheel"
    ], check=True)
    
    # Instalar PyTorch
    pytorch_success = install_pytorch()
    
    if not pytorch_success:
        print("\n⚠️  Error instalando PyTorch")
        return False
    
    # Instalar otros requirements
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        # Crear requirements mínimo si no existe
        print(f"⚠️  No se encontró {req_file}, creando versión mínima...")
        with open(req_file, 'w') as f:
            f.write("""# Requirements mínimos para Ubuntu/WSL
transformers>=4.36.0
accelerate>=0.25.0
huggingface-hub>=0.20.0
numpy>=1.26.0
matplotlib>=3.8.0
tqdm>=4.66.0
safetensors>=0.4.0
""")
    
    print(f"\n📦 Instalando dependencias desde {req_file}...")
    
    # Leer requirements
    with open(req_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    failed_packages = []
    
    for requirement in requirements:
        if 'torch' in requirement.lower():
            continue  # Ya instalado
        
        try:
            print(f"  - Instalando {requirement}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", requirement],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            print(f"    ⚠️  Fallo al instalar {requirement}")
            failed_packages.append(requirement)
    
    if failed_packages:
        print(f"\n⚠️  Algunos paquetes fallaron:")
        for pkg in failed_packages:
            print(f"    - {pkg}")
    
    # Verificar instalación
    print("\n🔍 Verificando instalación...")
    return verify_installation()

def verify_installation():
    """Verifica la instalación con información específica de WSL"""
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'numpy': 'NumPy',
        'tqdm': 'tqdm'
    }
    
    all_good = True
    
    for package, name in required_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'instalado')
            print(f"✅ {name}: {version}")
            
            if package == 'torch':
                import torch
                cuda_available = torch.cuda.is_available()
                print(f"   CUDA disponible: {cuda_available}")
                
                if cuda_available:
                    print(f"   CUDA version: {torch.version.cuda}")
                    print(f"   GPU: {torch.cuda.get_device_name(0)}")
                    print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                elif is_wsl():
                    print("   💡 Para GPU en WSL2: instala drivers NVIDIA para WSL")
                    
        except ImportError:
            print(f"❌ {name}: no instalado")
            all_good = False
    
    # Información adicional para WSL
    if is_wsl():
        print("\n📊 Información de WSL:")
        
        # Memoria disponible
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb / (1024 * 1024)
                        print(f"   Memoria total: {mem_gb:.2f} GB")
                        break
        except:
            pass
        
        # Espacio en disco
        try:
            result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 4:
                    print(f"   Espacio disponible: {parts[3]}")
        except:
            pass
    
    return all_good

def main():
    print("🚀 Instalador LLM Compression Toolkit para Ubuntu/WSL")
    print("="*60)
    
    # Verificar que estamos en Linux
    if platform.system().lower() != 'linux':
        print("❌ Este script es para Ubuntu/WSL")
        print("   Para Windows, usa install_windows.py")
        sys.exit(1)
    
    # Detectar WSL
    if is_wsl():
        wsl_version = get_wsl_version()
        print(f"🐧 WSL{wsl_version} detectado")
        
        if wsl_version == 1:
            print("\n⚠️  WSL1 tiene limitaciones:")
            print("   - No soporta GPU")
            print("   - Rendimiento I/O reducido")
            print("   Considera actualizar a WSL2: wsl --set-version Ubuntu 2")
    
    # Verificar Python
    python_version = get_python_version()
    print(f"🐍 Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("\n❌ Python 3.8+ requerido")
        print("   Instala con: sudo apt install python3.11")
        sys.exit(1)
    
    # Advertencia para Python 3.13
    if python_version.major == 3 and python_version.minor >= 13:
        print("\n⚠️  Python 3.13 es muy reciente")
        print("   Recomendamos Python 3.10 o 3.11")
        response = input("\n¿Continuar? (s/n): ")
        if response.lower() != 's':
            sys.exit(0)
    
    # Instalar
    success = install_requirements()
    
    if success:
        print("\n✅ Instalación completada!")
        print("\n📚 Para comenzar:")
        print("  python analyze_model.py --help")
        
        if is_wsl():
            print("\n💡 Tips para WSL:")
            print("  - Los modelos se guardarán en el sistema de archivos de Linux")
            print("  - Para mejor rendimiento, trabaja en /home/usuario no en /mnt/c")
            print("  - Si tienes problemas de memoria, ajusta .wslconfig")
    else:
        print("\n⚠️  Instalación con advertencias")
        print("  Algunos componentes pueden faltar")

if __name__ == "__main__":
    main()