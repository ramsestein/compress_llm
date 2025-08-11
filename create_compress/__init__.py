"""
Módulo de compresión de modelos
"""

import logging

logger = logging.getLogger(__name__)

# Imports con mejor manejo de errores
try:
    # Imports relativos (cuando se usa como paquete)
    from .compression_methods import (
        COMPRESSION_METHODS, 
        get_compression_method, 
        get_available_methods,
        apply_compression,
        estimate_compression_ratio,
        validate_compression_config
    )
    from .compression_profiles import COMPRESSION_PROFILES, get_profile
    from .compression_config_manager import CompressionConfigManager
    
    # Si necesitas importar clases específicas de métodos
    from .compression_methods import (
        QuantizationMethod,
        PruningMethod, 
        LowRankApproximation,
        AttentionPruning
    )
    
except ImportError as e:
    logger.debug(f"Error en imports relativos: {e}, intentando imports directos")
    try:
        # Fallback para imports directos
        from compression_methods import (
            COMPRESSION_METHODS, 
            get_compression_method, 
            get_available_methods,
            apply_compression,
            estimate_compression_ratio,
            validate_compression_config,
            QuantizationMethod,
            PruningMethod,
            LowRankApproximation,
            AttentionPruning
        )
        from compression_profiles import COMPRESSION_PROFILES, get_profile
        from compression_config_manager import CompressionConfigManager
    except ImportError as e:
        logger.error(f"Error importando módulos de compresión: {e}")
        raise

# Versión del módulo
__version__ = "1.0.0"

# Exports públicos
__all__ = [
    # Funciones principales
    'COMPRESSION_METHODS',
    'get_compression_method',
    'get_available_methods',
    'apply_compression',
    'estimate_compression_ratio',
    'validate_compression_config',
    
    # Perfiles
    'COMPRESSION_PROFILES', 
    'get_profile',
    
    # Manager
    'CompressionConfigManager',
    
    # Clases de métodos (opcional)
    'QuantizationMethod',
    'PruningMethod',
    'LowRankApproximation', 
    'AttentionPruning'
]

# Verificación de disponibilidad
def check_module_health():
    """Verifica que todos los componentes estén disponibles"""
    required_components = {
        'COMPRESSION_METHODS': COMPRESSION_METHODS,
        'COMPRESSION_PROFILES': COMPRESSION_PROFILES,
        'CompressionConfigManager': CompressionConfigManager
    }
    
    missing = []
    for name, component in required_components.items():
        if component is None:
            missing.append(name)
    
    if missing:
        logger.warning(f"Componentes faltantes: {', '.join(missing)}")
        return False
    
    return True

# Verificar al importar
if not check_module_health():
    logger.warning("El módulo de compresión tiene componentes faltantes")