"""Model compression module."""

import logging

logger = logging.getLogger(__name__)

# Imports with better error handling
try:
    # Relative imports (when used as a package)
    from .compression_config_manager import CompressionConfigManager

    # Import specific method classes if needed
    from .compression_methods import (
        COMPRESSION_METHODS,
        AttentionPruning,
        LowRankApproximation,
        PruningMethod,
        QuantizationMethod,
        apply_compression,
        estimate_compression_ratio,
        get_available_methods,
        get_compression_method,
        validate_compression_config,
    )
    from .compression_profiles import COMPRESSION_PROFILES, get_profile

except ImportError as e:
    logger.debug(f"Error in relative imports: {e}, trying direct imports")
    try:
        # Fallback for direct imports
        from compression_config_manager import CompressionConfigManager
        from compression_methods import (
            COMPRESSION_METHODS,
            AttentionPruning,
            LowRankApproximation,
            PruningMethod,
            QuantizationMethod,
            apply_compression,
            estimate_compression_ratio,
            get_available_methods,
            get_compression_method,
            validate_compression_config,
        )
        from compression_profiles import COMPRESSION_PROFILES, get_profile
    except ImportError as e:
        logger.error(f"Error importing compression modules: {e}")
        raise

# Module version
__version__ = "1.0.0"

# Public exports
__all__ = [
    # Main functions
    "COMPRESSION_METHODS",
    "get_compression_method",
    "get_available_methods",
    "apply_compression",
    "estimate_compression_ratio",
    "validate_compression_config",
    # Profiles
    "COMPRESSION_PROFILES",
    "get_profile",
    # Manager
    "CompressionConfigManager",
    # Method classes (optional)
    "QuantizationMethod",
    "PruningMethod",
    "LowRankApproximation",
    "AttentionPruning",
]


# Verificación de disponibilidad
def check_module_health():
    """Verifies that all components are available."""
    required_components = {
        "COMPRESSION_METHODS": COMPRESSION_METHODS,
        "COMPRESSION_PROFILES": COMPRESSION_PROFILES,
        "CompressionConfigManager": CompressionConfigManager,
    }

    missing = []
    for name, component in required_components.items():
        if component is None:
            missing.append(name)

    if missing:
        logger.warning(f"Missing components: {', '.join(missing)}")
        return False

    return True


# Verify on import
if not check_module_health():
    logger.warning("The compression module has missing components")
