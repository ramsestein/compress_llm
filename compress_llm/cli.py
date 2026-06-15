"""Zero-code CLI entry points for Compress LLM.

Provides simple commands for users without programming knowledge:
- compress-llm: Interactive model compression
- finetune-llm: Interactive fine-tuning wizard
- ollama-serve: Start the Ollama-compatible API server
"""

import sys
from pathlib import Path


def main() -> None:
    """Entry point for compress-llm CLI."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "apply_compression.py"
    if script.exists():
        exec(script.read_text(encoding="utf-8"), {"__name__": "__main__", "__file__": str(script)})
    else:
        print("Error: scripts/apply_compression.py not found.")
        sys.exit(1)


def finetune_main() -> None:
    """Entry point for finetune-llm CLI."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "finetune_peft.py"
    if script.exists():
        exec(script.read_text(encoding="utf-8"), {"__name__": "__main__", "__file__": str(script)})
    else:
        print("Error: scripts/finetune_peft.py not found.")
        sys.exit(1)


def serve_main() -> None:
    """Entry point for ollama-serve CLI."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "ollama_compact_server.py"
    if script.exists():
        exec(script.read_text(encoding="utf-8"), {"__name__": "__main__", "__file__": str(script)})
    else:
        print("Error: scripts/ollama_compact_server.py not found.")
        sys.exit(1)


def distill_main() -> None:
    """Entry point for distill-llm CLI."""
    from compress_llm.distillation_cli import main as _main

    _main()


if __name__ == "__main__":
    main()
