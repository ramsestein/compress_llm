# Contributing to Compress LLM

Thank you for your interest in contributing to Compress LLM. This document provides guidelines to help you get started.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ramsestein/compress_llm.git
   cd compress_llm
   ```

2. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Code Style

- Format code with `ruff`:
  ```bash
  ruff format .
  ```

- Lint with `ruff`:
  ```bash
  ruff check .
  ```

- Type-check with `mypy`:
  ```bash
  mypy create_compress LoRa_train compress_llm
  ```

## Testing

- Run the test suite:
  ```bash
  python -m unittest discover tests
  ```

- Ensure all tests pass before submitting a pull request.

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Make your changes and add tests if applicable.
3. Update documentation and the `CHANGELOG.md`.
4. Ensure `ruff format`, `ruff check`, and `python -m unittest discover tests` all pass locally.
5. Submit a pull request with a clear description of the changes.

## Reporting Issues

Please use the [GitHub Issues](https://github.com/ramsestein/compress_llm/issues) page to report bugs or request features. Include:
- A clear description of the issue
- Steps to reproduce
- Environment details (OS, Python version, GPU if applicable)
- Relevant error messages or logs

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.
