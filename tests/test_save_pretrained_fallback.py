import os
import sys
import pytest

# Ensure project root in path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from apply_compression import save_pretrained_with_fallback, logger


def test_save_pretrained_with_fallback_raises_after_three_recursion_errors(monkeypatch, tmp_path):
    class DummyModel:
        def __init__(self):
            self.calls = 0

        def save_pretrained(self, output_dir, safe_serialization=True):
            self.calls += 1
            raise RecursionError("failing save")

    dummy = DummyModel()
    # Force fallback path without safetensors
    monkeypatch.setattr("apply_compression.is_safetensors_available", lambda: False)

    import sys as real_sys
    base_limit = real_sys.getrecursionlimit()
    limits = []
    orig_set = real_sys.setrecursionlimit

    def fake_set(limit):
        limits.append(limit)
        orig_set(limit)

    monkeypatch.setattr(real_sys, "setrecursionlimit", fake_set)

    with pytest.raises(RuntimeError):
        save_pretrained_with_fallback(dummy, None, tmp_path, logger=logger)

    # Restore recursion limit to avoid side effects
    orig_set(base_limit)

    assert dummy.calls == 3
    assert limits == [base_limit * 2, base_limit * 4, base_limit * 8]


def test_save_pretrained_with_fallback_succeeds_after_retries(monkeypatch, tmp_path):
    class DummyModel:
        def __init__(self):
            self.calls = 0

        def save_pretrained(self, output_dir, safe_serialization=True):
            self.calls += 1
            if self.calls < 3:
                raise RecursionError("temporary fail")

    dummy = DummyModel()
    monkeypatch.setattr("apply_compression.is_safetensors_available", lambda: False)

    import sys as real_sys
    base_limit = real_sys.getrecursionlimit()
    limits = []
    orig_set = real_sys.setrecursionlimit

    def fake_set(limit):
        limits.append(limit)
        orig_set(limit)

    monkeypatch.setattr(real_sys, "setrecursionlimit", fake_set)

    save_pretrained_with_fallback(dummy, None, tmp_path, logger=logger)

    # Restore recursion limit
    orig_set(base_limit)

    assert dummy.calls == 3
    assert limits == [base_limit * 2, base_limit * 4]
