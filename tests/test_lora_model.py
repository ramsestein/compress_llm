import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lora_model_tester import LoRAModelTester


@pytest.fixture
def dummy_model_dir(tmp_path):
    adapter_config = {"base_model_name_or_path": "dummy-base"}
    (tmp_path / "adapter_config.json").write_text(json.dumps(adapter_config))
    return tmp_path


def test_batch_test_returns_expected_keys(dummy_model_dir, monkeypatch):
    tester = LoRAModelTester(str(dummy_model_dir))
    monkeypatch.setattr(LoRAModelTester, "load_model", lambda self: None)
    monkeypatch.setattr(LoRAModelTester, "generate", lambda self, prompt, max_length=200, temperature=0.7: f"echo: {prompt}")

    prompts = ["hi", "there"]
    results = tester.batch_test(prompts)

    assert len(results) == len(prompts)
    for res, prompt in zip(results, prompts):
        assert res["prompt"] == prompt
        assert set(["prompt", "response", "success"]).issubset(res.keys())
        assert res["success"] is True


def test_batch_test_handles_generation_errors(dummy_model_dir, monkeypatch):
    tester = LoRAModelTester(str(dummy_model_dir))
    monkeypatch.setattr(LoRAModelTester, "load_model", lambda self: None)

    def faulty_generate(self, prompt, max_length=200, temperature=0.7):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(LoRAModelTester, "generate", faulty_generate)

    prompts = ["fail"]
    results = tester.batch_test(prompts)

    assert len(results) == 1
    entry = results[0]
    assert entry["prompt"] == "fail"
    assert entry["success"] is False
    assert "error" in entry
