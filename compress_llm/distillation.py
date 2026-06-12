"""Knowledge distillation module for Compress LLM.

Generates a synthetic instruction dataset using a large teacher model and
trains a smaller student model via parameter-efficient fine-tuning (PEFT).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from LoRa_train.peft_methods_config import BasePEFTConfig
from LoRa_train.peft_universal_trainer import PEFTUniversalTrainer

logger = logging.getLogger(__name__)


class KnowledgeDistiller:
    """Orchestrates teacher-guided data generation and student training."""

    def __init__(
        self,
        teacher_model_name: str,
        student_model_name: str,
        device: Optional[str] = None,
    ):
        """Initialize the distiller with teacher and student model names."""
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = None
        self.tokenizer = None

    def _load_teacher(self) -> None:
        """Load the teacher model and tokenizer once."""
        if self.teacher is not None:
            return
        logger.info(f"Loading teacher model: {self.teacher_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.teacher_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.teacher = AutoModelForCausalLM.from_pretrained(
            self.teacher_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.teacher = self.teacher.to(self.device)
        logger.info("Teacher model loaded.")

    def _generate_text(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate text from the teacher given a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.teacher.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_synthetic_dataset(
        self,
        topic: str,
        num_samples: int,
        output_path: Path,
    ) -> Path:
        """Generate a CSV dataset of instruction/response pairs via the teacher.

        Args:
            topic: Subject matter for synthetic data generation.
            num_samples: Number of instruction-response pairs to generate.
            output_path: Destination CSV path.

        Returns:
            Path to the generated CSV file.

        """
        self._load_teacher()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        system_prompt = (
            f"You are an expert on '{topic}'. Generate a diverse set of "
            f"{num_samples} user questions and detailed answers about this topic. "
            "Format each example strictly as:\n"
            "Instruction: <user question>\n"
            "Response: <detailed answer>\n\n"
            "Ensure high quality, factual accuracy, and varied difficulty."
        )

        logger.info(f"Generating {num_samples} synthetic samples on '{topic}'...")
        raw_text = self._generate_text(system_prompt, max_new_tokens=num_samples * 180)

        # Parse the generated text for Instruction/Response pairs
        data: List[Dict[str, str]] = []
        blocks = raw_text.split("Instruction:")
        for block in blocks[1:]:
            if "Response:" in block:
                parts = block.split("Response:", 1)
                instruction = parts[0].strip()
                response = parts[1].split("Instruction:")[0].strip()
                if instruction and response:
                    data.append({"instruction": instruction, "response": response})
            if len(data) >= num_samples:
                break

        # Fallback if parsing yields too few samples
        if len(data) < num_samples:
            logger.warning(
                f"Parsed only {len(data)} samples; generating additional individual items."
            )
            while len(data) < num_samples:
                item_prompt = (
                    f"Write one question and a detailed answer about '{topic}'.\n"
                    "Format:\nInstruction: <question>\nResponse: <answer>"
                )
                item_text = self._generate_text(item_prompt, max_new_tokens=200)
                if "Response:" in item_text:
                    parts = item_text.split("Response:", 1)
                    instruction = parts[0].replace("Instruction:", "").strip()
                    response = parts[1].strip()
                    if instruction and response:
                        data.append({"instruction": instruction, "response": response})

        # Write CSV
        import csv

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["instruction", "response"])
            writer.writeheader()
            writer.writerows(data[:num_samples])

        logger.info(f"Synthetic dataset saved to {output_path} ({len(data[:num_samples])} rows).")
        return output_path

    def distill(
        self,
        topic: str,
        num_samples: int = 100,
        output_dir: Path = Path("./distilled_models"),
        peft_config: Optional[BasePEFTConfig] = None,
    ) -> Dict[str, Any]:
        """End-to-end knowledge distillation pipeline.

        Args:
            topic: Target subject for specialization.
            num_samples: Number of synthetic training examples.
            output_dir: Directory to save the trained student.
            peft_config: PEFT configuration for student fine-tuning.

        Returns:
            Dictionary with training metrics and output paths.

        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate dataset via teacher
        dataset_path = output_dir / f"{topic.replace(' ', '_')}_synthetic.csv"
        self.generate_synthetic_dataset(topic, num_samples, dataset_path)

        # Load the synthetic data for training
        import csv

        training_data: List[Dict[str, str]] = []
        with open(dataset_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                training_data.append(row)

        # Default to LoRA if no config provided
        if peft_config is None:
            from LoRa_train.peft_methods_config import LoRAConfig, PEFTMethod

            peft_config = LoRAConfig(
                method=PEFTMethod.LORA,
                learning_rate=2e-4,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
            )

        # Train student
        logger.info(f"Training student model on {len(training_data)} examples.")
        student_path = Path(self.student_model_name)
        if not student_path.exists():
            # Assume HuggingFace Hub model ID
            student_path = Path(f"./models/{self.student_model_name.replace('/', '_')}")

        trainer = PEFTUniversalTrainer(
            model_name=self.student_model_name,
            model_path=student_path,
            output_dir=output_dir / "student",
            peft_config=peft_config,
        )
        results = trainer.train(training_data)

        logger.info(f"Distillation complete. Model saved to {output_dir / 'student'}.")
        return {
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir / "student"),
            "metrics": results.get("metrics", {}),
        }
