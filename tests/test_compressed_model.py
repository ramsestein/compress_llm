#!/usr/bin/env python3
"""Script simple para testear el modelo comprimido."""

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def test_compressed_model():
    """Test simple del modelo comprimido."""
    model_dir = Path("models/microsoft_DialoGPT-small_compressed")

    if not model_dir.exists():
        print("Not found el modelo comprimido")
        return

    print(" Testeando modelo comprimido...")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("models/microsoft_DialoGPT-small")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Tokenizer cargado")

        # Load config
        config = AutoConfig.from_pretrained(str(model_dir))
        print("Config cargado")

        # Create modelo
        model = AutoModelForCausalLM.from_config(config)
        print("Modelo creado")

        # Load parameters
        state_dict = {}
        param_files = list(model_dir.glob("*.pt"))
        print(f"Encontrados {len(param_files)} archivos de parameters")

        # Mapeo de nombres de archivo a nombres de parameters
        name_mapping = {}
        for param_file in param_files:
            simple_name = param_file.stem

            # Mapear nombres correctamente
            if "transformer_h_" in simple_name:
                parts = simple_name.split("_")
                if len(parts) >= 4:
                    layer_num = parts[2]
                    layer_type = parts[3]

                    if layer_type == "attn":
                        if "c_attn" in simple_name:
                            if "bias" in simple_name:
                                param_name = f"transformer.h.{layer_num}.attn.c_attn.bias"
                            else:
                                param_name = f"transformer.h.{layer_num}.attn.c_attn.weight"
                        elif "c_proj" in simple_name:
                            if "bias" in simple_name:
                                param_name = f"transformer.h.{layer_num}.attn.c_proj.bias"
                            else:
                                param_name = f"transformer.h.{layer_num}.attn.c_proj.weight"
                    elif layer_type == "mlp":
                        if "c_fc" in simple_name:
                            if "bias" in simple_name:
                                param_name = f"transformer.h.{layer_num}.mlp.c_fc.bias"
                            else:
                                param_name = f"transformer.h.{layer_num}.mlp.c_fc.weight"
                        elif "c_proj" in simple_name:
                            if "bias" in simple_name:
                                param_name = f"transformer.h.{layer_num}.mlp.c_proj.bias"
                            else:
                                param_name = f"transformer.h.{layer_num}.mlp.c_proj.weight"
                    elif layer_type == "ln":
                        if "ln_1" in simple_name:
                            if "bias" in simple_name:
                                param_name = f"transformer.h.{layer_num}.ln_1.bias"
                            else:
                                param_name = f"transformer.h.{layer_num}.ln_1.weight"
                        elif "ln_2" in simple_name:
                            if "bias" in simple_name:
                                param_name = f"transformer.h.{layer_num}.ln_2.bias"
                            else:
                                param_name = f"transformer.h.{layer_num}.ln_2.weight"
                    else:
                        param_name = simple_name
                else:
                    param_name = simple_name
            elif "transformer_ln_f" in simple_name:
                if "bias" in simple_name:
                    param_name = "transformer.ln_f.bias"
                else:
                    param_name = "transformer.ln_f.weight"
            elif "transformer_wte" in simple_name:
                param_name = "transformer.wte.weight"
            elif "transformer_wpe" in simple_name:
                param_name = "transformer.wpe.weight"
            else:
                param_name = simple_name

            name_mapping[simple_name] = param_name

        # Load parameters
        for param_file in param_files:
            try:
                simple_name = param_file.stem
                param_name = name_mapping.get(simple_name, simple_name)

                param_tensor = torch.load(param_file, map_location="cpu")

                # Verify que el tensor tiene la forma correcta
                if param_name in model.state_dict():
                    expected_shape = model.state_dict()[param_name].shape
                    if param_tensor.shape == expected_shape:
                        state_dict[param_name] = param_tensor
                        print(f"{simple_name} -> {param_name} ({param_tensor.shape})")
                    else:
                        print(
                            f"Forma incorrecta: {simple_name} -> {param_name} (esperado: {expected_shape}, actual: {param_tensor.shape})"
                        )
                else:
                    print(f"Parámetro not found: {param_name}")

            except Exception as e:
                print(f"Error cargando {param_file.name}: {e}")
                continue

        print(f"\nParameters cargados: {len(state_dict)}")

        # Load estado del modelo
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Claves faltantes: {len(missing_keys)}")
        print(f"Claves inesperadas: {len(unexpected_keys)}")

        if len(missing_keys) < 10:  # Si faltan pocas claves, intentar generar
            print("\nTrying generación de texto...")

            model.eval()
            prompt = "Hello"
            inputs = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=len(inputs["input_ids"][0]) + 20,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Texto generado: {generated_text}")

        else:
            print("Demasiadas claves faltantes para generar texto")
            print("Primeras 10 claves faltantes:")
            for i, key in enumerate(missing_keys[:10]):
                print(f"   {i + 1}. {key}")

        print("\nTest completado!")

    except Exception as e:
        print(f"Error durante el test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_compressed_model()
