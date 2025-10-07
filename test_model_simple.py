#!/usr/bin/env python3
"""
Script simple para probar el modelo LoRA sin emojis (compatible con Windows)
"""
import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

def test_model_compare():
    """Compara el modelo base con el modelo LoRA entrenado"""
    
    print("=" * 60)
    print("COMPARANDO MODELO BASE vs MODELO LoRA ENTRENADO")
    print("=" * 60)
    
    # Verificar que existe el directorio output
    output_dir = Path("output")
    if not output_dir.exists():
        print("ERROR: No se encontró el directorio 'output'")
        return
    
    print(f"OK: Directorio de salida encontrado: {output_dir}")
    
    # Frases de prueba
    test_prompts = [
        "Hola, com estàs?",
        "Gràcies per la teva ajuda", 
        "Vull aprendre xinès",
        "Aquest és un test de traducció"
    ]
    
    try:
        print("\nCargando modelo base (DistilGPT2)...")
        
        # Cargar modelo base
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("OK: Modelo base cargado")
        
        print("\nCargando modelo LoRA entrenado...")
        
        # Cargar modelo LoRA
        lora_model = PeftModel.from_pretrained(base_model, str(output_dir))
        
        print("OK: Modelo LoRA cargado")
        
        print("\n" + "=" * 60)
        print("COMPARACION DE RESPUESTAS")
        print("=" * 60)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Prompt: {prompt}")
            print("-" * 40)
            
            # Preparar input
            input_text = f"Traduir al xinès: {prompt}\nXinès:"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            
            # Generar con modelo base
            with torch.no_grad():
                base_outputs = base_model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Generar con modelo LoRA
            with torch.no_grad():
                lora_outputs = lora_model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decodificar respuestas
            base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
            lora_response = tokenizer.decode(lora_outputs[0], skip_special_tokens=True)
            
            # Extraer solo la parte generada
            base_generated = base_response[len(input_text):].strip()
            lora_generated = lora_response[len(input_text):].strip()
            
            print(f"Modelo BASE:    {base_generated}")
            print(f"Modelo LoRA:    {lora_generated}")
            
            # Verificar si son diferentes
            if base_generated != lora_generated:
                print(">>> DIFERENTE - El entrenamiento cambió el comportamiento")
            else:
                print(">>> IGUAL - El entrenamiento no cambió el comportamiento")
        
        print("\n" + "=" * 60)
        print("ANALISIS FINAL")
        print("=" * 60)
        print("- Si ves diferencias, el entrenamiento funcionó")
        print("- Si son iguales, puede que el modelo base no sea adecuado")
        print("- DistilGPT2 no está diseñado para traducción catalán-chino")
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nPosibles soluciones:")
        print("1. Verificar que el entrenamiento se completó")
        print("2. Revisar que todos los archivos están en 'output'")
        print("3. El modelo base puede no ser adecuado para traducción")

if __name__ == "__main__":
    test_model_compare()


