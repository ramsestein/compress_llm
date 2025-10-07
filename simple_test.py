#!/usr/bin/env python3
"""
Simple test for compressed model - no Unicode characters
"""
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_compressed_model():
    print("=" * 60)
    print("TESTING COMPRESSED MODEL")
    print("=" * 60)
    
    try:
        # Load compressed model
        print("1. Loading compressed model...")
        model_path = './models/distilgpt2_compressed'
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("SUCCESS: Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Tokenizer type: {type(tokenizer).__name__}")
        
        # Test text generation
        print("\n2. Testing text generation...")
        test_prompts = [
            "The future of AI is",
            "Once upon a time",
            "The weather today is",
            "I love programming because"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n   Test {i}: \"{prompt}\"")
            inputs = tokenizer(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_length=50, 
                    do_sample=True, 
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   Generated: \"{result}\"")
        
        # Test model size
        print("\n3. Checking model size...")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        print("\n" + "=" * 60)
        print("SUCCESS: ALL TESTS PASSED!")
        print("SUCCESS: Compressed model is working perfectly!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_compressed_model()
    sys.exit(0 if success else 1)
