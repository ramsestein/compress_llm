#!/usr/bin/env python3
"""
Test script for translation model
Tests the fine-tuned model's ability to translate from Catalan to Chinese
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

def load_model_and_tokenizer(model_path: str):
    """Load the fine-tuned model and tokenizer"""
    console.print(f"[blue]Loading model from: {model_path}[/blue]")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        console.print("[green]✅ Model loaded successfully![/green]")
        return model, tokenizer
        
    except Exception as e:
        console.print(f"[red]❌ Error loading model: {e}[/red]")
        return None, None

def translate_text(model, tokenizer, text: str, max_length: int = 200):
    """Translate Catalan text to Chinese"""
    # Create the prompt in the same format as training
    prompt = f"Translate from Catalan to Chinese:\n{text}\n"
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Move to same device as model
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the translation part (remove the prompt)
    translation = generated_text.replace(prompt, "").strip()
    
    return translation

def interactive_test(model, tokenizer):
    """Interactive testing mode"""
    console.print("\n[bold green]🎯 Interactive Translation Test[/bold green]")
    console.print("Enter Catalan text to translate (type 'quit' to exit)")
    console.print("=" * 50)
    
    while True:
        try:
            catalan_text = Prompt.ask("\n[cyan]Catalan text")
            
            if catalan_text.lower() in ['quit', 'exit', 'q']:
                break
                
            if not catalan_text.strip():
                continue
                
            console.print("[yellow]Translating...[/yellow]")
            translation = translate_text(model, tokenizer, catalan_text)
            
            # Display results
            console.print(Panel(
                f"[bold]Catalan:[/bold] {catalan_text}\n[bold]Chinese:[/bold] {translation}",
                title="Translation Result",
                border_style="green"
            ))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

def batch_test(model, tokenizer, test_cases: list):
    """Test with predefined examples"""
    console.print("\n[bold green]🧪 Batch Translation Test[/bold green]")
    console.print("=" * 50)
    
    for i, catalan_text in enumerate(test_cases, 1):
        console.print(f"\n[bold]Test {i}:[/bold]")
        console.print(f"[cyan]Catalan:[/cyan] {catalan_text}")
        
        try:
            translation = translate_text(model, tokenizer, catalan_text)
            console.print(f"[green]Chinese:[/green] {translation}")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(description="Test translation model")
    parser.add_argument("model_path", help="Path to the fine-tuned model")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--batch", "-b", action="store_true", help="Batch test mode")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    if model is None or tokenizer is None:
        return
    
    # Test cases for batch mode
    test_cases = [
        "Hola, com estàs?",
        "Gràcies per la teva ajuda",
        "Quina hora és?",
        "On és l'estació de tren?",
        "Vull aprendre xinès"
    ]
    
    if args.interactive:
        interactive_test(model, tokenizer)
    elif args.batch:
        batch_test(model, tokenizer, test_cases)
    else:
        # Default: show both options
        console.print("\n[bold]Choose test mode:[/bold]")
        console.print("1. Interactive mode (type your own text)")
        console.print("2. Batch mode (test with predefined examples)")
        
        choice = Prompt.ask("Select option", choices=["1", "2"], default="1")
        
        if choice == "1":
            interactive_test(model, tokenizer)
        else:
            batch_test(model, tokenizer, test_cases)

if __name__ == "__main__":
    main()





