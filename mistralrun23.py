import os
import pandas as pd
import json
import re
import argparse
from tqdm import tqdm
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Run Mistral inference on dataset queries")
    parser.add_argument("--input_file", type=str, default="/users/s/a/sakk/llm/2k-3k.csv",
                        help="Path to CSV file with queries")
    parser.add_argument("--output_file", type=str, default="/users/s/a/sakk/llm/mistral_responses_2-3.json",
                        help="Path to save Mistral responses last")
    parser.add_argument("--model_name", type=str, default="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",  # Changed to instruct model
                        help="Model name or path")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,  # Increased temperature
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.95,  # Increased top-p
                        help="Top-p for generation")
    return parser.parse_args()

def load_mistral(model_name: str):
    """Load the Mistral model optimized for GPU."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def run_mistral(model, tokenizer, text_input, max_tokens=512, temperature=0.7, top_p=0.95):  # Adjusted default params
    """Generate a response from Mistral for a given query."""
    # Simplified prompt for base models or adjusted for instruct
    prompt = f"""<s>[INST] Answer the question and provide a clear reasoning. Be concise. Question: {text_input} [/INST]"""
    
    for _ in range(3):
        start_time = time.time()
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad token
            )
        
        generated_tokens = generated_ids[:, encoded.input_ids.shape[1]:]  # Slicing after input length
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        # Check for non-empty content (including markdown/whitespace)
        clean_decoded = re.sub(r'\s+', ' ', decoded).strip()
        if clean_decoded:
            return decoded, time.time() - start_time
    
    return "Error: No valid response generated after multiple attempts.", time.time() - start_time

def run_inference():
    args = parse_args()
    
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    queries_df = pd.read_csv(args.input_file)
    print(f"Loaded {len(queries_df)} queries from {args.input_file}")
    
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_mistral(args.model_name)
    
    results = []
    checkpoint_interval = 100
    checkpoint_file = args.output_file.replace(".json", "_checkpoint.json")
    
    try:
        for i, row in tqdm(queries_df.iterrows(), total=len(queries_df)):
            query_id = row['query_id']
            query = row['query']
            
            response_text, inference_time = run_mistral(
                model, tokenizer, query, 
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            results.append({
                "query_id": int(query_id),
                "query": query,
                "full_response": response_text,
                "inference_time": inference_time,
                "model_name": "Mistral"
            })
            
            if (i + 1) % checkpoint_interval == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Checkpoint saved at query {i+1}/{len(queries_df)}")
                
    except KeyboardInterrupt:
        print("Interrupted. Saving checkpoint...")
    finally:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")
    
    return results

if __name__ == "__main__":
    run_inference()