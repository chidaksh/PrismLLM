import os
import pandas as pd
import json
import re
import argparse
from tqdm import tqdm
import time

# Import functions from llama.py
from llama import load_llama_vllm, run_llama_vllm

def parse_args():
    parser = argparse.ArgumentParser(description="Run Llama inference on test queries using imported llama.py functions")
    parser.add_argument("--output_file", type=str, default="test_responses.json",
                        help="Path to save Llama responses")
    parser.add_argument("--model_name", type=str, default="astronomer/Llama-3-8B-GPTQ-4-Bit",
                        help="Model name or path")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    return parser.parse_args()

def generate_test_data():
    questions = [
        "What is the capital of France? A) Berlin B) Madrid C) Paris D) Rome",
        "Who wrote 'To Kill a Mockingbird'? A) J.K. Rowling B) Harper Lee C) Mark Twain D) Ernest Hemingway",
        "What is the largest planet in our solar system? A) Earth B) Venus C) Jupiter D) Mars",
        "Which element has the atomic number 1? A) Oxygen B) Carbon C) Hydrogen D) Helium",
        "Who painted the Mona Lisa? A) Vincent van Gogh B) Pablo Picasso C) Leonardo da Vinci D) Claude Monet"
    ]
    queries = [{"query_id": i+1, "query": question} for i, question in enumerate(questions)]
    return pd.DataFrame(queries)

def extract_answer_and_reasoning(response_text):
    # Simple pattern to just extract the first letter choice that appears in the response
    answer_pattern = r"([A-D])"
    
    answer_match = re.search(answer_pattern, response_text)
    answer = answer_match.group(1) if answer_match else "Unknown"
    
    # Return the entire response as the reasoning without preprocessing
    reasoning = response_text
    
    return answer, reasoning

def run_inference():
    args = parse_args()
    
    print(f"Loading model: {args.model_name}")
    llm = load_llama_vllm(args.model_name)
    
    # Run a warm-up query
    _ = run_llama_vllm(llm, "Warmup", max_tokens=10)
    
    queries_df = generate_test_data()
    print(f"Processing {len(queries_df)} test queries...")
    
    results = []
    total_time = 0
    
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df)):
        query_id = row['query_id']
        query = row['query']
        
        # Set parameters optimized for extracting answers and reasoning
        start_time = time.time()
        model_response = run_llama_vllm(
            llm, 
            query, 
            max_tokens=args.max_tokens,
            temperature=0.1,  # Lower temperature for more deterministic answers
            top_p=0.9
        )
        inference_time = time.time() - start_time
        total_time += inference_time
        
        answer, reasoning = extract_answer_and_reasoning(model_response)
        
        results.append({
            "query_id": query_id,
            "query": query,
            "llama_answer": answer,
            "llama_reasoning": reasoning,
            "full_response": model_response,
            "inference_time": inference_time
        })
        
        print(f"Query {query_id}: Answer: {answer} | Time: {inference_time:.2f}s")
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
    avg_time = total_time / len(queries_df)
    print(f"\nSummary:")
    print(f"Total inference time: {total_time:.2f}s")
    print(f"Average time per query: {avg_time:.2f}s")
    print(f"Results saved to {args.output_file}")
    
    return results

if __name__ == "__main__":
    run_inference()