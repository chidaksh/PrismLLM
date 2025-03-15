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
    parser = argparse.ArgumentParser(description="Run Llama inference on MMLU queries")
    parser.add_argument("--input_file", type=str, default="mmlu_extracts/llama_queries.csv",
                        help="Path to CSV file with MMLU queries")
    parser.add_argument("--output_file", type=str, default="mmlu_extracts/llama_responses.json",
                        help="Path to save Llama responses")
    parser.add_argument("--model_name", type=str, default="astronomer/Llama-3-8B-GPTQ-4-Bit",
                        help="Model name or path")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of queries to process in parallel (if supported by vLLM)")
    return parser.parse_args()

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
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load MMLU queries
    queries_df = pd.read_csv(args.input_file)
    print(f"Loaded {len(queries_df)} queries from {args.input_file}")
    
    print(f"Loading model: {args.model_name}")
    llm = load_llama_vllm(args.model_name)
    
    # Run a warm-up query
    print("Running warm-up query...")
    _ = run_llama_vllm(llm, "Warmup query", max_tokens=10)
    
    print(f"Processing MMLU queries...")
    
    results = []
    total_time = 0
    
    # Checkpoint functionality to save progress periodically
    checkpoint_interval = 100
    checkpoint_file = args.output_file.replace(".json", "_checkpoint.json")
    
    try:
        for i, row in tqdm(queries_df.iterrows(), total=len(queries_df)):
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
            
            result = {
                "query_id": int(query_id),
                "query": query,
                "llama_answer": answer,
                "llama_reasoning": reasoning,
                "full_response": model_response,
                "inference_time": inference_time
            }
            
            results.append(result)
            
            if (i + 1) % 10 == 0:  # Print status every 10 queries
                print(f"Query {query_id}: Answer: {answer} | Time: {inference_time:.2f}s")
            
            # Save checkpoint periodically
            if (i + 1) % checkpoint_interval == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Checkpoint saved at query {i+1}/{len(queries_df)}")
        
    except KeyboardInterrupt:
        print("Interrupted by user. Saving results so far...")
    
    finally:
        # Save final results
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary statistics
        queries_processed = len(results)
        if queries_processed > 0:
            avg_time = total_time / queries_processed
            print(f"\nSummary:")
            print(f"Processed {queries_processed}/{len(queries_df)} queries")
            print(f"Total inference time: {total_time:.2f}s")
            print(f"Average time per query: {avg_time:.2f}s")
            print(f"Results saved to {args.output_file}")
        else:
            print("No queries were processed.")
    
    return results

if __name__ == "__main__":
    run_inference()
