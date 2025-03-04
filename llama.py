import torch
from vllm import LLM, SamplingParams
import time
import sys

def load_llama_vllm(model_name="astronomer/Llama-3-8B-GPTQ-4-Bit"):
    llm = LLM(model_name, tensor_parallel_size=torch.cuda.device_count(), dtype="float16")
    return llm

def format_prompt(user_query):
    return f"""<|system|>
You are a helpful AI assistant. Provide detailed answers using clear reasoning.
<|user|>
{user_query}
<|assistant|>
"""

def run_llama_vllm(llm, user_query, max_tokens=500, temperature=0.7, top_p=0.9):
    prompt = format_prompt(user_query)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=1.1
    )
    
    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    output_time = time.time() - start_time
    
    response = outputs[0].outputs[0].text.strip()
    print(f"\nInference time: {output_time:.2f}s ({max_tokens/output_time:.2f} tokens/s)")
    return response

if __name__ == "__main__":
    try:
        llm = load_llama_vllm()
        _ = run_llama_vllm(llm, "Warmup", max_tokens=10)
        query = sys.argv[1]
        response = run_llama_vllm(llm, query, max_tokens=500)
        print("\nResponse:", response)
        
    except Exception as e:
        print(f"Error: {e}")
        if "CUDA out of memory" in str(e):
            print("Try reducing max_tokens or using a smaller model.")

