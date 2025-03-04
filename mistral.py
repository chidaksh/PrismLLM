import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import perf_counter
import sys

def load_pre_quantized_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    return model

def initialize_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def run_mistral(text_input, model, max_tokens=500, temperature=0.7, top_p=0.9, deterministic=False, tokenizer=None, device=None):
    prompt = f"""<s>[INST]You are a helpful AI agent. Answer the question and also provide a clear and logical reasoning.\n User query: {text_input}[/INST]"""

    start_time = perf_counter()
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=not deterministic,
            temperature=temperature if not deterministic else None,
            top_p=top_p if not deterministic else None,
            repetition_penalty=1.1
        )

    generated_tokens = generated_ids[:, len(encoded.input_ids[0]):]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    output_time = perf_counter() - start_time
    print(f"Time taken for inference: {round(output_time, 2)} seconds")
    return decoded[0]

def load_mistral():
    MODEL_NAME = "unsloth/mistral-7b-bnb-4bit"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_pre_quantized_model(MODEL_NAME).to(device)
    tokenizer = initialize_tokenizer(MODEL_NAME)
    return model, tokenizer, device

if __name__ == "__main__":
    user_input = sys.argv[1]
    model, tokenizer, device = load_mistral()
    output = run_mistral(user_input, model, max_tokens=50, temperature=0.7, top_p=0.9, deterministic=False, tokenizer=tokenizer, device=device)
    print(output)
