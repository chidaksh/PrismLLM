import json
import os
import random

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def process_squad(squad_data):
    mistral_samples = []
    llama_samples = []
    for item in squad_data['responses']:
        context = item.get("context", "").strip()
        question = item.get("question", "").strip()
        prompt = f"Context: {context}\nQuestion: {question}"
        sample = {
            "prompt": prompt,
            "chosen": item["chosen_response"].strip(),
            "rejected": item["rejected_response"].strip(),
            "chosen_name": item.get("chosen_model", ""),
            "rejected_name": item.get("rejected_model", ""),
            "dataset": 'squad'
        }
        if item.get("chosen_model", "").lower() == "mistral":
            mistral_samples.append(sample)
        elif item.get("chosen_model", "").lower() == "llama":
            llama_samples.append(sample)
    num_samples = min(len(mistral_samples), len(llama_samples))
    return random.sample(mistral_samples, num_samples) + random.sample(llama_samples, num_samples)

def process_swag(swag_data):
    mistral_samples = []
    llama_samples = []
    for item in swag_data['responses']:
        ctx = item.get("ctx", "").strip()
        choices = item.get("choices", {})

        choice_str = "\n".join([
            f"{label}. {text.strip()}"
            for label, text in sorted(choices.items())
        ])

        prompt = f"Context: {ctx}\nChoices:\n{choice_str}"
        sample = {
            "prompt": prompt,
            "chosen": item["chosen_response"].strip(),
            "rejected": item["rejected_response"].strip(),
            "chosen_name": item.get("chosen_model", ""),
            "rejected_name": item.get("rejected_model", ""),
            "dataset": 'hellaswag'
        }

        if item.get("chosen_model", "").lower() == "mistral":
            mistral_samples.append(sample)
        elif item.get("chosen_model", "").lower() == "llama":
            llama_samples.append(sample)
    num_samples = min(len(mistral_samples), len(llama_samples))
    return random.sample(mistral_samples, num_samples) + random.sample(llama_samples, num_samples)

def combine_datasets(squad_path, swag_path):
    squad_data = load_json(squad_path)
    swag_data = load_json(swag_path)
    
    processed_squad = process_squad(squad_data)
    processed_swag = process_swag(swag_data)
    
    combined = processed_squad + processed_swag
    return combined

def save_combined_dataset(data, output_path="combined_dpo_data.json"):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Combined dataset saved to {output_path}")

def main():
    squad_path = "/work/users/c/h/chidaksh/hackathons/hack2/squad_ann.json"
    swag_path = "/work/users/c/h/chidaksh/hackathons/hack2/swag_ann.json"
    output_path = "/work/users/c/h/chidaksh/hackathons/hack2/combined_dpo_data.json"

    print("Loading and processing datasets...")
    combined = combine_datasets(squad_path, swag_path)
    save_combined_dataset(combined, output_path)

if __name__ == "__main__":
    main()
