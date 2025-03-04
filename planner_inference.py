import torch
import torch.nn as nn
from unsloth import FastLanguageModel
from peft import PeftModel
from llama import run_llama_vllm, load_llama_vllm
from mistral import run_mistral, load_mistral
import sys

def load_model_and_classifier(model_path, classifier_path, device="cuda"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/DeepSeek-R1-Distill-Llama-8B",
        device_map="auto",
        load_in_4bit=True
    )
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    hidden_size = model.config.hidden_size
    classifier = MLPClassifier(hidden_size).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()

    return model, tokenizer, classifier

class MLPClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes=2):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_size, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_classes))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        return self.softmax(self.fc(x))

def preprocess_query(query, tokenizer, device):
    input_text = f"Given the following query, decide whether Mistral or Llama would produce the best response.\n\nQuery: {query}\n\nYour answer should be 0 if Mistral is better, 1 if Llama is better."
    inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=1024, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}

def infer_routing(model, classifier, tokenizer, query, device="cuda"):
    inputs = preprocess_query(query, tokenizer, device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, 0, :].to(torch.float32)
        preds = classifier(hidden)
        predicted_label = torch.argmax(preds, dim=1).item()
    
    return "Mistral" if predicted_label == 0 else "Llama"

if __name__ == "__main__":
    model_path = "./ckpts/finetuned_llama_lora" 
    classifier_path = "./ckpts/finetuned_mlp.pth"
    model, tokenizer, classifier = load_model_and_classifier(model_path, classifier_path)
    
    random_query = sys.argv[1]
    print("Query:", random_query)
    predicted_llm = infer_routing(model, classifier, tokenizer, random_query)
    response = None
    
    if predicted_llm == "Llama":
        llm = load_llama_vllm()
        response = run_llama_vllm(llm, user_query=random_query)
    else:
        model, tokenizer, device = load_mistral()
        response = run_mistral(random_query, model, tokenizer=tokenizer, device=device)
        
    print(predicted_llm, response)