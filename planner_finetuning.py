import torch
import torch.nn as nn
from unsloth import FastLanguageModel
from datasets import Dataset
from torch.utils.data import DataLoader
import json
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import random

def load_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def load_data(datareq1, phase="full"):
    data = []
    for item in datareq1["responses"]:
        query = item["query"]
        correct_answer = item["correct_answer"]
        llama_response = item["llama_response"]
        mistral_response = item["mistral_response"]
        label = 0 if item["label"] == 'Mistral' else 1
        if phase == "full":
            input_text = ("You are an expert evaluator trained to assess AI responses. Given a query, ground truth answer, and LLM responses, decide which LLM is more aligned with the correct answer.\n\n"
                          f"Query: {query}\n\nGround Truth Answer: {correct_answer}\n\nMistral Response: {mistral_response}\n\nLlama Response: {llama_response}\n\nYour output should be 0 if Mistral is better and 1 if Llama is better.")
        else:
            input_text = ("Given the following query, decide whether Mistral or Llama would produce the best response.\n\n"
                          f"Query: {query}\n\nYour answer should be 0 if Mistral is better, 1 if Llama is better.")
        data.append({"input_text": input_text, "label": label})
    return Dataset.from_list(data)

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=1024, return_tensors="pt")

def convert_to_tensors(batch):
    return {
        "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
        "label": torch.tensor(batch["label"], dtype=torch.long)
    }

def collate_fn(batch, device):
    input_ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    attention_mask = [torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {
        "input_ids": torch.stack(input_ids).to(device),
        "attention_mask": torch.stack(attention_mask).to(device),
        "label": labels.to(device)
    }

class MLPClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes=2):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_size, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_classes))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        return self.softmax(self.fc(x))

def train_epoch(model, classifier, dataloader, optimizer, scheduler, device):
    model.train()
    classifier.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        labels = batch["label"]
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, 0, :]
        hidden = hidden.to(torch.float32)
        preds = classifier(hidden)
        loss = nn.CrossEntropyLoss(weight=torch.tensor([0.625, 2.5]).to(device))(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_epoch(model, classifier, dataloader, device):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
            labels = batch["label"]
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, 0, :]
            hidden = hidden.to(torch.float32)
            preds = classifier(hidden)
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def get_dataloader(dataset, batch_size, device):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, device))

def main():
    device = "cuda"
    datareq = load_json_data('./rearranged_resp.json')
    # breakpoint()
    all_responses = datareq['responses']
    random.shuffle(all_responses)
    split_idx = int(0.7 * len(all_responses))  # 80% for full context, 20% for query-only
    full_context_data = {"responses": all_responses[:split_idx]}
    query_only_data = {"responses": all_responses[split_idx:]}
    full_dataset = load_data(full_context_data, phase="full")
    query_dataset = load_data(query_only_data, phase="query_only")
    model, tokenizer = FastLanguageModel.from_pretrained("unsloth/DeepSeek-R1-Distill-Llama-8B", device_map="auto", load_in_4bit=True)
    for param in model.parameters():
        param.requires_grad = False
    lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    hidden_size = model.config.hidden_size
    classifier = MLPClassifier(hidden_size).to(device)
    tokenized_full = full_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_full = tokenized_full.map(convert_to_tensors, batched=True)
    tokenized_query = query_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_query = tokenized_query.map(convert_to_tensors, batched=True)
    full_split = tokenized_full.train_test_split(test_size=0.1)
    query_split = tokenized_query.train_test_split(test_size=0.1)
    train_full = full_split["train"]
    eval_full = full_split["test"]
    train_query = query_split["train"]
    eval_query = query_split["test"]
    train_loader_full = get_dataloader(train_full, batch_size=4, device=device)
    eval_loader_full = get_dataloader(eval_full, batch_size=4, device=device)
    train_loader_query = get_dataloader(train_query, batch_size=4, device=device)
    eval_loader_query = get_dataloader(eval_query, batch_size=4, device=device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000)
    num_epochs_full = 5
    num_epochs_query = 1
    for epoch in range(num_epochs_full):
        loss = train_epoch(model, classifier, train_loader_full, optimizer, scheduler, device)
        acc = evaluate_epoch(model, classifier, eval_loader_full, device)
        torch.save({
            'model_state': model.state_dict(),
            'classifier_state': classifier.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, f"ckpts/checkpoint_full_epoch{epoch+1}.pth")
        print(f"Full Context Phase Epoch {epoch+1}: Loss {loss:.4f}, Accuracy {acc*100:.2f}%")
    for epoch in range(num_epochs_query):
        loss = train_epoch(model, classifier, train_loader_query, optimizer, scheduler, device)
        acc = evaluate_epoch(model, classifier, eval_loader_query, device)
        torch.save({
            'model_state': model.state_dict(),
            'classifier_state': classifier.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, f"ckpts/checkpoint_query_epoch{epoch+1}.pth")
        print(f"Query Only Phase Epoch {epoch+1}: Loss {loss:.4f}, Accuracy {acc*100:.2f}%")
    final_acc = evaluate_epoch(model, classifier, eval_loader_query, device)
    model.save_pretrained("ckpts/finetuned_llama_lora")
    torch.save(classifier.state_dict(), "ckpts/finetuned_mlp.pth")
    print(f"Final Query Only Phase Accuracy: {final_acc*100:.2f}%")
    
if __name__ == "__main__":
    main()
