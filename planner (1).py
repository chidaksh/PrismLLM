import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup,AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
import time
import datetime
from sklearn.model_selection import train_test_split
import sys
from peft import get_peft_model, LoraConfig, TaskType

print("hi")
# Create log file with timestamp for uniqueness
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"planner_training_{timestamp}.log"

# Redirect stdout and stderr to the same log file
log_file = open(log_filename, 'w')
sys.stdout = log_file
sys.stderr = log_file

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log_filename'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Constants
MODEL_NAME = "google-bert/bert-large-uncased"
EMBEDDING_DIM = 256
PROJECTION_DIM = 128
CLASSIFIER_DIM = 64
MARGIN = 0.3
DROPOUT_RATE = 0.1
MAX_SEQ_LENGTH = 1024
print("hellow")
# Dataset for triplet contrastive learning
class TripletContrastiveDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        anchor = self.tokenizer(sample["question"] + " " + sample["ground_truth"], return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
        positive = self.tokenizer(sample["positive_response"], return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
        negative = self.tokenizer(sample["negative_response"], return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
        return {
            "input_ids": anchor["input_ids"].squeeze(0),
            "attention_mask": anchor["attention_mask"].squeeze(0)
        }, {
            "input_ids": positive["input_ids"].squeeze(0),
            "attention_mask": positive["attention_mask"].squeeze(0)
        }, {
            "input_ids": negative["input_ids"].squeeze(0),
            "attention_mask": negative["attention_mask"].squeeze(0)
        }

def collate_triplet(batch):
    anchors, positives, negatives = zip(*batch)

    def stack(tensors):
        return {k: torch.stack([x[k] for x in tensors]) for k in tensors[0]}

    return {
        "anchor": stack(anchors),
        "positive": stack(positives),
        "negative": stack(negatives)
    }

# Dataset for classifier fine-tuning
class ClassificationDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoded = self.tokenizer(sample["question"] + " " + sample["ground_truth"], padding='max_length', truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.long)
        }

# Model for triplet contrastive learning
class TripletContrastiveLLM(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()

        m = AutoModel.from_pretrained(model_name)
        def show_trainable_params(model):
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {trainable:,} / {total:,} ({(trainable / total) * 100:.2f}%)")
        
        show_trainable_params(m)

        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],  # check model to adapt
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,  # or TaskType.FEATURE_EXTRACTION if using triplet
        )
        self.backbone = get_peft_model(m, lora_config)
        show_trainable_params(self.backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False
        show_trainable_params(self.backbone)
        self.projection = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, EMBEDDING_DIM),
            nn.GELU(),
            nn.LayerNorm(EMBEDDING_DIM),
            nn.Linear(EMBEDDING_DIM, PROJECTION_DIM),
            nn.LayerNorm(PROJECTION_DIM)
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def embed(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden.shape).float()
        mean_pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
        return self.projection(mean_pooled)

    def forward(self, anchor, positive, negative):
        anchor_emb = F.normalize(self.embed(**anchor), dim=1)
        positive_emb = F.normalize(self.embed(**positive), dim=1)
        negative_emb = F.normalize(self.embed(**negative), dim=1)
        pos_sim = F.cosine_similarity(anchor_emb, positive_emb) / self.temperature
        neg_sim = F.cosine_similarity(anchor_emb, negative_emb) / self.temperature
        return F.softplus(neg_sim - pos_sim + MARGIN).mean()

# Model for classification fine-tuning
class PlannerClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes: int = 2):
        super().__init__()
        self.encoder = pretrained_model.backbone
        self.projection = pretrained_model.projection
        self.classifier = nn.Sequential(
            nn.Linear(PROJECTION_DIM, CLASSIFIER_DIM),
            nn.SiLU(),
            nn.LayerNorm(CLASSIFIER_DIM),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(CLASSIFIER_DIM, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden.shape).float()
        mean_pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
        projected = self.projection(mean_pooled)
        return self.classifier(projected)

# Trainer
class LLMPlannerTrainer:
    def __init__(self, model, device, optimizer, scheduler=None, use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        self.use_amp = use_amp
        
    def evaluate_contrastive(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                anchor = {k: v.to(self.device) for k, v in batch["anchor"].items()}
                positive = {k: v.to(self.device) for k, v in batch["positive"].items()}
                negative = {k: v.to(self.device) for k, v in batch["negative"].items()}
                loss = self.model(anchor, positive, negative)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate_classifier(self, dataloader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total
        
    def train_contrastive_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        print("\n🚀 Starting Contrastive Training Epoch", epoch)
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Contrastive Epoch {epoch}", leave=True, ncols=100, dynamic_ncols=False, file=sys.__stdout__)
        cnt = 0
        for batch in pbar:
            anchor = {k: v.to(self.device) for k, v in batch["anchor"].items()}
            positive = {k: v.to(self.device) for k, v in batch["positive"].items()}
            negative = {k: v.to(self.device) for k, v in batch["negative"].items()}

            loss = self.model(anchor, positive, negative)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"✅ Finished Contrastive Epoch {epoch} | Avg Loss: {total_loss / len(dataloader):.4f}")
        return total_loss / len(dataloader)

    def train_classifier_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, float]:
        print("\n🎯 Starting Classifier Fine-Tuning Epoch", epoch)
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(dataloader, desc=f"Classifier Epoch {epoch}", leave=True, ncols=100, dynamic_ncols=False, file=sys.__stdout__)
        cnt =0 
        for batch in pbar:
            #print(batch)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler:
                self.scheduler.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        accuracy = correct / total
        print(f"✅ Finished Classifier Epoch {epoch} | Loss: {total_loss / len(dataloader):.4f} | Accuracy: {accuracy:.2%}")
        return total_loss / len(dataloader), accuracy

# Entry function
if __name__ == "__main__":
    print("🔥 Initializing Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with open("/users/a/k/akkineni/LLMs/hack3/dataset/contrastivefin.jsonl") as f:
        contrastive_samples = [json.loads(line) for line in f]
    train_c, temp_c = train_test_split(contrastive_samples, test_size=0.2, random_state=42)
    val_c, test_c = train_test_split(temp_c, test_size=0.5, random_state=42)

    train_contrastive_ds = TripletContrastiveDataset(train_c, tokenizer)
    val_contrastive_ds = TripletContrastiveDataset(val_c, tokenizer)
    test_contrastive_ds = TripletContrastiveDataset(test_c, tokenizer)
    
    train_c_loader = DataLoader(train_contrastive_ds, batch_size=4, shuffle=True, collate_fn=collate_triplet)
    val_c_loader = DataLoader(val_contrastive_ds, batch_size=4, shuffle=False, collate_fn=collate_triplet)
    test_c_loader = DataLoader(test_contrastive_ds, batch_size=4, shuffle=False, collate_fn=collate_triplet)

    model = TripletContrastiveLLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)

    trainer = LLMPlannerTrainer(model, device, optimizer, scheduler)
    for epoch in range(1, 11):  # 10 contrastive epochs
        
        train_loss = trainer.train_contrastive_epoch(train_c_loader, epoch)
        val_loss = trainer.evaluate_contrastive(val_c_loader)
        logger.info(f"[Contrastive] Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
        if epoch % 5 == 0:
            test_loss = trainer.evaluate_contrastive(test_c_loader)
            logger.info(f"[Contrastive] TEST Loss after Epoch {epoch}: {test_loss:.4f}")
            
        
    final_test_loss = trainer.evaluate_contrastive(test_c_loader)
    logger.info(f"✅ Final Contrastive Test Loss: {final_test_loss:.4f}")
    print("\n📚 Contrastive training complete. Proceeding to classifier fine-tuning...")

    with open("/users/a/k/akkineni/LLMs/hack3/dataset/classificationfin.jsonl") as f:
        classification_samples = [json.loads(line) for line in f]
    
    train_cls, temp_cls = train_test_split(classification_samples, test_size=0.2, random_state=42)
    val_cls, test_cls = train_test_split(temp_cls, test_size=0.5, random_state=42)
    
    train_cls_ds = ClassificationDataset(train_cls, tokenizer)
    val_cls_ds = ClassificationDataset(val_cls, tokenizer)
    test_cls_ds = ClassificationDataset(test_cls, tokenizer)
    
    train_cls_loader = DataLoader(train_cls_ds, batch_size=4, shuffle=True)
    val_cls_loader = DataLoader(val_cls_ds, batch_size=4, shuffle=False)
    test_cls_loader = DataLoader(test_cls_ds, batch_size=4, shuffle=False)
    
    
    classifier_model = PlannerClassifier(model)
    classifier_optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=1e-3)
    classifier_scheduler = get_linear_schedule_with_warmup(classifier_optimizer, num_warmup_steps=10, num_training_steps=100)

    trainer = LLMPlannerTrainer(classifier_model, device, classifier_optimizer, classifier_scheduler)
    
    for epoch in range(1, 50):  # 10 classifier epochs
        train_loss, train_acc = trainer.train_classifier_epoch(train_cls_loader, epoch)
        val_acc = trainer.evaluate_classifier(val_cls_loader)
        logger.info(f"[Classifier] Epoch {epoch}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Loss = {train_loss:.4f}")
    
        if epoch % 5 == 0:
            test_acc = trainer.evaluate_classifier(test_cls_loader)
            logger.info(f"[Classifier] TEST Accuracy after Epoch {epoch}: {test_acc:.4f}")
            torch.save(classifier_model.state_dict(), f"final_classifier_model{epoch}.pt")
            
    final_test_acc = trainer.evaluate_classifier(test_cls_loader)
    logger.info(f"✅ Final Classifier Test Accuracy: {final_test_acc:.4f}")
    
    
    print("\n🎯 FINAL TEST RESULTS:")
    print(f"Contrastive Test Loss: {final_test_loss:.4f}")
    print(f"Classifier Test Accuracy: {final_test_acc:.2%}")
    logger.info(f"📊 Combined Final Test Summary | Contrastive Loss: {final_test_loss:.4f} | Classification Accuracy: {final_test_acc:.2%}")

    torch.save(classifier_model.state_dict(), "final_classifier_model.pt")
    
    print("\n🏁 Training pipeline completed.")
    logger.info("🎉 Training complete. Model saved.")

    # Inference
    classifier_model.eval()
    sample = val_cls[0]
    encoding = tokenizer(sample["question"] + " " + sample["ground_truth"], return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH).to(device)
    with torch.no_grad():
        logits = classifier_model(encoding["input_ids"], encoding["attention_mask"])
        prediction = torch.argmax(logits, dim=1).item()

    print(f"\n🔍 Inference Sample:")
    print(f"Question: {sample['question']}")
    print(f"Ground Truth: {sample['ground_truth']}")
    print(f"Predicted Class: {prediction}, Actual Label: {sample['label']}")
    