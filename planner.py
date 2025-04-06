import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
import time
import datetime

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('planner_training.log'),
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
MAX_SEQ_LENGTH = 512

# Dataset for triplet contrastive learning
class TripletContrastiveDataset(Dataset):
    def __init__(self, file_path: str, tokenizer):
        with open(file_path) as f:
            self.samples = [json.loads(line) for line in f]
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
    def __init__(self, file_path: str, tokenizer):
        with open(file_path) as f:
            self.samples = [json.loads(line) for line in f]
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
        self.backbone = AutoModel.from_pretrained(model_name)
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

    def train_contrastive_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        print("\n🚀 Starting Contrastive Training Epoch", epoch)
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Contrastive Epoch {epoch}", leave=False)

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
        pbar = tqdm(dataloader, desc=f"Classifier Epoch {epoch}", leave=False)

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

    contrastive_dataset = TripletContrastiveDataset("/users/a/k/akkineni/LLMs/hack3/data/contrastive.jsonl", tokenizer)
    contrastive_loader = DataLoader(contrastive_dataset, batch_size=1, shuffle=True, collate_fn=collate_triplet)

    model = TripletContrastiveLLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)

    trainer = LLMPlannerTrainer(model, device, optimizer, scheduler)
    trainer.train_contrastive_epoch(contrastive_loader, epoch=10)

    print("\n📚 Contrastive training complete. Proceeding to classifier fine-tuning...")

    classifier_dataset = ClassificationDataset("/users/a/k/akkineni/LLMs/hack3/data/classification.jsonl", tokenizer)
    classifier_loader = DataLoader(classifier_dataset, batch_size=1, shuffle=True)

    classifier_model = PlannerClassifier(model)
    classifier_optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=1e-5)
    classifier_scheduler = get_linear_schedule_with_warmup(classifier_optimizer, num_warmup_steps=10, num_training_steps=100)

    trainer = LLMPlannerTrainer(classifier_model, device, classifier_optimizer, classifier_scheduler)
    trainer.train_classifier_epoch(classifier_loader, epoch=10)
    print("\n🏁 Training pipeline completed.")