# -*- coding: utf-8 -*-
"""
PyTorch MuRIL-BERT Fine-Tuning Pipeline for Multilingual Emotion Classification.
Trains on 60/25/15 English, Hindi, and Hinglish dataset using BCEWithLogitsLoss.
Saves fine-tuned weights directly to backend/muril_emotion_model.pth.
"""

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
from mindmate_integration import MuRILEmotionClassifier, EMOTION_LABELS, MURIL_BASE

class MultilingualEmotionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        labels = torch.tensor(item["labels"], dtype=torch.float32)
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels
        }


def train_model(epochs=3, batch_size=16, lr=2e-5):
    print("=" * 65)
    print("STARTING MURIL-BERT MULTILINGUAL FINE-TUNING")
    print("=" * 65)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device selected for training: {device}")
    
    dataset_path = os.path.join(os.path.dirname(__file__), "multilingual_emotion_dataset.json")
    model_save_path = os.path.join(os.path.dirname(__file__), "muril_emotion_model.pth")
    local_tok_dir = os.path.join(os.path.dirname(__file__), "local_tokenizer")
    
    if os.path.exists(local_tok_dir):
        tokenizer = AutoTokenizer.from_pretrained(local_tok_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MURIL_BASE)
        
    dataset = MultilingualEmotionDataset(dataset_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = MuRILEmotionClassifier().to(device)
    
    # Freeze MuRIL backbone to save RAM and prevent CPU OOM errors
    for param in model.muril.parameters():
        param.requires_grad = False
        
    print("[Train] Frozen MuRIL backbone parameters. Training classification head only.")
    
    # Load existing state dict if present to perform domain adaptation
    if os.path.exists(model_save_path):
        try:
            ckpt = torch.load(model_save_path, map_location=device)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state, strict=False)
            print("[Train] Loaded existing weights for incremental domain adaptation.")
        except Exception as e:
            print(f"[Train] Initializing fresh classifier head: {e}")
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch [{epoch}/{epochs}] Complete — Average BCE Loss: {avg_loss:.4f}")
        
    # Save fine-tuned checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epochs,
        "labels": EMOTION_LABELS
    }, model_save_path)
    
    print("\n" + "=" * 65)
    print("SUCCESS: FINE-TUNING COMPLETE!")
    print(f"📁 Updated weights saved to: {model_save_path}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    train_model(epochs=3, batch_size=16)
