"""
Fine-tune deepfake detector on Celeb-DF-v2 dataset using LoRA
Uses prithivMLmods/deepfake-detector-model-v1 as base model
LoRA enables efficient fine-tuning with lower memory and faster training
"""

import os
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

# Configuration
BASE_MODEL = "prithivMLmods/deepfake-detector-model-v1"
DATA_DIR = "/mnt/e/szakdoga/Deepfake/celeb_df_processed"
OUTPUT_DIR = "/mnt/e/szakdoga/Deepfake/models/celeb_df_finetuned"
LOGS_DIR = "/mnt/e/szakdoga/Deepfake/training_logs"

# Training hyperparameters
BATCH_SIZE = 16  # Adjust based on GPU memory
LEARNING_RATE = 2e-5  # Lower learning rate for fine-tuning
NUM_EPOCHS = 5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIMIT = 3  # Keep only best 3 checkpoints

class DeepfakeDataset(Dataset):
    """Custom dataset for deepfake detection"""

    def __init__(self, data_dir, split='train', processor=None):
        """
        Args:
            data_dir: Root directory containing train/val splits
            split: 'train' or 'val'
            processor: HuggingFace image processor
        """
        self.data_dir = Path(data_dir) / split
        self.processor = processor
        self.images = []
        self.labels = []

        # Load real images (label=1)
        real_dir = self.data_dir / 'real'
        for img_path in real_dir.glob('*.jpg'):
            self.images.append(str(img_path))
            self.labels.append(1)  # Real = 1

        # Load fake images (label=0)
        fake_dir = self.data_dir / 'fake'
        for img_path in fake_dir.glob('*.jpg'):
            self.images.append(str(img_path))
            self.labels.append(0)  # Fake = 0

        print(f"Loaded {split} set: {len(self.images)} images")
        print(f"  Real: {sum(self.labels)} images")
        print(f"  Fake: {len(self.labels) - sum(self.labels)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        # Process image
        if self.processor:
            encoding = self.processor(image, return_tensors='pt')
            # Remove batch dimension
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        else:
            encoding = {'pixel_values': image}

        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        return encoding

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print("="*60)
    print("Fine-Tuning Deepfake Detector on Celeb-DF-v2")
    print("="*60)

    # Create output directories
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n1. Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load processor and model
    print(f"\n2. Loading base model: {BASE_MODEL}")
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,  # Binary classification: real vs fake
        ignore_mismatched_sizes=True  # Allow different number of labels
    )

    # Update label mappings
    model.config.id2label = {0: "fake", 1: "real"}
    model.config.label2id = {"fake": 0, "real": 1}

    print(f"   Base model loaded successfully")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Total parameters: {total_params:.2f}M")

    # Configure LoRA
    print(f"\n3. Configuring LoRA adapters")
    lora_config = LoraConfig(
        r=16,  # Rank of LoRA matrices (higher = more capacity, 8-16 typical)
        lora_alpha=32,  # Scaling factor (typically 2*r)
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention Q and V projections
        lora_dropout=0.1,  # Dropout for LoRA layers
        bias="none",  # Don't train bias terms
    )

    # Wrap model with LoRA
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    print(f"   LoRA configuration:")
    print(f"   - Rank: {lora_config.r}")
    print(f"   - Alpha: {lora_config.lora_alpha}")
    print(f"   - Target modules: {lora_config.target_modules}")
    print(f"   - Trainable parameters: {trainable_params:.2f}M ({trainable_params/total_params*100:.1f}% of total)")
    print(f"   - Memory savings: ~{(1 - trainable_params/total_params)*100:.0f}%")

    # Load datasets
    print(f"\n4. Loading datasets from {DATA_DIR}")
    train_dataset = DeepfakeDataset(DATA_DIR, split='train', processor=processor)
    val_dataset = DeepfakeDataset(DATA_DIR, split='val', processor=processor)

    # Training arguments
    print(f"\n5. Setting up training configuration")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOGS_DIR,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",  # Disable wandb/tensorboard
        fp16=device == "cuda",  # Use mixed precision if GPU available
    )

    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Mixed precision (FP16): {training_args.fp16}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train
    print(f"\n6. Starting LoRA training...")
    print("="*60)
    train_result = trainer.train()

    # Save final model (LoRA adapters + base model)
    print("\n7. Saving LoRA adapters and model...")
    model.save_pretrained(OUTPUT_DIR)  # Saves LoRA adapters
    processor.save_pretrained(OUTPUT_DIR)
    print(f"   LoRA adapters saved to: {OUTPUT_DIR}")

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Final evaluation
    print("\n8. Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Save configuration
    config = {
        "base_model": BASE_MODEL,
        "fine_tuning_method": "LoRA",
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "dataset": "Celeb-DF-v2",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "trainable_params_million": trainable_params,
        "final_metrics": eval_metrics
    }

    with open(Path(OUTPUT_DIR) / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nFinal Metrics:")
    print(f"  Accuracy: {eval_metrics['eval_accuracy']:.4f}")
    print(f"  F1 Score: {eval_metrics['eval_f1']:.4f}")
    print(f"  Precision: {eval_metrics['eval_precision']:.4f}")
    print(f"  Recall: {eval_metrics['eval_recall']:.4f}")
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
