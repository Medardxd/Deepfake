"""
Compare Base Model vs Fine-Tuned Model Performance
Uses already-extracted frames from celeb_df_processed/val/
"""

import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
from peft import PeftModel
from tqdm import tqdm
import json


class ModelComparison:
    """Compare base model vs fine-tuned model on validation frames"""

    def __init__(self):
        self.base_model_name = "prithivMLmods/deepfake-detector-model-v1"
        self.finetuned_path = "models/celeb_df_finetuned"

        print("="*60)
        print("Model Comparison: Base vs Fine-Tuned")
        print("Using validation frames from celeb_df_processed/val/")
        print("="*60)

        # Load both models
        self.base_pipeline = self._load_base_model()
        self.finetuned_pipeline = self._load_finetuned_model()

    def _load_base_model(self):
        """Load original base model"""
        print("\n1. Loading BASE model (original)...")
        pipeline_model = pipeline(
            'image-classification',
            model=self.base_model_name,
            device=-1  # CPU
        )
        print("   ✅ Base model loaded")
        return pipeline_model

    def _load_finetuned_model(self):
        """Load fine-tuned LoRA model"""
        print("\n2. Loading FINE-TUNED model (LoRA adapted)...")

        # Load base model
        base_model = AutoModelForImageClassification.from_pretrained(
            self.base_model_name,
            num_labels=2,
            ignore_mismatched_sizes=True
        )

        # Set labels
        base_model.config.id2label = {0: "fake", 1: "real"}
        base_model.config.label2id = {"fake": 0, "real": 1}

        # Load and merge LoRA
        peft_model = PeftModel.from_pretrained(base_model, str(self.finetuned_path))
        merged_model = peft_model.merge_and_unload()

        # Ensure labels
        merged_model.config.id2label = {0: "fake", 1: "real"}
        merged_model.config.label2id = {"fake": 0, "real": 1}

        # Create pipeline
        processor = AutoImageProcessor.from_pretrained(str(self.finetuned_path))
        pipeline_model = pipeline(
            'image-classification',
            model=merged_model,
            image_processor=processor,
            device=-1  # CPU
        )

        print("   ✅ Fine-tuned model loaded")
        return pipeline_model

    def analyze_frame(self, image_path):
        """Run both models on the same frame"""
        img = Image.open(image_path).convert('RGB')

        # Base model prediction
        base_preds = self.base_pipeline(img)
        base_result = self._parse_predictions(base_preds)

        # Fine-tuned model prediction
        finetuned_preds = self.finetuned_pipeline(img)
        finetuned_result = self._parse_predictions(finetuned_preds)

        return {
            'base': base_result,
            'finetuned': finetuned_result
        }

    def _parse_predictions(self, predictions):
        """Parse model predictions into fake/real scores"""
        fake_score = 0.0
        real_score = 0.0

        for pred in predictions:
            label = pred['label'].lower()
            score = pred['score']

            if 'fake' in label or 'label_0' in label or label == '0':
                fake_score = score
            elif 'real' in label or 'label_1' in label or label == '1':
                real_score = score

        is_fake = fake_score > 0.7  # Using same threshold as detector

        return {
            'prediction': 'FAKE' if is_fake else 'REAL',
            'fake_score': fake_score,
            'real_score': real_score,
            'confidence': fake_score if is_fake else real_score
        }


def main():
    """Run comprehensive comparison on validation frames"""
    comparator = ModelComparison()

    # Load validation frames
    VAL_DIR = Path("celeb_df_processed/val")

    print(f"\n📋 Loading validation frames from {VAL_DIR}...")

    # Collect all frames
    real_frames = list((VAL_DIR / "real").glob("*.jpg"))
    fake_frames = list((VAL_DIR / "fake").glob("*.jpg"))

    print(f"   Real frames: {len(real_frames)}")
    print(f"   Fake frames: {len(fake_frames)}")
    print(f"   Total: {len(real_frames) + len(fake_frames)} frames")

    # Statistics
    base_stats = {'real_correct': 0, 'real_total': 0, 'fake_correct': 0, 'fake_total': 0}
    ft_stats = {'real_correct': 0, 'real_total': 0, 'fake_correct': 0, 'fake_total': 0}

    detailed_results = []

    # Test on real frames
    print(f"\n🔬 Testing on REAL frames...")
    for frame_path in tqdm(real_frames, desc="Real frames"):
        result = comparator.analyze_frame(str(frame_path))

        base_stats['real_total'] += 1
        ft_stats['real_total'] += 1

        if result['base']['prediction'] == 'REAL':
            base_stats['real_correct'] += 1
        if result['finetuned']['prediction'] == 'REAL':
            ft_stats['real_correct'] += 1

        detailed_results.append({
            'frame': frame_path.name,
            'ground_truth': 'real',
            'base': result['base'],
            'finetuned': result['finetuned']
        })

    # Test on fake frames
    print(f"\n🔬 Testing on FAKE frames...")
    for frame_path in tqdm(fake_frames, desc="Fake frames"):
        result = comparator.analyze_frame(str(frame_path))

        base_stats['fake_total'] += 1
        ft_stats['fake_total'] += 1

        if result['base']['prediction'] == 'FAKE':
            base_stats['fake_correct'] += 1
        if result['finetuned']['prediction'] == 'FAKE':
            ft_stats['fake_correct'] += 1

        detailed_results.append({
            'frame': frame_path.name,
            'ground_truth': 'fake',
            'base': result['base'],
            'finetuned': result['finetuned']
        })

    # Calculate metrics
    total_frames = base_stats['real_total'] + base_stats['fake_total']

    base_real_acc = base_stats['real_correct'] / base_stats['real_total'] if base_stats['real_total'] > 0 else 0
    base_fake_acc = base_stats['fake_correct'] / base_stats['fake_total'] if base_stats['fake_total'] > 0 else 0
    base_overall = (base_stats['real_correct'] + base_stats['fake_correct']) / total_frames

    ft_real_acc = ft_stats['real_correct'] / ft_stats['real_total'] if ft_stats['real_total'] > 0 else 0
    ft_fake_acc = ft_stats['fake_correct'] / ft_stats['fake_total'] if ft_stats['fake_total'] > 0 else 0
    ft_overall = (ft_stats['real_correct'] + ft_stats['fake_correct']) / total_frames

    # Print results
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)
    print(f"Total frames tested: {total_frames}")
    print(f"  Real: {base_stats['real_total']}")
    print(f"  Fake: {base_stats['fake_total']}")
    print("-"*60)

    print(f"\nBASE MODEL:")
    print(f"  Real frames: {base_stats['real_correct']}/{base_stats['real_total']} correct ({base_real_acc:.1%})")
    print(f"  Fake frames: {base_stats['fake_correct']}/{base_stats['fake_total']} correct ({base_fake_acc:.1%})")
    print(f"  Overall accuracy: {base_overall:.1%}")

    print(f"\nFINE-TUNED MODEL:")
    print(f"  Real frames: {ft_stats['real_correct']}/{ft_stats['real_total']} correct ({ft_real_acc:.1%})")
    print(f"  Fake frames: {ft_stats['fake_correct']}/{ft_stats['fake_total']} correct ({ft_fake_acc:.1%})")
    print(f"  Overall accuracy: {ft_overall:.1%}")

    print(f"\nIMPROVEMENT:")
    print(f"  Real frame accuracy: {base_real_acc:.1%} → {ft_real_acc:.1%} (+{(ft_real_acc - base_real_acc):.1%})")
    print(f"  Fake frame accuracy: {base_fake_acc:.1%} → {ft_fake_acc:.1%} (+{(ft_fake_acc - base_fake_acc):.1%})")
    print(f"  Overall accuracy: {base_overall:.1%} → {ft_overall:.1%} (+{(ft_overall - base_overall):.1%})")
    print("="*60)

    # Save results
    output_file = "model_comparison_results.json"
    results_data = {
        'summary': {
            'total_frames': total_frames,
            'real_frames': base_stats['real_total'],
            'fake_frames': base_stats['fake_total'],
            'base_model': {
                'real_accuracy': float(base_real_acc),
                'fake_accuracy': float(base_fake_acc),
                'overall_accuracy': float(base_overall),
                'real_correct': base_stats['real_correct'],
                'fake_correct': base_stats['fake_correct']
            },
            'finetuned_model': {
                'real_accuracy': float(ft_real_acc),
                'fake_accuracy': float(ft_fake_acc),
                'overall_accuracy': float(ft_overall),
                'real_correct': ft_stats['real_correct'],
                'fake_correct': ft_stats['fake_correct']
            },
            'improvement': {
                'real_accuracy': float(ft_real_acc - base_real_acc),
                'fake_accuracy': float(ft_fake_acc - base_fake_acc),
                'overall_accuracy': float(ft_overall - base_overall)
            }
        },
        'detailed_results': detailed_results
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n✅ Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
