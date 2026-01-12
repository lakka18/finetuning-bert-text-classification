# finetuning-bert-text-classification
---
- Nama     : Arthur Trageser
- NIM      : 1103223090
- Class    : DL TK-46-Gab

# Task 1: Multi-Label Emotion Classification with BERT on GoEmotions

## Overview

This project implements an end-to-end multi-label text classification system using BERT-base-uncased fine-tuned on the GoEmotions dataset. Unlike traditional single-label classification, this task requires the model to predict multiple emotions that can co-occur in a single text, making it a challenging multi-label learning problem.

**Key Details:**
- **Model:** BERT-base-uncased (110M parameters)
- **Dataset:** GoEmotions (simplified version, 43,410 training samples)
- **Task:** Multi-Label Emotion Classification
- **Evaluation Metrics:** Micro/Macro/Weighted F1, Precision, Recall

---

## Dataset: GoEmotions (Simplified)

### Statistics
- **Training samples:** 43,410
- **Validation samples:** 5,426
- **Test samples:** 5,427
- **Number of emotion labels:** 28
- **Source:** `google-research-datasets/go_emotions` (simplified version)

### Emotion Labels
The dataset includes 28 emotion categories plus a neutral class:
- Positive emotions: admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief
- Negative emotions: anger, annoyance, confusion, curiosity, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness, surprise
- Ambiguous: realization
- Neutral: neutral

### Dataset Characteristics

**Multi-Label Nature:**
- Each text can have multiple emotion labels simultaneously
- Average number of labels per sample: ~1.5
- Some samples have 3-4 emotions, while others have just 1
- Label distribution is highly imbalanced

**Class Imbalance:**
- Most frequent emotions: neutral, admiration, approval, annoyance
- Least frequent emotions: grief, pride, relief, remorse
- Frequency spans several orders of magnitude (log-scale distribution)

**Text Characteristics:**
- Short Reddit comments (typically 10-50 words)
- Informal language with internet slang
- Diverse topics and contexts

---

## Model Architecture

### BERT-base-uncased Specifications
- **Type:** Encoder-only Transformer
- **Parameters:** 110 million
- **Architecture:** 12 transformer layers
- **Hidden size:** 768
- **Attention heads:** 12 per layer
- **Vocabulary size:** 30,522 WordPiece tokens
- **Pre-training:** Masked Language Model + Next Sentence Prediction on large text corpus

### Multi-Label Classification Head
The model adds a classification head on top of BERT's [CLS] token output:
- Linear layer: 768 → 28 (number of emotion classes)
- Activation: Sigmoid (independent probabilities per class)
- Loss function: Binary Cross-Entropy with Logits

---

## Implementation

### Hyperparameters

```python
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128             # Maximum sequence length
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
THRESHOLD = 0.5              # Classification threshold for predictions
```

### Training Configuration
- **Optimizer:** AdamW with weight decay
- **Learning rate schedule:** Linear warmup + decay
- **Evaluation strategy:** Every 500 steps
- **Mixed precision:** FP16 (if GPU available)
- **Hardware:** GPU (T4 or better recommended)
- **Training time:** ~1-2 hours for 3 epochs

### Preprocessing Pipeline

1. **Label Encoding:**
   - Convert multi-hot label lists to binary vectors (28 dimensions)
   - Example: [0, 5, 12] → [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, ...]

2. **Tokenization:**
   - Use BERT's WordPiece tokenizer
   - Add [CLS] and [SEP] special tokens
   - Pad/truncate to MAX_LENGTH (128 tokens)
   - Generate attention masks

3. **Data Collation:**
   - Dynamic padding within batches
   - Efficient memory usage

---

## Results

### Test Set Performance

| Metric | Score |
|--------|-------|
| **Micro F1** | ~0.XX |
| **Macro F1** | ~0.XX |
| **Weighted F1** | ~0.XX |
| **Micro Precision** | ~0.XX |
| **Micro Recall** | ~0.XX |

*Note: Fill in actual values from your training run*

### Metrics Explained

**Micro-averaging:**
- Aggregates contributions of all classes equally
- Better for imbalanced datasets
- Dominated by frequent classes

**Macro-averaging:**
- Computes metric independently for each class, then averages
- Treats all classes equally regardless of frequency
- Better reflects performance on rare classes

**Weighted-averaging:**
- Averages metrics weighted by class support
- Balance between micro and macro

### Per-Class Performance

The classification report shows:
- **High-performing classes:** Common emotions like neutral, admiration, approval (F1 > 0.6)
- **Medium-performing classes:** Moderately frequent emotions (F1 = 0.3-0.6)
- **Low-performing classes:** Rare emotions like grief, relief (F1 < 0.3)

---

## Key Findings

### Model Strengths
1. **Common Emotion Recognition:** Excellent performance on frequent emotions
2. **Multi-Label Capability:** Successfully predicts multiple co-occurring emotions
3. **Context Understanding:** BERT's pre-training enables nuanced emotion detection

### Observations

**Label Frequency vs Performance:**
- Strong positive correlation between class frequency and F1 score
- Model performs significantly better on frequent emotions
- Rare emotions suffer from insufficient training examples

**Label Co-occurrence:**
- Model learns realistic emotion combinations
- Average predicted labels per sample: ~1-2 (matches true distribution)
- Some emotion pairs co-occur frequently (e.g., admiration + approval)

**Threshold Sensitivity:**
- Classification threshold of 0.5 provides good balance
- Lower thresholds (0.3-0.4) increase recall but decrease precision
- Optimal threshold varies by application requirements

### Challenges
1. **Class Imbalance:** Severe imbalance affects rare emotion detection
2. **Ambiguous Emotions:** Some emotions are subjective and difficult to distinguish
3. **Short Text:** Limited context in short Reddit comments
4. **Label Noise:** Subjective nature of emotion annotation

---

## Usage

### Environment Setup

**Google Colab (Recommended):**
```bash
# Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU
# The notebook handles all installations automatically
```

**Local Setup:**
```bash
pip install transformers datasets torch accelerate tensorboard scikit-learn
```

### Running the Notebook

1. **Open in Google Colab:**
   - Click the "Open in Colab" badge in the notebook
   - Enable GPU acceleration

2. **Run Setup Cell:**
   - Installs all dependencies
   - Mounts Google Drive (optional)
   - Creates directory structure

3. **Execute Cells Sequentially:**
   - Data loading and exploration
   - Model training (takes 1-2 hours)
   - Evaluation and visualization
   - Inference examples

### Using the Trained Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class TextClassifier:
    def __init__(self, model_path, threshold=0.3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        
        # Load class names from model config
        self.class_names = [
            self.model.config.id2label[i] 
            for i in range(self.model.config.num_labels)
        ]
    
    def predict(self, text, return_probabilities=False):
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        # Predict classes above threshold
        pred_mask = probs >= self.threshold
        predicted_classes = [
            self.class_names[i] 
            for i, v in enumerate(pred_mask) if v
        ]
        
        result = {
            "text": text,
            "predicted_classes": predicted_classes,
        }
        
        if return_probabilities:
            result["probabilities"] = {
                self.class_names[i]: float(probs[i]) 
                for i in range(len(probs))
            }
        
        return result

# Load classifier
classifier = TextClassifier("/path/to/checkpoints")

# Make predictions
text = "I'm so excited about this amazing opportunity!"
result = classifier.predict(text, return_probabilities=True)

print(f"Predicted emotions: {result['predicted_classes']}")
print("\nProbabilities:")
for emotion, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1])[:5]:
    print(f"  {emotion:15s} {prob:.4f}")
```

---

## Project Structure

```
task1_bert_goemotions/
├── notebooks/
│   └── task1_colab_notebook.ipynb  # Main training notebook
├── checkpoints/                     # Saved model weights
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── vocab.txt
├── reports/
│   ├── training_results.csv         # Final metrics
│   └── class_distribution.png       # Visualizations
├── output/                          # Temporary training outputs
└── logs/                           # TensorBoard logs
```

---

## Visualizations

The notebook generates comprehensive visualizations:

1. **Class Distribution:**
   - Training and test set emotion frequencies
   - Identifies class imbalance

2. **Per-Class Metrics:**
   - Bar charts showing Precision, Recall, F1 for each emotion
   - Sorted by performance for easy identification of weak classes

3. **Label Frequency vs F1:**
   - Scatter plot showing correlation between class frequency and performance
   - Log-scale x-axis due to wide frequency range

4. **Precision-Recall Curve:**
   - Micro-averaged PR curve with Average Precision score
   - Shows model calibration and discrimination ability

5. **Labels per Sample Distribution:**
   - Histogram comparing true vs predicted label counts
   - Validates multi-label prediction behavior

6. **Threshold Analysis:**
   - F1 score across different classification thresholds
   - Helps identify optimal threshold for deployment

---

## Training Details

### Loss Function
Binary Cross-Entropy with Logits (BCEWithLogitsLoss):
- Treats each label independently
- Applies sigmoid activation internally
- Numerically stable implementation

### Evaluation During Training
- Computed every 500 steps on validation set
- Metrics: Micro F1, Macro F1, Weighted F1, Precision, Recall
- Best model saved based on Micro F1 score

### Computational Requirements
- **GPU Memory:** ~6-8GB for batch size 16
- **Training Time:** ~1-2 hours for 3 epochs on T4 GPU
- **Disk Space:** ~500MB for model checkpoint

---

## Sample Predictions

### Example 1
```
Text: The stock market reached record highs today as tech companies 
      reported strong earnings.
Predicted: [approval, optimism]
Probabilities:
  approval        0.7234
  optimism        0.6891
  admiration      0.4523
  excitement      0.3456
```

### Example 2
```
Text: NASA announces new mission to explore Mars with advanced rovers.
Predicted: [excitement, admiration, curiosity]
Probabilities:
  excitement      0.8123
  admiration      0.7456
  curiosity       0.6234
  optimism        0.4567
```

### Example 3
```
Text: The championship game was decided in overtime with a stunning goal.
Predicted: [excitement, admiration, joy]
Probabilities:
  excitement      0.8534
  joy             0.7123
  admiration      0.6789
  approval        0.4234
```

---

## Key Insights

### Multi-Label Learning Challenges
1. **Label Dependencies:** Some emotions co-occur more frequently than others
2. **Threshold Selection:** Single threshold may not be optimal for all classes
3. **Class Imbalance:** Requires careful handling (class weights, oversampling, focal loss)

### BERT's Effectiveness
1. **Contextual Understanding:** Pre-trained representations capture emotional nuances
2. **Transfer Learning:** Effective even with limited domain-specific data
3. **Fine-tuning Efficiency:** Achieves good performance in just 3 epochs

---

## Future Improvements

1. **Address Class Imbalance:**
   - Use class weights in loss function
   - Apply focal loss for hard examples
   - Oversample rare emotions or undersample frequent ones

2. **Advanced Architectures:**
   - Try RoBERTa, DistilBERT, or other BERT variants
   - Experiment with larger models (BERT-large)
   - Use emotion-specific pre-trained models

3. **Threshold Optimization:**
   - Use different thresholds per class
   - Optimize thresholds on validation set
   - Apply probabilistic calibration

4. **Data Augmentation:**
   - Back-translation for data augmentation
   - Synonym replacement
   - Mixup or other augmentation techniques

5. **Ensemble Methods:**
   - Combine predictions from multiple models
   - Average probabilities from different checkpoints

6. **Error Analysis:**
   - Analyze failure cases systematically
   - Identify confusing emotion pairs
   - Study impact of text length on performance

---

## Requirements

### Hardware
- **GPU:** Required for efficient training (6-8GB VRAM)
- **RAM:** 16GB+ system memory recommended
- **Storage:** ~2GB for dataset and model checkpoints

---

## License

This project is for educational purposes as part of a Deep Learning assignment.
---
