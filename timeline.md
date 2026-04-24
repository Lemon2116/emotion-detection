# Mini-Project: Sentence-Level Emotion Classification

---

## Overview

**Title:** Sentence-Level Emotion Classification with BERT

### Problem Statement

Given a sentence, classify it into one of 28 emotion categories from the GoEmotions taxonomy. The goal is to fine-tune a BERT model on the GoEmotions dataset and evaluate it rigorously against a pre-trained baseline using standard classification metrics.

---

### Model Architecture

```
GoEmotions Dataset (flat, sentence-level)
        ↓
   BERT Encoder (fine-tuned)
        ↓
  [CLS] embedding
        ↓
  Linear head → 28 emotion labels
```

**Baseline:** `bhadresh-savani/bert-base-go-emotion` — zero-shot evaluation on the validation set  
**Proposed:** Fine-tune `google-bert/bert-base-uncased` on GoEmotions train split, compare metrics

---

### Dataset

- **Source:** GoEmotions (`datasets` library) — 58k Reddit comments across 28 emotion labels
- **Splits:** Pre-defined train / validation / test splits used as-is

---

### Key Challenges

1. **Multi-label ambiguity:** Many sentences carry multiple emotion labels simultaneously. Strategy: use top-1 label (highest confidence) for classification; report both micro and macro F1 to capture per-class performance.

2. **Class imbalance:** Label 27 (`neutral`) dominates the dataset. Weighted F1 and per-class breakdown will expose model bias toward frequent labels.

---

### Expected Outcomes

- Fine-tuned BERT matches or outperforms the pre-trained baseline on validation F1
- Per-class analysis reveals which emotions are easy vs. hard to classify
- ROC/AUC curves show model confidence calibration across all 28 classes

---

## 5-Week Timeline

### Week 1 — Environment Setup & Data Exploration
**Goal:** Have a clean, working environment and a thorough understanding of the raw data.

- [ ] Set up Python environment: install `transformers`, `torch`, `datasets`, `pandas`, `scikit-learn`, `matplotlib`
- [ ] Load GoEmotions via `datasets` library; confirm splits and columns (`text`, `labels`)
- [ ] Run exploratory data analysis (EDA):
  - Label distribution (identify dominant vs. rare emotions)
  - Sentence length distribution
  - Number of multi-label samples
- [ ] Confirm the HuggingFace baseline model (`bhadresh-savani/bert-base-go-emotion`) loads and runs on a few sample sentences

**Deliverable:** EDA notebook + confirmed baseline pipeline running

---

### Week 2 — Data Preprocessing
**Goal:** Prepare the dataset for fine-tuning.

- [ ] Resolve multi-label ambiguity: select top-1 label per sentence (highest index or first label)
- [ ] Tokenize all splits using `BertTokenizer` with padding and truncation
- [ ] Create `Dataset` / `DataLoader` objects for train, validation, and test sets
- [ ] Save processed data to disk for reproducibility

**Deliverable:** `data_prep.py` script + processed dataset

---

### Week 3 — Baseline Evaluation & Fine-Tuning
**Goal:** Establish a quantitative baseline and fine-tune BERT.

- [ ] **Baseline:** Evaluate `bhadresh-savani/bert-base-go-emotion` on the validation set
  - Record accuracy, macro F1, weighted F1, and per-class precision & recall
- [ ] **Fine-tuning:** Load `google-bert/bert-base-uncased` with a linear classification head (28 classes)
  - Write training loop with cross-entropy loss, AdamW optimizer, and learning-rate scheduler
  - Run a sanity-check pass on a small subset (200 samples) to confirm no shape errors
- [ ] Begin full training; log loss and validation F1 per epoch

**Deliverable:** `baseline_eval.py` + `train.py` + first training run

---

### Week 4 — Hyperparameter Tuning & Final Evaluation
**Goal:** Optimize the model and produce final metrics.

- [ ] Tune key hyperparameters:
  - BERT fine-tuning depth (frozen / top-2-layers / all layers)
  - Learning rate (1e-4, 5e-5, 3e-5)
  - Batch size (16 / 32)
- [ ] Evaluate best checkpoint on the test set; compare against baseline
- [ ] Compute full evaluation suite on validation set:
  - Overall accuracy
  - Per-class precision, recall, and F1
  - Macro and weighted F1
  - ROC/AUC (one-vs-rest, per class)

**Deliverable:** Trained model checkpoint + `eval.py` + results comparison table

---

### Week 5 — Analysis, Visualization & Write-Up
**Goal:** Interpret results, produce visualizations, and finalize the project.

- [ ] **Visualizations:**
  - Bar chart: per-class F1 scores (baseline vs. fine-tuned)
  - Confusion matrix: predicted vs. true labels (top-N most frequent classes)
  - Precision-Recall curves per emotion class
  - ROC curves with AUC scores (one-vs-rest for all 28 classes)
  - Training curve: loss and F1 over epochs
- [ ] Error analysis: identify patterns in misclassified samples (e.g., confusion between *sadness* and *disappointment*)
- [ ] Write final report covering:
  - Problem statement & dataset description
  - Model architecture and training setup
  - Results table (baseline vs. fine-tuned): accuracy, precision, recall, F1, AUC
  - Visualization figures
  - Limitations & future work
- [ ] Clean up code: finalize `README.md`, ensure scripts are runnable end-to-end

**Deliverable:** Final report + visualization figures + clean codebase

---

## Summary Table

| Week | Focus | Key Output |
|------|-------|------------|
| 1 | Setup & EDA | Working environment, EDA notebook |
| 2 | Data Preprocessing | Tokenized dataset, `data_prep.py` |
| 3 | Baseline + Fine-Tuning | Baseline metrics, `train.py` |
| 4 | Tuning & Evaluation | Trained model, accuracy / F1 / AUC results |
| 5 | Analysis & Write-Up | Visualizations, final report |
