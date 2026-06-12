# VoiceShield AI — Deepfake Voice Detection

Detects whether a speech audio clip is **real (human)** or **AI-generated (synthetic/spoofed)** using a fine-tuned Wav2Vec2 model.

## Overview

AI voice cloning and TTS systems are increasingly used in scams (fake calls impersonating relatives, executives, etc.). This project builds and evaluates a binary classifier that distinguishes authentic human speech from AI-generated speech, using self-supervised audio representations.

## Approach

- **Model**: Pretrained `facebook/wav2vec2-base` (frozen) + lightweight classification head (Linear -> ReLU -> Dropout -> Linear)
- **Dataset**: [garystafford/deepfake-audio-detection](https://huggingface.co/datasets/garystafford/deepfake-audio-detection) — 1,866 balanced samples (933 real / 933 fake)
- **Preprocessing**: resample to 16kHz, pad/truncate to 4-second fixed length
- **Training**: 4 epochs, AdamW, lr=1e-4, batch size 16 — only the 197K-parameter classification head is trained (Wav2Vec2 encoder frozen)
- **Split**: 80/20 stratified (1,492 train / 374 test)

## Results

| Metric | Value |
|---|---|
| Test Accuracy (default threshold) | 85.03% |
| ROC-AUC | 0.992 |
| Equal Error Rate (EER) | 3.21% |
| Accuracy at EER threshold | 96.79% |

The high ROC-AUC relative to default-threshold accuracy revealed decision boundary miscalibration — tuning the threshold to the EER point improved accuracy from 85% to 96.79% with perfectly balanced precision/recall (0.97) across both classes.

![ROC Curve](results/figures/roc_curve.png)
![Confusion Matrix](results/figures/confusion_matrix.png)

## Error Analysis

Misclassifications at the default threshold were concentrated in real samples predicted as fake, with probabilities clustered near the decision boundary — indicating genuine ambiguity rather than confident errors.

![Error Analysis](results/figures/error_analysis_spectrograms.png)

**Hypothesis**: Misclassified real samples show denser, more continuous spectral energy with fewer silence gaps. This may reflect the model associating natural pauses/breathing with bonafide speech — consistent with prior findings that TTS-generated speech typically lacks the leading/trailing silences present in human recordings.

## Limitations & Future Work

- Single architecture evaluated; a CNN-on-spectrogram baseline would enable architecture comparison
- Cross-dataset evaluation (ASVspoof benchmark) would test generalization
- Future work: multi-feature fusion (pitch, prosody, spectral features), multimodal (audio+video) deepfake detection

## Tech Stack

Python, PyTorch, Hugging Face Transformers (Wav2Vec2), librosa, scikit-learn, Google Colab
