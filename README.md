# VoiceShield AI — Deepfake Voice Detection

Detects whether a speech audio clip is **real (human)** or **AI-generated (synthetic/spoofed)** using two architectures: a fine-tuned Wav2Vec2 transformer and a CNN trained on Mel spectrograms.

## Overview

AI voice cloning and TTS systems are increasingly used in scams (fake calls impersonating relatives, executives, etc.). This project builds, compares, and evaluates two detection architectures that distinguish authentic human speech from AI-generated speech.

## Architecture Comparison

| Metric | Wav2Vec2 | CNN (Spectrogram) |
|---|---|---|
| Test Accuracy (default threshold) | 85.03% | **97.59%** |
| ROC-AUC | 0.992 | **0.995** |
| EER | 3.21% | **2.67%** |
| Accuracy @ EER threshold | **96.79%** | 94.39% |

**Key finding**: CNN outperforms Wav2Vec2 on default-threshold accuracy and EER, but Wav2Vec2 shows better calibration at the optimal threshold — suggesting the two architectures capture complementary signal. CNN learns explicit spectro-temporal patterns; Wav2Vec2 leverages deep speech representations pretrained on 960h of human audio.

![Architecture Comparison](results/figures/architecture_comparison.png)
![ROC Curve](results/figures/roc_curve.png)
![Confusion Matrix](results/figures/confusion_matrix.png)

## Dataset

[garystafford/deepfake-audio-detection](https://huggingface.co/datasets/garystafford/deepfake-audio-detection)
- 1,866 balanced samples (933 real / 933 fake)
- 80/20 stratified split: 1,492 train / 374 test
- Preprocessing: resample to 16kHz, pad/truncate to 4s fixed length

## Models

**Wav2Vec2 Classifier**
- Pretrained `facebook/wav2vec2-base` encoder (frozen, 94.5M params)
- Trainable classification head: Linear(768->256) -> ReLU -> Dropout(0.3) -> Linear(256->2)
- 197K trainable parameters
- Training: 4 epochs, AdamW lr=1e-4, batch size 16

**CNN Spectrogram Classifier**
- Input: 64-band Mel spectrogram (64x126)
- 3 conv blocks (32->64->128 filters) + AdaptiveAvgPool + classifier head
- 618K parameters
- Training: 10 epochs, AdamW lr=1e-3, batch size 32

## Error Analysis

Misclassifications were concentrated in real samples predicted as fake, with probabilities near the decision boundary — genuine ambiguity rather than confident errors.

![Error Analysis](results/figures/error_analysis_spectrograms.png)

**Hypothesis**: Misclassified real samples show denser, more continuous spectral energy with fewer silence gaps — consistent with prior findings that TTS-generated speech typically lacks the natural pauses present in human recordings.

## Limitations & Future Work

- Cross-dataset evaluation (ASVspoof benchmark) would test generalization
- Ensemble fusion of both architectures could combine complementary strengths
- Future work: pitch/prosody features, multimodal (audio+video) deepfake detection

## Tech Stack

Python, PyTorch, Hugging Face Transformers (Wav2Vec2), librosa, scikit-learn, Google Colab
