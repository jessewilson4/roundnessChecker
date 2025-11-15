# 1D U‑Net — EEG Denoising & Correction

**Purpose:** Train a 1D U‑Net to take noisy 7‑channel EEG input and output corrected (cleaned) 7‑channel EEG. The model uses temporal context (past + future) and inter‑channel correlations to reconstruct the clean waveform and optionally produce per‑sample confidence or artifact masks.

---

## 1. Task definition

* **Input:** multichannel noisy EEG window, shape `(batch, 7, T_in)` (example T_in = 500 samples = 2 s @ 250 Hz)
* **Output:** cleaned EEG window for the center segment, shape `(batch, 7, T_out)` (example T_out = 250 samples = 1 s centered)
* **Goal:** Minimize reconstruction error of the clean waveform while preserving spectral/phase fidelity and avoiding removal of true brain signal.

---

## 2. High‑level approach

* Use a **1D U‑Net / encoder–decoder** architecture operating on time axis with channels as feature maps.
* Train with paired clean/noisy examples (supervised). Use synthetic artifact injection into clean data and available real dirty/clean segments.
* Optionally multi‑task: main regression head (clean waveform) + auxiliary classification head (artifact probability per sample) to aid learning.

---

## 3. Data preparation

**Windowing & alignment**

* Use overlapping windows. Example: 2 s input window (500 samples) → predict center 1 s (250 samples). Stride = 125 samples (50% of output) recommended for smooth stitching.

**Normalization**

* Per‑channel z‑score normalization using running stats computed from training data. Store per‑subject offsets for online calibration.

**Augmentation**

* Synthetic artifacts: eyeblinks (low‑freq spikes), EMG bursts (broadband high‑freq), electrode pops (transient spikes), channel dropout, baseline drift, additive AWGN.
* Time stretching/compression small amounts; channel mixing; polarity flips for robustness.

**Labels**

* Primary label = ground truth clean waveform for central window.
* Optional: ground truth artifact mask (binary per sample) when available.

---

## 4. Architecture (recommended starter)

**Input:** `(7, 500)`

Encoder (example 4 levels):

* Conv1D(64, kernel=15, padding=7) → BN → ReLU

* Conv1D(64, kernel=15, padding=7) → BN → ReLU

* MaxPool1D(2)

* Conv1D(128, kernel=15, padding=7) → BN → ReLU

* Conv1D(128, kernel=15, padding=7) → BN → ReLU

* MaxPool1D(2)

* Conv1D(256, kernel=15, padding=7) → BN → ReLU

* Conv1D(256, kernel=15, padding=7) → BN → ReLU

* MaxPool1D(2)

* Conv1D(512, kernel=15, padding=7) → BN → ReLU

* Conv1D(512, kernel=15, padding=7) → BN → ReLU

Bottleneck: Conv1D(512, kernel=15) → ReLU → Dropout(0.3)

Decoder (mirrors encoder with transposed conv or upsample+conv):

* UpSample/ConvTranspose to restore temporal resolution
* Concatenate skip connection from matching encoder level
* Two Conv1D+BN+ReLU blocks per level

Final layers:

* Conv1D(7, kernel=1) → Linear output (no activation) to predict real‑valued waveforms
* Optional auxiliary head: Conv1D(1, kernel=1) → Sigmoid for per‑sample artifact probability

**Notes on kernels & receptive field:** large kernels (e.g., 9–15) increase receptive field quickly; stacking + pooling yields wide temporal context. Adjust to ensure receptive field ≥ T_out.

---

## 5. Losses & training objectives

* **Primary loss:** L1 (MAE) or L2 (MSE) between predicted clean waveform and ground truth (per channel). MAE often yields sharper reconstructions; MSE penalizes large errors more.
* **Spectral loss:** STFT magnitude loss or multi‑resolution STFT loss to preserve frequency content.
* **Auxiliary loss (optional):** binary cross‑entropy for artifact mask.
* **Total loss:** weighted sum, e.g. `L = α·time_loss + β·spectral_loss + γ·artifact_loss`.

Optimization:

* Optimizer: Adam or AdamW
* LR: 1e‑3 → reduce on plateau; warmup for large batches possible
* Batch size: 16–64 depending on GPU
* Regularization: dropout in bottleneck, weight decay

---

## 6. Training regime

* Train on synthetic + real clean segments. For each clean sample, randomly inject artifacts to generate noisy input; use original clean as target.
* Mix in unmodified real dirty segments with pseudo‑labels (if you have clean nearby) for robustness.
* Validation: hold out subjects (leave‑one‑subject‑out) to test generalization.
* Early stopping on validation spectral+time loss.

---

## 7. Inference & stitching

* Run the model on overlapping windows; for each window predict center T_out samples.
* Reconstruct full stream by overlap‑adding with a smooth window (e.g., raised cosine) or median blending where overlapping predictions disagree.
* Optionally produce a per‑sample confidence from auxiliary head and use it to blend measured vs predicted signal (e.g., `output = conf*pred + (1−conf)*measured`).

---

## 8. Evaluation metrics

* **Time‑domain:** MAE, RMSE per channel, percentile error
* **Spectral:** multi‑resolution STFT loss, spectral distortion (SDR-like metrics)
* **Detection:** precision/recall/F1 for artifact localization if masks available
* **Clinical utility:** evaluate downstream neurofeedback metrics or task performance differences pre/post denoising

---

## 9. Online calibration & personalization

* Collect 1–5 minutes of subject baseline. Update per‑channel normalization stats.
* Optionally fine‑tune the last decoder block or adapter layers for a small number of gradient steps with low LR (e.g., 1e‑5) using subject baseline.
* Use parameter‑efficient adapters or LoRA‑style modules if on‑device fine‑tuning is desired.

---

## 10. Computational considerations

* 1‑D convolutions are computationally cheap; latency dominated by model depth and channel width.
* For POC on desktop: use float32 training. For eventual edge: convert to FP16 and use TensorRT or ONNX runtime with quantization.
* Profile CWT/STFT only if you add spectral losses or auxiliary image branches; otherwise time‑domain U‑Net avoids this overhead.

---

## 11. Suggested hyperparameter starting point

* Window: `T_in=500` (2 s), `T_out=250` (1 s)
* Encoder channels: [64, 128, 256, 512]
* Kernel size: 15 (or 9 if memory constrained)
* Optimizer: Adam, lr=1e‑3, batch=32
* Loss weights: α=time_loss=1.0, β=spectral_loss=0.5 (if used), γ=artifact_loss=0.2 (if used)

---

## 12. Next steps (practical)

1. Implement data pipeline: windowing, normalization, synthetic artifact injection.
2. Implement starter 1D U‑Net in PyTorch with shapes above. Unit test forward/backward on dummy data.
3. Train on small subset (1–2 hours) to validate learning signal. Visualize reconstructions.
4. Add spectral loss and auxiliary artifact head if recon loses frequency content.
5. Evaluate on held‑out subjects and iterate.

---

*End of document — updated to reflect EEG denoising and multichannel correction goal.*