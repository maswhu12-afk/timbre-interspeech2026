#!/usr/bin/env python3
"""
Layer-wise Probing for Laryngeal Bandwidth Hypothesis.

Extract embeddings from each transformer layer of wav2vec2-xls-r-300m (24 layers, 128 languages)
or wav2vec2-xls-r-1b (48 layers), train linear probe on each, compare accuracy curves.

HYPOTHESIS:
- Romance languages: perturbation effect persists in LOWER layers (acoustic)
- Tonal languages: best emotion accuracy at LOWER layers (acoustic)
- Non-tonal non-Romance: accuracy INCREASES at upper layers (linguistic)

Models:
  - facebook/wav2vec2-xls-r-300m  (300M params, 24 layers, 1024-dim, ~1.2GB) — DEFAULT
  - facebook/wav2vec2-xls-r-1b    (1B params, 48 layers, 1280-dim, ~3.8GB) — for 48GB GPU
  - facebook/wav2vec2-base-960h   (95M params, 12 layers, 768-dim) — English-only baseline

Usage:
  # On GPU (A6000 48GB recommended for 1B model)
  python layerwise_probe.py --model facebook/wav2vec2-xls-r-300m

  # All 22 corpora with 300M model (~4h on A6000)
  python layerwise_probe.py --model facebook/wav2vec2-xls-r-300m --max-per-class 400

  # Quick test (2 corpora, ~15 min on T4)
  python layerwise_probe.py --corpora ravdess esd_zh --max-per-class 100

  # 1B model for maximum depth (~12h on A6000 48GB)
  python layerwise_probe.py --model facebook/wav2vec2-xls-r-1b --max-per-class 300
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "cache"

# Import corpus configs from main experiment
sys.path.insert(0, str(BASE_DIR))
from deep_ser_experiment import CORPUS_CONFIGS, UNIFIED_5


def get_wav_paths_and_labels(corpus_key: str, max_per_class: int = 500):
    """Load WAV paths and emotion labels for a corpus."""
    cache_file = CACHE_DIR / f"{corpus_key}_phoniatric.csv"
    if not cache_file.exists():
        print(f"  [SKIP] No cached features for {corpus_key}. Run extract phase first.")
        return None, None

    df = pd.read_csv(cache_file)
    if 'emotion' not in df.columns or 'file' not in df.columns:
        print(f"  [SKIP] Missing emotion/file columns in {corpus_key}")
        return None, None

    # Filter to unified emotions
    df = df[df['emotion'].isin(UNIFIED_5)].copy()

    # Subsample per class
    if max_per_class and max_per_class > 0:
        df = df.groupby('emotion').apply(
            lambda x: x.sample(min(len(x), max_per_class), random_state=42)
        ).reset_index(drop=True)

    # Resolve WAV paths
    cfg = CORPUS_CONFIGS.get(corpus_key, {})
    data_subdir = cfg.get("data_subdir", corpus_key)
    data_dir = DATA_DIR / data_subdir

    wav_paths = []
    labels = []
    for _, row in df.iterrows():
        # Try to find the WAV file
        fname = row['file']
        candidates = [
            data_dir / fname,
            data_dir / "extracted" / fname,
            data_dir / "extracted" / "wav" / fname,
            data_dir / "wav" / fname,
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break

        if found is None:
            # Try recursive search
            matches = list(data_dir.rglob(fname))
            if matches:
                found = matches[0]

        if found:
            wav_paths.append(str(found))
            labels.append(row['emotion'])

    return wav_paths, labels


def extract_layerwise_embeddings(
    wav_paths: list,
    model_name: str = "facebook/wav2vec2-xls-r-300m",
    batch_size: int = 8,
    max_length_sec: float = 10.0,
) -> dict:
    """
    Extract embeddings from EACH transformer layer.
    Returns dict: {layer_idx: np.ndarray of shape (N, hidden_dim)}
    Layer 0 = CNN feature extractor output
    Layer 1..N = transformer layers
    """
    import torch
    import librosa
    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  [layerwise] Device: {device}, Model: {model_name}, batch_size: {batch_size}")

    # Use BF16 on Ampere+ GPUs (A100, H100, H200) for 2x speedup
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    if use_bf16:
        print(f"  [layerwise] Using BF16 (GPU compute capability {torch.cuda.get_device_capability()})")

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name, output_hidden_states=True, torch_dtype=dtype).to(device)
    model.eval()

    # torch.compile() for 20-30% speedup on H200/A100 (PyTorch 2.x)
    if device == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("  [layerwise] torch.compile() enabled")
        except Exception:
            pass

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    sr_target = 16000
    max_samples = int(max_length_sec * sr_target)
    n_layers = model.config.num_hidden_layers + 1  # +1 for CNN output

    layer_embeddings = {i: [] for i in range(n_layers)}

    with torch.no_grad():
        for i in range(0, len(wav_paths), batch_size):
            batch_paths = wav_paths[i:i + batch_size]
            waveforms = []
            for p in batch_paths:
                try:
                    y, sr = librosa.load(p, sr=sr_target, mono=True)
                    if len(y) > max_samples:
                        y = y[:max_samples]
                    waveforms.append(y)
                except Exception:
                    waveforms.append(np.zeros(sr_target, dtype=np.float32))

            inputs = processor(
                waveforms, sampling_rate=sr_target,
                return_tensors="pt", padding=True, truncation=True,
                max_length=max_samples
            ).to(device)

            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # tuple of (batch, time, hidden)

            for layer_idx, hs in enumerate(hidden_states):
                # Mean pool over time dimension
                pooled = hs.mean(dim=1).cpu().numpy()  # (batch, hidden)
                layer_embeddings[layer_idx].append(pooled)

            if (i // batch_size) % 5 == 0:
                print(f"    [layerwise] {min(i + batch_size, len(wav_paths))}/{len(wav_paths)}")

    # Stack all batches
    for layer_idx in layer_embeddings:
        layer_embeddings[layer_idx] = np.vstack(layer_embeddings[layer_idx])

    return layer_embeddings


def probe_layer(embeddings: np.ndarray, labels: list, n_splits: int = 5) -> dict:
    """Train linear probe on embeddings, return metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import f1_score, accuracy_score

    le = LabelEncoder()
    y = le.fit_transform(labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1s = []
    accs = []

    for train_idx, test_idx in skf.split(embeddings, y):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1s.append(f1_score(y_test, y_pred, average='weighted'))
        accs.append(accuracy_score(y_test, y_pred))

    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
    }


def run_layerwise_experiment(
    corpus_keys: list,
    max_per_class: int = 400,
    model_name: str = "facebook/wav2vec2-xls-r-300m",
    batch_size: int = 8,
):
    """Run layer-wise probing for multiple corpora."""
    results = {}

    for corpus_key in corpus_keys:
        cfg = CORPUS_CONFIGS.get(corpus_key, {})
        name = cfg.get("name", corpus_key)
        lang = cfg.get("language", "?")
        ptype = cfg.get("prosodic_type", "?")

        print(f"\n{'='*60}")
        print(f"CORPUS: {name} ({lang}, {ptype})")
        print(f"{'='*60}")

        wav_paths, labels = get_wav_paths_and_labels(corpus_key, max_per_class)
        if wav_paths is None or len(wav_paths) < 50:
            print(f"  [SKIP] Not enough samples ({len(wav_paths) if wav_paths else 0})")
            continue

        print(f"  Samples: {len(wav_paths)}, Classes: {len(set(labels))}")

        # Extract layer-wise embeddings
        layer_embs = extract_layerwise_embeddings(wav_paths, model_name, batch_size=batch_size)

        # Probe each layer
        corpus_results = {
            "corpus": name,
            "language": lang,
            "prosodic_type": ptype,
            "n_samples": len(wav_paths),
            "n_classes": len(set(labels)),
            "layers": {}
        }

        for layer_idx, embs in sorted(layer_embs.items()):
            print(f"  Layer {layer_idx:2d}: probing...", end=" ")
            metrics = probe_layer(embs, labels)
            corpus_results["layers"][layer_idx] = metrics
            print(f"F1={metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}")

        results[corpus_key] = corpus_results

        # Find best layer
        best_layer = max(
            corpus_results["layers"].items(),
            key=lambda x: x[1]["f1_mean"]
        )
        print(f"\n  BEST LAYER: {best_layer[0]} (F1={best_layer[1]['f1_mean']:.3f})")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / "layerwise_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Best layers by prosodic type")
    print(f"{'='*60}")
    for key, res in results.items():
        layers = res["layers"]
        best = max(layers.items(), key=lambda x: x[1]["f1_mean"])
        worst = min(layers.items(), key=lambda x: x[1]["f1_mean"])
        # Compute "upper vs lower" ratio (adaptive to model depth)
        n_total = len(layers)
        mid = n_total // 2
        lower_mean = np.mean([layers[str(i)]["f1_mean"] for i in range(0, mid) if str(i) in layers])
        upper_mean = np.mean([layers[str(i)]["f1_mean"] for i in range(mid, n_total) if str(i) in layers])
        print(f"  {res['corpus']:20s} ({res['prosodic_type']:15s}): "
              f"best=L{best[0]}({best[1]['f1_mean']:.3f}) "
              f"lower={lower_mean:.3f} upper={upper_mean:.3f} "
              f"ratio={upper_mean/max(lower_mean,0.001):.2f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer-wise probing for LBH")
    parser.add_argument("--corpora", nargs="+", default=[
        # Romance (HIGH perturbation)
        "emouerj_portuguese", "cafe", "emozionalmente", "mesd",
        # Non-Romance stress-timed
        "ravdess", "crema_d", "emodb", "aesdd", "nemo", "resd", "dusha_russian",
        # Tonal
        "esd_zh", "caves", "thai_ser", "visec_vietnamese",
        # Turkic control
        "turevdb_turkish", "kazemotts_kazakh",
        # Other (ejective, pitch-accent, etc.)
        "ased_amharic", "jvnv", "mder_arabic", "urdu_dataset", "subesco",
        # Stød
        "emotale_danish",
    ])
    parser.add_argument("--max-per-class", type=int, default=400)
    parser.add_argument("--model", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for embedding extraction (32 for H200 141GB, 8 for T4 16GB)")

    args = parser.parse_args()
    run_layerwise_experiment(args.corpora, args.max_per_class, args.model, args.batch_size)
