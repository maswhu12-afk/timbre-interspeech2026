#!/usr/bin/env python3
"""
Cross-Corpus Speech Emotion Recognition Experiment.
Paper: "Do Voice Perturbation Metrics Still Matter?"

Corpora:
  - RAVDESS (English, acted, 24 actors, 8 emotions)
  - EmoDB (German, acted, 10 actors, 7 emotions)
  - CREMA-D (English, crowd-acted, 91 actors, 6 emotions)

Features: 27 handcrafted (6 groups) + optionally wav2vec 2.0 embeddings (GPU)

Usage:
  python cross_corpus_ser.py --download          # download all corpora
  python cross_corpus_ser.py --extract           # extract features
  python cross_corpus_ser.py --classify          # run XGBoost + ablation
  python cross_corpus_ser.py --all               # do everything
  python cross_corpus_ser.py --all --gpu         # include wav2vec (needs GPU)
"""

import argparse
import json
import os
import sys
import warnings
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Emotion mapping to unified 5-class scheme
# We map all corpora to: angry, happy, sad, neutral, fear
# (5 classes = fair comparison across corpora)
UNIFIED_EMOTIONS = ["angry", "happy", "sad", "neutral", "fear"]

RAVDESS_EMOTION_MAP = {
    1: "neutral", 2: "neutral",  # calm → neutral
    3: "happy", 4: "sad", 5: "angry",
    6: "fear", 7: "disgust", 8: "surprise"
}

EMODB_EMOTION_MAP = {
    "W": "angry", "L": "neutral",  # boredom → neutral
    "E": "disgust", "A": "fear",
    "F": "happy", "T": "sad", "N": "neutral"
}

CREMAD_EMOTION_MAP = {
    "ANG": "angry", "HAP": "happy", "SAD": "sad",
    "NEU": "neutral", "FEA": "fear", "DIS": "disgust"
}


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_ravdess(data_dir: Path):
    """Download RAVDESS audio-only files from Zenodo."""
    import urllib.request

    ravdess_dir = data_dir / "ravdess"
    if ravdess_dir.exists() and any(ravdess_dir.rglob("*.wav")):
        n = len(list(ravdess_dir.rglob("*.wav")))
        print(f"[RAVDESS] Already downloaded ({n} files)")
        return

    ravdess_dir.mkdir(parents=True, exist_ok=True)
    print("[RAVDESS] Downloading from Zenodo...")

    # RAVDESS audio-speech: 24 actors
    base_url = "https://zenodo.org/record/1188976/files"
    for actor_num in range(1, 25):
        actor_id = f"{actor_num:02d}"
        filename = f"Audio_Speech_Actors_01-24.zip"
        zip_path = ravdess_dir / filename

        if not zip_path.exists():
            url = f"{base_url}/{filename}?download=1"
            print(f"  Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, zip_path)
            except Exception as e:
                print(f"  Error: {e}")
                # Try alternative: individual actor downloads
                break

        if zip_path.exists():
            print(f"  Extracting {filename}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(ravdess_dir)
            zip_path.unlink()
            break  # Single zip contains all actors

    n = len(list(ravdess_dir.rglob("*.wav")))
    print(f"[RAVDESS] Done: {n} audio files")


def download_emodb(data_dir: Path):
    """Download Berlin EmoDB."""
    import urllib.request

    emodb_dir = data_dir / "emodb"
    if emodb_dir.exists() and any(emodb_dir.rglob("*.wav")):
        n = len(list(emodb_dir.rglob("*.wav")))
        print(f"[EmoDB] Already downloaded ({n} files)")
        return

    emodb_dir.mkdir(parents=True, exist_ok=True)
    print("[EmoDB] Downloading from TU Berlin...")

    url = "http://emodb.bilderbar.info/download/download.zip"
    zip_path = emodb_dir / "emodb.zip"

    try:
        urllib.request.urlretrieve(url, zip_path)
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(emodb_dir)
        zip_path.unlink()
    except Exception as e:
        print(f"  Error: {e}")
        print("  Manual download: http://emodb.bilderbar.info/download/download.zip")

    n = len(list(emodb_dir.rglob("*.wav")))
    print(f"[EmoDB] Done: {n} audio files")


def download_cremad(data_dir: Path):
    """Download CREMA-D via direct links or HuggingFace."""
    cremad_dir = data_dir / "cremad"
    if cremad_dir.exists() and any(cremad_dir.rglob("*.wav")):
        n = len(list(cremad_dir.rglob("*.wav")))
        print(f"[CREMA-D] Already downloaded ({n} files)")
        return

    cremad_dir.mkdir(parents=True, exist_ok=True)
    print("[CREMA-D] Downloading... (large dataset, ~2.6 GB)")

    try:
        from datasets import load_dataset
        print("  Using HuggingFace datasets...")
        ds = load_dataset("SpeechBrain/CREMA-D", split="train", trust_remote_code=True)
        print(f"  Loaded {len(ds)} samples")
        # Save to wav files
        import soundfile as sf
        for i, sample in enumerate(ds):
            emotion = sample.get("emotion", sample.get("label", "unknown"))
            audio = sample["audio"]
            wav_path = cremad_dir / f"cremad_{i:05d}_{emotion}.wav"
            sf.write(str(wav_path), audio["array"], audio["sampling_rate"])
            if (i + 1) % 500 == 0:
                print(f"  Saved {i+1} files...")
    except ImportError:
        print("  HuggingFace datasets not available.")
        print("  Install: pip install datasets")
        print("  Or download manually from: https://github.com/CheyneyComputerScience/CREMA-D")
    except Exception as e:
        print(f"  Error: {e}")

    n = len(list(cremad_dir.rglob("*.wav")))
    print(f"[CREMA-D] Done: {n} audio files")


def download_all(data_dir: Path):
    """Download all corpora."""
    data_dir.mkdir(parents=True, exist_ok=True)
    download_ravdess(data_dir)
    download_emodb(data_dir)
    download_cremad(data_dir)


# ---------------------------------------------------------------------------
# Corpus loading (file → emotion label)
# ---------------------------------------------------------------------------

def load_ravdess(data_dir: Path) -> list[tuple[Path, str]]:
    """Load RAVDESS files with emotion labels.
    Filename: {modality}-{channel}-{emotion}-{intensity}-{statement}-{rep}-{actor}.wav
    """
    ravdess_dir = data_dir / "ravdess"
    samples = []
    for wav in sorted(ravdess_dir.rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue
        modality = int(parts[0])
        if modality != 3:  # audio-only
            continue
        emotion_code = int(parts[2])
        emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
        if emotion and emotion in UNIFIED_EMOTIONS:
            samples.append((wav, emotion))
    print(f"[RAVDESS] Loaded {len(samples)} samples ({len(set(e for _, e in samples))} emotions)")
    return samples


def load_emodb(data_dir: Path) -> list[tuple[Path, str]]:
    """Load EmoDB files with emotion labels.
    Filename: {speaker}{text}{emotion}.wav  (e.g., 03a01Fa.wav)
    Emotion is the 6th character (0-indexed: 5th).
    """
    emodb_dir = data_dir / "emodb"
    samples = []
    # EmoDB wav files are in a 'wav' subdirectory
    wav_dirs = [emodb_dir / "wav", emodb_dir]
    for d in wav_dirs:
        for wav in sorted(d.rglob("*.wav")):
            name = wav.stem
            if len(name) < 6:
                continue
            # Emotion code is typically the 6th character
            emotion_code = name[5]
            emotion = EMODB_EMOTION_MAP.get(emotion_code)
            if emotion and emotion in UNIFIED_EMOTIONS:
                samples.append((wav, emotion))
    print(f"[EmoDB] Loaded {len(samples)} samples ({len(set(e for _, e in samples))} emotions)")
    return samples


def load_cremad(data_dir: Path) -> list[tuple[Path, str]]:
    """Load CREMA-D files with emotion labels.
    Original filename: {actor}_{sentence}_{emotion}_{intensity}.wav
    Our saved filename: cremad_{idx}_{emotion}.wav
    """
    cremad_dir = data_dir / "cremad"
    samples = []
    for wav in sorted(cremad_dir.rglob("*.wav")):
        name = wav.stem
        # Try original CREMA-D naming: 1001_DFA_ANG_XX.wav
        parts = name.split("_")
        if len(parts) >= 3:
            # Check if it matches CREMA-D original format
            emotion_code = None
            for p in parts:
                if p in CREMAD_EMOTION_MAP:
                    emotion_code = p
                    break
            if emotion_code:
                emotion = CREMAD_EMOTION_MAP[emotion_code]
                if emotion in UNIFIED_EMOTIONS:
                    samples.append((wav, emotion))
                continue
            # Our saved format: cremad_00001_angry.wav
            if parts[0] == "cremad" and len(parts) >= 3:
                emotion = parts[-1].lower()
                if emotion in UNIFIED_EMOTIONS:
                    samples.append((wav, emotion))
    print(f"[CREMA-D] Loaded {len(samples)} samples ({len(set(e for _, e in samples))} emotions)")
    return samples


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(wav_path: Path, sr_target: int = 16000) -> Optional[dict]:
    """Extract 27 handcrafted features from a single audio file.

    Returns dict with keys:
      Frequency (3): pitch_hz, pitch_std, voiced_ratio
      Perturbation (3): jitter, shimmer, hnr
      MFCC (13): mfcc_0 ... mfcc_12
      Spectral (4): spectral_centroid, spectral_bandwidth, spectral_rolloff, zcr
      Formant (3): f1, f2, f3
      Temporal (1): energy_db
    """
    import librosa
    import parselmouth

    try:
        y, sr = librosa.load(str(wav_path), sr=sr_target, mono=True)
        if len(y) < sr_target * 0.3:  # skip < 0.3 sec
            return None

        features = {}

        # --- Frequency features (Parselmouth) ---
        snd = parselmouth.Sound(str(wav_path))
        # Resample if needed
        if snd.sampling_frequency != sr_target:
            snd = snd.resample_to(sr_target)

        pitch_obj = parselmouth.praat.call(snd, "To Pitch", 0.0, 75, 600)
        pitch_values = pitch_obj.selected_array['frequency']
        pitch_voiced = pitch_values[pitch_values > 0]

        features['pitch_hz'] = float(np.mean(pitch_voiced)) if len(pitch_voiced) > 0 else 0.0
        features['pitch_std'] = float(np.std(pitch_voiced)) if len(pitch_voiced) > 1 else 0.0
        features['voiced_ratio'] = float(len(pitch_voiced) / len(pitch_values)) if len(pitch_values) > 0 else 0.0

        # --- Perturbation features (Parselmouth/Praat) ---
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)...", 75, 600)

        jitter_val = parselmouth.praat.call(
            point_process, "Get jitter (local)...", 0, 0, 0.0001, 0.02, 1.3
        )
        shimmer_val = parselmouth.praat.call(
            [snd, point_process], "Get shimmer (local)...", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)...", 0.01, 75, 0.1, 1.0)
        hnr_val = parselmouth.praat.call(harmonicity, "Get mean...", 0, 0)

        features['jitter'] = float(jitter_val) if not np.isnan(jitter_val) else 0.0
        features['shimmer'] = float(shimmer_val) if not np.isnan(shimmer_val) else 0.0
        features['hnr'] = float(hnr_val) if not np.isnan(hnr_val) else 0.0

        # --- MFCC features (librosa) ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr_target, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        for i in range(13):
            features[f'mfcc_{i}'] = float(mfcc_mean[i])

        # --- Spectral features (librosa) ---
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr_target)[0]
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr_target)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr_target)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        features['spectral_centroid'] = float(np.mean(spec_centroid))
        features['spectral_bandwidth'] = float(np.mean(spec_bandwidth))
        features['spectral_rolloff'] = float(np.mean(spec_rolloff))
        features['zcr'] = float(np.mean(zcr))

        # --- Formant features (Parselmouth/Praat Burg) ---
        formant_obj = parselmouth.praat.call(snd, "To Formant (burg)...", 0.0, 5, 5500, 0.025, 50)
        for i in range(1, 4):
            f_val = parselmouth.praat.call(formant_obj, "Get mean...", i, 0, 0, "hertz")
            features[f'f{i}'] = float(f_val) if not np.isnan(f_val) else 0.0

        # --- Temporal features (librosa) ---
        rms = librosa.feature.rms(y=y)[0]
        energy_db = float(np.mean(librosa.amplitude_to_db(rms + 1e-10)))
        features['energy_db'] = energy_db

        return features

    except Exception as e:
        print(f"  [error] {wav_path.name}: {e}")
        return None


def extract_corpus_features(
    samples: list[tuple[Path, str]],
    corpus_name: str,
    cache_path: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Extract features for entire corpus, with caching."""

    if cache_path and cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"[{corpus_name}] Loaded cached features: {len(df)} samples")
        return df

    if max_samples and len(samples) > max_samples:
        # Stratified sampling
        from collections import Counter
        emotion_counts = Counter(e for _, e in samples)
        per_class = max_samples // len(emotion_counts)
        selected = []
        for emotion in emotion_counts:
            emotion_samples = [(p, e) for p, e in samples if e == emotion]
            np.random.shuffle(emotion_samples)
            selected.extend(emotion_samples[:per_class])
        samples = selected
        print(f"[{corpus_name}] Sampled {len(samples)} (max {max_samples})")

    rows = []
    total = len(samples)
    for i, (wav_path, emotion) in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{corpus_name}] Extracting {i+1}/{total}...")

        feats = extract_features(wav_path)
        if feats is not None:
            feats['emotion'] = emotion
            feats['corpus'] = corpus_name
            feats['file'] = wav_path.name
            rows.append(feats)

    df = pd.DataFrame(rows)
    print(f"[{corpus_name}] Extracted features for {len(df)}/{total} samples")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"[{corpus_name}] Cached to {cache_path}")

    return df


# ---------------------------------------------------------------------------
# Classification & Ablation
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "Frequency": ["pitch_hz", "pitch_std", "voiced_ratio"],
    "Perturbation": ["jitter", "shimmer", "hnr"],
    "MFCC": [f"mfcc_{i}" for i in range(13)],
    "Spectral": ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zcr"],
    "Formant": ["f1", "f2", "f3"],
    "Temporal": ["energy_db"],
}

ALL_FEATURES = []
for feats in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(feats)


def classify_and_ablate(
    df: pd.DataFrame,
    corpus_name: str,
    n_folds: int = 5,
    n_permutations: int = 100,
) -> dict:
    """Run XGBoost classification + ablation study on a single corpus."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
    from xgboost import XGBClassifier

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].values
    y_labels = df['emotion'].values

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    classes = le.classes_

    print(f"\n{'='*60}")
    print(f"CLASSIFICATION: {corpus_name}")
    print(f"  Samples: {len(X)}, Features: {len(feature_cols)}, Classes: {len(classes)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y_labels, return_counts=True)))}")
    print(f"{'='*60}")

    # --- Full model (5-fold CV) ---
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_pred_all = np.zeros_like(y)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=42, verbosity=0,
        )
        clf.fit(X[train_idx], y[train_idx])
        y_pred_all[test_idx] = clf.predict(X[test_idx])

    acc = accuracy_score(y, y_pred_all)
    f1_w = f1_score(y, y_pred_all, average='weighted')
    f1_m = f1_score(y, y_pred_all, average='macro')

    # Per-class report
    report = classification_report(y, y_pred_all, target_names=classes, output_dict=True)
    cm = confusion_matrix(y, y_pred_all)

    print(f"\n  Accuracy: {acc:.3f}")
    print(f"  F1 (weighted): {f1_w:.3f}")
    print(f"  F1 (macro): {f1_m:.3f}")
    print(f"\n  Per-class:")
    for cls in classes:
        r = report[cls]
        print(f"    {cls:>10}: P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1-score']:.2f} N={r['support']}")

    # --- Permutation test ---
    print(f"\n  Running permutation test ({n_permutations} iterations)...")
    perm_accs = []
    skf_perm = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for perm_i in range(n_permutations):
        y_perm = np.random.permutation(y)
        fold_accs = []
        for train_idx, test_idx in skf_perm.split(X, y_perm):
            clf_perm = XGBClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                use_label_encoder=False, eval_metric='mlogloss',
                random_state=perm_i, verbosity=0,
            )
            clf_perm.fit(X[train_idx], y_perm[train_idx])
            fold_accs.append(accuracy_score(y_perm[test_idx], clf_perm.predict(X[test_idx])))
        perm_accs.append(np.mean(fold_accs))

    p_value = np.mean([pa >= acc for pa in perm_accs])
    print(f"  p-value: {'< 0.01' if p_value < 0.01 else f'{p_value:.3f}'}")

    # --- Feature importance ---
    clf_full = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, verbosity=0,
    )
    clf_full.fit(X, y)

    importance = clf_full.feature_importances_
    feat_importance = sorted(
        zip(feature_cols, importance),
        key=lambda x: x[1], reverse=True
    )

    print(f"\n  Top-10 features by gain:")
    for fname, gain in feat_importance[:10]:
        group = next((g for g, fs in FEATURE_GROUPS.items() if fname in fs), "?")
        print(f"    {fname:>20}: {gain*100:.2f}%  [{group}]")

    # --- Ablation study ---
    print(f"\n  Ablation study:")
    ablation_results = {}
    for group_name, group_feats in FEATURE_GROUPS.items():
        remaining = [f for f in feature_cols if f not in group_feats]
        if not remaining:
            continue
        X_abl = df[remaining].values

        y_pred_abl = np.zeros_like(y)
        for train_idx, test_idx in skf.split(X_abl, y):
            clf_abl = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric='mlogloss',
                random_state=42, verbosity=0,
            )
            clf_abl.fit(X_abl[train_idx], y[train_idx])
            y_pred_abl[test_idx] = clf_abl.predict(X_abl[test_idx])

        f1_abl = f1_score(y, y_pred_abl, average='weighted')
        delta = f1_abl - f1_w
        ablation_results[group_name] = {
            'f1_without': round(f1_abl, 3),
            'delta_f1': round(delta, 3),
            'n_features': len(group_feats),
        }
        print(f"    Remove {group_name:>15} ({len(group_feats)} feats): "
              f"F1={f1_abl:.3f}  ΔF1={delta:+.3f}")

    # --- Compile results ---
    results = {
        'corpus': corpus_name,
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'n_classes': len(classes),
        'classes': list(classes),
        'class_distribution': {str(c): int(n) for c, n in zip(*np.unique(y_labels, return_counts=True))},
        'accuracy': round(acc, 3),
        'f1_weighted': round(f1_w, 3),
        'f1_macro': round(f1_m, 3),
        'p_value': '< 0.001' if p_value < 0.001 else f'{p_value:.3f}',
        'random_baseline': round(1.0 / len(classes), 3),
        'per_class': {cls: {
            'precision': round(report[cls]['precision'], 3),
            'recall': round(report[cls]['recall'], 3),
            'f1': round(report[cls]['f1-score'], 3),
            'support': int(report[cls]['support']),
        } for cls in classes},
        'confusion_matrix': cm.tolist(),
        'feature_importance': [
            {'feature': f, 'gain_pct': round(g * 100, 2),
             'group': next((grp for grp, fs in FEATURE_GROUPS.items() if f in fs), "?")}
            for f, g in feat_importance[:15]
        ],
        'ablation': ablation_results,
    }

    return results


# ---------------------------------------------------------------------------
# Cross-corpus summary
# ---------------------------------------------------------------------------

def cross_corpus_summary(all_results: list[dict], output_dir: Path):
    """Generate cross-corpus comparison summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("CROSS-CORPUS SUMMARY")
    print(f"{'='*70}")

    # Table: corpus, N, classes, accuracy, F1_w, F1_m, p-value
    print(f"\n{'Corpus':<12} {'N':>6} {'Cls':>4} {'Acc':>6} {'F1_w':>6} {'F1_m':>6} {'p':>8}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['corpus']:<12} {r['n_samples']:>6} {r['n_classes']:>4} "
              f"{r['accuracy']:>6.3f} {r['f1_weighted']:>6.3f} {r['f1_macro']:>6.3f} "
              f"{r['p_value']:>8}")

    # Ablation comparison across corpora
    print(f"\n{'Ablation: ΔF1 when removing group'}")
    print(f"{'Group':<16}", end="")
    for r in all_results:
        print(f"  {r['corpus']:>10}", end="")
    print()
    print("-" * (16 + 12 * len(all_results)))

    for group_name in FEATURE_GROUPS:
        print(f"{group_name:<16}", end="")
        for r in all_results:
            abl = r['ablation'].get(group_name, {})
            delta = abl.get('delta_f1', 0)
            print(f"  {delta:>+10.3f}", end="")
        print()

    # Top feature across corpora
    print(f"\n{'Top feature per corpus:'}")
    for r in all_results:
        top = r['feature_importance'][0]
        print(f"  {r['corpus']}: {top['feature']} ({top['gain_pct']:.1f}%, {top['group']})")

    # Save JSON
    json_path = output_dir / "cross_corpus_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {json_path}")

    # Save CSV summary
    rows = []
    for r in all_results:
        row = {
            'corpus': r['corpus'],
            'n_samples': r['n_samples'],
            'n_classes': r['n_classes'],
            'accuracy': r['accuracy'],
            'f1_weighted': r['f1_weighted'],
            'f1_macro': r['f1_macro'],
            'p_value': r['p_value'],
        }
        for group_name in FEATURE_GROUPS:
            abl = r['ablation'].get(group_name, {})
            row[f'ablation_{group_name}'] = abl.get('delta_f1', 0)
        rows.append(row)

    csv_path = output_dir / "cross_corpus_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-Corpus SER Experiment")
    parser.add_argument("--download", action="store_true", help="Download all corpora")
    parser.add_argument("--extract", action="store_true", help="Extract features")
    parser.add_argument("--classify", action="store_true", help="Run classification + ablation")
    parser.add_argument("--all", action="store_true", help="Do everything")
    parser.add_argument("--gpu", action="store_true", help="Include wav2vec features (needs GPU)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per corpus (for quick test)")
    parser.add_argument("--corpora", nargs="+", default=["ravdess", "emodb", "cremad"],
                        help="Which corpora to use")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)

    args = parser.parse_args()

    if args.all:
        args.download = args.extract = args.classify = True

    np.random.seed(42)

    # --- Download ---
    if args.download:
        print("\n" + "=" * 60)
        print("STEP 1: DOWNLOADING CORPORA")
        print("=" * 60)
        if "ravdess" in args.corpora:
            download_ravdess(args.data_dir)
        if "emodb" in args.corpora:
            download_emodb(args.data_dir)
        if "cremad" in args.corpora:
            download_cremad(args.data_dir)

    # --- Load & Extract ---
    corpus_dfs = {}
    if args.extract or args.classify:
        print("\n" + "=" * 60)
        print("STEP 2: LOADING & EXTRACTING FEATURES")
        print("=" * 60)

        loaders = {
            "ravdess": ("RAVDESS", load_ravdess),
            "emodb": ("EmoDB", load_emodb),
            "cremad": ("CREMA-D", load_cremad),
        }

        for corpus_key in args.corpora:
            if corpus_key not in loaders:
                print(f"[warn] Unknown corpus: {corpus_key}")
                continue

            corpus_name, loader = loaders[corpus_key]
            try:
                samples = loader(args.data_dir)
                if not samples:
                    print(f"[{corpus_name}] No samples found, skipping")
                    continue

                cache_path = args.data_dir / f"features_{corpus_key}.csv"
                df = extract_corpus_features(
                    samples, corpus_name,
                    cache_path=cache_path if not args.extract else None,  # force re-extract if --extract
                    max_samples=args.max_samples,
                )
                if len(df) > 0:
                    corpus_dfs[corpus_name] = df
            except Exception as e:
                print(f"[{corpus_name}] Error: {e}")
                import traceback
                traceback.print_exc()

    # --- Load from cache if only --classify ---
    if args.classify and not args.extract:
        for corpus_key in args.corpora:
            cache_path = args.data_dir / f"features_{corpus_key}.csv"
            name_map = {"ravdess": "RAVDESS", "emodb": "EmoDB", "cremad": "CREMA-D"}
            corpus_name = name_map.get(corpus_key, corpus_key)
            if corpus_name not in corpus_dfs and cache_path.exists():
                corpus_dfs[corpus_name] = pd.read_csv(cache_path)
                print(f"[{corpus_name}] Loaded cached: {len(corpus_dfs[corpus_name])} samples")

    # --- Classify ---
    if args.classify and corpus_dfs:
        print("\n" + "=" * 60)
        print("STEP 3: CLASSIFICATION & ABLATION")
        print("=" * 60)

        all_results = []
        for corpus_name, df in corpus_dfs.items():
            try:
                results = classify_and_ablate(df, corpus_name)
                all_results.append(results)
            except Exception as e:
                print(f"[{corpus_name}] Classification error: {e}")
                import traceback
                traceback.print_exc()

        if all_results:
            cross_corpus_summary(all_results, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
