#!/usr/bin/env python3
"""
Deep Cross-Corpus SER Experiment for INTERSPEECH 2026.

"Do Voice Perturbation Metrics Still Matter?"
— The largest multi-corpus ablation study of phoniatric features for SER.

10 corpora, 5+ languages, 50K+ utterances.
3 feature levels: phoniatric (27) + eGeMAPS (88) + wav2vec 2.0 (768).
5 classifiers: XGBoost, SVM, RF, MLP, wav2vec fine-tuned.
Cross-corpus generalization protocol.

Usage:
  # Quick local test (RAVDESS + EmoDB only, small samples)
  python deep_ser_experiment.py --phase download --corpora ravdess emodb
  python deep_ser_experiment.py --phase extract --corpora ravdess emodb --max-per-class 50
  python deep_ser_experiment.py --phase classify --corpora ravdess emodb

  # Full GPU run (all corpora, all features)
  python deep_ser_experiment.py --phase all --gpu
  python deep_ser_experiment.py --phase all --gpu --wav2vec
"""

import argparse
import json
import os
import sys
import time
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "cache"


# ============================================================================
# UNIFIED EMOTION TAXONOMY
# ============================================================================
# Map all corpora to 5 basic emotions (Ekman subset) for fair comparison.
# angry, happy, sad, neutral, fear — present in most corpora.

UNIFIED_5 = {"angry", "happy", "sad", "neutral", "fear"}

CORPUS_CONFIGS = {
    # ── STRESS-TIMED ──────────────────────────────────────────
    "ravdess": {
        "name": "RAVDESS",
        "language": "English",
        "prosodic_type": "stress-timed",
        "type": "acted",
        "source": "zenodo",
        "url": "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
        "data_subdir": "ravdess",
        "emotion_map": {
            1: "neutral", 2: "neutral",  # calm→neutral
            3: "happy", 4: "sad", 5: "angry",
            6: "fear", 7: None, 8: None  # disgust, surprise → exclude
        },
        "parse_fn": "parse_ravdess",
    },
    "emodb": {
        "name": "EmoDB",
        "language": "German",
        "prosodic_type": "stress-timed",
        "type": "acted",
        "source": "direct",
        "url": "http://emodb.bilderbar.info/download/download.zip",
        "data_subdir": "emodb",
        "emotion_map": {
            "W": "angry", "L": None,  # boredom → exclude
            "E": None, "A": "fear",  # disgust → exclude
            "F": "happy", "T": "sad", "N": "neutral"
        },
        "parse_fn": "parse_emodb",
    },
    "cremad": {
        "name": "CREMA-D",
        "language": "English",
        "prosodic_type": "stress-timed",
        "type": "crowd-acted",
        "source": "huggingface",
        "hf_id": "ml-superb/crema_d",
        "data_subdir": "cremad",
        "emotion_map": {
            "ANG": "angry", "HAP": "happy", "SAD": "sad",
            "NEU": "neutral", "FEA": "fear", "DIS": None  # disgust → exclude
        },
        "parse_fn": "parse_cremad",
    },
    "crema_d_english": {
        "name": "CREMA-D",
        "language": "English",
        "prosodic_type": "stress-timed",
        "type": "crowd-acted",
        "source": "local",
        "data_subdir": "crema_d_english",
        "emotion_map": {
            "anger": "angry", "happy": "happy", "sadness": "sad",
            "sad": "sad", "neutral": "neutral", "fear": "fear",
            "angry": "angry", "happiness": "happy",
            "disgust": None, "surprise": None,
        },
        "parse_fn": "parse_prefix",
    },
    "savee": {
        "name": "SAVEE",
        "language": "English",
        "prosodic_type": "stress-timed",
        "type": "acted",
        "source": "huggingface",
        "hf_id": "Ar4ikov/savee",
        "data_subdir": "savee",
        "emotion_map": {
            "anger": "angry", "happiness": "happy", "sadness": "sad",
            "neutral": "neutral", "fear": "fear",
            "disgust": None, "surprise": None
        },
        "parse_fn": "parse_hf_generic",
    },
    "tess": {
        "name": "TESS",
        "language": "English",
        "prosodic_type": "stress-timed",
        "type": "acted",
        "source": "huggingface",
        "hf_id": "Ar4ikov/tess",
        "data_subdir": "tess",
        "emotion_map": {
            "angry": "angry", "happy": "happy", "sad": "sad",
            "neutral": "neutral", "fear": "fear",
            "disgust": None, "surprise": None, "pleasant_surprise": None
        },
        "parse_fn": "parse_hf_generic",
    },
    "aesdd": {
        "name": "AESDD",
        "language": "Greek",
        "prosodic_type": "stress-timed",
        "type": "acted",
        "source": "huggingface",
        "hf_id": "Ar4ikov/acted_emotional_speech_dynamic_database",
        "data_subdir": "aesdd",
        "emotion_map": {
            "anger": "angry", "happiness": "happy", "sadness": "sad",
            "fear": "fear", "disgust": None
        },
        "parse_fn": "parse_suffix",
    },
    "nemo_polish": {
        "name": "nEMO",
        "language": "Polish",
        "prosodic_type": "stress-timed",
        "type": "acted",
        "source": "local",
        "data_subdir": "nemo_polish",
        "emotion_map": {
            "angry": "angry", "happy": "happy", "sad": "sad",
            "neutral": "neutral", "fear": "fear",
            "anger": "angry", "happiness": "happy", "sadness": "sad",
            "disgust": None, "surprise": None,
        },
        "parse_fn": "parse_prefix",
    },
    "resd_russian": {
        "name": "RESD",
        "language": "Russian",
        "prosodic_type": "stress-timed",
        "type": "acted",
        "source": "local",
        "data_subdir": "resd_russian",
        "emotion_map": {
            "anger": "angry", "happy": "happy", "sad": "sad",
            "neutral": "neutral", "fear": "fear",
            "angry": "angry", "happiness": "happy", "sadness": "sad",
            "disgust": None, "surprise": None,
        },
        "parse_fn": "parse_prefix",
    },
    "dusha_russian": {
        "name": "DUSHA",
        "language": "Russian",
        "prosodic_type": "stress-timed",
        "type": "crowd-sourced",
        "source": "local",
        "data_subdir": "dusha_russian",
        "emotion_map": {
            "angry": "angry", "positive": "happy", "sad": "sad",
            "neutral": "neutral", "other": "fear",
            "anger": "angry", "happy": "happy", "fear": "fear",
        },
        "parse_fn": "parse_prefix",
    },
    "subesco_bengali": {
        "name": "SUBESCO",
        "language": "Bengali",
        "prosodic_type": "stress-timed",
        "type": "acted",
        "source": "local",
        "data_subdir": "subesco_bengali",
        "emotion_map": {
            "anger": "angry", "happy": "happy", "sad": "sad",
            "neutral": "neutral", "fear": "fear",
            "angry": "angry", "happiness": "happy", "sadness": "sad",
            "disgust": None, "surprise": None,
        },
        "parse_fn": "parse_prefix",
    },
    # ── TONAL ─────────────────────────────────────────────────
    "esd_zh": {
        "name": "ESD-ZH",
        "language": "Mandarin",
        "prosodic_type": "tonal",
        "type": "multi-speaker",
        "source": "local",
        "data_subdir": "esd_mandarin",
        "emotion_map": {
            "emo1": "angry", "emo4": "happy", "emo5": "neutral",
            "emo6": "sad", "emo7": None,  # surprise → exclude
        },
        "speaker_filter": lambda s: int(s) >= 11,  # 0011-0020 = Chinese
        "parse_fn": "parse_yukat237",
    },
    "esd": {
        "name": "ESD",
        "language": "English+Chinese",
        "prosodic_type": "stress-timed",
        "type": "multi-speaker",
        "source": "huggingface",
        "hf_id": "Emotech/esd",
        "data_subdir": "esd",
        "emotion_map": {
            "Angry": "angry", "Happy": "happy", "Sad": "sad",
            "Neutral": "neutral", "Surprise": None
        },
        "parse_fn": "parse_hf_generic",
    },
    "caves": {
        "name": "CAVES",
        "language": "Cantonese",
        "prosodic_type": "tonal",
        "type": "acted",
        "source": "local",
        "data_subdir": "caves_cantonese_flat",
        "emotion_map": {
            "angry": "angry", "happy": "happy", "sad": "sad",
            "neutral": "neutral", "fear": "fear",
            "disgust": None, "surprise": None,
        },
        "parse_fn": "parse_prefix",
    },
    "thai_ser": {
        "name": "Thai-SER",
        "language": "Thai",
        "prosodic_type": "tonal",
        "type": "acted",
        "source": "local",
        "data_subdir": "thai_ser",
        "emotion_map": {
            "Angry": "angry", "Happy": "happy", "Sad": "sad",
            "Neutral": "neutral", "Frustrated": None,
            "angry": "angry", "happy": "happy", "sad": "sad",
            "neutral": "neutral",
        },
        "parse_fn": "parse_prefix_ci",
    },
    # ── PITCH-ACCENT ──────────────────────────────────────────
    "jvnv_japanese": {
        "name": "JVNV",
        "language": "Japanese",
        "prosodic_type": "pitch-accent",
        "type": "acted",
        "source": "local",
        "data_subdir": "jvnv_japanese",
        "emotion_map": {
            "emo1": "angry", "emo3": "fear", "emo4": "happy",
            "emo5": "neutral", "emo6": "sad", "emo7": None,
        },
        "parse_fn": "parse_yukat237",
    },
    # ── SYLLABLE-TIMED ────────────────────────────────────────
    "cafe_french": {
        "name": "CaFE",
        "language": "French",
        "prosodic_type": "syllable-timed",
        "type": "acted",
        "source": "local",
        "data_subdir": "cafe_french",
        "emotion_map": {
            "anger": "angry", "happy": "happy", "sad": "sad",
            "neutral": "neutral", "fear": "fear",
            "angry": "angry", "happiness": "happy", "sadness": "sad",
            "joy": "happy", "surprise": None, "disgust": None,
        },
        "parse_fn": "parse_prefix",
    },
    "emozionalmente_italian": {
        "name": "Emozionalmente",
        "language": "Italian",
        "prosodic_type": "syllable-timed",
        "type": "acted",
        "source": "local",
        "data_subdir": "emozionalmente_italian",
        "emotion_map": {
            "anger": "angry", "happy": "happy", "sad": "sad",
            "neutral": "neutral", "fear": "fear",
            "angry": "angry", "happiness": "happy", "sadness": "sad",
            "joy": "happy", "disgust": None, "surprise": None,
        },
        "parse_fn": "parse_prefix",
    },
    "emovo": {
        "name": "EMOVO",
        "language": "Italian",
        "prosodic_type": "syllable-timed",
        "type": "acted",
        "source": "huggingface",
        "hf_id": "Ar4ikov/emovo",
        "data_subdir": "emovo",
        "emotion_map": {
            "anger": "angry", "joy": "happy", "sadness": "sad",
            "neutral": "neutral", "fear": "fear",
            "disgust": None, "surprise": None
        },
        "parse_fn": "parse_hf_generic",
    },
    "mesd_spanish": {
        "name": "MESD",
        "language": "Spanish",
        "prosodic_type": "syllable-timed",
        "type": "acted",
        "source": "local",
        "data_subdir": "mesd_spanish",
        "emotion_map": {
            "anger": "angry", "happy": "happy", "sad": "sad",
            "neutral": "neutral", "fear": "fear",
            "angry": "angry", "happiness": "happy", "sadness": "sad",
            "disgust": None, "surprise": None,
        },
        "parse_fn": "parse_prefix",
    },
    "turevdb_turkish": {
        "name": "TurEV-DB",
        "language": "Turkish",
        "prosodic_type": "syllable-timed",
        "type": "acted",
        "source": "local",
        "data_subdir": "turkish_turevdb/Sound Source",
        "emotion_map": {
            "angry": "angry", "calm": "neutral",
            "happy": "happy", "sad": "sad",
        },
        "parse_fn": "parse_folder_emotion",
    },
    "mder_arabic": {
        "name": "MDER-MA",
        "language": "Arabic-Moroccan",
        "prosodic_type": "stress-timed",
        "type": "acted",
        "source": "local",
        "data_subdir": "mder_moroccan_arabic",
        "emotion_map": {
            "angry": "angry", "happy": "happy", "sad": "sad",
            "neutral": "neutral",
        },
        "parse_fn": "parse_prefix",
    },
    "urdu_dataset": {
        "name": "URDU-Dataset",
        "language": "Urdu",
        "prosodic_type": "stress-timed",
        "type": "natural",
        "source": "local",
        "data_subdir": "urdu_dataset",
        "emotion_map": {
            "angry": "angry", "happy": "happy", "sad": "sad",
            "neutral": "neutral",
        },
        "parse_fn": "parse_folder_emotion",
    },
    "emouerj_portuguese": {
        "name": "emoUERJ",
        "language": "Portuguese-BR",
        "prosodic_type": "syllable-timed",
        "type": "acted",
        "source": "local",
        "data_subdir": "emouerj_portuguese/emoUERJ",
        "emotion_map": {
            "a": "angry", "h": "happy",
            "s": "sad", "n": "neutral",
        },
        "parse_fn": "parse_char_emotion",
        "emotion_char_pos": 3,
    },
    "indowavesentiment_indonesian": {
        "name": "IndoWaveSentiment",
        "language": "Indonesian",
        "prosodic_type": "syllable-timed",
        "type": "acted",
        "source": "local",
        "data_subdir": "indowavesentiment_indonesian",
        "emotion_map": {
            "angry": "angry", "happy": "happy", "sad": "sad",
            "neutral": "neutral",
        },
        "parse_fn": "parse_prefix",
    },
    "ased_amharic": {
        "name": "ASED",
        "language": "Amharic",
        "prosodic_type": "stress-timed",
        "type": "acted",
        "source": "local",
        "data_subdir": "ased_amharic",
        "emotion_map": {
            "01neutral": "neutral", "02fearful": "fear",
            "03happy": "happy", "04sad": "sad", "05angry": "angry",
        },
        "parse_fn": "parse_folder_emotion",
    },
    # ── TONAL (additional) ──────────────────────────────────────
    "visec_vietnamese": {
        "name": "VISEC",
        "language": "Vietnamese",
        "prosodic_type": "tonal",
        "type": "acted",
        "source": "local",
        "data_subdir": "visec_vietnamese/wav",
        "emotion_map": {"angry": "angry", "happy": "happy", "sad": "sad", "neutral": "neutral"},
        "parse_fn": "parse_visec",
    },
    "kazemotts_kazakh": {
        "name": "KazEmoTTS",
        "language": "Kazakh",
        "prosodic_type": "stress-timed",
        "type": "TTS",
        "source": "local",
        "data_subdir": "kazemotts_kazakh/EmoKaz",
        "emotion_map": {"angry": "angry", "happy": "happy", "sad": "sad", "neutral": "neutral", "fear": "fear"},
        "parse_fn": "parse_kazemotts",
    },
    "oreau_french": {
        "name": "Oreau",
        "language": "French",
        "prosodic_type": "syllable-timed",
        "type": "acted",
        "source": "local",
        "data_subdir": "oreau_french/wav",
        "emotion_map": {"anger": "angry", "happiness": "happy", "sadness": "sad", "neutral": "neutral", "fear": "fear"},
        "parse_fn": "parse_prefix",
    },
    # ── STØD (medium-low laryngeal) ─────────────────────────
    "emotale_danish": {
        "name": "EmoTale",
        "language": "Danish",
        "prosodic_type": "stress-timed",
        "type": "acted",
        "source": "local",
        "data_subdir": "emotale_danish",
        "emotion_map": {
            "A": "angry", "H": "happy", "S": "sad",
            "N": "neutral", "B": None,  # boredom → exclude
        },
        "parse_fn": "parse_emotale",
    },
    # ── 3-WAY LARYNGEAL (medium load) ──────────────────────
    "hikia_korean": {
        "name": "Hi,KIA",
        "language": "Korean",
        "prosodic_type": "stress-timed",
        "type": "acted",
        "source": "local",
        "data_subdir": "hikia_korean/HIKIA_NEW/wav",
        "emotion_map": {
            "a": "angry", "h": "happy",
            "s": "sad", "n": "neutral",
        },
        "parse_fn": "parse_hikia",
    },
}


# ============================================================================
# FEATURE GROUPS
# ============================================================================

PHONIATRIC_GROUPS = {
    "Perturbation": ["jitter", "shimmer", "hnr"],
    "Frequency": ["pitch_hz", "pitch_std", "voiced_ratio"],
    "MFCC": [f"mfcc_{i}" for i in range(13)],
    "Spectral": ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zcr"],
    "Formant": ["f1", "f2", "f3"],
    "Temporal": ["energy_db"],
}

ALL_PHONIATRIC = []
for feats in PHONIATRIC_GROUPS.values():
    ALL_PHONIATRIC.extend(feats)


# ============================================================================
# DOWNLOAD CORPORA
# ============================================================================

def download_corpus(corpus_key: str, data_dir: Path):
    """Download a single corpus."""
    cfg = CORPUS_CONFIGS[corpus_key]
    corpus_dir = data_dir / corpus_key
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Check if already has WAV files
    wavs = list(corpus_dir.rglob("*.wav"))
    if wavs:
        print(f"  [{cfg['name']}] Already have {len(wavs)} files")
        return

    source = cfg["source"]

    if source == "zenodo" or source == "direct":
        import urllib.request
        url = cfg["url"]
        zip_path = corpus_dir / f"{corpus_key}.zip"
        print(f"  [{cfg['name']}] Downloading from {url[:50]}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(corpus_dir)
            zip_path.unlink()
        except Exception as e:
            print(f"  [{cfg['name']}] Download error: {e}")

    elif source == "huggingface":
        try:
            from datasets import load_dataset
            import soundfile as sf

            hf_id = cfg["hf_id"]
            print(f"  [{cfg['name']}] Loading from HuggingFace: {hf_id}")
            ds = load_dataset(hf_id, trust_remote_code=True)

            # Flatten splits
            count = 0
            for split_name in ds:
                split = ds[split_name]
                for i, sample in enumerate(split):
                    # Get emotion label
                    label = None
                    for key in ["emotion", "label", "labels"]:
                        if key in sample:
                            label = sample[key]
                            break
                    if label is None:
                        continue

                    # Get audio
                    audio = sample.get("audio", sample.get("speech"))
                    if audio is None:
                        continue

                    if isinstance(audio, dict):
                        arr = np.array(audio["array"], dtype=np.float32)
                        sr = audio["sampling_rate"]
                    else:
                        continue

                    # Map emotion label
                    if isinstance(label, int):
                        # Some datasets use integer labels
                        label_names = split.features.get("emotion", split.features.get("label"))
                        if hasattr(label_names, "names"):
                            label = label_names.names[label]

                    label_str = str(label).lower()

                    wav_path = corpus_dir / f"{split_name}_{i:05d}_{label_str}.wav"
                    sf.write(str(wav_path), arr, sr)
                    count += 1

                    if count % 1000 == 0:
                        print(f"    Saved {count} files...")

            print(f"  [{cfg['name']}] Saved {count} files")

        except ImportError:
            print(f"  [{cfg['name']}] Need: pip install datasets soundfile")
        except Exception as e:
            print(f"  [{cfg['name']}] Error: {e}")

    wavs = list(corpus_dir.rglob("*.wav"))
    print(f"  [{cfg['name']}] Total: {len(wavs)} WAV files")


# ============================================================================
# PARSE CORPORA → (wav_path, emotion_label)
# ============================================================================

def parse_ravdess(corpus_dir: Path, emotion_map: dict) -> list:
    """RAVDESS: {mod}-{chan}-{emo}-{int}-{stmt}-{rep}-{actor}.wav"""
    samples = []
    for wav in sorted(corpus_dir.rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue
        if int(parts[0]) != 3:  # audio-only
            continue
        emo_code = int(parts[2])
        emotion = emotion_map.get(emo_code)
        if emotion:
            samples.append((wav, emotion))
    return samples


def parse_emodb(corpus_dir: Path, emotion_map: dict) -> list:
    """EmoDB: {speaker}{text}{emotion}.wav — emotion at position 5."""
    samples = []
    for wav in sorted(corpus_dir.rglob("*.wav")):
        name = wav.stem
        if len(name) < 6:
            continue
        emo_code = name[5]
        emotion = emotion_map.get(emo_code)
        if emotion:
            samples.append((wav, emotion))
    return samples


def parse_cremad(corpus_dir: Path, emotion_map: dict) -> list:
    """CREMA-D: {actor}_{sentence}_{EMO}_{intensity}.wav"""
    samples = []
    for wav in sorted(corpus_dir.rglob("*.wav")):
        parts = wav.stem.split("_")
        for p in parts:
            p_upper = p.upper()
            if p_upper in emotion_map:
                emotion = emotion_map[p_upper]
                if emotion:
                    samples.append((wav, emotion))
                break
        else:
            # Try lowercase match (our HF format: split_00001_angry.wav)
            for p in parts:
                if p.lower() in {"angry", "happy", "sad", "neutral", "fear"}:
                    samples.append((wav, p.lower()))
                    break
    return samples


def parse_hf_generic(corpus_dir: Path, emotion_map: dict, **kw) -> list:
    """Generic parser for HuggingFace-downloaded files: split_00001_label.wav"""
    samples = []
    for wav in sorted(corpus_dir.rglob("*.wav")):
        parts = wav.stem.split("_")
        label = parts[-1].lower()
        # Try direct match
        emotion = emotion_map.get(label)
        if emotion is None:
            # Try capitalized
            emotion = emotion_map.get(label.capitalize())
        if emotion is None:
            # Try the label as-is if it's in unified set
            if label in UNIFIED_5:
                emotion = label
        if emotion:
            samples.append((wav, emotion))
    return samples


def parse_prefix(corpus_dir: Path, emotion_map: dict, **kw) -> list:
    """Parser for {emotion}_{rest}.wav — emotion is first underscore-delimited token."""
    samples = []
    for wav in sorted(corpus_dir.rglob("*.wav")):
        label = wav.stem.split("_")[0].lower()
        emotion = emotion_map.get(label)
        if emotion is None:
            emotion = emotion_map.get(label.capitalize())
        if emotion is None and label in UNIFIED_5:
            emotion = label
        if emotion:
            samples.append((wav, emotion))
    return samples


def parse_char_emotion(corpus_dir: Path, emotion_map: dict, **kw) -> list:
    """Parser for {gender}{speakerID}{emotionChar}{utterance}.wav — emotion is single char at position 3."""
    pos = kw.get("emotion_char_pos", 3)
    samples = []
    for wav in sorted(corpus_dir.rglob("*.wav")):
        stem = wav.stem
        if len(stem) > pos:
            label = stem[pos].lower()
            emotion = emotion_map.get(label)
            if emotion:
                samples.append((wav, emotion))
    return samples


def parse_visec(corpus_dir: Path, emotion_map: dict, **kw) -> list:
    """Parser for VISEC Vietnamese — reads data.csv from parent dir."""
    import pandas as pd
    csv_path = corpus_dir.parent / "data.csv"
    if not csv_path.exists():
        csv_path = corpus_dir / "data.csv"
    samples = []
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            emo = emotion_map.get(row["emotion"])
            if emo:
                wav = corpus_dir / row["file"]
                if wav.exists():
                    samples.append((wav, emo))
    return samples


def parse_kazemotts(corpus_dir: Path, emotion_map: dict, **kw) -> list:
    """Parser for KazEmoTTS — files named {speakerID}_{emotion}_{number}.wav in nested dirs."""
    samples = []
    for wav in sorted(corpus_dir.rglob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) >= 3:
            label = parts[1].lower()  # emotion is second part: speakerID_emotion_number
            emotion = emotion_map.get(label)
            if emotion:
                samples.append((wav, emotion))
    return samples


def parse_hikia(corpus_dir: Path, emotion_map: dict, **kw) -> list:
    """Parser for Hi,KIA Korean — files named {G}{id}_S{scene}_{trial}_{emo_letter}.wav."""
    samples = []
    for wav in sorted(corpus_dir.rglob("*.wav")):
        label = wav.stem.split("_")[-1].lower()  # last char after last underscore
        emotion = emotion_map.get(label)
        if emotion:
            samples.append((wav, emotion))
    return samples


def parse_emovo(corpus_dir: Path, emotion_map: dict, **kw) -> list:
    """Parser for EMOVO Italian — files named {emotion}_{speaker}_{sentence}.wav."""
    samples = []
    for wav in sorted(corpus_dir.rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) >= 1:
            label = parts[0].lower()  # emotion is first part
            emotion = emotion_map.get(label)
            if emotion:
                samples.append((wav, emotion))
    return samples


def parse_emotale(corpus_dir: Path, emotion_map: dict, **kw) -> list:
    """Parser for EmoTale Danish — reads annotations.csv, WAVs in wav/ subfolder."""
    import pandas as pd
    csv_path = corpus_dir / "annotations.csv"
    samples = []
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            emo = emotion_map.get(row["gt_emotion"])
            if emo:
                wav = corpus_dir / "wav" / row["file"]
                if wav.exists():
                    samples.append((wav, emo))
    return samples


def parse_prefix_ci(corpus_dir: Path, emotion_map: dict, **kw) -> list:
    """Case-insensitive prefix parser: {Emotion}_{rest}.wav"""
    samples = []
    for wav in sorted(corpus_dir.rglob("*.wav")):
        label = wav.stem.split("_")[0]
        # Try as-is, lowercase, capitalized
        emotion = emotion_map.get(label) or emotion_map.get(label.lower()) or emotion_map.get(label.capitalize())
        if emotion is None and label.lower() in UNIFIED_5:
            emotion = label.lower()
        if emotion:
            samples.append((wav, emotion))
    return samples


def parse_suffix(corpus_dir: Path, emotion_map: dict, **kw) -> list:
    """Parser for {prefix}_{index}_{emotion}.wav — emotion is LAST token."""
    samples = []
    for wav in sorted(corpus_dir.rglob("*.wav")):
        label = wav.stem.split("_")[-1].lower()
        emotion = emotion_map.get(label)
        if emotion is None:
            emotion = emotion_map.get(label.capitalize())
        if emotion is None and label in UNIFIED_5:
            emotion = label
        if emotion:
            samples.append((wav, emotion))
    return samples


def parse_yukat237(corpus_dir: Path, emotion_map: dict, **kw) -> list:
    """Parser for yukat237 format: emo{N}_{speaker}_{index}.wav"""
    speaker_filter = kw.get('speaker_filter')
    samples = []
    for wav in sorted(corpus_dir.rglob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) < 2:
            continue
        emo_code = parts[0].lower()  # e.g., "emo1"
        speaker = parts[1] if len(parts) > 1 else "0"
        # Apply speaker filter if present (e.g., ESD Chinese only)
        if speaker_filter and not speaker_filter(speaker):
            continue
        emotion = emotion_map.get(emo_code)
        if emotion:
            samples.append((wav, emotion))
    return samples


def parse_folder_emotion(corpus_dir: Path, emotion_map: dict, **kw) -> list:
    """Parser for folder-based corpora: {corpus_dir}/{Emotion}/*.wav"""
    samples = []
    for subdir in sorted(corpus_dir.iterdir()):
        if not subdir.is_dir():
            continue
        label = subdir.name.lower()
        emotion = emotion_map.get(label) or emotion_map.get(subdir.name)
        if emotion is None and label in UNIFIED_5:
            emotion = label
        if emotion:
            for wav in sorted(subdir.glob("*.wav")):
                samples.append((wav, emotion))
    return samples


def load_corpus(corpus_key: str, data_dir: Path) -> list:
    """Load corpus: returns [(wav_path, emotion), ...]"""
    cfg = CORPUS_CONFIGS[corpus_key]
    # Use data_subdir if specified, otherwise fall back to corpus_key
    subdir = cfg.get("data_subdir", corpus_key)
    corpus_dir = data_dir / subdir
    emotion_map = cfg["emotion_map"]
    parse_fn = globals()[cfg["parse_fn"]]
    # Pass extra config kwargs (e.g., speaker_filter)
    extra = {k: v for k, v in cfg.items()
             if k not in ("name", "language", "prosodic_type", "type", "source",
                          "url", "hf_id", "data_subdir", "emotion_map", "parse_fn")}
    samples = parse_fn(corpus_dir, emotion_map, **extra)

    # Stats
    emotions = Counter(e for _, e in samples)
    print(f"  [{cfg['name']}] {len(samples)} samples: {dict(emotions)}")
    return samples


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_phoniatric_features(wav_path: Path, sr_target: int = 16000) -> Optional[dict]:
    """Extract 27 phoniatric features from audio file."""
    import librosa
    import parselmouth

    try:
        y, sr = librosa.load(str(wav_path), sr=sr_target, mono=True)
        if len(y) < sr_target * 0.3:
            return None

        features = {}

        # Praat Sound object
        snd = parselmouth.Sound(values=y, sampling_frequency=sr_target)

        # --- Frequency ---
        pitch_obj = parselmouth.praat.call(snd, "To Pitch", 0.0, 75, 600)
        pitch_values = pitch_obj.selected_array['frequency']
        pitch_voiced = pitch_values[pitch_values > 0]

        features['pitch_hz'] = float(np.mean(pitch_voiced)) if len(pitch_voiced) > 0 else 0.0
        features['pitch_std'] = float(np.std(pitch_voiced)) if len(pitch_voiced) > 1 else 0.0
        features['voiced_ratio'] = float(len(pitch_voiced) / max(len(pitch_values), 1))

        # --- Perturbation ---
        pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)...", 75, 600)

        jit = parselmouth.praat.call(pp, "Get jitter (local)...", 0, 0, 0.0001, 0.02, 1.3)
        shim = parselmouth.praat.call([snd, pp], "Get shimmer (local)...", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harm = parselmouth.praat.call(snd, "To Harmonicity (cc)...", 0.01, 75, 0.1, 1.0)
        hnr = parselmouth.praat.call(harm, "Get mean...", 0, 0)

        features['jitter'] = float(jit) if np.isfinite(jit) else 0.0
        features['shimmer'] = float(shim) if np.isfinite(shim) else 0.0
        features['hnr'] = float(hnr) if np.isfinite(hnr) else 0.0

        # --- MFCC ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr_target, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        for i in range(13):
            features[f'mfcc_{i}'] = float(mfcc_mean[i]) if np.isfinite(mfcc_mean[i]) else 0.0

        # --- Spectral ---
        sc = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_target)[0])
        sb = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr_target)[0])
        sr_feat = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr_target)[0])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])

        features['spectral_centroid'] = float(sc) if np.isfinite(sc) else 0.0
        features['spectral_bandwidth'] = float(sb) if np.isfinite(sb) else 0.0
        features['spectral_rolloff'] = float(sr_feat) if np.isfinite(sr_feat) else 0.0
        features['zcr'] = float(zcr) if np.isfinite(zcr) else 0.0

        # --- Formant ---
        formant_obj = parselmouth.praat.call(snd, "To Formant (burg)...", 0.0, 5, 5500, 0.025, 50)
        for i in range(1, 4):
            f_val = parselmouth.praat.call(formant_obj, "Get mean...", i, 0, 0, "hertz")
            features[f'f{i}'] = float(f_val) if np.isfinite(f_val) else 0.0

        # --- Temporal ---
        rms = librosa.feature.rms(y=y)[0]
        energy = float(np.mean(librosa.amplitude_to_db(rms + 1e-10)))
        features['energy_db'] = energy if np.isfinite(energy) else -60.0

        return features

    except Exception as e:
        return None


def extract_wav2vec_embeddings(wav_paths: list, batch_size: int = 32) -> np.ndarray:
    """Extract wav2vec 2.0 embeddings on GPU. Returns (N, 768) array."""
    import torch
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    import librosa

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [wav2vec] Device: {device}")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model.eval()

    embeddings = []
    for i in range(0, len(wav_paths), batch_size):
        batch_paths = wav_paths[i:i+batch_size]
        batch_audio = []
        for p in batch_paths:
            y, _ = librosa.load(str(p), sr=16000, mono=True)
            # Pad/truncate to 5 seconds
            target_len = 16000 * 5
            if len(y) > target_len:
                y = y[:target_len]
            else:
                y = np.pad(y, (0, max(0, target_len - len(y))))
            batch_audio.append(y)

        inputs = processor(batch_audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pool over time dimension
            hidden = outputs.last_hidden_state  # (B, T, 768)
            pooled = hidden.mean(dim=1)  # (B, 768)
            embeddings.append(pooled.cpu().numpy())

        if (i + batch_size) % 500 < batch_size:
            print(f"    [wav2vec] {min(i+batch_size, len(wav_paths))}/{len(wav_paths)}")

    return np.vstack(embeddings)


def extract_all_features(
    corpus_key: str,
    samples: list,
    data_dir: Path,
    max_per_class: Optional[int] = None,
    use_wav2vec: bool = False,
) -> pd.DataFrame:
    """Extract features for a corpus. Returns DataFrame."""

    cfg = CORPUS_CONFIGS[corpus_key]
    cache_path = CACHE_DIR / f"{corpus_key}_phoniatric.csv"

    # Balance classes if needed
    if max_per_class:
        by_emotion = defaultdict(list)
        for path, emo in samples:
            by_emotion[emo].append((path, emo))
        balanced = []
        for emo, items in by_emotion.items():
            np.random.shuffle(items)
            balanced.extend(items[:max_per_class])
        samples = balanced
        print(f"  [{cfg['name']}] Balanced to {len(samples)} ({max_per_class}/class)")

    # Check cache
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        if len(df) >= len(samples) * 0.8:  # allow 20% tolerance
            print(f"  [{cfg['name']}] Using cached features ({len(df)} samples)")
            return df

    # Extract phoniatric features
    rows = []
    n = len(samples)
    t0 = time.time()
    for i, (wav_path, emotion) in enumerate(samples):
        feats = extract_phoniatric_features(wav_path)
        if feats:
            feats['emotion'] = emotion
            feats['corpus'] = cfg['name']
            feats['language'] = cfg['language']
            feats['file'] = wav_path.name
            rows.append(feats)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"    [{cfg['name']}] {i+1}/{n} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    df = pd.DataFrame(rows)

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  [{cfg['name']}] Extracted {len(df)}/{n} samples in {time.time()-t0:.0f}s")

    # wav2vec embeddings (GPU only)
    if use_wav2vec:
        wav_paths = [p for p, _ in samples[:len(df)]]  # match extracted
        w2v_cache = CACHE_DIR / f"{corpus_key}_wav2vec.npy"
        if w2v_cache.exists():
            w2v = np.load(w2v_cache)
            print(f"  [{cfg['name']}] wav2vec cached ({w2v.shape})")
        else:
            print(f"  [{cfg['name']}] Extracting wav2vec embeddings...")
            w2v = extract_wav2vec_embeddings(wav_paths)
            np.save(w2v_cache, w2v)
            print(f"  [{cfg['name']}] wav2vec: {w2v.shape}")

        # Add to DataFrame
        for i in range(w2v.shape[1]):
            df[f'w2v_{i}'] = w2v[:len(df), i]

    return df


# ============================================================================
# CLASSIFICATION ENGINE
# ============================================================================

def run_classification(
    df: pd.DataFrame,
    feature_cols: list,
    label_col: str = 'emotion',
    n_folds: int = 5,
    model_type: str = 'xgboost',
    name: str = '',
) -> dict:
    """Run classification with cross-validation. Returns results dict."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col].values)
    classes = le.classes_

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_pred = np.zeros_like(y)

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        if model_type == 'xgboost':
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric='mlogloss',
                random_state=42, verbosity=0,
            )
            clf.fit(X[tr], y[tr])
            y_pred[te] = clf.predict(X[te])

        elif model_type == 'svm':
            from sklearn.svm import SVC
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])
            clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
            clf.fit(X_tr, y[tr])
            y_pred[te] = clf.predict(X_te)

        elif model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            )
            clf.fit(X[tr], y[tr])
            y_pred[te] = clf.predict(X[te])

        elif model_type == 'mlp':
            from sklearn.neural_network import MLPClassifier
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 128), max_iter=300,
                random_state=42, early_stopping=True,
            )
            clf.fit(X_tr, y[tr])
            y_pred[te] = clf.predict(X_te)

    acc = accuracy_score(y, y_pred)
    f1_w = f1_score(y, y_pred, average='weighted')
    f1_m = f1_score(y, y_pred, average='macro')
    report = classification_report(y, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    # Feature importance (XGBoost only)
    feat_imp = None
    if model_type == 'xgboost':
        clf_full = __import__('xgboost').XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=42, verbosity=0,
        )
        clf_full.fit(X, y)
        imp = clf_full.feature_importances_
        feat_imp = sorted(zip(feature_cols, imp.tolist()), key=lambda x: x[1], reverse=True)

    return {
        'name': name,
        'model': model_type,
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'n_classes': len(classes),
        'classes': list(classes),
        'accuracy': round(acc, 4),
        'f1_weighted': round(f1_w, 4),
        'f1_macro': round(f1_m, 4),
        'per_class': {c: {k: round(v, 3) for k, v in report[c].items()} for c in classes},
        'confusion_matrix': cm.tolist(),
        'feature_importance': feat_imp[:20] if feat_imp else None,
    }


def run_ablation(
    df: pd.DataFrame,
    base_f1: float,
    n_folds: int = 5,
) -> dict:
    """Run group ablation. Returns {group: {f1, delta}}."""
    feature_cols = [c for c in ALL_PHONIATRIC if c in df.columns]
    results = {}

    for group_name, group_feats in PHONIATRIC_GROUPS.items():
        remaining = [f for f in feature_cols if f not in group_feats]
        if not remaining:
            continue
        res = run_classification(df, remaining, n_folds=n_folds, model_type='xgboost',
                                 name=f"ablation_no_{group_name}")
        delta = res['f1_weighted'] - base_f1
        results[group_name] = {
            'f1_without': res['f1_weighted'],
            'delta_f1': round(delta, 4),
            'n_removed': len(group_feats),
        }

    return results


def run_permutation_test(df: pd.DataFrame, observed_acc: float, n_perm: int = 100) -> float:
    """Permutation test for statistical significance."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier

    feature_cols = [c for c in ALL_PHONIATRIC if c in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X)
    y = pd.factorize(df['emotion'])[0]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    perm_accs = []

    for pi in range(n_perm):
        y_perm = np.random.permutation(y)
        fold_accs = []
        for tr, te in skf.split(X, y_perm):
            clf = XGBClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                use_label_encoder=False, eval_metric='mlogloss',
                random_state=pi, verbosity=0,
            )
            clf.fit(X[tr], y_perm[tr])
            fold_accs.append(accuracy_score(y_perm[te], clf.predict(X[te])))
        perm_accs.append(np.mean(fold_accs))

    p_value = np.mean([pa >= observed_acc for pa in perm_accs])
    return p_value


# ============================================================================
# CROSS-CORPUS GENERALIZATION
# ============================================================================

def cross_corpus_eval(all_dfs: dict, feature_cols: list) -> list:
    """Train on each corpus, test on all others."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import f1_score
    from xgboost import XGBClassifier

    # Find common emotions across all corpora
    all_emotions = set.intersection(*[set(df['emotion'].unique()) for df in all_dfs.values()])
    print(f"  Common emotions across all corpora: {all_emotions}")

    results = []
    for train_name, train_df in all_dfs.items():
        train_df_filt = train_df[train_df['emotion'].isin(all_emotions)]
        le = LabelEncoder()
        le.fit(list(all_emotions))

        X_train = train_df_filt[feature_cols].values.astype(np.float32)
        X_train = np.nan_to_num(X_train)
        y_train = le.transform(train_df_filt['emotion'])

        clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=42, verbosity=0,
        )
        clf.fit(X_train, y_train)

        for test_name, test_df in all_dfs.items():
            test_df_filt = test_df[test_df['emotion'].isin(all_emotions)]
            X_test = test_df_filt[feature_cols].values.astype(np.float32)
            X_test = np.nan_to_num(X_test)
            y_test = le.transform(test_df_filt['emotion'])

            y_pred = clf.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results.append({
                'train': train_name,
                'test': test_name,
                'f1_weighted': round(f1, 4),
                'n_train': len(X_train),
                'n_test': len(X_test),
            })

    return results


# ============================================================================
# F0 BANDWIDTH HYPOTHESIS — PROSODIC TYPE ANALYSIS
# ============================================================================

PROSODIC_ORDER = ["tonal", "pitch-accent", "syllable-timed", "stress-timed"]

def get_prosodic_type(corpus_key: str) -> str:
    """Get prosodic type for a corpus."""
    return CORPUS_CONFIGS.get(corpus_key, {}).get("prosodic_type", "unknown")


def run_f0_bandwidth_analysis(all_results: dict, all_dfs: dict) -> dict:
    """
    Core analysis for F0 Bandwidth Hypothesis.

    Groups corpora by prosodic type and compares:
    - Feature importance of F0 (Frequency group) vs Perturbation group
    - Ablation deltas: which group matters more per prosodic type?

    Hypothesis: In tonal languages, F0 importance is LOW (occupied by lexical tone),
    so emotion info "spills" into perturbation metrics.
    """
    from scipy import stats

    analysis = {"by_prosodic_type": {}, "statistical_tests": {}}

    # Collect ablation deltas grouped by prosodic type
    type_ablations = defaultdict(lambda: {"frequency": [], "perturbation": [], "mfcc": [],
                                           "corpora": [], "f1_scores": []})

    for corpus_name, cr in all_results.items():
        if corpus_name == 'cross_corpus':
            continue
        # Find corpus key by name
        ckey = None
        for k, v in CORPUS_CONFIGS.items():
            if v["name"] == corpus_name:
                ckey = k
                break
        if not ckey:
            continue

        ptype = get_prosodic_type(ckey)
        if ptype == "unknown":
            continue

        abl = cr.get("ablation", {})
        if not abl:
            continue

        entry = type_ablations[ptype]
        entry["corpora"].append(corpus_name)
        entry["f1_scores"].append(cr.get("xgboost_f1w", 0))

        for group in ["Frequency", "Perturbation", "MFCC", "Spectral", "Formant", "Temporal"]:
            if group in abl:
                entry[group.lower()].append(abl[group]["delta_f1"])

        # Feature importance (top features from XGBoost)
        fi = cr.get("feature_importance", [])
        if fi:
            freq_feats = set(PHONIATRIC_GROUPS["Frequency"])
            pert_feats = set(PHONIATRIC_GROUPS["Perturbation"])
            freq_imp = sum(imp for feat, imp in fi if feat in freq_feats)
            pert_imp = sum(imp for feat, imp in fi if feat in pert_feats)
            entry.setdefault("freq_importance", []).append(freq_imp)
            entry.setdefault("pert_importance", []).append(pert_imp)

    # Summarize per prosodic type
    for ptype in PROSODIC_ORDER:
        if ptype not in type_ablations:
            continue
        d = type_ablations[ptype]
        summary = {
            "n_corpora": len(d["corpora"]),
            "corpora": d["corpora"],
            "mean_f1": round(np.mean(d["f1_scores"]), 4) if d["f1_scores"] else None,
        }
        for group in ["frequency", "perturbation", "mfcc", "spectral", "formant", "temporal"]:
            vals = d.get(group, [])
            if vals:
                summary[f"ablation_{group}_mean"] = round(np.mean(vals), 4)
                summary[f"ablation_{group}_std"] = round(np.std(vals), 4)
                summary[f"ablation_{group}_values"] = [round(v, 4) for v in vals]

        if d.get("freq_importance"):
            summary["freq_importance_mean"] = round(np.mean(d["freq_importance"]), 4)
        if d.get("pert_importance"):
            summary["pert_importance_mean"] = round(np.mean(d["pert_importance"]), 4)

        analysis["by_prosodic_type"][ptype] = summary

    # === PRIMARY: 2-way contrast (tonal vs non-tonal) ===
    # Non-tonal includes pitch-accent, stress-timed, syllable-timed
    tonal_data = type_ablations.get("tonal", {})
    nontonal_data = defaultdict(list)
    nontonal_corpora = []
    for ptype in ["pitch-accent", "stress-timed", "syllable-timed"]:
        d = type_ablations.get(ptype, {})
        for key in ["frequency", "perturbation", "mfcc", "spectral", "formant", "temporal"]:
            nontonal_data[key].extend(d.get(key, []))
        nontonal_corpora.extend(d.get("corpora", []))
        nontonal_data["freq_importance"].extend(d.get("freq_importance", []))
        nontonal_data["pert_importance"].extend(d.get("pert_importance", []))

    analysis["two_way"] = {
        "tonal": {"n_corpora": len(tonal_data.get("corpora", [])),
                  "corpora": tonal_data.get("corpora", [])},
        "non_tonal": {"n_corpora": len(nontonal_corpora),
                      "corpora": nontonal_corpora},
    }

    def cohens_d(a, b):
        """Effect size: Cohen's d."""
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return None
        pooled_std = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
        if pooled_std == 0:
            return 0.0
        return (np.mean(a) - np.mean(b)) / pooled_std

    for group in ["frequency", "perturbation", "mfcc"]:
        tonal_vals = tonal_data.get(group, [])
        nontonal_vals = nontonal_data.get(group, [])
        if len(tonal_vals) >= 2 and len(nontonal_vals) >= 2:
            u_stat, p_val = stats.mannwhitneyu(tonal_vals, nontonal_vals, alternative='two-sided')
            d = cohens_d(tonal_vals, nontonal_vals)
            analysis["statistical_tests"][f"2way_{group}_tonal_vs_nontonal"] = {
                "test": "Mann-Whitney U (PRIMARY)",
                "tonal_mean": round(np.mean(tonal_vals), 4),
                "nontonal_mean": round(np.mean(nontonal_vals), 4),
                "tonal_values": [round(v, 4) for v in tonal_vals],
                "nontonal_values": [round(v, 4) for v in nontonal_vals],
                "U": round(u_stat, 2),
                "p_value": round(p_val, 4),
                "cohens_d": round(d, 3) if d is not None else None,
                "significant": p_val < 0.05,
            }

    # Feature importance comparison (2-way)
    for imp_key in ["freq_importance", "pert_importance"]:
        tonal_imp = tonal_data.get(imp_key, [])
        nontonal_imp = nontonal_data.get(imp_key, [])
        if len(tonal_imp) >= 2 and len(nontonal_imp) >= 2:
            u_stat, p_val = stats.mannwhitneyu(tonal_imp, nontonal_imp, alternative='two-sided')
            analysis["statistical_tests"][f"2way_{imp_key}"] = {
                "test": "Mann-Whitney U",
                "tonal_mean": round(np.mean(tonal_imp), 4),
                "nontonal_mean": round(np.mean(nontonal_imp), 4),
                "U": round(u_stat, 2),
                "p_value": round(p_val, 4),
                "significant": p_val < 0.05,
            }

    # === SECONDARY: 4-way Kruskal-Wallis ===
    for group in ["frequency", "perturbation", "mfcc"]:
        kw_groups = []
        kw_labels = []
        for ptype in PROSODIC_ORDER:
            vals = type_ablations.get(ptype, {}).get(group, [])
            if vals:
                kw_groups.append(vals)
                kw_labels.append(ptype)
        if len(kw_groups) >= 3:
            h_stat, p_val = stats.kruskal(*kw_groups)
            analysis["statistical_tests"][f"4way_{group}_kruskal_wallis"] = {
                "test": "Kruskal-Wallis H (SECONDARY)",
                "groups": {pt: [round(v, 4) for v in type_ablations.get(pt, {}).get(group, [])]
                           for pt in kw_labels},
                "H": round(h_stat, 2),
                "p_value": round(p_val, 4),
                "significant": p_val < 0.05,
            }

    # Print summary
    print(f"\n{'='*60}")
    print("F0 BANDWIDTH HYPOTHESIS — RESULTS")
    print(f"{'='*60}")

    # 2-way summary
    print(f"\n  ── PRIMARY: TONAL vs NON-TONAL ──")
    for group in ["frequency", "perturbation", "mfcc"]:
        t = analysis["statistical_tests"].get(f"2way_{group}_tonal_vs_nontonal")
        if t:
            sig = "***" if t["significant"] else "n.s."
            d_str = f"d={t['cohens_d']:.2f}" if t["cohens_d"] else ""
            print(f"    Ablation -{group:>12}: tonal={t['tonal_mean']:+.4f}  non-tonal={t['nontonal_mean']:+.4f}  p={t['p_value']:.4f} {sig}  {d_str}")

    # 4-way detail
    print(f"\n  ── SECONDARY: 4-WAY PROSODIC TYPE ──")
    for ptype in PROSODIC_ORDER:
        s = analysis["by_prosodic_type"].get(ptype)
        if not s:
            continue
        print(f"\n  {ptype.upper()} ({s['n_corpora']} corpora: {', '.join(s['corpora'])})")
        print(f"    Mean F1 = {s['mean_f1']}")
        for group in ["frequency", "perturbation", "mfcc"]:
            m = s.get(f"ablation_{group}_mean")
            if m is not None:
                print(f"    Ablation -{group:>12}: ΔF1 = {m:+.4f} (±{s.get(f'ablation_{group}_std', 0):.4f})")
        fi_f = s.get("freq_importance_mean")
        fi_p = s.get("pert_importance_mean")
        if fi_f is not None:
            print(f"    Feature importance: Frequency={fi_f:.4f}  Perturbation={fi_p:.4f}")

    return analysis


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deep Cross-Corpus SER Experiment")
    parser.add_argument("--phase", required=True,
                        choices=["download", "extract", "classify", "f0", "all"])
    parser.add_argument("--corpora", nargs="+", default=list(CORPUS_CONFIGS.keys()))
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--wav2vec", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--models", nargs="+", default=["xgboost", "svm", "rf"])

    args = parser.parse_args()
    np.random.seed(42)

    do_download = args.phase in ("download", "all")
    do_extract = args.phase in ("extract", "all")
    do_classify = args.phase in ("classify", "f0", "all")
    do_f0 = args.phase in ("f0", "all")

    # ---- DOWNLOAD ----
    if do_download:
        print("\n" + "=" * 60)
        print("PHASE 1: DOWNLOAD CORPORA")
        print("=" * 60)
        for ck in args.corpora:
            if ck in CORPUS_CONFIGS:
                download_corpus(ck, args.data_dir)

    # ---- EXTRACT ----
    all_dfs = {}
    if do_extract or do_classify:
        print("\n" + "=" * 60)
        print("PHASE 2: FEATURE EXTRACTION")
        print("=" * 60)
        for ck in args.corpora:
            if ck not in CORPUS_CONFIGS:
                continue
            try:
                samples = load_corpus(ck, args.data_dir)
                if not samples:
                    continue
                df = extract_all_features(
                    ck, samples, args.data_dir,
                    max_per_class=args.max_per_class,
                    use_wav2vec=args.wav2vec,
                )
                if len(df) > 0:
                    all_dfs[CORPUS_CONFIGS[ck]['name']] = df
            except Exception as e:
                print(f"  [{ck}] Error: {e}")
                import traceback; traceback.print_exc()

    # Load from cache if only classify
    if do_classify and not do_extract:
        for ck in args.corpora:
            if ck not in CORPUS_CONFIGS:
                continue
            name = CORPUS_CONFIGS[ck]['name']
            if name not in all_dfs:
                cache = CACHE_DIR / f"{ck}_phoniatric.csv"
                if cache.exists():
                    all_dfs[name] = pd.read_csv(cache)
                    print(f"  [{name}] Loaded cache: {len(all_dfs[name])}")

    # ---- CLASSIFY ----
    if do_classify and all_dfs:
        print("\n" + "=" * 60)
        print("PHASE 3: CLASSIFICATION & ABLATION")
        print("=" * 60)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Load existing results to MERGE (not overwrite)
        existing_path = args.output_dir / "experiment_results.json"
        if existing_path.exists():
            with open(existing_path) as f:
                all_results = json.load(f)
            print(f"  Loaded {len(all_results)} existing results from {existing_path}")
        else:
            all_results = {}

        for corpus_name, df in all_dfs.items():
            print(f"\n{'─'*60}")
            print(f"  CORPUS: {corpus_name} ({len(df)} samples)")
            print(f"{'─'*60}")

            # Find config for this corpus to get metadata
            _cfg_key = next((k for k, v in CORPUS_CONFIGS.items() if v["name"] == corpus_name), None)
            _cfg = CORPUS_CONFIGS.get(_cfg_key, {}) if _cfg_key else {}
            corpus_results = {
                'corpus': corpus_name,
                'language': df['language'].iloc[0] if 'language' in df else _cfg.get('language', '?'),
                'prosodic_type': _cfg.get('prosodic_type', '?'),
                'data_type': _cfg.get('type', '?'),
                'source': _cfg.get('source', '?'),
            }
            feature_cols = [c for c in ALL_PHONIATRIC if c in df.columns]

            # Multiple models
            for model_type in args.models:
                print(f"\n  Model: {model_type}")
                res = run_classification(
                    df, feature_cols, model_type=model_type,
                    name=f"{corpus_name}_{model_type}"
                )
                corpus_results[f'{model_type}_acc'] = res['accuracy']
                corpus_results[f'{model_type}_f1w'] = res['f1_weighted']
                corpus_results[f'{model_type}_f1m'] = res['f1_macro']
                corpus_results[f'{model_type}_per_class'] = res['per_class']
                corpus_results[f'{model_type}_cm'] = res['confusion_matrix']

                if res['feature_importance']:
                    corpus_results['feature_importance'] = res['feature_importance']

                print(f"    Acc={res['accuracy']:.3f}  F1w={res['f1_weighted']:.3f}  F1m={res['f1_macro']:.3f}")

            # Ablation (XGBoost only)
            print(f"\n  Ablation study:")
            base_f1 = corpus_results.get('xgboost_f1w', 0)
            abl = run_ablation(df, base_f1)
            corpus_results['ablation'] = abl
            for g, r in sorted(abl.items(), key=lambda x: x[1]['delta_f1']):
                print(f"    Remove {g:>15}: ΔF1 = {r['delta_f1']:+.4f}")

            # Permutation test
            print(f"\n  Permutation test (100 iter)...")
            p_val = run_permutation_test(df, corpus_results.get('xgboost_acc', 0))
            corpus_results['p_value'] = f"< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
            print(f"    p-value: {corpus_results['p_value']}")

            all_results[corpus_name] = corpus_results

        # ---- CROSS-CORPUS ----
        if len(all_dfs) > 1:
            print(f"\n{'='*60}")
            print("PHASE 4: CROSS-CORPUS GENERALIZATION")
            print(f"{'='*60}")

            feature_cols = [c for c in ALL_PHONIATRIC if c in list(all_dfs.values())[0].columns]
            cross_results = cross_corpus_eval(all_dfs, feature_cols)

            # Print matrix
            corpus_names = list(all_dfs.keys())
            print(f"\n  {'Train↓ Test→':<14}", end="")
            for cn in corpus_names:
                print(f"  {cn:>10}", end="")
            print()
            for train_cn in corpus_names:
                print(f"  {train_cn:<14}", end="")
                for test_cn in corpus_names:
                    r = next((x for x in cross_results
                              if x['train'] == train_cn and x['test'] == test_cn), None)
                    if r:
                        f1 = r['f1_weighted']
                        marker = "*" if train_cn == test_cn else " "
                        print(f"  {f1:>9.3f}{marker}", end="")
                    else:
                        print(f"  {'---':>10}", end="")
                print()

            all_results['cross_corpus'] = cross_results

        # ---- F0 BANDWIDTH ANALYSIS ----
        if do_f0 and len(all_results) > 1:
            f0_analysis = run_f0_bandwidth_analysis(all_results, all_dfs)
            all_results['f0_bandwidth_hypothesis'] = f0_analysis

        # ---- SAVE ----
        output_path = args.output_dir / "experiment_results.json"
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[SAVED] {output_path}")

        # Summary table with prosodic_type
        summary_path = args.output_dir / "summary.csv"
        rows = []
        for cn, cr in all_results.items():
            if cn in ('cross_corpus', 'f0_bandwidth_hypothesis'):
                continue
            # Find config metadata
            _ck = next((k for k, v in CORPUS_CONFIGS.items() if v["name"] == cn), None)
            _cc = CORPUS_CONFIGS.get(_ck, {}) if _ck else {}
            row = {
                'corpus': cn,
                'language': cr.get('language', _cc.get('language', '?')),
                'prosodic_type': cr.get('prosodic_type', _cc.get('prosodic_type', '?')),
                'data_type': cr.get('data_type', _cc.get('type', '?')),
                'source': cr.get('source', _cc.get('source', '?')),
            }
            for mt in args.models:
                row[f'{mt}_f1w'] = cr.get(f'{mt}_f1w', '')
            if 'ablation' in cr:
                for g, a in cr['ablation'].items():
                    row[f'abl_{g}'] = a['delta_f1']
            rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"[SAVED] {summary_path}")

    print("\n✓ EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
