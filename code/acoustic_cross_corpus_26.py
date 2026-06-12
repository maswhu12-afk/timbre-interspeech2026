#!/usr/bin/env python3
"""Exp 1: acoustic 26x26 cross-corpus LogReg transfer matrix.

Reproduces data/cpu_acoustic_cross_26_logreg.json (grand avg 0.252, Table 1)
from the per-corpus 27-dim feature CSVs produced by deep_ser_experiment.py
(per-class-balanced, max 500 utterances per emotion; sizes as in Table S1).

Protocol (matches the original February 2026 run exactly):
  - raw feature values, NO normalization (no test-corpus statistics)
  - LogisticRegression(L2, C=1.0, lbfgs, max_iter=1000, random_state=42)
  - weighted F1 on the full test corpus

Usage: python3 acoustic_cross_corpus_26.py <features_dir> [out.json]
  <features_dir> contains one CSV per corpus: 27 feature columns
  (pitch_hz..rms/energy_db) + 'emotion'.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

CANON = (
    ["pitch_hz", "pitch_std", "voiced_ratio", "jitter", "shimmer", "hnr"]
    + [f"mfcc_{i}" for i in range(13)]
    + ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zcr",
       "f1", "f2", "f3", "rms"]
)


def main():
    feat_dir = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("cpu_acoustic_cross_26_logreg.json")
    data = {}
    for f in sorted(feat_dir.glob("*.csv")):
        df = pd.read_csv(f)
        if "rms" not in df.columns:
            df = df.rename(columns={"energy_db": "rms"})
        X = np.nan_to_num(df[CANON].values.astype(float))
        data[f.stem] = (X, df.emotion.values)
    corpora = sorted(data)
    res = {}
    for src in corpora:
        Xs, ys = data[src]
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(Xs, ys)
        scores = {}
        for tgt in corpora:
            if tgt == src:
                scores[tgt] = None
                continue
            Xt, yt = data[tgt]
            scores[tgt] = float(f1_score(yt, clf.predict(Xt), average="weighted"))
        vals = [v for v in scores.values() if v is not None]
        res[src] = {"avg_f1": round(float(np.mean(vals)), 4), "scores": scores}
    allv = [v for s in res.values() for v in s["scores"].values() if v is not None]
    print(f"grand average F1 = {np.mean(allv):.5f}  (paper: 0.252)")
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
