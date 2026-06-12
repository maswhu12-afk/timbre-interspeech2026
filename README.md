# TIMBRE: Layer-Wise Cross-Lingual Speech Emotion Recognition Across 49 Layers and 26 Corpora

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20584918.svg)](https://doi.org/10.5281/zenodo.20584918)

Code and data to reproduce the results of the Interspeech 2026 paper *TIMBRE* (#579).
Archived release (citable): **https://doi.org/10.5281/zenodo.20584918**

The study probes all 49 layers of `wav2vec2-xls-r-1b` for cross-lingual speech
emotion transfer across 26 corpora (23 languages, 14 families), a 49×26×26
transfer tensor of 31,850 cross-corpus experiments, and finds the **CRIS layer**
(Cross-lingual Representational Integration Stratum) at layer 15.

## Quick reproduction (no GPU, seconds)

```bash
pip install numpy scipy matplotlib
python3 reproduce.py
```

This reads `data/` and reproduces the headline numbers from the included data: the
layer curve and CRIS L15 peak, within/between-type & -family contrasts with
corpus-clustered bootstrap CIs, typological advantages, incoming transfer, the
wav2vec2 mean-pool cross-corpus average (0.355), and the acoustic-feature ablation
(Section 4.4); it regenerates `fig1_layer_curve.pdf` and `fig2_typological_profiles.pdf`.

The acoustic-feature cross-corpus matrix (Exp 1, grand 0.252) is included as
`data/cpu_acoustic_cross_26_logreg.json`; the Table 1 cells (incoming transfer
per test corpus, wins 3/23) derive from it together with the wav2vec2 matrix.
Protocol: raw 27-dim feature values (no normalization), LogisticRegression
(L2, C=1.0, lbfgs, max_iter=1000, random_state=42), weighted F1 : see
`code/acoustic_cross_corpus_26.py`. To regenerate from audio, run
`code/deep_ser_experiment.py` on the corpora (audio not redistributed : see
`CORPUS_SOURCES.md`).

## What's included

```
reproduce.py                     one-command reproduction from the included data
data/
  gpu_cross_lw_cross_lw_lr_26.json    the 49×26×26 layer-wise transfer tensor (main result)
  w2v2_meanpool_cross_corpus_26.json  wav2vec2 mean-pool cross-corpus matrix (Exp 2, 0.355)
  w2v2_meanpool_cross_corpus_26_early_extraction.json  superseded Exp 2 matrix from an
                                      early extraction pass (grand 0.270); kept for provenance
  cpu_acoustic_cross_26_logreg.json   acoustic 26×26 cross-corpus matrix (Exp 1, 0.252)
  summary.csv                         per-corpus 6-group acoustic ablation (25 corpora)
  master_26_corpora.json              per-corpus metadata (language, elicitation, size, type)
code/                            full pipeline that produced data/ (needs GPU + corpora)
  layerwise_probe.py                 extract wav2vec2 layer embeddings + probe
  cross_corpus_ser.py                cross-corpus transfer experiments
  acoustic_cross_corpus_26.py        Exp 1 matrix from per-corpus feature CSVs
  deep_ser_experiment.py             per-corpus acoustic features + ablation
  statistical_analysis.py            significance tests / effect sizes
CORPUS_SOURCES.md                where to obtain each of the 26 corpora + licenses
```

## Model

The base model is the public **wav2vec2-xls-r-1b** checkpoint, available on Hugging Face (model id in the code).
It is not redistributed here. The probing classifiers (logistic regression / SVM) are
trained by the code; their results are the small files in `data/`.

## Data / corpora

The audio of the 26 corpora is **not** redistributed : each corpus has its own license.
`CORPUS_SOURCES.md` lists the source and license for every corpus so they can be obtained
from the original providers. `data/` contains only derived numerical results and metadata.

**Note:** in `master_26_corpora.json` the `data_type` field for KazEmoTTS reads `TTS`; this label
refers to the corpus's purpose (a dataset *for* Kazakh emotional text-to-speech). The recordings
are human speech (narrators), not synthetic, and are treated as recorded speech throughout.

## Citation

```
@inproceedings{Marchenko2026TIMBRE,
  author    = {Marchenko, Anatoly},
  title     = {{TIMBRE}: Layer-Wise Cross-Lingual Speech Emotion Recognition Across 49 Layers and 26 Corpora},
  booktitle = {Proc. Interspeech},
  year      = {2026}
}
```

## License

Code: MIT (see `LICENSE`). Derived data in `data/`: CC BY 4.0. Underlying corpora retain
their original licenses (see `CORPUS_SOURCES.md`).
