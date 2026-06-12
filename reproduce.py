#!/usr/bin/env python3
"""Reproduce all key numbers and figures of the TIMBRE paper from included data.

No GPU, no audio needed — runs on data/ in seconds.
    python3 reproduce.py
Outputs: prints every headline number; writes fig1_layer_curve.pdf, fig2_typological_profiles.pdf.
Full pipeline that produced data/ (needs GPU + the 26 corpora): see code/ and CORPUS_SOURCES.md.
"""
import json, csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
TENSOR = ROOT / "data" / "gpu_cross_lw_cross_lw_lr_26.json"
SUMMARY = ROOT / "data" / "summary.csv"

data = json.load(open(TENSOR))
LAYERS = list(range(49))
corpora = list(data["layer_15"]["matrix"].keys())
def M(i): return data[f"layer_{i}"]["matrix"]
def grand(i): return data[f"layer_{i}"]["grand_avg_f1"]

TYPE = {"cafe_french":"Romance","emouerj_portuguese":"Romance","emozionalmente_italian":"Romance",
        "mesd_spanish":"Romance","oreau_french":"Romance","caves":"Tonal","esd_zh":"Tonal",
        "thai_ser":"Tonal","visec_vietnamese":"Tonal","emodb":"Germanic","ravdess":"Germanic",
        "pavoque_german":"Germanic","dusha_russian":"Slavic","resd_russian":"Slavic","nemo_polish":"Slavic"}
FAM = {"ased_amharic":"Semitic","cafe_french":"Romance","caves":"Sino-Tibetan","dusha_russian":"Slavic",
       "emodb":"Germanic","emouerj_portuguese":"Romance","emozionalmente_italian":"Romance","esd_zh":"Sino-Tibetan",
       "estonian_eekk":"Uralic","hikia_korean":"Koreanic","jvnv_japanese":"Japonic","kannada_ser":"Dravidian",
       "kazemotts_kazakh":"Turkic","mder_arabic":"Semitic","mesd_spanish":"Romance","nemo_polish":"Slavic",
       "oreau_french":"Romance","pavoque_german":"Germanic","ravdess":"Germanic","resd_russian":"Slavic",
       "shemo_persian":"Iranian","subesco_bengali":"Indo-Aryan","thai_ser":"Kra-Dai","turevdb_turkish":"Turkic",
       "urdu_dataset":"Indo-Aryan","visec_vietnamese":"Austroasiatic"}
def typ(c): return TYPE.get(c, "Other")
def fam(c): return FAM.get(c, "Other")
rng = np.random.default_rng(20260607); B = 2000

print(f"== Corpus sets ==\n main cross-corpus N={len(corpora)} (expect 26); pairs={len(corpora)*(len(corpora)-1)}; experiments={49*len(corpora)*(len(corpora)-1)}")

print("\n== Layer curve (CRIS) ==")
for i in [0,10,15,45]:
    print(f" L{i:2d} grand F1 = {grand(i):.3f}")
print(f" L45 vs L15 = {(grand(45)-grand(15))/grand(15)*100:.1f}%")

def pairs(i):
    for c in corpora:
        for tg,sc in M(i)[c]["scores"].items():
            if sc is not None and tg != c: yield c, tg, sc

def wb(label_fn, i=15):
    from collections import Counter
    cnt = Counter(label_fn(c) for c in corpora)
    elig = [c for c in corpora if label_fn(c) not in ("Other","?") and cnt[label_fn(c)]>=2]
    wi=[]; be=[]
    for c,tg,sc in pairs(i):
        if c in elig and tg in elig:
            (wi if label_fn(c)==label_fn(tg) else be).append(sc)
    # cluster bootstrap of the difference
    diffs=[]
    for _ in range(B):
        s=list(rng.choice(elig,size=len(elig),replace=True)); w=[];b=[]
        for c in s:
            for tg in elig:
                if tg==c: continue
                sc=M(i)[c]["scores"].get(tg)
                if sc is None: continue
                (w if label_fn(c)==label_fn(tg) else b).append(sc)
        if w and b: diffs.append(np.mean(w)-np.mean(b))
    ci=np.percentile(diffs,[2.5,97.5])
    return np.mean(wi),np.mean(be),ci

print("\n== Membership contrasts @L15 (corpus-clustered bootstrap) ==")
for name,fn in [("type",typ),("family",fam)]:
    w,b,ci=wb(fn); print(f" {name}: within={w:.3f} between={b:.3f}  diff 95% CI [{ci[0]:+.3f},{ci[1]:+.3f}]")
allp=[sc for _,_,sc in pairs(15)]; print(f" grand avg = {np.mean(allp):.3f}")

def within_adv(i,t):
    v=[sc for c,tg,sc in pairs(i) if typ(c)==t and typ(tg)==t]
    return np.mean(v)-grand(i)
print("\n== Typological within-type advantage (L0/L15/L35) ==")
for t in ["Romance","Tonal"]:
    print(f" {t}: "+" / ".join(f"{within_adv(i,t):+.3f}" for i in [0,15,35]))

def incoming(test,i=15):
    v=[M(i)[tr]["scores"].get(test) for tr in corpora if tr!=test and M(i)[tr]["scores"].get(test) is not None]
    return np.mean(v)
print("\n== Incoming transfer @L15 ==")
print(f" EmoDB={incoming('emodb'):.3f}  DUSHA={incoming('dusha_russian'):.3f}  Thai->EmoDB={M(15)['thai_ser']['scores']['emodb']:.3f}")

W2 = ROOT / "data" / "w2v2_meanpool_cross_corpus_26.json"
print("\n== Acoustic vs wav2vec2 mean-pool (Section 4.3) ==")
if W2.exists():
    w2 = json.load(open(W2))
    vv = [s for tr in w2 for tg, s in w2[tr]["scores"].items() if s is not None and tg != tr]
    print(f" wav2vec2 mean-pool grand avg = {np.mean(vv):.3f}  [paper 0.355]")
AC = ROOT / "data" / "cpu_acoustic_cross_26_logreg.json"
if AC.exists() and W2.exists():
    ac = json.load(open(AC))
    av = [s for tr in ac for tg, s in ac[tr]["scores"].items() if s is not None and tg != tr]
    print(f" acoustic grand avg = {np.mean(av):.4f}  [paper 0.2521]")
    print(f" ratio = {np.mean(av)/np.mean(vv)*100:.1f}%  [paper 71.0%]")
    cs = sorted(ac)
    def inc(d, t): return np.mean([d[s]["scores"][t] for s in cs if s != t])
    wins = [c for c in cs if inc(ac, c) > inc(w2, c)]
    print(f" acoustic wins (incoming basis) = {len(wins)} of 26  [paper 3]")
    marg = sorted(cs, key=lambda c: inc(ac, c) - inc(w2, c))
    t3a = [(c, round(inc(ac, c), 3), round(inc(w2, c), 3)) for c in marg[::-1][:3]]
    t3w = [(c, round(inc(ac, c), 3), round(inc(w2, c), 3)) for c in marg[:3]]
    print(f" Table 1 top-3 acoustic: {t3a}")
    print(f" Table 1 top-3 wav2vec2: {t3w}")
else:
    print(" data/cpu_acoustic_cross_26_logreg.json missing; regenerate via")
    print(" code/acoustic_cross_corpus_26.py or deep_ser_experiment.py (see CORPUS_SOURCES.md).")

print("\n== Ablation (summary.csv) ==")
rows=[r for r in csv.DictReader(open(SUMMARY)) if r['corpus']!='corpus']
ROM={'CaFE','Oreau','Emozionalmente','MESD','emoUERJ'}
mfcc=[float(r['abl_MFCC']) for r in rows]
rp=[float(r['abl_Perturbation']) for r in rows if r['corpus'] in ROM]
nr=[float(r['abl_Perturbation']) for r in rows if r['corpus'] not in ROM]
print(f" N={len(rows)}  MFCC mean={np.mean(mfcc):.3f}")
print(f" Romance dPert={np.mean(rp):.3f} (all neg={all(x<0 for x in rp)})  non-Romance={np.mean(nr):.3f}")
try:
    from scipy import stats
    u,p=stats.mannwhitneyu(rp,nr,alternative='two-sided')
    ps=np.sqrt((np.std(rp,ddof=1)**2+np.std(nr,ddof=1)**2)/2)
    print(f" Mann-Whitney p={p:.5f}  Cohen d={(np.mean(rp)-np.mean(nr))/ps:.2f}")
except ImportError:
    print(" (install scipy for p-value / Cohen d)")

# figures
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    g=[grand(i) for i in LAYERS]
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(LAYERS,g,'b-',lw=2)
    ax.scatter([15],[g[15]],color='red',s=100,label=f'L15 (CRIS) F1={g[15]:.3f}')
    ax.axvspan(13,18,alpha=.1,color='red'); ax.set_xlabel('Layer'); ax.set_ylabel('Grand Average F1')
    ax.legend(); ax.grid(alpha=.3); ax.set_xlim(0,48); plt.tight_layout()
    plt.savefig(ROOT/"fig1_layer_curve.pdf",bbox_inches='tight')
    cols={"Romance":"#e74c3c","Tonal":"#2ecc71","Germanic":"#3498db","Slavic":"#9b59b6"}
    mk={"Romance":"o","Tonal":"^","Germanic":"s","Slavic":"D"}
    fig,ax=plt.subplots(figsize=(7,4))
    for t in cols:
        tc=[c for c in corpora if typ(c)==t]
        mean=[];lo=[];hi=[]
        per={c:{i:[s for tg,s in M(i)[c]["scores"].items() if s is not None and tg!=c] for i in LAYERS} for c in tc}
        for i in LAYERS:
            bs=[]
            for _ in range(B):
                pool=[];
                for c in rng.choice(tc,size=len(tc),replace=True): pool+=per[c][i]
                bs.append(np.mean(pool) if pool else np.nan)
            mean.append(np.nanmean(bs)); lo.append(np.nanpercentile(bs,2.5)); hi.append(np.nanpercentile(bs,97.5))
        ax.fill_between(LAYERS,lo,hi,color=cols[t],alpha=.15)
        ax.plot(LAYERS,mean,color=cols[t],lw=2,marker=mk[t],markevery=5,label=t)
    ax.set_xlabel('Layer'); ax.set_ylabel('Mean cross-corpus F1 by prosodic type')
    ax.legend(); ax.grid(alpha=.3); ax.set_xlim(0,48); plt.tight_layout()
    plt.savefig(ROOT/"fig2_typological_profiles.pdf",bbox_inches='tight')
    print("\nWrote fig1_layer_curve.pdf, fig2_typological_profiles.pdf")
except ImportError:
    print("\n(install matplotlib to regenerate figures)")
