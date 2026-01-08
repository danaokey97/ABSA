import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import time
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

# =====================================================
# PATH – DISKONFIG SESUAI FOLDER ABSA
# =====================================================
st.set_page_config(page_title="ABSA – LDA + Logistic Regression", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.join(BASE_DIR, "MODEL_LDA_5ASPEK_NEW")
MODEL_DIR   = os.path.join(ROOT, "Model_LDA")
ARTEFAK_DIR = os.path.join(ROOT, "artefak")
SENT_MODEL_DIR = os.path.join(BASE_DIR, "MODEL")

# === FILE LEXICON (SESUAIKAN NAMA FILE DI SINI) ===
SLANG_FILE      = os.path.join(BASE_DIR, "kamus_slang.txt")   # TODO: sesuaikan
KATADASAR_FILE  = os.path.join(BASE_DIR, "kata_dasar.txt")   # TODO: sesuaikan

ASPEK = ["Kemasan", "Aroma", "Tekstur", "Harga", "Efek"]

# =====================================================
# UTIL PREPROCESSING DASAR
# =====================================================

def _simple_clean(text: str) -> str:
    t = str(text).lower()
    return re.sub(r"[^a-z0-9_ ]+", " ", t)

def _root_id(token: str) -> str:
    t = str(token).lower().strip()
    t = re.sub(r'(ku|mu|nya)$', '', t)
    t = re.sub(r'^([a-z0-9]+)_\1$', r'\1', t)
    return t

def tokenize_from_val(val, bigram=None):
    """
    Tokenisasi 'standar' untuk korpus LDA (tanpa lexicon).
    Dipakai di jalur dataset (bukan ulasan tunggal dengan lexicon).
    """
    if isinstance(val, list):
        toks = [str(t) for t in val if t]
    else:
        toks = _simple_clean(val).split()
    if bigram is not None:
        try:
            toks = list(bigram[toks])
        except Exception:
            pass
    return toks

def _expand_for_seed(tokens):
    parts = []
    for tok in tokens:
        if "_" in tok:
            parts.extend(tok.split("_"))
        parts.append(tok)
    return {_root_id(p) for p in parts}

def split_into_sentences(text: str):
    if not isinstance(text, str):
        text = str(text)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sentences = []

    for line in lines:
        # ✅ HANYA split . ! ?
        parts = re.split(r"([.!?])", line)
        buf = ""

        for part in parts:
            if part in [".", "!", "?"]:
                buf += part
                if buf.strip():
                    sentences.append(buf.strip())
                buf = ""
            else:
                buf += part

        if buf.strip():
            sentences.append(buf.strip())

    return sentences

# =====================================================
# LEXICON: SLANG & KATA DASAR UNTUK PREPROSES SINGLE TEXT
# =====================================================

@st.cache_resource
def load_lexicons():
    """
    Load slang dictionary saja (tanpa kata dasar).
    Format slang yang didukung:
    1) TXT/CSV biasa, satu pasangan per baris:
         ga, tidak
         gk\t tidak
    2) Potongan dictionary/JSON:
         "ga": "tidak", "gk": "tidak", ...
    """
    slang_dict = {}

    # ================== LOAD SLANG ==================
    if os.path.exists(SLANG_FILE):
        with open(SLANG_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        # --- Mode 2: pattern "slang": "baku" ---
        pair_pattern = re.compile(r'"([^"]+)"\s*:\s*"([^"]+)"')
        for s, n in pair_pattern.findall(content):
            slang_dict[s.strip().lower()] = n.strip().lower()

        # --- Mode 1: satu pasangan per baris, dipisah koma / TAB ---
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" in line and '"' in line:
                continue

            if "," in line:
                s, n = [p.strip().lower() for p in line.split(",", 1)]
            elif "\t" in line:
                s, n = [p.strip().lower() for p in line.split("\t", 1)]
            else:
                continue

            if s and n:
                slang_dict[s] = n

    # --- Fallback wajib: pastikan kata ini SELALU ada ---
    forced_pairs = {
        "ga": "tidak",
        "g": "tidak",
        "gak": "tidak",
        "gk": "tidak",
        "enggak": "tidak",
        "nggak": "tidak",
    }
    for s, n in forced_pairs.items():
        slang_dict.setdefault(s, n)

    return slang_dict


def _normalize_repeated_chars(token: str, max_repeat: int = 2) -> str:
    """
    Contoh: 'bangett' -> 'banget', 'lucu bangeeet' -> 'lucu banget'
    """
    return re.sub(r"(.)\1{%d,}" % max_repeat, r"\1" * max_repeat, token)

def _lemmatize_token_with_katadasar(token: str, kata_dasar: set) -> str:
    """
    Lematisasi ringan berbasis kata_dasar.
    Heuristik sederhana saja.
    """
    t = _normalize_repeated_chars(token)

    if t in kata_dasar:
        return t

    suffixes = ["nya", "in", "kan", "lah", "ku", "mu"]
    for suf in suffixes:
        if t.endswith(suf) and len(t) > len(suf) + 2:
            base = t[:-len(suf)]
            if base in kata_dasar:
                return base

    return t

def normalize_tokens_with_lexicon(tokens):
    """
    Normalisasi token dengan slang saja (tanpa kata_dasar).
    Dipakai untuk ulasan tunggal (segmentasi + sentimen) jika use_lexicon=True.
    """
    slang_dict = load_lexicons()

    norm_tokens = []
    for tok in tokens:
        t = tok.lower().strip()
        if not t:
            continue

        # 🔒 WAJIB: semua variasi 'ga' selalu jadi 'tidak'
        if t in {"ga", "gak", "gk", "engga", "enggak", "nggak", "g"}:
            t = "tidak"
        else:
            t = slang_dict.get(t, t)

        norm_tokens.append(t)

    return norm_tokens

def _apply_negation_rules(tokens):
    """
    - Jika (NEG_WORD + kata) ada di POLAR_SWAP → ganti dengan antonim.
      contoh: tidak mahal -> murah
    - Jika tidak ada di POLAR_SWAP → biarkan apa adanya (tidak bikin *_neg).
    """
    new_tokens = []
    skip_next = False

    for i, t in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue

        if t in NEG_WORDS and i + 1 < len(tokens):
            nxt = tokens[i + 1]

            if nxt in POLAR_SWAP:
                # Ada antonim di kamus → pakai antonim
                new_tokens.append(POLAR_SWAP[nxt])
                skip_next = True
            else:
                # Tidak ada di kamus → simpan kata negasi dan kata aslinya
                new_tokens.append(t)
                # boleh juga tambahkan nxt, kalau mau:
                # new_tokens.append(nxt)
                skip_next = False   # kita tidak skip kata berikut
        else:
            new_tokens.append(t)

    return new_tokens

# =====================================================
# LOAD RESOURCES LDA (dictionary, lda, mapping, seeds)
# =====================================================

@st.cache_resource
def load_resources():
    dictionary = Dictionary.load(os.path.join(MODEL_DIR, "dictionary.gensim"))
    lda = LdaModel.load(os.path.join(MODEL_DIR, "lda_model.gensim"))

    bigram = None
    bg_path = os.path.join(MODEL_DIR, "bigram_phraser.pkl")
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            bigram = pickle.load(f)

    df_map = pd.read_excel(os.path.join(MODEL_DIR, "mapping_aspek_auto.xlsx"))
    topic2aspect = dict(zip(df_map["topic"], df_map["assigned_aspect"]))

    with open(os.path.join(ARTEFAK_DIR, "seeds.json"), "r", encoding="utf-8") as f:
        sj = json.load(f)

    SEED_DICT = {
        "Kemasan": set(sj.get("Kemasan", [])),
        "Aroma":   set(sj.get("Aroma", [])),
        "Tekstur": set(sj.get("Tekstur", [])),
        "Harga":   set(sj.get("Harga", [])),
        "Efek":    set(sj.get("Efek", [])),
    }

    SEED_ROOTS = {
        aspek: {
            _root_id(part)
            for w in SEED_DICT[aspek]
            for part in (w.split("_") + [w])
        }
        for aspek in ASPEK
    }

    return dictionary, lda, bigram, topic2aspect, SEED_DICT, SEED_ROOTS

# =====================================================
# LOAD SENTIMENT MODELS (Logistic Regression)
# =====================================================

@st.cache_resource
def load_sentiment_models():
    """
    Load model sentimen per aspek:
    - logreg_kemasan.pkl, tfidf_kemasan.pkl
    - logreg_aroma.pkl, tfidf_aroma.pkl
    - dst.
    """
    models = {}

    for aspek in ASPEK:
        key = aspek.lower().replace(" ", "_")

        model_path = os.path.join(SENT_MODEL_DIR, f"logreg_{key}.pkl")
        tfidf_path = os.path.join(SENT_MODEL_DIR, f"tfidf_{key}.pkl")

        if os.path.exists(model_path) and os.path.exists(tfidf_path):
            with open(model_path, "rb") as fm:
                clf = pickle.load(fm)
            with open(tfidf_path, "rb") as fv:
                vec = pickle.load(fv)

            models[aspek] = (clf, vec)

    return models

def gabung_negasi(text: str) -> str:
    """
    Gabungkan negasi + 1 kata setelahnya menjadi satu token dengan underscore.
    Contoh: 'tidak kasar' -> 'tidak_kasar'
    Ini harus konsisten dengan training LogReg kamu.
    """
    t = str(text).lower()

    neg = r"(tidak|ga|gak|nggak|enggak|tak|tdk|bukan|kurang|gk|g)"
    t = re.sub(rf"\b{neg}\s+(\w+)\b", r"\1_\2", t)

    return t

def preprocess_for_sentiment(text: str, use_lexicon: bool = False) -> str:
    cleaned = _simple_clean(text)
    tokens = cleaned.split()

    if use_lexicon:
        tokens = normalize_tokens_with_lexicon(tokens)
        tokens = _apply_negation_rules(tokens)

    out = " ".join(tokens)

    # ✅ gabung negasi hanya jika memang ada kata negasi
    if re.search(r"\b(tidak|ga|gak|nggak|enggak|tak|tdk|bukan|kurang|gk|g)\b", out):
        out = gabung_negasi(out)

    return out

def predict_sentiment_for_segment(seg_text: str, aspek: str, sent_models: dict, use_lexicon: bool = False):
    if aspek not in sent_models:
        return None, None

    clf, vec = sent_models[aspek]
    X = vec.transform([preprocess_for_sentiment(seg_text, use_lexicon=use_lexicon)])

    y_pred = clf.predict(X)[0]

    prob_pos = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        if "Positive" in clf.classes_:
            prob_pos = float(proba[list(clf.classes_).index("Positive")])

    return y_pred, prob_pos

# =====================================================
# FUNGSI INTI: DETEKSI ASPEK + SEGMENTASI (LDA)
# =====================================================
# =====================================================
#  NEGATION HANDLING (untuk ulasan tunggal + sentimen)
# =====================================================

NEG_WORDS = {"ga", "gak", "gk", "tidak", "enggak", "nggak","g"}

# Kamus antonim sederhana untuk kata-kata yang sering muncul di ulasan
# (bisa kamu tambah sendiri kapan saja)
POLAR_SWAP = {
    # Harga
    "mahal": "murah",
    "murah": "mahal",

    # Kesan umum
    "bagus": "jelek",
    "jelek": "bagus",
    "buruk": "bagus",
    "oke"  : "tidak_oke",

    # Tekstur
    "berat": "ringan",
    "ringan": "berat",
    "lengket": "tidak_lengket",

    # Efek
    "perih": "nyaman",
    "iritasi": "nyaman",
    "jerawatan": "tidak_jerawatan",  # contoh: "tidak jerawatan" -> netral/positif
}


SEGMENT_STOPWORDS = {
    "tidak", "gak", "nggak", "enggak", "ga","g",
    "banget", "aja", "sih", "dong", "kok",
    "dan", "atau", "yang", "itu", "ini",
    "enak", "dipake", "pake", "nyaman", "kurang"
}

BASE_ROOT = {
    "Kemasan": "kemas",
    "Aroma":   "aroma",
    "Tekstur": "tekstur",
    "Harga":   "harga",
    "Efek":    "efek",
}

def detect_aspect_from_token(tok: str):
    _, _, _, _, _, SEED_ROOTS = load_resources()

    root = _root_id(_simple_clean(tok)).strip()
    if not root or root in SEGMENT_STOPWORDS:
        return None

    for a in ASPEK:
        if root in SEED_ROOTS[a]:
            return a
    return None

def bow_of(tokens, dictionary):
    return dictionary.doc2bow([t for t in tokens if t in dictionary.token2id])

def predict_aspect_boosted(
    tokens,
    lambda_boost=0.9,
    gamma=2.0,
    seed_bonus=0.03,
    dampen_price_if_no_seed=True,
    price_delta=0.7,
    prefer_seed_for_top1=True,
):
    dictionary, lda, _, topic2aspect, SEED_DICT, _ = load_resources()

    bow = bow_of(tokens, dictionary)
    dist_pairs = lda.get_document_topics(bow, minimum_probability=0.0)

    p_aspek = {a: 0.0 for a in ASPEK}
    for k, p in dist_pairs:
        a = topic2aspect.get(k, f"T{k}")
        p_aspek[a] += p

    toks_for_seed = _expand_for_seed(tokens) | _expand_for_seed(
        _simple_clean(" ".join(tokens)).split()
    )
    seed_hits = {
        a: len({_root_id(w) for w in SEED_DICT[a]} & toks_for_seed)
        for a in ASPEK
    }

    p_boost = {
        a: p_aspek[a] * (1.0 + lambda_boost * seed_hits[a]) ** gamma
        for a in ASPEK
    }
    for a, h in seed_hits.items():
        if h >= 1:
            p_boost[a] += seed_bonus

    if dampen_price_if_no_seed and seed_hits["Harga"] == 0 and max(seed_hits.values()) > 0:
        p_boost["Harga"] *= price_delta

    Z = sum(p_boost.values()) or 1.0
    p_boost = {a: v / Z for a, v in p_boost.items()}

    # fallback: kalau semua kecil dan tidak ada seed → anggap Efek (umum)
    if max(p_boost.values()) < 0.35 and sum(seed_hits.values()) == 0:
        aspect_final = "Efek"
    else:
        if prefer_seed_for_top1 and any(h > 0 for h in seed_hits.values()):
            seeded_aspects = [a for a, h in seed_hits.items() if h > 0]
            aspect_final = max(seeded_aspects, key=lambda a: p_boost[a])
        else:
            aspect_final = max(p_boost, key=p_boost.get)

    aspect_top1_plain = max(p_boost, key=p_boost.get)

    return p_aspek, seed_hits, p_boost, aspect_final, aspect_top1_plain

PUNCT_SPLIT_REGEX = r"[.!?;:]+"

def split_by_punctuation(text: str):
    text = str(text)
    text = text.replace("\n", ". ")
    parts = re.split(PUNCT_SPLIT_REGEX, text)
    return [p.strip() for p in parts if p.strip()]

CONJ_SPLIT_WORDS2 = {"tapi", "namun", "tetapi", "sedangkan", "walaupun", "meskipun", "cuma", "hanya"}

def split_by_conjunction(seg: str):
    toks = _simple_clean(seg).split()
    if not toks:
        return [seg]

    for i, t in enumerate(toks):
        if t in CONJ_SPLIT_WORDS2 and 0 < i < len(toks)-1:
            left = " ".join(toks[:i]).strip()
            right = " ".join(toks[i+1:]).strip()
            out = []
            if left:
                out.append(left)
            if right:
                out.append(right)
            return out

    return [seg]

def detect_aspect_by_seed(tokens):
    """
    Deteksi aspek dari tokens:
    1) BASE_ROOT (harga/aroma/tekstur/kemas/efek) untuk menangkap "harganya", "aromanya"
    2) SEED_ROOTS (seeds.json)
    Return: (best_aspect or None, hits_per_aspect)
    """
    _, _, _, _, _, SEED_ROOTS = load_resources()

    roots = [_root_id(t) for t in tokens]
    roots_set = set(roots)

    # 1) BASE_ROOT dulu
    for r in roots:
        for a in ASPEK:
            if BASE_ROOT[a] in r:
                return a, {asp: 0 for asp in ASPEK}

    # 2) Seed hits
    hits = {a: len(SEED_ROOTS[a] & roots_set) for a in ASPEK}

    best_aspect = None
    best_score = 0
    for a, sc in hits.items():
        if sc > best_score:
            best_score = sc
            best_aspect = a

    if best_score == 0:
        return None, hits

    return best_aspect, hits

CONJ_JUNK = {"tapi", "namun", "tetapi", "sedangkan", "walaupun", "meskipun", "cuma", "hanya"}

def segment_text_aspect_aware(text: str, use_lexicon=False):
    """
    Segmentasi RULE-BASED sesuai maumu:
    - potong segmen hanya jika ditemukan aspek baru (seed/base-root) yang BEDA
    - bigram phraser dipakai hanya untuk tokens LDA (bukan untuk tampilan segmen & sentimen)
    - buang konjungsi yang nyangkut di akhir segmen kiri (mis. "aroma tapi")
    """
    _, _, bigram, _, _, _ = load_resources()

    # --- tokens "plain" untuk menentukan batas segmen (tanpa bigram, tanpa underscore)
    plain_tokens = _simple_clean(text).split()
    if use_lexicon:
        plain_tokens = normalize_tokens_with_lexicon(plain_tokens)

    if not plain_tokens:
        return []

    # helper: aspek per token (base_root/seed roots)
    def aspect_of_token(tok: str):
        _, _, _, _, _, SEED_ROOTS = load_resources()
        r = _root_id(tok)

        # base_root
        for a in ASPEK:
            if BASE_ROOT[a] in r:
                return a

        # seed
        for a in ASPEK:
            if r in SEED_ROOTS[a]:
                return a

        return None

    segments = []
    start = 0
    current_aspect = None

    for i, tok in enumerate(plain_tokens):
        a_tok = aspect_of_token(tok)

        if current_aspect is None and a_tok is not None:
            current_aspect = a_tok

        # jika ketemu aspek baru yang beda -> CUT
        if a_tok is not None and current_aspect is not None and a_tok != current_aspect:
            cut = i

            # buang konjungsi di ujung kiri segmen (contoh: "... aroma tapi | harga ...")
            while cut > start and plain_tokens[cut - 1] in CONJ_JUNK:
                cut -= 1

            left_tokens = plain_tokens[start:cut]
            left_text = " ".join(left_tokens).strip()

            if left_text:
                # tokens untuk LDA boleh pakai bigram
                lda_tokens = tokenize_from_val(left_text, bigram=bigram)
                if use_lexicon:
                    lda_tokens = normalize_tokens_with_lexicon(lda_tokens)

                anchor, hits = detect_aspect_by_seed(left_tokens)
                if anchor is None:
                    anchor = current_aspect

                segments.append({
                    "seg_text": left_text,         # ✅ tampil bersih (tanpa underscore)
                    "tokens": lda_tokens,          # ✅ untuk LDA boleh bigram
                    "anchor_aspect": anchor,
                    "seed_hits": hits
                })

            start = i
            current_aspect = a_tok

    # segmen terakhir
    last_tokens = plain_tokens[start:]
    last_text = " ".join(last_tokens).strip()

    if last_text:
        lda_tokens = tokenize_from_val(last_text, bigram=bigram)
        if use_lexicon:
            lda_tokens = normalize_tokens_with_lexicon(lda_tokens)

        anchor, hits = detect_aspect_by_seed(last_tokens)
        if anchor is None:
            anchor = current_aspect

        segments.append({
            "seg_text": last_text,
            "tokens": lda_tokens,
            "anchor_aspect": anchor,
            "seed_hits": hits
        })

    return segments


CONJ_SPLIT_WORDS = {"tapi", "namun", "tetapi", "sedangkan", "walaupun", "meskipun"}

def segment_text_for_aspect(text: str, use_lexicon=False):
    """
    Segmentasi berdasarkan anchor aspek (BASE_ROOT/SEED_ROOTS).
    Output segmen besar: dari anchor ke anchor berikutnya.
    """

    _, _, bigram, _, _, SEED_ROOTS = load_resources()

    sentences = split_into_sentences(text)
    segments = []

    def token_aspect(tok: str):
        r = _root_id(_simple_clean(tok))
        if not r or r in SEGMENT_STOPWORDS:
            return None

        # BASE_ROOT dulu
        for a in ASPEK:
            if BASE_ROOT[a] in r:
                return a

        # seed
        for a in ASPEK:
            if r in SEED_ROOTS[a]:
                return a

        return None

    for sent in sentences:
        toks_plain = _simple_clean(sent).split()
        if use_lexicon:
            toks_plain = normalize_tokens_with_lexicon(toks_plain)
        if not toks_plain:
            continue

        # cari anchor posisi aspek
        anchors = []
        for i, t in enumerate(toks_plain):
            a = token_aspect(t)
            if a is not None:
                # compress anchor berurutan aspek sama
                if not anchors or anchors[-1][1] != a:
                    anchors.append((i, a))

        # kalau tidak ada anchor -> segmen umum
        if not anchors:
            segments.append({"seg_text": " ".join(toks_plain), "anchor_aspect": None})
            continue

        # kalau ada anchor -> potong dari anchor ke anchor berikutnya
        for idx, (pos, asp) in enumerate(anchors):
            end = anchors[idx + 1][0] if idx + 1 < len(anchors) else len(toks_plain)
            seg_tokens = toks_plain[pos:end]
            seg_text = " ".join(seg_tokens).strip()
            if seg_text:
                segments.append({"seg_text": seg_text, "anchor_aspect": asp})

    # attach segmen tanpa anchor ke segmen anchor terdekat (lanjutan)
    attached = []
    last_anchor = None
    for s in segments:
        if s["anchor_aspect"] is not None:
            attached.append(s)
            last_anchor = s["anchor_aspect"]
        else:
            # kalau ada anchor sebelumnya, attach ke dia
            if last_anchor is not None and attached:
                attached[-1]["seg_text"] += " " + s["seg_text"]
            else:
                attached.append(s)

    # buat tokens LDA untuk tiap segmen (pakai bigram disini saja)
    out = []
    for s in attached:
        seg_text = s["seg_text"]
        toks = tokenize_from_val(seg_text, bigram=bigram)
        if use_lexicon:
            toks = normalize_tokens_with_lexicon(toks)

        out.append({
            "seg_text": seg_text,
            "tokens": toks,
            "anchor_aspect": s["anchor_aspect"],
        })

    return out
CONJ_SPLIT_WORDS = {
    "tapi", "namun", "tetapi", "sedangkan",
    "walaupun", "meskipun", "cuma", "hanya"
}

def split_by_punct_and_conj(text: str):
    """
    Split kasar:
    - tanda baca: . ! ? ; :
    - konjungsi: tapi/namun/dst (konjungsi dibuang)
    Koma tidak memotong.
    """
    text = str(text).replace("\n", ". ")
    # split tanda baca dulu
    parts = re.split(r"[.!?;:]+", text)
    parts = [p.strip() for p in parts if p.strip()]

    out = []
    for p in parts:
        toks = _simple_clean(p).split()
        if not toks:
            continue

        # split sekali pada konjungsi pertama yang ketemu
        cut = None
        for i, t in enumerate(toks):
            if t in CONJ_SPLIT_WORDS and 0 < i < len(toks) - 1:
                cut = i
                break

        if cut is None:
            out.append(" ".join(toks).strip())
        else:
            left = " ".join(toks[:cut]).strip()
            right = " ".join(toks[cut+1:]).strip()
            if left:
                out.append(left)
            if right:
                out.append(right)

    return out
def split_by_aspect_anchor_inside_chunk(chunk: str):
    """
    Kalau ada kata aspek eksplisit (BASE_ROOT) muncul di tengah chunk,
    potong chunk jadi beberapa bagian.
    """
    toks = _simple_clean(chunk).split()
    if not toks:
        return []

    cuts = [0]   # start selalu 0

    for i, tok in enumerate(toks):
        r = _root_id(tok)
        for a in ASPEK:
            if BASE_ROOT[a] in r:
                if i not in cuts:
                    cuts.append(i)

    cuts = sorted(set(cuts))

    parts = []
    for j in range(len(cuts)):
        start = cuts[j]
        end = cuts[j+1] if j+1 < len(cuts) else len(toks)

        seg = " ".join(toks[start:end]).strip()
        if seg:
            parts.append(seg)

    return parts
def split_by_seed_shift(chunk: str, min_hits=1):
    """
    Potong chunk jika seed aspek lain muncul setelah anchor awal.
    RETURN list of dict:
    [{"text": ..., "forced_aspect": ...}, ...]
    """

    _, _, _, _, _, SEED_ROOTS = load_resources()

    toks = _simple_clean(chunk).split()
    if not toks:
        return []

    # cari anchor awal dari BASE_ROOT dulu (tekstur/harga/aroma/kemas/efek)
    current_aspect = None
    for tok in toks:
        r = _root_id(tok)
        for a in ASPEK:
            if BASE_ROOT[a] in r:
                current_aspect = a
                break
        if current_aspect:
            break

    # kalau tidak ada anchor awal, pakai seed dominan saja
    if current_aspect is None:
        asp_seed, _ = detect_aspect_simple(toks)
        return [{"text": chunk, "forced_aspect": asp_seed}]

    parts = []
    start = 0
    buffer = []

    for i, tok in enumerate(toks):
        buffer.append(tok)
        roots_buf = {_root_id(t) for t in buffer}

        # cek seed aspek lain
        best_shift = None
        best_hits = 0
        for a in ASPEK:
            if a == current_aspect:
                continue
            hits = len(SEED_ROOTS[a] & roots_buf)
            if hits >= min_hits and hits > best_hits:
                best_hits = hits
                best_shift = a

        if best_shift is not None:
            # segmen sebelum shift
            seg = " ".join(toks[start:i]).strip()
            if seg:
                parts.append({"text": seg, "forced_aspect": current_aspect})

            # reset
            start = i
            buffer = []
            current_aspect = best_shift

    # segmen terakhir
    seg_last = " ".join(toks[start:]).strip()
    if seg_last:
        parts.append({"text": seg_last, "forced_aspect": current_aspect})

    return parts

def detect_aspect_simple(tokens):
    """
    Deteksi aspek dominan dari tokens:
    - BASE_ROOT (harga/aroma/tekstur/kemas/efek) kasih skor kuat
    - SEED_ROOTS dari seeds.json
    """
    _, _, _, _, _, SEED_ROOTS = load_resources()

    roots = {_root_id(t) for t in tokens}
    score = {a: 0 for a in ASPEK}

    # skor dari seed roots
    for a in ASPEK:
        score[a] += len(SEED_ROOTS[a] & roots)

    # bonus kuat dari base_root substring (mis: harganya -> harga)
    for r in roots:
        for a in ASPEK:
            if BASE_ROOT[a] in r:
                score[a] += 3

    best_a = max(score, key=score.get)
    if score[best_a] == 0:
        return None, score
    return best_a, score
def detect_aspect_negation_pattern(tokens_plain):
    """
    Kalau ada pola: NEG_WORD + kata_seed_aspek
    maka aspek = aspek seed tersebut.
    Contoh: tidak mahal -> Harga
    """
    _, _, _, _, _, SEED_ROOTS = load_resources()

    roots = [_root_id(t) for t in tokens_plain]

    for i in range(len(roots)-1):
        if roots[i] in NEG_WORDS:
            nxt = roots[i+1]
            for a in ASPEK:
                if nxt in SEED_ROOTS[a]:
                    return a
    return None

def has_aspect_evidence(tokens_plain, SEED_ROOTS):
    roots = {_root_id(t) for t in tokens_plain}

    # cek BASE_ROOT eksplisit
    for r in roots:
        for a in ASPEK:
            if BASE_ROOT[a] in r:
                return True

    # cek seed hits
    for a in ASPEK:
        if len(SEED_ROOTS[a] & roots) > 0:
            return True

    return False


def merge_short_tail_segments(segs, SEED_ROOTS, use_lexicon=False, max_words=2):
    """
    Gabungkan segmen yang terlalu pendek (<= max_words)
    dan tidak punya seed/base_root ke segmen sebelumnya.
    Ini mencegah segmen seperti: "akan", "tetap", "rekomen", dll berdiri sendiri.
    """
    if not segs or len(segs) < 2:
        return segs

    merged = [segs[0]]

    for i in range(1, len(segs)):
        curr = segs[i]
        prev = merged[-1]

        toks_plain = _simple_clean(curr["seg_text"]).split()
        if use_lexicon:
            toks_plain = normalize_tokens_with_lexicon(toks_plain)

        # kalau segmen pendek dan tidak punya evidence aspek -> merge
        if len(toks_plain) <= max_words and not has_aspect_evidence(toks_plain, SEED_ROOTS):
            prev["seg_text"] = prev["seg_text"].rstrip(" ,") + " " + curr["seg_text"].lstrip(" ,")
            prev["tokens"].extend(curr["tokens"])
        else:
            merged.append(curr)

    return merged


def segment_text_merge_by_aspect(text: str, use_lexicon=False):
    """
    LOGIKA FINAL:
    1) split kasar (tanda baca + konjungsi)
    2) kalau chunk mengandung kata aspek eksplisit (kemas/aroma/tekstur/harga/efek) -> pakai itu (switch boleh)
    3) kalau tidak ada kata aspek eksplisit -> WARISKAN aspek sebelumnya (JANGAN switch pakai seed)
    4) kalau awal teks dan tidak ada aspek eksplisit -> baru pakai seed
    5) merge jika aspek sama
    """
    _, _, bigram, _, _, _ = load_resources()

    chunks_raw = split_by_punct_and_conj(text)
    chunks = []
    for ch in chunks_raw:
        chunks.extend(split_by_seed_shift(ch, min_hits=1))

    if not chunks:
        return []

    def explicit_aspect_from_tokens(tokens_plain):
        # cek substring base_root (harganya -> harga, kemasannya -> kemas, dst)
        for tok in tokens_plain:
            r = _root_id(tok)
            for a in ASPEK:
                if BASE_ROOT[a] in r:
                    return a
        return None

    segs = []
    last_aspect = None

    for ch in chunks:
    ch_text = ch["text"]
    forced_aspect = ch.get("forced_aspect", None)

    toks_plain = _simple_clean(ch_text).split()
    if use_lexicon:
        toks_plain = normalize_tokens_with_lexicon(toks_plain)

    if not toks_plain:
        continue

    asp_seed, hits = detect_aspect_simple(toks_plain)
    asp_explicit = explicit_aspect_from_tokens(toks_plain)

    # ✅ forced aspect override (hasil split_by_seed_shift)
    if forced_aspect is not None:
        asp = forced_aspect
        last_aspect = asp
    else:
        if asp_explicit is not None:
            asp = asp_explicit
            last_aspect = asp_explicit
        else:
            if last_aspect is not None:
                asp = last_aspect
            else:
                asp = asp_seed

    toks_lda = tokenize_from_val(ch_text, bigram=bigram)
    if use_lexicon:
        toks_lda = normalize_tokens_with_lexicon(toks_lda)

    item = {
        "seg_text": ch_text.strip(),
        "tokens": toks_lda,
        "anchor_aspect": asp,
        "seed_hits": hits
    }

    if segs and asp is not None and segs[-1]["anchor_aspect"] == asp:
        segs[-1]["seg_text"] += " " + item["seg_text"]
        segs[-1]["tokens"].extend(item["tokens"])
    else:
        segs.append(item)
    # ✅ merge segmen super pendek (mis: "akan", "tetap") jika tidak punya seed/base_root
    _, _, _, _, _, SEED_ROOTS = load_resources()
    segs = merge_short_tail_segments(segs, SEED_ROOTS, use_lexicon=use_lexicon, max_words=3)

    return segs



def test_segmented_text(
    text,
    lambda_boost=0.9,
    gamma=2.0,
    seed_bonus=0.03,
    dampen_price_if_no_seed=True,
    price_delta=0.7,
    prefer_seed_for_top1=True,
    use_lexicon=False,
):
    _, _, bigram, _, _, _ = load_resources()

    def dominant_seed_aspect(seed_hits: dict):
        if not seed_hits:
            return None
        best_a, best_v = None, 0
        for a, v in seed_hits.items():
            if v > best_v:
                best_a, best_v = a, v
        return best_a if best_v > 0 else None

    def can_merge(prev_item: dict, curr_item: dict) -> bool:
        ap = prev_item.get("anchor_aspect")
        ac = curr_item.get("anchor_aspect")

        if ap is not None and ac is not None and ap != ac:
            return False

        dp = dominant_seed_aspect(prev_item.get("seed_hits"))
        dc = dominant_seed_aspect(curr_item.get("seed_hits"))
        if dp is not None and dc is not None and dp != dc:
            return False

        return True

    # ✅ pakai segmenter kamu (yang merge_by_aspect)
    seg_infos = segment_text_merge_by_aspect(text, use_lexicon=use_lexicon)

    if not seg_infos:
        seg_infos = [{
            "seg_text": text,
            "anchor_aspect": None,
            "tokens": tokenize_from_val(text, bigram=bigram),
            "seed_hits": {a: 0 for a in ASPEK}
        }]

    labeled = []
    for info in seg_infos:
        seg = info.get("seg_text", "")
        anchor = info.get("anchor_aspect", None)
        toks = info.get("tokens", None)

        if not toks:
            toks = tokenize_from_val(seg, bigram=bigram)
            if use_lexicon:
                toks = normalize_tokens_with_lexicon(toks)

        p_raw, hits_lda, p_boost, aspect_pred, _ = predict_aspect_boosted(
            toks,
            lambda_boost=lambda_boost,
            gamma=gamma,
            seed_bonus=seed_bonus,
            dampen_price_if_no_seed=dampen_price_if_no_seed,
            price_delta=price_delta,
            prefer_seed_for_top1=prefer_seed_for_top1
        )

        # ✅ FIX UTAMA: seed_hits harus dari hasil LDA (hits_lda), bukan info lama
        seed_hits = hits_lda

        seed_dom = dominant_seed_aspect(seed_hits)

        # ✅ LOGIKA FINAL: anchor tetap prioritas
        if anchor is not None:
            aspect_final = anchor
        else:
            # ✅ override ringan kalau seed_dom cukup kuat
            if seed_dom is not None and p_boost.get(seed_dom, 0) >= 0.20:
                aspect_final = seed_dom
            else:
                aspect_final = aspect_pred

        prob_final = p_boost.get(aspect_final, 0.0)

        labeled.append({
            "seg_text": seg,
            "anchor_aspect": anchor,
            "tokens": toks,
            "p_boost": p_boost,
            "seed_hits": seed_hits,
            "aspect_final": aspect_final,
            "aspect_prob_final": prob_final,
        })

    # =========================
    # Merge segmen sangat pendek (aman)
    # =========================
    merged_short = []
    for item in labeled:
        if not merged_short:
            merged_short.append(item)
            continue

        prev = merged_short[-1]

        tok_len = len(item["tokens"])
        no_anchor = item.get("anchor_aspect") is None
        total_seed_hits = sum(item["seed_hits"].values()) if item.get("seed_hits") else 0

        short_anchorless = (tok_len <= 4 and no_anchor and total_seed_hits == 0)
        very_short_any = (tok_len <= 2)

        if (short_anchorless or very_short_any) and can_merge(prev, item):
            combined_text = prev["seg_text"].rstrip(" ,") + " " + item["seg_text"].lstrip(" ,")

            toks2 = tokenize_from_val(combined_text, bigram=bigram)
            if use_lexicon:
                toks2 = normalize_tokens_with_lexicon(toks2)

            p_raw2, hits2, p_boost2, aspect2, _ = predict_aspect_boosted(
                toks2,
                lambda_boost=lambda_boost,
                gamma=gamma,
                seed_bonus=seed_bonus,
                dampen_price_if_no_seed=dampen_price_if_no_seed,
                price_delta=price_delta,
                prefer_seed_for_top1=prefer_seed_for_top1
            )

            anchor_combined = prev.get("anchor_aspect", None)
            if anchor_combined is None:
                anchor_combined = item.get("anchor_aspect", None)

            aspect_final2 = anchor_combined if anchor_combined is not None else aspect2

            merged_short[-1] = {
                "seg_text": combined_text,
                "anchor_aspect": anchor_combined,
                "tokens": toks2,
                "p_boost": p_boost2,
                "seed_hits": hits2,
                "aspect_final": aspect_final2,
                "aspect_prob_final": p_boost2.get(aspect_final2, 0.0),
            }
        else:
            merged_short.append(item)

    # =========================
    # Merge segmen dengan aspek sama (aman)
    # =========================
    merged = []
    for item in merged_short:
        if not merged:
            merged.append(item)
            continue

        prev = merged[-1]
        same_aspect = (item["aspect_final"] == prev["aspect_final"])

        anchor_prev = prev.get("anchor_aspect", None)
        anchor_curr = item.get("anchor_aspect", None)
        anchor_conflict = (
            anchor_prev is not None and
            anchor_curr is not None and
            anchor_prev != anchor_curr
        )

        if same_aspect and (not anchor_conflict) and can_merge(prev, item):
            combined_text = prev["seg_text"].rstrip(" ,") + " " + item["seg_text"].lstrip(" ,")

            toks2 = tokenize_from_val(combined_text, bigram=bigram)
            if use_lexicon:
                toks2 = normalize_tokens_with_lexicon(toks2)

            p_raw2, hits2, p_boost2, aspect2, _ = predict_aspect_boosted(
                toks2,
                lambda_boost=lambda_boost,
                gamma=gamma,
                seed_bonus=seed_bonus,
                dampen_price_if_no_seed=dampen_price_if_no_seed,
                price_delta=price_delta,
                prefer_seed_for_top1=prefer_seed_for_top1
            )

            anchor_combined = anchor_prev if anchor_prev is not None else anchor_curr
            aspect_final2 = anchor_combined if anchor_combined is not None else aspect2

            merged[-1] = {
                "seg_text": combined_text,
                "anchor_aspect": anchor_combined,
                "tokens": toks2,
                "p_boost": p_boost2,
                "seed_hits": hits2,
                "aspect_final": aspect_final2,
                "aspect_prob_final": p_boost2.get(aspect_final2, 0.0),
            }
        else:
            merged.append(item)

    # =========================
    # Output final
    # =========================
    results = []
    for i, r in enumerate(merged, start=1):
        results.append({
            "seg_index": i,
            "seg_text": r["seg_text"],
            "p_boost": r["p_boost"],
            "seed_hits": r["seed_hits"],
            "aspect_final": r["aspect_final"],
            "aspect_prob_final": r["aspect_prob_final"],
        })

    return results



# =====================================================
# HELPER: PROSES DATASET MENJADI SEGMENT-LEVEL
# =====================================================

@st.cache_data
def run_absa_on_dataframe(df_raw, _sent_models):

    data_rows = []

    for idx, row in df_raw.iterrows():
        text = str(row["text-content"])

        # Untuk dataset → gunakan use_lexicon=False (konsisten dengan korpus LDA)
        segments = test_segmented_text(text, use_lexicon=True)

        for seg in segments:
            aspek = seg["aspect_final"]
            seg_text = seg["seg_text"]

            # Dataset: preprocessing sentimen juga tetap "ringan" (tanpa lexicon)
            sent_label, _ = predict_sentiment_for_segment(
                seg_text, aspek, _sent_models, use_lexicon=False
            )

            data_rows.append({
                "original_index": idx,
                "Segmen": seg["seg_index"],
                "Teks Segmen": seg_text,
                "Aspek": aspek,
                "Sentimen": sent_label,
                "SkinType": row.get("profile-description", None),
                "Age": row.get("profile-age", None),
                "username": row.get("profile-username", None),
            })

    return pd.DataFrame(data_rows)


# =====================================================
# STREAMLIT UI
# =====================================================
st.markdown("""
<style>
.metric-card {
    background-color: #1e1e1e;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #333;
    text-align: center;
    color: white;
}
.metric-value {
    font-size: 32px;
    font-weight: bold;
}
.metric-label {
    font-size: 16px;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)



def main():

    # ========================== SIDEBAR ==========================
    with st.sidebar:

        ICON_DASHBOARD_MENU = "https://img.icons8.com/?size=100&id=94097&format=png&color=#A3A3A3"

        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                <img src="{ICON_DASHBOARD_MENU}" width="26">
                <h3 style="margin:0; padding:0;">Dashboard Menu</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        menu = st.radio(
            "",
            ["Ulasan Tunggal", "Dashboard Dataset"],
            index=0
        )

    # ==================== LOAD RESOURCES (MODEL) ========================
    try:
        dictionary, lda, bigram, topic2aspect, SEED_DICT, SEED_ROOTS = load_resources()
    except Exception as e:
        st.error(f"Gagal memuat model LDA: {e}")
        st.stop()

    sent_models = load_sentiment_models()
    if not sent_models:
        st.error("Model sentimen Logistic Regression tidak ditemukan. Periksa folder MODEL.")
        st.stop()

    # =====================================================================
    #                           ULASAN TUNGGAL
    # =====================================================================
    if menu == "Ulasan Tunggal":
        st.title("Analisis Ulasan Tunggal")

        text = st.text_area(
            "Masukkan teks ulasan:",
            value="",
            height=160,
            placeholder="Masukkan ulasan..."
        )

        if st.button("🚀 Deteksi Aspek + Sentimen"):
            if not text.strip():
                st.warning("Teks kosong.")
                st.stop()

            # Tampilkan versi preproses (slang + kata dasar) sekadar info ke user
            pre_single = preprocess_for_sentiment(text, use_lexicon=True)
            st.markdown("**Teks setelah preprocessing (slang → baku + kata dasar):**")
            st.code(pre_single, language="text")

            # Segmentasi + LDA pakai lexicon
            results = test_segmented_text(text, use_lexicon=True)

            rows = []
            for r in results:
                aspek = r["aspect_final"]
                seg_text = r["seg_text"]

                # Sentimen pun pakai preproses yang sama (lexicon=True)
                sent_label, _ = predict_sentiment_for_segment(
                    seg_text, aspek, sent_models, use_lexicon=True
                )

                rows.append({
                    "Segmen": r["seg_index"],
                    "Teks Segmen": seg_text,
                    "Aspek": aspek,
                    "Sentimen": sent_label,
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # =====================================================================
    #                        DASHBOARD DATASET
    # =====================================================================
    if menu == "Dashboard Dataset":
        st.title("Female Daily Product Sentiment Overview")

        uploaded = st.file_uploader(
            "Upload file CSV/Excel Female Daily:",
            type=["csv", "xlsx"]
        )

        if uploaded is not None:
            if uploaded.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)

            st.success(f"File berhasil dimuat: {df_raw.shape[0]} baris")

            with st.spinner("Memproses ABSA seluruh dataset..."):
                df_seg = run_absa_on_dataframe(df_raw, sent_models)

            # ===================== DASHBOARD SUMMARY CARDS =====================
            st.markdown("### Quick Dataset Overview")

            c1, c2, c3, c4 = st.columns(4)

            total_ulasan = df_raw.shape[0]
            total_segmen = df_seg.shape[0]

            sentiment_counts = df_seg["Sentimen"].value_counts()

            pos_count = sentiment_counts.get("Positive", 0)
            neg_count = sentiment_counts.get("Negative", 0)

            pos_percent = (pos_count / total_segmen) * 100 if total_segmen > 0 else 0
            neg_percent = (neg_count / total_segmen) * 100 if total_segmen > 0 else 0

            with c1:
                st.markdown(f"""
                <div style="padding:20px; background:#00c0ef; border-radius:12px; text-align:center;">
                    <h2 style="color:white; margin-bottom:0;">{total_ulasan}</h2>
                    <p style="color:#CCCCCC;">Total Ulasan</p>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div style="padding:20px; background:#f39c12; border-radius:12px; text-align:center;">
                    <h2 style="color:white; margin-bottom:0;">{total_segmen}</h2>
                    <p style="color:#CCCCCC;">Total Segmen</p>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div style="padding:20px; background:#00a65a; border-radius:12px; text-align:center;">
                    <h2 style="color:white; margin-bottom:0;">{pos_percent:.1f}%</h2>
                    <p style="color:#CCCCCC;">Sentimen Positif</p>
                </div>
                """, unsafe_allow_html=True)

            with c4:
                st.markdown(f"""
                <div style="padding:20px; background:#dd4b39; border-radius:12px; text-align:center;">
                    <h2 style="color:white; margin-bottom:0;">{neg_percent:.1f}%</h2>
                    <p style="color:#CCCCCC;">Sentimen Negatif</p>
                </div>
                """, unsafe_allow_html=True)

            # Tambah kolom SkinTypeMain
            df_seg["SkinTypeMain"] = df_seg["SkinType"].astype(str).apply(lambda x: x.split(",")[0].strip())

            # ====================== KPI CARDS ======================
            st.markdown("### Key Metrics")

            sent_counts = df_seg["Sentimen"].value_counts()
            pos_count = sent_counts.get("Positive", 0)
            neg_count = sent_counts.get("Negative", 0)
            total_aspek = df_seg["Aspek"].nunique()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                label="Jumlah Ulasan Positif",
                value=pos_count
            )

            c2.metric(
                label="Jumlah Ulasan Negatif",
                value=neg_count
            )

            c3.metric(
                label="Total Aspek",
                value=total_aspek
            )

            # ====================== INSIGHT 1 ======================

            df_filtered = df_seg[df_seg["Sentimen"].isin(["Positive", "Negative"])]

            dist_aspek = (
                df_filtered.groupby(["Aspek", "Sentimen"])
                .size()
                .reset_index(name="count")
            )

            list_aspek = dist_aspek["Aspek"].unique()
            
            st.markdown("###### Distribusi Sentimen per Aspek")

            color_map_aspek = {
                "Aroma":   ["#2ecc71", "#1c973b"],
                "Kemasan": ["#61bdfb", "#1672c8"],
                "Harga":   ["#c390d8", "#b73ce7"],
                "Tekstur": ["#ff983d", "#f27333"],
                "Efek":    ["#fd79a8", "#dd339c"],
            }

            cols = st.columns(5)

            for i, aspek in enumerate(list_aspek):
                df_aspek = dist_aspek[dist_aspek["Aspek"] == aspek]

                fig_donut = px.pie(
                    df_aspek,
                    values="count",
                    names="Sentimen",
                    hole=0.55,
                    title=f"{aspek}",
                    color="Sentimen",
                    color_discrete_map={
                        "Positive": color_map_aspek[aspek][0],
                        "Negative": color_map_aspek[aspek][1]
                    }
                )

                fig_donut.update_layout(
                    margin=dict(l=0, r=0, t=70, b=0),
                    height=240
                )

                fig_donut.update_traces(textinfo="percent", textposition="inside")

                cols[i].plotly_chart(fig_donut, use_container_width=True)

            fig_bar = px.bar(
                dist_aspek,
                x="Aspek",
                y="count",
                color="Sentimen",
                barmode="group",
                text="count",
                color_discrete_map={
                    "Positive": "#2ecc71",
                    "Negative": "#e74c3c"
                }
            )
            fig_bar.update_layout(
                margin=dict(l=20, r=20, t=20, b=20)
            )

            st.plotly_chart(fig_bar, use_container_width=True)

            # ====================== INSIGHT 2 ======================

            efek = df_seg[df_seg["Aspek"] == "Efek"]

            dist_skin = (
                efek.groupby(["SkinTypeMain", "Sentimen"])
                .size()
                .reset_index(name="count")
            )

            fig2 = px.bar(
                dist_skin,
                x="SkinTypeMain",
                y="count",
                color="Sentimen",
                barmode="group",
                title="SkinType vs Sentimen (Efek)",
                text="count",
                color_discrete_map={
                    "Positive": "#fbb3ff",
                    "Negative": "#df3cc6"
                }

            )

            fig2.update_layout(
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig2, use_container_width=True)

            # ====================== INSIGHT 3 ======================

            dist_age_simple = (
                df_seg.groupby(["Age", "Sentimen"])
                .size().reset_index(name="count")
            )

            fig3 = px.bar(
                dist_age_simple,
                x="Age",
                y="count",
                color="Sentimen",
                barmode="group",
                title="Distribusi Sentimen Berdasarkan Usia",
                text="count",
                color_discrete_map={
                    "Positive": "#ff9860",
                    "Negative": "#f96c15"
                }
            )

            fig3.update_layout(
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig3, use_container_width=True)

            # ====================== INSIGHT 4 ======================
            st.markdown("######  Sentimen Efek berdasarkan SkinType & Usia")

            skin_list = sorted(df_seg["SkinTypeMain"].dropna().unique())
            choose_skin = st.selectbox("Pilih jenis kulit:", skin_list)

            efek2 = df_seg[
                (df_seg["Aspek"] == "Efek") &
                (df_seg["SkinTypeMain"] == choose_skin)
            ]

            dist_age_skin = (
                efek2.groupby(["Age", "Sentimen"])
                .size().reset_index(name="count")
            )

            fig4 = px.bar(
                dist_age_skin,
                x="Age",
                y="count",
                color="Sentimen",
                barmode="group",
                title=f"Sentimen Efek untuk SkinType {choose_skin}",
                text="count"
            )
            st.plotly_chart(fig4, use_container_width=True)

            # ====================== INSIGHT 5 ======================
            st.markdown("##### WordCloud Sentimen Positif & Negatif")

            df_seg["Sentimen"] = (
                df_seg["Sentimen"]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.capitalize()
            )

            positif_text = " ".join(df_seg[df_seg["Sentimen"] == "Positive"]["Teks Segmen"])
            negatif_text = " ".join(df_seg[df_seg["Sentimen"] == "Negative"]["Teks Segmen"])

            color_pos = "Greens"
            color_neg = "Reds"

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### WordCloud Positif")

                wc_pos = WordCloud(
                    width=900,
                    height=500,
                    background_color="white",
                    colormap=color_pos,
                    max_words=150
                ).generate(positif_text)

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(wc_pos, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

            with col2:
                st.markdown("### WordCloud Negatif")

                wc_neg = WordCloud(
                    width=900,
                    height=500,
                    background_color="white",
                    colormap=color_neg,
                    max_words=150
                ).generate(negatif_text)

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.imshow(wc_neg, interpolation="bilinear")
                ax2.axis("off")
                st.pyplot(fig2)

            # ====================== INSIGHT 6 ======================
            st.markdown("###  Dataframe Segmen (Aspek + Sentimen)")

            cols_show = ["original_index", "Segmen", "Teks Segmen", "Aspek", "Sentimen"]
            st.dataframe(df_seg[cols_show], use_container_width=True)

    st.markdown("""
    <hr style="margin-top:40px;">

    <div style="text-align:center; font-size:13px; color:gray; padding:10px;">
        © 2025 Danskuy — Female Daily Review Analysis Dashboard  
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
