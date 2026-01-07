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
        parts = re.split(r"([.,!?])", line)
        buf = ""

        for part in parts:
            if part in [".", "!", "?", ","]:
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
    """
    Preprocessing untuk input Logistic Regression.
    - Default (dataset): _simple_clean saja
    - use_lexicon=True (ulasan tunggal): slang -> baku + negation rules (POLAR_SWAP)
    - Selalu terapkan gabung_negasi() agar konsisten dengan training LogReg.
    """
    cleaned = _simple_clean(text)
    tokens = cleaned.split()

    if use_lexicon:
        # slang saja
        tokens = normalize_tokens_with_lexicon(tokens)
        # biarkan fitur negation POLAR_SWAP kamu tetap ada
        tokens = _apply_negation_rules(tokens)

    out = " ".join(tokens)

    # ✅ konsisten dengan training
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
    Deteksi aspek BERDASARKAN KEYWORD:
    1) BASE_ROOT (harga/aroma/tekstur/kemas/efek)  -> kuat buat "harganya"
    2) SEED_ROOTS (dari seeds.json)
    Return: (best_aspect or None, hits_per_aspect)
    """
    _, _, _, _, _, SEED_ROOTS = load_resources()

    roots = [_root_id(t) for t in tokens]
    roots_set = set(roots)

    # 1) BASE_ROOT dulu (paling eksplisit)
    for r in roots:
        for a in ASPEK:
            if BASE_ROOT[a] in r:
                # hits diset 0 saja (tidak wajib dipakai di sini)
                return a, {asp: 0 for asp in ASPEK}

    # 2) Seed-based hits
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

def segment_text_aspect_aware(text: str, use_lexicon=False):
    """
    Segmentasi sesuai logika kamu:
    - Segmen dipotong HANYA saat ditemukan keyword aspek yang BERBEDA dari aspek aktif.
    - Keyword aspek = BASE_ROOT atau SEED_ROOTS (via detect_aspect_by_seed()).
    - Kalau tidak ada keyword baru -> lanjutkan segmen yang sama.
    - Output: list of dict {seg_text, tokens, anchor_aspect, seed_hits}
    """

    _, _, bigram, _, _, _ = load_resources()

    # Tokenisasi konsisten
    toks_all = tokenize_from_val(text, bigram=bigram)
    if use_lexicon:
        toks_all = normalize_tokens_with_lexicon(toks_all)

    # Edge case: kosong
    if not toks_all:
        return [{
            "seg_text": "",
            "tokens": [],
            "anchor_aspect": None,
            "seed_hits": {a: 0 for a in ASPEK}
        }]

    # Helper: aspek token tunggal (base_root / seed)
    def aspect_of_token(tok: str):
        # Cek BASE_ROOT cepat
        r = _root_id(tok)
        for a in ASPEK:
            if BASE_ROOT[a] in r:
                return a
        # Cek seed roots
        _, _, _, _, _, SEED_ROOTS = load_resources()
        for a in ASPEK:
            if r in SEED_ROOTS[a]:
                return a
        return None

    # Scan token, potong kalau aspek berubah
    segments = []
    start = 0
    current_aspect = None

    for i, tok in enumerate(toks_all):
        a_tok = aspect_of_token(tok)

        # set aspek pertama kali ketemu
        if current_aspect is None and a_tok is not None:
            current_aspect = a_tok

        # kalau ketemu aspek baru yang beda -> cut segmen
        if a_tok is not None and current_aspect is not None and a_tok != current_aspect:
            seg_tokens = toks_all[start:i]
            seg_text = " ".join(seg_tokens).strip()

            anchor, hits = detect_aspect_by_seed(seg_tokens)
            # anchor fallback: kalau detect_aspect_by_seed gagal, pakai current_aspect lama
            if anchor is None:
                anchor = current_aspect

            if seg_text:
                segments.append({
                    "seg_text": seg_text,
                    "tokens": seg_tokens,
                    "anchor_aspect": anchor,
                    "seed_hits": hits
                })

            # mulai segmen baru
            start = i
            current_aspect = a_tok

    # segmen terakhir
    seg_tokens = toks_all[start:]
    seg_text = " ".join(seg_tokens).strip()

    anchor, hits = detect_aspect_by_seed(seg_tokens)
    if anchor is None:
        anchor = current_aspect  # boleh None kalau memang tidak ada keyword sama sekali

    if seg_text:
        segments.append({
            "seg_text": seg_text,
            "tokens": seg_tokens,
            "anchor_aspect": anchor,
            "seed_hits": hits
        })

    # Merge terakhir: kalau dua segmen berturut aspek sama -> gabung (sesuai maumu)
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        if seg["anchor_aspect"] is not None and seg["anchor_aspect"] == merged[-1]["anchor_aspect"]:
            merged[-1]["seg_text"] += " " + seg["seg_text"]
            merged[-1]["tokens"].extend(seg["tokens"])
            # seed_hits boleh dibiarkan (atau recompute), gak kritikal
        else:
            merged.append(seg)

    return merged

CONJ_SPLIT_WORDS = {"tapi", "namun", "tetapi", "sedangkan", "walaupun", "meskipun"}

def segment_text_for_aspect(text: str, use_lexicon=False):
    """
    Segmentasi berbasis:
    1) Split per kalimat
    2) Deteksi anchor via BASE_ROOT (aroma/harga/kemas/tekstur/efek)
    3) Jika > 1 anchor -> split antar anchor
    4) Jika anchor gagal tapi seed > 1 aspek -> split via conjunction words
    5) Sinkron dengan lexicon untuk ulasan tunggal
    """

    sentences = split_into_sentences(text)
    segments = []

    # ambil SEED_ROOTS untuk fallback split
    _, _, _, _, _, SEED_ROOTS = load_resources()

    for sent in sentences:
        # ✅ TOKEN BERSIH
        tokens = _simple_clean(sent).split()

        # ✅ sinkron dengan lexicon (mode ulasan tunggal)
        if use_lexicon:
            tokens = normalize_tokens_with_lexicon(tokens)

        if not tokens:
            continue

        # ============================================================
        # 1) DETEKSI ANCHOR LIST BERDASARKAN BASE_ROOT
        # ============================================================
        anchor_list = []
        for idx, tok in enumerate(tokens):
            root = _root_id(tok)

            anchored = False
            for aspek in ASPEK:
                base = BASE_ROOT[aspek]
                if base in root:  # contoh: "harganya" mengandung "harga"
                    start_pos = idx

                    # rule "segi harga" / "segi aroma"
                    if idx > 0 and tokens[idx - 1] == "segi":
                        start_pos = idx - 1

                    anchor_list.append((start_pos, aspek))
                    anchored = True
                    break

            # anchor khusus: "cocok" -> Efek
            if not anchored and root == "cocok":
                anchor_list.append((idx, "Efek"))

        # ============================================================
        # 2) FALLBACK: kalau anchor tidak cukup tapi seed muncul > 1 aspek
        # ============================================================
        detected_seed_aspects = set()
        for tok in tokens:
            r = _root_id(tok)
            for a in ASPEK:
                if r in SEED_ROOTS[a]:
                    detected_seed_aspects.add(a)

        # Kalau kalimat punya seed dari >=2 aspek tapi anchor tidak terdeteksi
        # → split berdasarkan conjunction words seperti "tapi/namun"
        if len(detected_seed_aspects) >= 2 and len(anchor_list) <= 1:
            split_pos = None
            for i, t in enumerate(tokens):
                if t in CONJ_SPLIT_WORDS:
                    split_pos = i
                    break

            if split_pos is not None:
                left = " ".join(tokens[:split_pos]).strip()
                right = " ".join(tokens[split_pos + 1:]).strip()

                if left:
                    segments.append({"seg_text": left, "anchor_aspect": None})
                if right:
                    segments.append({"seg_text": right, "anchor_aspect": None})

                continue  # lanjut ke kalimat berikutnya

        # ============================================================
        # 3) JIKA TIDAK ADA ANCHOR: MASUKKAN SEBAGAI SEGMENT UMUM
        # ============================================================
        if not anchor_list:
            segments.append({
                "seg_text": " ".join(tokens).strip(),
                "anchor_aspect": None
            })
            continue

        # ============================================================
        # 4) COMPRESS ANCHOR BERURUTAN DENGAN ASPEK SAMA
        # ============================================================
        compressed = []
        for pos, asp in sorted(anchor_list, key=lambda x: x[0]):
            if not compressed or compressed[-1][1] != asp:
                compressed.append((pos, asp))

        # ============================================================
        # 5) RULE: JIKA HANYA 1 ANCHOR -> SATU SEGMENT (ASPEK ITU)
        # ============================================================
        if len(compressed) == 1:
            _, asp = compressed[0]
            segments.append({
                "seg_text": " ".join(tokens).strip(),
                "anchor_aspect": asp
            })
            continue

        # ============================================================
        # 6) SPLIT ANTAR ANCHOR
        # ============================================================
        prev_end = 0
        for i, (pos, asp) in enumerate(compressed):
            if prev_end < pos:
                seg_tokens = tokens[prev_end:pos]
                seg_text = " ".join(seg_tokens).strip()
                if seg_text:
                    segments.append({"seg_text": seg_text, "anchor_aspect": None})

            end = compressed[i + 1][0] if i + 1 < len(compressed) else len(tokens)
            seg_tokens = tokens[pos:end]
            seg_text = " ".join(seg_tokens).strip()
            if seg_text:
                segments.append({"seg_text": seg_text, "anchor_aspect": asp})

            prev_end = end

    # ============================================================
    # 7) GABUNG SEGMENT TANPA ANCHOR KE SEGMENT ANCHOR SETELAHNYA
    # ============================================================
    attached = []
    seen_anchor = False
    i = 0

    while i < len(segments):
        curr = segments[i]
        asp_curr = curr.get("anchor_aspect", None)

        if asp_curr is not None:
            combined_text = curr["seg_text"]
            j = i + 1
            while j < len(segments) and segments[j].get("anchor_aspect") is None:
                combined_text += " " + segments[j]["seg_text"]
                j += 1

            attached.append({
                "seg_text": combined_text.strip(),
                "anchor_aspect": asp_curr,
            })
            seen_anchor = True
            i = j
        else:
            attached.append(curr)
            i += 1

    return attached


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
    """
    Versi FIX:
    - Menghormati segmentasi dari segment_text_aspect_aware()
    - TIDAK akan menggabungkan segmen jika anchor_aspect beda
    - TIDAK akan menggabungkan segmen jika seed-dominant beda (mis. Aroma vs Harga)
    - Merge segmen pendek hanya jika masih 1 aspek
    """
    _, _, bigram, _, _, _ = load_resources()

    def dominant_seed_aspect(seed_hits: dict):
        """Ambil aspek dengan seed hit terbesar. None kalau semua 0/None."""
        if not seed_hits:
            return None
        best_a = None
        best_v = 0
        for a, v in seed_hits.items():
            if v > best_v:
                best_v = v
                best_a = a
        return best_a if best_v > 0 else None

    def can_merge(prev_item: dict, curr_item: dict) -> bool:
        """
        TRUE hanya jika aman digabung:
        - kalau dua-duanya punya anchor_aspect dan beda -> FALSE
        - kalau seed dominant beda -> FALSE
        """
        ap = prev_item.get("anchor_aspect")
        ac = curr_item.get("anchor_aspect")

        if ap is not None and ac is not None and ap != ac:
            return False

        dp = dominant_seed_aspect(prev_item.get("seed_hits"))
        dc = dominant_seed_aspect(curr_item.get("seed_hits"))

        if dp is not None and dc is not None and dp != dc:
            return False

        # kalau salah satu None, kita anggap masih mungkin lanjutan
        return True

    # =========================
    # 1) ambil segmen dari segmenter
    # =========================
    seg_infos = segment_text_aspect_aware(text, use_lexicon=use_lexicon)

    if not seg_infos:
        seg_infos = [{
            "seg_text": text,
            "anchor_aspect": None,
            "tokens": tokenize_from_val(text, bigram=bigram),
            "seed_hits": {a: 0 for a in ASPEK}
        }]

    # =========================
    # 2) label per segmen
    # =========================
    labeled = []
    for info in seg_infos:
        seg = info["seg_text"]
        anchor = info.get("anchor_aspect", None)

        toks = info.get("tokens")
        if not toks:
            toks = tokenize_from_val(seg, bigram=bigram)
            if use_lexicon:
                toks = normalize_tokens_with_lexicon(toks)

        p_raw, hits_lda, p_boost, aspect_pred, aspect_top1_plain = predict_aspect_boosted(
            toks,
            lambda_boost=lambda_boost,
            gamma=gamma,
            seed_bonus=seed_bonus,
            dampen_price_if_no_seed=dampen_price_if_no_seed,
            price_delta=price_delta,
            prefer_seed_for_top1=prefer_seed_for_top1
        )

        # kalau segmenter sudah kasih seed_hits, pakai itu; kalau tidak ada, pakai hits dari predict_aspect_boosted
        seed_hits = info.get("seed_hits", None)
        if seed_hits is None:
            seed_hits = hits_lda

        aspect_final = anchor if anchor is not None else aspect_pred
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
    # 3) gabung segmen sangat pendek (TAPI hanya jika aman)
    # =========================
    merged_short = []
    for item in labeled:
        if not merged_short:
            merged_short.append(item)
            continue

        prev = merged_short[-1]

        # aturan pendek yang kamu punya
        tok_len = len(item["tokens"])
        no_anchor = item.get("anchor_aspect") is None
        total_seed_hits = sum(item["seed_hits"].values()) if item.get("seed_hits") else 0

        short_anchorless = (tok_len <= 4 and no_anchor and total_seed_hits == 0)
        very_short_any = (tok_len <= 2)

        # ✅ kunci: hanya merge kalau "aman" (tidak beda aspek via anchor/seed)
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

            # anchor tidak boleh berubah
            anchor_combined = prev.get("anchor_aspect", None) if prev.get("anchor_aspect", None) is not None else item.get("anchor_aspect", None)
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
    # 4) gabung segmen ber-aspek sama (TAPI hanya jika aman)
    # =========================
    merged = []
    for item in merged_short:
        if not merged:
            merged.append(item)
            continue

        prev = merged[-1]

        # kalau aspek final sama + tidak ada konflik anchor + aman dari seed/anchor -> merge
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
    # 5) output
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
        segments = test_segmented_text(text, use_lexicon=False)

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
