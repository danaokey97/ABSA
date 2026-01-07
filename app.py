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
KATADASAR_FILE  = os.path.join(BASE_DIR, "kata_dasar.txt")   # TODO: tidak dipakai saat ini

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
    Tokenisasi standar untuk korpus LDA (tanpa lexicon).
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
# LEXICON: SLANG UNTUK PREPROSES SINGLE TEXT
# =====================================================
@st.cache_resource
def load_lexicons():
    """
    Load slang dictionary saja (tanpa kata dasar).
    Format slang yang didukung:
    1) TXT/CSV biasa: ga, tidak  / gk\t tidak
    2) potongan JSON: "ga": "tidak"
    """
    slang_dict = {}

    if os.path.exists(SLANG_FILE):
        with open(SLANG_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        pair_pattern = re.compile(r'"([^"]+)"\s*:\s*"([^"]+)"')
        for s, n in pair_pattern.findall(content):
            slang_dict[s.strip().lower()] = n.strip().lower()

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


def normalize_tokens_with_lexicon(tokens):
    """
    Normalisasi token dengan slang saja.
    Dipakai untuk ulasan tunggal (segmentasi + sentimen) jika use_lexicon=True.
    """
    slang_dict = load_lexicons()
    norm_tokens = []

    for tok in tokens:
        t = tok.lower().strip()
        if not t:
            continue

        if t in {"ga", "gak", "gk", "engga", "enggak", "nggak", "g"}:
            t = "tidak"
        else:
            t = slang_dict.get(t, t)

        norm_tokens.append(t)

    return norm_tokens


# =====================================================
# NEGATION HANDLING
# =====================================================
NEG_WORDS = {"ga", "gak", "gk", "tidak", "enggak", "nggak", "g"}

POLAR_SWAP = {
    "mahal": "murah",
    "murah": "mahal",

    "bagus": "jelek",
    "jelek": "bagus",
    "buruk": "bagus",
    "oke": "tidak_oke",

    "berat": "ringan",
    "ringan": "berat",
    "lengket": "tidak_lengket",

    "perih": "nyaman",
    "iritasi": "nyaman",
    "jerawatan": "tidak_jerawatan",
}

def _apply_negation_rules(tokens):
    """
    - Jika (NEG_WORD + kata) ada di POLAR_SWAP -> ganti kata jadi antonim.
    - Jika tidak ada -> biarkan negasi tetap muncul.
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
                new_tokens.append(POLAR_SWAP[nxt])
                skip_next = True
            else:
                new_tokens.append(t)
                skip_next = False
        else:
            new_tokens.append(t)

    return new_tokens


def gabung_negasi(text: str) -> str:
    """
    Gabungkan negasi + 1 kata setelahnya jadi token underscore.
    Contoh: tidak kasar -> tidak_kasar
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

    if re.search(r"\b(tidak|ga|gak|nggak|enggak|tak|tdk|bukan|kurang|gk|g)\b", out):
        out = gabung_negasi(out)

    return out


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

    SEED_DICT = {a: set(sj.get(a, [])) for a in ASPEK}

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


def predict_sentiment_for_segment(seg_text: str, aspek: str, sent_models: dict, use_lexicon: bool = False):
    if aspek not in sent_models:
        return None, None

    clf, vec = sent_models[aspek]
    X = vec.transform([preprocess_for_sentiment(seg_text, use_lexicon=use_lexicon)])
    y_pred = clf.predict(X)[0]

    prob_pos = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        classes = list(clf.classes_)

        # robust positive label
        pos_labels = ["Positive", "positive", "Positif", "positif", 1]
        pos_index = None
        for lab in pos_labels:
            if lab in classes:
                pos_index = classes.index(lab)
                break

        if pos_index is not None:
            prob_pos = float(proba[pos_index])

    return y_pred, prob_pos


# =====================================================
# DETEKSI ASPEK + SEGMENTASI
# =====================================================
SEGMENT_STOPWORDS = {
    "tidak", "gak", "nggak", "enggak", "ga", "g",
    "banget", "aja", "sih", "dong", "kok",
    "dan", "atau", "yang", "itu", "ini",
    "enak", "dipake", "pake", "nyaman", "kurang"
}

BASE_ROOT = {
    "Kemasan": "kemas",
    "Aroma": "aroma",
    "Tekstur": "tekstur",
    "Harga": "harga",
    "Efek": "efek",
}

CONJ_SPLIT_WORDS = {
    "tapi", "namun", "tetapi", "sedangkan",
    "walaupun", "meskipun", "cuma", "hanya",
    "trs", "terus", "lalu", "kemudian"
}


def bow_of(tokens, dictionary):
    return dictionary.doc2bow([t for t in tokens if t in dictionary.token2id])


def predict_aspect_boosted(
    tokens,
    dictionary,
    lda,
    topic2aspect,
    SEED_DICT,
    lambda_boost=0.9,
    gamma=2.0,
    seed_bonus=0.03,
    dampen_price_if_no_seed=True,
    price_delta=0.7,
    prefer_seed_for_top1=True,
):
    bow = bow_of(tokens, dictionary)
    dist_pairs = lda.get_document_topics(bow, minimum_probability=0.0)

    p_aspek = {a: 0.0 for a in ASPEK}
    for k, p in dist_pairs:
        a = topic2aspect.get(k, f"T{k}")
        if a in p_aspek:
            p_aspek[a] += p

    toks_for_seed = _expand_for_seed(tokens) | _expand_for_seed(_simple_clean(" ".join(tokens)).split())
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


def split_by_punct_and_conj(text: str):
    text = str(text).replace("\n", ". ")
    parts = re.split(r"[.!?;:]+", text)
    parts = [p.strip() for p in parts if p.strip()]

    out = []
    for p in parts:
        toks = _simple_clean(p).split()
        if not toks:
            continue

        cut = None
        for i, t in enumerate(toks):
            if t in CONJ_SPLIT_WORDS and 0 < i < len(toks) - 1:
                cut = i
                break

        if cut is None:
            out.append(" ".join(toks).strip())
        else:
            left = " ".join(toks[:cut]).strip()
            right = " ".join(toks[cut + 1:]).strip()
            if left:
                out.append(left)
            if right:
                out.append(right)

    return out


def detect_aspect_simple(tokens, SEED_ROOTS):
    roots = {_root_id(t) for t in tokens}
    score = {a: 0 for a in ASPEK}

    for a in ASPEK:
        score[a] += len(SEED_ROOTS[a] & roots)

    for r in roots:
        for a in ASPEK:
            if BASE_ROOT[a] in r:
                score[a] += 3

    best_a = max(score, key=score.get)
    if score[best_a] == 0:
        return None, score
    return best_a, score


def segment_text_merge_by_aspect(text: str, bigram, SEED_ROOTS, use_lexicon=False):
    """
    SEGMENTASI STABIL + CUT NATURAL:
    - Split kasar punctuation + conjunction
    - Switch aspek hanya jika:
        1) BASE_ROOT eksplisit
        2) seed aspect kuat (>=2 hits dalam window)
    - KHUSUS switch ke Efek:
        cut tidak langsung di token sekarang, tapi digeser ke kata pembuka efek:
        {"bikin","jadi","hasilnya","membuat","menjadikan"}
    - Segmen tidak boleh terlalu pendek
    """

    chunks = split_by_punct_and_conj(text)
    if not chunks:
        return []

    WINDOW = 6
    SEED_MIN_HITS = 2
    MIN_WORDS = 4

    # marker yang biasanya jadi awal "Efek"
    EFFECT_ANCHORS = {"bikin", "jadi", "hasilnya", "membuat", "menjadikan", "efeknya", "hasil"}

    def aspect_from_base_root(tok):
        r = _root_id(tok)
        for a in ASPEK:
            if BASE_ROOT[a] in r:
                return a
        return None

    def seed_hits_in_window(tokens_window):
        roots = {_root_id(t) for t in tokens_window}
        score = {a: len(SEED_ROOTS[a] & roots) for a in ASPEK}
        best_a = max(score, key=score.get)
        return best_a, score[best_a], score

    segs = []
    last_aspect = None

    for ch in chunks:
        toks_plain = _simple_clean(ch).split()
        if use_lexicon:
            toks_plain = normalize_tokens_with_lexicon(toks_plain)

        if not toks_plain:
            continue

        # tentukan aspek awal
        current_aspect = None
        start = 0

        # base-root dulu
        for t in toks_plain:
            a0 = aspect_from_base_root(t)
            if a0 is not None:
                current_aspect = a0
                break

        # seed kuat awal chunk
        if current_aspect is None:
            best_a, best_hits, _ = seed_hits_in_window(toks_plain[:WINDOW])
            if best_hits >= SEED_MIN_HITS:
                current_aspect = best_a
            else:
                current_aspect = last_aspect

        i = 0
        while i < len(toks_plain):
            tok = toks_plain[i]

            # === (1) BASE ROOT SWITCH ===
            a_base = aspect_from_base_root(tok)
            if a_base is not None and current_aspect is not None and a_base != current_aspect:
                left_tokens = toks_plain[start:i]
                if len(left_tokens) >= MIN_WORDS:
                    left_text = " ".join(left_tokens).strip()
                    toks_lda = tokenize_from_val(left_text, bigram=bigram)
                    if use_lexicon:
                        toks_lda = normalize_tokens_with_lexicon(toks_lda)

                    segs.append({
                        "seg_text": left_text,
                        "tokens": toks_lda,
                        "anchor_aspect": current_aspect,
                        "seed_hits": {a: 0 for a in ASPEK}
                    })

                    start = i
                    current_aspect = a_base
                    last_aspect = current_aspect

                i += 1
                continue

            # === (2) SEED STRONG SWITCH ===
            win = toks_plain[i:i+WINDOW]
            best_a, best_hits, _ = seed_hits_in_window(win)

            if (best_a is not None and current_aspect is not None and best_a != current_aspect and best_hits >= SEED_MIN_HITS):
                
                cut_pos = i

                # ✅ KHUSUS jika aspek baru = Efek
                # geser cut ke kata anchor efek supaya natural
                if best_a == "Efek":
                    for j in range(i, min(i+WINDOW, len(toks_plain))):
                        if _root_id(toks_plain[j]) in EFFECT_ANCHORS:
                            cut_pos = j
                            break

                left_tokens = toks_plain[start:cut_pos]
                if len(left_tokens) >= MIN_WORDS:
                    left_text = " ".join(left_tokens).strip()
                    toks_lda = tokenize_from_val(left_text, bigram=bigram)
                    if use_lexicon:
                        toks_lda = normalize_tokens_with_lexicon(toks_lda)

                    segs.append({
                        "seg_text": left_text,
                        "tokens": toks_lda,
                        "anchor_aspect": current_aspect,
                        "seed_hits": {a: 0 for a in ASPEK}
                    })

                    start = cut_pos
                    current_aspect = best_a
                    last_aspect = current_aspect

            i += 1

        # segmen terakhir
        last_tokens = toks_plain[start:]
        if last_tokens:
            last_text = " ".join(last_tokens).strip()
            toks_lda = tokenize_from_val(last_text, bigram=bigram)
            if use_lexicon:
                toks_lda = normalize_tokens_with_lexicon(toks_lda)

            segs.append({
                "seg_text": last_text,
                "tokens": toks_lda,
                "anchor_aspect": current_aspect,
                "seed_hits": {a: 0 for a in ASPEK}
            })

    # merge aspek sama
    merged = []
    for item in segs:
        if not merged:
            merged.append(item)
            continue

        if item["anchor_aspect"] is not None and merged[-1]["anchor_aspect"] == item["anchor_aspect"]:
            merged[-1]["seg_text"] += " " + item["seg_text"]
            merged[-1]["tokens"].extend(item["tokens"])
        else:
            merged.append(item)

    return merged

def test_segmented_text(
    text,
    dictionary,
    lda,
    bigram,
    topic2aspect,
    SEED_DICT,
    SEED_ROOTS,
    lambda_boost=0.9,
    gamma=2.0,
    seed_bonus=0.03,
    dampen_price_if_no_seed=True,
    price_delta=0.7,
    prefer_seed_for_top1=True,
    use_lexicon=False,
):
    seg_infos = segment_text_merge_by_aspect(text, bigram, SEED_ROOTS, use_lexicon=use_lexicon)

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
            dictionary=dictionary,
            lda=lda,
            topic2aspect=topic2aspect,
            SEED_DICT=SEED_DICT,
            lambda_boost=lambda_boost,
            gamma=gamma,
            seed_bonus=seed_bonus,
            dampen_price_if_no_seed=dampen_price_if_no_seed,
            price_delta=price_delta,
            prefer_seed_for_top1=prefer_seed_for_top1
        )

        seed_hits = info.get("seed_hits", hits_lda)
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

    results = []
    for i, r in enumerate(labeled, start=1):
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
def run_absa_on_dataframe(df_raw, _sent_models, dictionary, lda, bigram, topic2aspect, SEED_DICT, SEED_ROOTS):
    """
    FIX: dataset pakai use_lexicon=False agar konsisten dengan training LDA (dictionary)
    Sentimen dataset default use_lexicon=False (ubah jika training logreg pakai lexicon)
    """
    data_rows = []

    for idx, row in df_raw.iterrows():
        text = str(row["text-content"])

        # ✅ Dataset -> use_lexicon=False (LDA harus match dictionary)
        segments = test_segmented_text(
            text,
            dictionary=dictionary,
            lda=lda,
            bigram=bigram,
            topic2aspect=topic2aspect,
            SEED_DICT=SEED_DICT,
            SEED_ROOTS=SEED_ROOTS,
            use_lexicon=False
        )

        for seg in segments:
            aspek = seg["aspect_final"]
            seg_text = seg["seg_text"]

            # ✅ Dataset sentimen -> default False (ubah kalau training logreg pakai lexicon)
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

        menu = st.radio("", ["Ulasan Tunggal", "Dashboard Dataset"], index=0)

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

            pre_single = preprocess_for_sentiment(text, use_lexicon=True)
            st.markdown("**Teks setelah preprocessing (slang → baku + negation handling):**")
            st.code(pre_single, language="text")

            results = test_segmented_text(
                text,
                dictionary=dictionary,
                lda=lda,
                bigram=bigram,
                topic2aspect=topic2aspect,
                SEED_DICT=SEED_DICT,
                SEED_ROOTS=SEED_ROOTS,
                use_lexicon=True
            )

            rows = []
            for r in results:
                aspek = r["aspect_final"]
                seg_text = r["seg_text"]

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

        uploaded = st.file_uploader("Upload file CSV/Excel Female Daily:", type=["csv", "xlsx"])

        if uploaded is not None:
            if uploaded.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)

            st.success(f"File berhasil dimuat: {df_raw.shape[0]} baris")

            with st.spinner("Memproses ABSA seluruh dataset..."):
                df_seg = run_absa_on_dataframe(
                    df_raw,
                    sent_models,
                    dictionary, lda, bigram, topic2aspect, SEED_DICT, SEED_ROOTS
                )

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
            c1.metric(label="Jumlah Ulasan Positif", value=pos_count)
            c2.metric(label="Jumlah Ulasan Negatif", value=neg_count)
            c3.metric(label="Total Aspek", value=total_aspek)

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

            # ✅ auto columns sesuai jumlah aspek
            cols = st.columns(len(list_aspek))

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
                        "Positive": color_map_aspek.get(aspek, ["#2ecc71","#1c973b"])[0],
                        "Negative": color_map_aspek.get(aspek, ["#2ecc71","#1c973b"])[1]
                    }
                )

                fig_donut.update_layout(margin=dict(l=0, r=0, t=70, b=0), height=240)
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
            fig_bar.update_layout(margin=dict(l=20, r=20, t=20, b=20))
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
            fig2.update_layout(margin=dict(l=20, r=20, t=40, b=20))
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
            fig3.update_layout(margin=dict(l=20, r=20, t=40, b=20))
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

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### WordCloud Positif")

                wc_pos = WordCloud(
                    width=900,
                    height=500,
                    background_color="white",
                    colormap="Greens",
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
                    colormap="Reds",
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
