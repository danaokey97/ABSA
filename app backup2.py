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

ASPEK = ["Kemasan", "Aroma", "Tekstur", "Harga", "Efek"]

# =====================================================
# UTIL PREPROCESSING
# =====================================================

def _simple_clean(text: str) -> str:
    t = str(text).lower()
    t = t.replace("enggak", "gak").replace("nggak", "gak")
    return re.sub(r"[^a-z0-9_ ]+", " ", t)

def _root_id(token: str) -> str:
    t = str(token).lower().strip()
    t = re.sub(r'(ku|mu|nya)$', '', t)
    t = re.sub(r'^([a-z0-9]+)_\1$', r'\1', t)
    return t

def tokenize_from_val(val, bigram=None):
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

def preprocess_for_sentiment(text: str) -> str:
    cleaned = _simple_clean(text)
    tokens = cleaned.split()
    return " ".join(tokens)

def predict_sentiment_for_segment(seg_text: str, aspek: str, sent_models: dict):
    if aspek not in sent_models:
        return None, None

    clf, vec = sent_models[aspek]
    X = vec.transform([preprocess_for_sentiment(seg_text)])

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

SEGMENT_STOPWORDS = {
    "tidak", "gak", "nggak", "enggak", "ga",
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


def segment_text_for_aspect(text: str):
    sentences = split_into_sentences(text)
    segments = []

    # --- 1) Segmentasi awal per kalimat + anchor BASE_ROOT + 'cocok' ---
    for sent in sentences:
        tokens = sent.split()
        if not tokens:
            continue

        anchor_list = []
        for idx, tok in enumerate(tokens):
            root = _root_id(_simple_clean(tok))

            anchored = False
            for aspek in ASPEK:
                base = BASE_ROOT[aspek]
                if base in root:
                    start_pos = idx
                    if idx > 0:
                        prev_root = _root_id(_simple_clean(tokens[idx - 1]))
                        if prev_root == "segi":
                            start_pos = idx - 1
                    anchor_list.append((start_pos, aspek))
                    anchored = True
                    break

            if not anchored and root == "cocok":
                anchor_list.append((idx, "Efek"))

        # tidak ada anchor → segmen umum (non-aspek)
        if not anchor_list:
            segments.append({
                "seg_text": sent.strip(),
                "anchor_aspect": None
            })
            continue

        # compress anchor yang berurutan dengan aspek sama
        compressed = []
        for pos, asp in sorted(anchor_list, key=lambda x: x[0]):
            if not compressed or compressed[-1][1] != asp:
                compressed.append((pos, asp))

        # RULE: jika hanya 1 anchor dalam kalimat → seluruh kalimat milik aspek itu
        if len(compressed) == 1:
            _, asp = compressed[0]
            segments.append({
                "seg_text": sent.strip(),
                "anchor_aspect": asp
            })
            continue

        # kalau anchor-nya > 1 → gunakan logika split
        prev_end = 0
        for i, (pos, asp) in enumerate(compressed):
            if prev_end < pos:
                seg_tokens = tokens[prev_end:pos]
                seg_text = " ".join(seg_tokens).strip(" ,")
                if seg_text:
                    segments.append({
                        "seg_text": seg_text,
                        "anchor_aspect": None
                    })

            end = compressed[i + 1][0] if i + 1 < len(compressed) else len(tokens)
            seg_tokens = tokens[pos:end]
            seg_text = " ".join(seg_tokens).strip(" ,")
            if seg_text:
                segments.append({
                    "seg_text": seg_text,
                    "anchor_aspect": asp
                })

            prev_end = end

    # --- 2) TANPA refinement ekor: langsung pakai segments apa adanya ---
    refined = segments

    # --- 3) Gabungkan segmen tanpa anchor ---
    attached = []
    seen_anchor = False

    i = 0
    while i < len(refined):
        curr = refined[i]
        asp_curr = curr.get("anchor_aspect", None)

        if asp_curr is not None:
            combined_text = curr["seg_text"]
            j = i + 1
            while j < len(refined) and refined[j].get("anchor_aspect") is None:
                combined_text += " " + refined[j]["seg_text"]
                j += 1

            attached.append({
                "seg_text": combined_text.strip(),
                "anchor_aspect": asp_curr,
            })
            seen_anchor = True
            i = j
        else:
            tokens = curr["seg_text"].split()
            if (
                not seen_anchor
                and len(tokens) <= 4
                and i < len(refined) - 1
                and refined[i + 1].get("anchor_aspect") is not None
            ):
                nxt = refined[i + 1]
                combined_text = curr["seg_text"] + " " + nxt["seg_text"]

                attached.append({
                    "seg_text": combined_text.strip(),
                    "anchor_aspect": nxt["anchor_aspect"],
                })
                seen_anchor = True
                i = i + 2
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
    prefer_seed_for_top1=True
):
    _, _, bigram, _, _, _ = load_resources()

    seg_infos = segment_text_for_aspect(text)

    labeled = []
    for info in seg_infos:
        seg = info["seg_text"]
        anchor = info.get("anchor_aspect", None)

        toks = tokenize_from_val(seg, bigram=bigram)

        p_raw, hits, p_boost, aspect_pred, aspect_top1_plain = predict_aspect_boosted(
            toks,
            lambda_boost=lambda_boost,
            gamma=gamma,
            seed_bonus=seed_bonus,
            dampen_price_if_no_seed=dampen_price_if_no_seed,
            price_delta=price_delta,
            prefer_seed_for_top1=prefer_seed_for_top1
        )

        aspect_final = anchor if anchor is not None else aspect_pred
        prob_final   = p_boost[aspect_final]

        labeled.append({
            "seg_text": seg,
            "anchor_aspect": anchor,
            "tokens": toks,
            "p_boost": p_boost,
            "seed_hits": hits,
            "aspect_final": aspect_final,
            "aspect_prob_final": prob_final,
        })

    # gabung segmen sangat pendek
    merged_short = []
    for item in labeled:
        tok_len = len(item["tokens"])
        if not merged_short:
            merged_short.append(item)
            continue

        no_anchor = item.get("anchor_aspect") is None
        total_seed_hits = sum(item["seed_hits"].values())

        short_anchorless = (tok_len <= 4 and no_anchor and total_seed_hits == 0)
        very_short_any  = (tok_len <= 2)

        if short_anchorless or very_short_any:
            prev = merged_short[-1]
            combined_text = prev["seg_text"].rstrip(" ,") + " " + item["seg_text"].lstrip(" ,")
            combined_tokens = tokenize_from_val(combined_text, bigram=bigram)

            p_raw2, hits2, p_boost2, aspect2, aspect_top1_plain2 = predict_aspect_boosted(
                combined_tokens,
                lambda_boost=lambda_boost,
                gamma=gamma,
                seed_bonus=seed_bonus,
                dampen_price_if_no_seed=dampen_price_if_no_seed,
                price_delta=price_delta,
                prefer_seed_for_top1=prefer_seed_for_top1
            )

            anchor_combined = prev.get("anchor_aspect", None)
            aspect_final2   = anchor_combined if anchor_combined is not None else aspect2

            merged_short[-1] = {
                "seg_text": combined_text,
                "anchor_aspect": anchor_combined,
                "tokens": combined_tokens,
                "p_boost": p_boost2,
                "seed_hits": hits2,
                "aspect_final": aspect_final2,
                "aspect_prob_final": p_boost2[aspect_final2],
            }
        else:
            merged_short.append(item)

    # gabung segmen ber-aspek sama
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

        if same_aspect and not anchor_conflict:
            combined_text = prev["seg_text"].rstrip(" ,") + " " + item["seg_text"].lstrip(" ,")
            combined_tokens = tokenize_from_val(combined_text, bigram=bigram)

            p_raw2, hits2, p_boost2, aspect2, aspect_top1_plain2 = predict_aspect_boosted(
                combined_tokens,
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
                "tokens": combined_tokens,
                "p_boost": p_boost2,
                "seed_hits": hits2,
                "aspect_final": aspect_final2,
                "aspect_prob_final": p_boost2[aspect_final2],
            }
        else:
            merged.append(item)

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

        segments = test_segmented_text(text)

        for seg in segments:
            aspek = seg["aspect_final"]
            seg_text = seg["seg_text"]

            sent_label, _ = predict_sentiment_for_segment(seg_text, aspek, _sent_models)

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

            results = test_segmented_text(text)

            rows = []
            for r in results:
                aspek = r["aspect_final"]
                seg_text = r["seg_text"]
                sent_label, _ = predict_sentiment_for_segment(seg_text, aspek, sent_models)

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
        © 2025 Rifqi — Female Daily Review Analysis Dashboard  
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
