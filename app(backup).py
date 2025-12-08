import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

# =====================================================
# PATH – DISKONFIG SESUAI FOLDER ABSA
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.join(BASE_DIR, "MODEL_LDA_5ASPEK_NEW")
MODEL_DIR   = os.path.join(ROOT, "Model_LDA")
ARTEFAK_DIR = os.path.join(ROOT, "artefak")

# folder tempat logreg_*.pkl dan tfidf_*.pkl
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
        "Aroma": set(sj.get("Aroma", [])),
        "Tekstur": set(sj.get("Tekstur", [])),
        "Harga": set(sj.get("Harga", [])),
        "Efek": set(sj.get("Efek", [])),
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
# LOAD Logistic Regression MODELS
# =====================================================

@st.cache_resource
def load_sentiment_models():
    """
    Load model sentimen per aspek.
    Ekspektasi nama file:
      - MODEL/logreg_kemasan.pkl, logreg_aroma.pkl, ...
      - MODEL/tfidf_kemasan.pkl, tfidf_aroma.pkl, ...
    """
    models = {}

    for aspek in ASPEK:
        key = aspek.lower().replace(" ", "_")

        # ✅ perbaikan: logrec_ → logreg_
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
    "Kemasan": "kemasan",
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
        if a in p_aspek:
            p_aspek[a] += p
        # kalau ada topic yang tidak dipetakan ke 5 aspek, di-skip saja

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

        if not anchor_list:
            segments.append({
                "seg_text": sent.strip(),
                "anchor_aspect": None
            })
            continue

        compressed = []
        for pos, asp in sorted(anchor_list, key=lambda x: x[0]):
            if not compressed or compressed[-1][1] != asp:
                compressed.append((pos, asp))

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

    # --- 2) Refinement ekor segmen milik aspek berikutnya ---
    refined = []
    BACK_WINDOW = 4

    i = 0
    while i < len(segments):
        curr = segments[i]

        if i < len(segments) - 1:
            nxt = segments[i + 1]
            a1 = curr.get("anchor_aspect", None)
            a2 = nxt.get("anchor_aspect", None)

            if a1 is not None and a2 is not None and a1 != a2:
                orig_tokens = curr["seg_text"].split()
                split_idx = None

                for j, tok in enumerate(orig_tokens):
                    asp_tok = detect_aspect_from_token(tok)
                    if asp_tok == a2:
                        split_idx = max(0, j - BACK_WINDOW)
                        break

                if split_idx is not None and 0 < split_idx < len(orig_tokens) - 1:
                    left_text  = " ".join(orig_tokens[:split_idx]).strip(" ,")
                    right_head = " ".join(orig_tokens[split_idx:]).strip(" ,")

                    if left_text:
                        refined.append({
                            "seg_text": left_text,
                            "anchor_aspect": a1
                        })

                    combined_text = (right_head + " " + nxt["seg_text"]).strip()

                    refined.append({
                        "seg_text": combined_text,
                        "anchor_aspect": a2
                    })

                    i += 2
                    continue

        refined.append(curr)
        i += 1

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

def main():
    st.set_page_config(page_title="ABSA – LDA + Logistic Regression", layout="wide")
    st.title("🔍 Aspect-Based Sentiment – Deteksi Aspek (LDA) + Sentimen (Logistic Regression)")

    st.markdown("""
    Aplikasi ini melakukan:

    1. **Segmentasi & deteksi aspek** menggunakan LDA + seed words.  
    2. **Klasifikasi sentimen per aspek** menggunakan **Logistic Regression + TF-IDF**.  
    3. **Dashboard analisis dataset** Female Daily berdasarkan aspek & sentimen.
    """)

    # load resources
    try:
        dictionary, lda, bigram, topic2aspect, SEED_DICT, SEED_ROOTS = load_resources()
        st.success("Model & artefak LDA berhasil dimuat.")
    except Exception as e:
        st.error(f"Gagal memuat model LDA: {e}")
        st.stop()

    sent_models = load_sentiment_models()
    if not sent_models:
        st.error("Model sentimen Logistic Regression tidak ditemukan. Cek lagi nama file logreg_*.pkl dan tfidf_*.pkl di folder MODEL.")
        st.stop()
    else:
        st.success("Model sentimen Logistic Regression per aspek berhasil dimuat.")

    # =================================================================
    #                     **TAB MENU 1 & 2**
    # =================================================================
    tab_single, tab_dashboard = st.tabs(
        ["✏️ Ulasan Tunggal", "📊 Dashboard Dataset"]
    )

    # -------------------------------------------------------------
    # TAB 1 — ULASAN TUNGGAL
    # -------------------------------------------------------------
    with tab_single:
        st.subheader("✏️ Analisis Ulasan Tunggal")

        text = st.text_area(
            "Masukkan teks ulasan:",
            value="",
            height=160,
            placeholder="Masukkan ulasan..."
        )

        if st.button("🚀 Deteksi Aspek + Sentimen", key="btn_single"):
            if not text.strip():
                st.warning("Teks kosong.")
                st.stop()

            results = test_segmented_text(text)

            rows = []
            for r in results:
                aspek = r["aspect_final"]
                seg_text = r["seg_text"]

                sent_label, prob_pos = predict_sentiment_for_segment(
                    seg_text, aspek, sent_models
                )

                rows.append({
                    "Segmen": r["seg_index"],
                    "Teks Segmen": seg_text,
                    "Aspek": aspek,
                    "Sentimen": sent_label,
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2 — DASHBOARD DATASET
    # -------------------------------------------------------------
    with tab_dashboard:
        st.subheader("📊 Dashboard Analisis Dataset")

        uploaded = st.file_uploader(
            "Upload file CSV/Excel Female Daily",
            type=["csv", "xlsx"]
        )

        if uploaded is not None:
            if uploaded.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)

            st.success(f"File dimuat: {df_raw.shape[0]} baris")
            st.write(df_raw.head())

            # 🧠 Proses ABSA dataset
            with st.spinner("Memproses ABSA seluruh dataset..."):
                df_seg = run_absa_on_dataframe(df_raw, sent_models)

            # ---------- Insight 1 ----------
            st.markdown("### 1️⃣ Jumlah Data Sebelum & Sesudah Segmentasi")
            c1, c2 = st.columns(2)
            c1.metric("Jumlah Ulasan", df_raw.shape[0])
            c2.metric("Jumlah Segmen", df_seg.shape[0])

            # ---------- Insight 2 ----------
            st.markdown("### 2️⃣ Distribusi Sentimen per Aspek")
            dist_aspek = (
                df_seg.groupby(["Aspek", "Sentimen"])
                .size().reset_index(name="count")
            )
            st.dataframe(dist_aspek)
            st.bar_chart(dist_aspek.pivot(index="Aspek", columns="Sentimen", values="count").fillna(0))

            # ---------- Insight 3 ----------
            st.markdown("### 3️⃣ Skin Type vs Sentimen (Aspek Efek)")
            efek = df_seg[df_seg["Aspek"] == "Efek"]
            if not efek.empty:
                dist_skin = (
                    efek.groupby(["SkinType", "Sentimen"])
                    .size().reset_index(name="count")
                )
                st.dataframe(dist_skin)
                st.bar_chart(dist_skin.pivot(index="SkinType", columns="Sentimen", values="count").fillna(0))

            # ---------- Insight 4 ----------
            st.markdown("### 4️⃣ Usia vs Sentimen per Aspek")
            dist_age = (
                df_seg.groupby(["Age", "Aspek", "Sentimen"])
                .size().reset_index(name="count")
            )
            st.dataframe(dist_age)

            # ---------- Insight 5 ----------
            st.markdown("### 5️⃣ Sentimen Efek berdasarkan SkinType & Usia")
            skin_list = sorted(df_seg["SkinType"].dropna().unique())

            if skin_list:
                choose_skin = st.selectbox("Pilih jenis kulit:", skin_list)

                efek2 = df_seg[(df_seg["Aspek"] == "Efek") & (df_seg["SkinType"] == choose_skin)]
                dist_age_skin = (
                    efek2.groupby(["Age", "Sentimen"])
                    .size().reset_index(name="count")
                )
                st.bar_chart(dist_age_skin.pivot(index="Age", columns="Sentimen", values="count").fillna(0))

            # ---------- Insight 6 ----------
            st.markdown("### 6️⃣ Dataframe Segmen (Aspek + Sentimen)")
            st.dataframe(df_seg, use_container_width=True)


if __name__ == "__main__":
    main()
