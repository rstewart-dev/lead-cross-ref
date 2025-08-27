import io
import re
import unicodedata
from typing import List

import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Lead Cross-Reference Tool", page_icon="🔎", layout="wide")
st.title("🔎 Lead Cross-Reference Tool")

st.markdown(
    "Upload **exactly 2 CSVs**, choose the **key column** in each file, select match type "
    "(Exact or Imperfect), and download the matched results. This app always uses an **inner join** "
    "(i.e., shows only rows that appear in both files)."
)

# ---------- Helpers ----------
def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_email(s: str) -> str:
    return normalize_text(s)

def normalize_phone(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"\D", "", s)  # keep digits
    if len(s) > 10:
        s = s[-10:]  # keep last 10 digits (US-ish)
    return s

NORMALIZERS = {
    "General (trim & lowercase)": normalize_text,
    "Email": normalize_email,
    "Phone (US-ish)": normalize_phone,
}

def cleaned_key_series(series: pd.Series, normalizer) -> pd.Series:
    return series.astype(str).map(normalizer)

def imperfect_link_two(df1: pd.DataFrame, df2: pd.DataFrame, k1: str, k2: str, threshold: int, scorer=fuzz.WRatio):
    candidates = df2[k2].tolist()
    matches = []
    for idx, val in df1[k1].items():
        if not val:
            matches.append((None, 0))
            continue
        best = process.extractOne(val, candidates, scorer=scorer)
        if best and best[1] >= threshold:
            match_val = best[0]
            score = best[1]
        else:
            match_val = None
            score = 0
        matches.append((match_val, score))
    link_df = df1.copy()
    link_df["_match_value"] = [m[0] for m in matches]
    link_df["_match_score"] = [m[1] for m in matches]
    merged = link_df.merge(df2, left_on="_match_value", right_on=k2, how="inner", suffixes=("", "__2"))
    return merged

# ---------- UI ----------
uploaded_files = st.file_uploader(
    "Upload two CSV files", type=["csv"], accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) < 2:
        st.info("Please upload **two** CSV files.")
        st.stop()
    if len(uploaded_files) > 2:
        st.error("You uploaded more than two files. Please remove extras and keep only **two** CSVs.")
        st.stop()

    st.success("Loaded 2 files. Configure settings below.")

    dfs: List[pd.DataFrame] = []
    names: List[str] = []
    for f in uploaded_files[:2]:
        try:
            df = pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""])
        except Exception:
            f.seek(0)
            df = pd.read_csv(f, dtype=str, encoding="utf-16", keep_default_na=False, na_values=[""])
        dfs.append(df)
        names.append(f.name)

    st.subheader("1) Choose a key column for each file")
    key_cols: List[str] = []
    for i, df in enumerate(dfs):
        with st.expander(f"Columns in {names[i]}", expanded=(i == 0)):
            st.dataframe(df.head(5), use_container_width=True)
        key = st.selectbox(
            f"Key column for **{names[i]}**",
            options=list(df.columns),
            key=f"key_{i}",
        )
        key_cols.append(key)

    st.subheader("2) Normalize keys")
    normalizer_name = st.selectbox("Choose a normalization preset", list(NORMALIZERS.keys()))
    normalizer = NORMALIZERS[normalizer_name]
    st.caption("Tip: Use Email or Phone normalizer if that’s your key. Otherwise General works well.")

    norm_dfs: List[pd.DataFrame] = []
    norm_keys: List[str] = []
    for i, df in enumerate(dfs):
        dfc = df.copy()
        norm_key = f"__norm_key_{i}"
        dfc[norm_key] = cleaned_key_series(dfc[key_cols[i]], normalizer)
        norm_dfs.append(dfc)
        norm_keys.append(norm_key)

    st.subheader("3) Match options")
    match_type = st.radio("Match type", ["Exact (2 files, inner join)", "Imperfect (2 files, with score)"])

    if match_type == "Imperfect (2 files, with score)":
        threshold = st.slider("Imperfect match threshold (0–100)", min_value=60, max_value=100, value=88, step=1)

    st.subheader("4) Run")
    run = st.button("Find matches")

    if run:
        with st.spinner("Matching..."):
            if match_type == "Exact (2 files, inner join)":
                # Always inner join on normalized keys
                left = norm_dfs[0]
                right = norm_dfs[1]
                result = left.merge(
                    right,
                    left_on=norm_keys[0],
                    right_on=norm_keys[1],
                    how="inner",
                    suffixes=("", "__2")
                )
            else:
                # Imperfect two-file link with threshold
                result = imperfect_link_two(norm_dfs[0], norm_dfs[1], norm_keys[0], norm_keys[1], threshold=threshold)

        st.success("Done! Preview below.")
        st.dataframe(result.head(500), use_container_width=True)

        buff = io.StringIO()
        result.to_csv(buff, index=False)
        st.download_button(
            "Download matches as CSV",
            buff.getvalue(),
            file_name="matches.csv",
            mime="text/csv",
        )

else:
    st.info("Upload two CSV files to begin.")
