# app.py — Cultura AI (MVP+ with RAG, audit, history, reload/re-embed)
import os
from datetime import datetime
from difflib import get_close_matches
import logging

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# ------------------ App / Style ------------------
st.set_page_config(page_title="Cultura AI", page_icon="🌍", layout="wide")
logging.basicConfig(level=logging.INFO)

PRIMARY = "#0E6299"
MUTED   = "#6B7280"
BG_SOFT = "#F7FAFC"

st.markdown(
    f"""
    <style>
      .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 2rem;
        max-width: 1100px;
      }}
      h1, h2, h3, h4 {{ color: {PRIMARY}; }}
      .subtitle {{ color: {MUTED}; font-size: 0.95rem; margin-top: -8px; }}
      .badge {{
        display:inline-block; padding: 4px 8px; border-radius: 10px;
        background: {BG_SOFT}; color: {PRIMARY}; border:1px solid #E5E7EB; font-weight:600;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Config ------------------
DATA_PATH  = "AI Newcomer Navigator.xlsx"
CHAT_MODEL = "gpt-4o-mini"
EMB_MODEL  = "text-embedding-3-small"

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_OK  = bool(OPENAI_KEY)

# ------------------ Sidebar Branding (Cloud-safe) ------------------
st.sidebar.markdown("### 🌍 Cultura AI")
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_column_width=True)
else:
    st.sidebar.markdown("<div style='font-size:48px;line-height:1'>🌍</div>", unsafe_allow_html=True)
st.sidebar.caption("Culturally intelligent, evidence-based care.")

if OPENAI_OK:
    st.sidebar.success("✅ OpenAI connected")
else:
    st.sidebar.warning("⚠️ OPENAI_API_KEY not set — AI features limited")

# ------------------ Helpers ------------------
def file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0

@st.cache_data
def load_data(path: str, _mtime: float) -> pd.DataFrame:
    return pd.read_excel(path)

def list_countries(df: pd.DataFrame) -> list[str]:
    series = df["Countries"].fillna("").astype(str).str.split(",")
    countries = {c.strip() for row in series for c in row if c.strip()}
    return sorted(countries)

def build_country_index(df: pd.DataFrame) -> dict[str, int]:
    idx: dict[str, int] = {}
    for i, row in df.iterrows():
        raw = str(row.get("Countries", "") or "")
        for c in raw.split(","):
            name = c.strip().lower()
            if name:
                idx[name] = i
    return idx

def resolve_row(country_input: str, df: pd.DataFrame, country_index: dict[str, int]) -> pd.Series | None:
    if not country_input:
        return None
    key = country_input.strip().lower()
    if key in country_index:
        return df.loc[country_index[key]]
    mask = df["Countries"].astype(str).str.contains(country_input, case=False, na=False) | \
           df["Region"].astype(str).str.contains(country_input, case=False, na=False)
    hits = df[mask]
    if hits.empty:
        return None
    return hits.iloc[0]

def suggest_country(name: str, options: list[str]) -> str | None:
    if not name:
        return None
    match = get_close_matches(name.strip(), options, n=1, cutoff=0.75)
    return match[0] if match else None

# RAG docs
def rows_to_docs(df: pd.DataFrame) -> list[dict]:
    docs: list[dict] = []
    for i, row in df.iterrows():
        text = "\n".join([
            f"Region: {row.get('Region','')}",
            f"Countries: {row.get('Countries','')}",
            f"Infectious Disease Screening: {row.get('Infectious Disease Screening','')}",
            f"Chronic/Metabolic Screening: {row.get('Chronic/Metabolic Screening','')}",
            f"Immunization Catch-Up: {row.get('Immunization Catch-Up','')}",
            f"Cultural/Contextual Considerations: {row.get('Cultural/Contextual Considerations','')}",
        ])
        docs.append({"id": int(i), "text": text})
    return docs

@st.cache_data(show_spinner=False)
def embed_texts_cached(texts: list[str], excel_mtime: float, salt: int, api_key_ok: bool) -> np.ndarray | None:
    """Returns normalized embeddings matrix or None on failure."""
    if not api_key_ok or not texts:
        return None
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        embs: list[list[float]] = []
        B = 64
        for i in range(0, len(texts), B):
            chunk = texts[i:i+B]
            resp = client.embeddings.create(model=EMB_MODEL, input=chunk)
            embs.extend([d.embedding for d in resp.data])
        X = np.array(embs, dtype="float32")
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        return X
    except Exception:
        logging.exception("Embedding build failed")
        return None

def embed_query(text: str, client: OpenAI) -> np.ndarray:
    v = client.embeddings.create(model=EMB_MODEL, input=[text]).data[0].embedding
    v = np.array(v, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-9)

def semantic_search(query: str, docs: list[dict], X: np.ndarray, client: OpenAI, top_k: int = 3) -> list[dict]:
    if X is None or len(docs) == 0:
        return []
    vq = embed_query(query, client)
    scores = X @ vq
    idx = np.argsort(-scores)[:top_k]
    return [docs[i] for i in idx]

def rules_modifiers(age: int, sex: str) -> list[str]:
    rules = []
    if age >= 50:
        rules.append("Consider colorectal cancer screening if not up to date (CTFPHC).")
    if sex.lower() == "female" and 25 <= age <= 69:
        rules.append("Pap/HPV screening per provincial schedule (CTFPHC).")
    if age >= 40:
        rules.append("Consider lipid screening per CTFPHC based on risk.")
    return rules

def must_include_from_text(screening: str, chronic: str) -> list[str]:
    mi = []
    sc = (screening or "").lower()
    ch = (chronic or "").lower()
    if "tb" in sc or "tuberculosis" in sc:
        mi.append("Specify TB test modality (IGRA vs TST) and chest X-ray if indicated.")
    if any(k in sc for k in ["hepatitis", "hbv", "hcv"]) or any(k in ch for k in ["hepatitis", "hbv", "hcv"]):
        mi.append("For HBV include HBsAg, anti-HBc, +/- anti-HBs; for HCV include anti-HCV +/- RNA.")
    return mi

# ------------------ Header & Controls ------------------
st.title("Cultura AI")
st.markdown('<div class="subtitle">Global evidence. Local insight. Culturally aware.</div>', unsafe_allow_html=True)
st.info("Decision support only; always apply clinical judgment and local/provincial guidance.")

c1, c2, _ = st.columns([1, 1, 5])
with c1:
    if st.button("🔄 Reload Data"):
        st.cache_data.clear()
        st.toast("Reloading dataset…", icon="🔄")
        st.rerun()
with c2:
    # re-embed salt increments to bust embedding cache
    if "reembed_salt" not in st.session_state:
        st.session_state["reembed_salt"] = 0
    if st.button("🧠 Force Re-embed"):
        st.session_state["reembed_salt"] += 1
        st.toast("Embeddings will rebuild on next generate.", icon="🧠")

# ------------------ Load Data ------------------
mtime = file_mtime(DATA_PATH)
try:
    df = load_data(DATA_PATH, mtime)
    st.success("✅ Dataset loaded.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Build country artifacts
all_countries = list_countries(df)
country_index = build_country_index(df)

# ------------------ Sidebar Inputs ------------------
st.sidebar.header("Patient")
country_pick = st.sidebar.selectbox("Country (from list)", [""] + all_countries, index=0)
country_free = st.sidebar.text_input("…or type a country")
country_input = country_free.strip() or country_pick

age  = st.sidebar.number_input("Age", min_value=0, max_value=120, value=35)
sex  = st.sidebar.selectbox("Sex", ["Female", "Male", "Other"])
arrv = st.sidebar.number_input("Year of Arrival", min_value=1980, max_value=2035, value=2023)
mode = st.sidebar.radio("Output style", ["Clinician note", "Patient handout", "Both"], index=2)

if country_input and country_input not in all_countries:
    sug = suggest_country(country_input, all_countries)
    if sug:
        st.sidebar.caption(f"Did you mean **{sug}**?")

# ------------------ Prepare RAG ------------------
docs = rows_to_docs(df)
texts = [d["text"] for d in docs]
salt  = st.session_state.get("reembed_salt", 0)
X = embed_texts_cached(texts, excel_mtime=mtime, salt=salt, api_key_ok=OPENAI_OK)
KB_READY = (X is not None) and OPENAI_OK

# ------------------ Generate ------------------
if st.button("✨ Generate Cultura AI Summary", key="generate"):
    row = resolve_row(country_input, df, country_index)
    if row is None:
        st.warning("No match found. Check spelling or confirm the country exists in your dataset.")
        st.stop()

    region    = str(row.get("Region", ""))
    screening = str(row.get("Infectious Disease Screening", ""))
    chronic   = str(row.get("Chronic/Metabolic Screening", ""))
    immun     = str(row.get("Immunization Catch-Up", ""))
    culture   = str(row.get("Cultural/Contextual Considerations", ""))

    # Optional columns
    sources = str(row.get("Sources", "") or "").strip()
    citeurl = str(row.get("Citation_URL", "") or "").strip()
    reviewed= str(row.get("Last_Reviewed", "") or "").strip()
    notes   = str(row.get("Notes", "") or "").strip()

    # Guardrails
    mods = rules_modifiers(age, sex)
    must = must_include_from_text(screening, chronic)

    if mods:
        st.info("Rule-based modifiers:\n- " + "\n- ".join(mods))
    if must:
        st.info("Must-include tests:\n- " + "\n- ".join(must))

    # Retrieval context
    retrieval_context = ""
    top_docs = []
    try:
        if KB_READY:
            client = OpenAI(api_key=OPENAI_KEY)
            rq = f"{country_input} {region} age:{age} sex:{sex} arrival:{arrv} screening:{screening} chronic:{chronic} immun:{immun} culture:{culture}"
            top_docs = semantic_search(rq, docs, X, client, top_k=3)
            if top_docs:
                retrieval_context = "\n\n---\n\n".join([d["text"] for d in top_docs])
        elif not OPENAI_OK:
            st.warning("OPENAI_API_KEY not set — semantic retrieval disabled.")
    except Exception as e:
        logging.exception("Semantic retrieval failed")
        st.warning(f"Semantic retrieval error: {e}")

    mode_text = {
        "Clinician note": "Return only the clinician summary in concise bullets.",
        "Patient handout": "Return only the patient-friendly summary in plain English.",
        "Both": "Return both sections."
    }[mode]

    user_prompt = f"""
You are an evidence-based Canadian family medicine assistant.

Patient:
- Country: {country_input}
- Region: {region}
- Age: {age}, Sex: {sex}, Arrival year: {arrv}

Guideline context (verbatim snippets from KB if available):
{retrieval_context if retrieval_context else "[No KB snippets available]"}

Structured dataset fields:
- Infectious disease screening: {screening}
- Chronic/metabolic screening: {chronic}
- Immunizations: {immun}
- Cultural considerations: {culture}

Instructions:
1) Synthesize recommendations consistent with CMAJ, PHO, WHO, CPS, and CTFPHC.
2) Prefer Canadian sources if conflicts arise; note any uncertainty briefly.
3) Include specific tests (e.g., IGRA vs TST, HBsAg vs anti-HBc vs anti-HBs) and age/sex modifiers.
4) Flag red flags or referrals if indicated.
5) Keep the output succinct (≈250–400 words). {mode_text}
"""

    if mods:
        user_prompt += "\nAdditional rule-based modifiers to consider:\n- " + "\n- ".join(mods) + "\n"
    if must:
        user_prompt += "\nAlways include:\n- " + "\n- ".join(must) + "\n"

    # Fallback summary
    def offline_summary() -> str:
        out = [
            "**Clinical Summary (Offline Fallback)**",
            f"- Region: {region}",
            f"- Infectious disease screening: {screening}",
            f"- Chronic/metabolic screening: {chronic}",
            f"- Immunization catch-up: {immun}",
            f"- Cultural/contextual notes: {culture}",
            "",
            "**Patient-friendly explanation:**",
            f"You are {age} years old and arrived in {arrv} from {country_input}.",
            f"We’ll review infection screening, chronic conditions, and vaccines, factoring in cultural preferences ({culture}).",
        ]
        return "\n".join(out)

    # AI call
    try:
        client = OpenAI(api_key=OPENAI_KEY)
        with st.spinner("🧠 Generating evidence-based guidance..."):
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a concise, evidence-based Canadian family medicine assistant."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=900,
            )
            summary_text = resp.choices[0].message.content or offline_summary()
    except Exception as e:
        logging.exception("OpenAI chat failed")
        st.warning(f"⚠️ AI unavailable ({e}). Showing offline summary instead.")
        summary_text = offline_summary()

    # Confidence heuristic
    score = 0
    if KB_READY: score += 40
    if top_docs: score += 30
    if sources:  score += 20
    if mods or must: score += 10
    score = min(score, 100)

    # Display
    st.subheader("🧠 Cultura AI Summary")
    st.progress(score / 100.0)
    st.caption(f"Confidence (heuristic): {score}%")
    st.markdown(summary_text)

    # Sources & Audit
    with st.expander("📚 Sources & audit trail"):
        st.markdown(f"**Sources:** {sources or '—'}")
        if citeurl:
            st.markdown(f"**Citation link:** {citeurl}")
        st.markdown(f"**Last reviewed:** {reviewed or '—'}")
        if notes:
            st.markdown(f"**Notes:** {notes}")

    # KB snippets
    if top_docs:
        with st.expander("🔎 Evidence snippets from knowledge base"):
            for j, d in enumerate(top_docs, start=1):
                st.markdown(f"**Snippet {j}**")
                st.markdown(d["text"])
                st.divider()

    # History
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "country": country_input, "age": age, "sex": sex, "arrival": arrv,
        "summary": summary_text, "score": score,
    })
    with st.expander("🕘 History (this session)"):
        for i, item in enumerate(reversed(st.session_state["history"]), start=1):
            st.markdown(
                f"**{i}. {item['country']} / {item['age']} / {item['sex']} / {item['arrival']}** "
                f"<span class='badge'>Confidence {item['score']}%</span>",
                unsafe_allow_html=True,
            )
            st.markdown(item["summary"])
            st.divider()

    # Downloads
    st.divider()
    st.caption("📋 Copy or download for your EMR")
    st.text_area("Copy from here:", value=summary_text, height=220, key="copybox")

    st.download_button(
        "⬇️ Download summary (.txt)",
        data=summary_text.encode("utf-8"),
        file_name=f"cultura_summary_{country_input}_{age}.txt",
        mime="text/plain",
        key="dl_txt",
    )
    st.download_button(
        "⬇️ Download summary (.md)",
        data=summary_text.encode("utf-8"),
        file_name=f"cultura_summary_{country_input}_{age}.md",
        mime="text/markdown",
        key="dl_md",
    )
    html = f"""
    <html><body>
    <h2 style="color:{PRIMARY};">Cultura AI Summary</h2>
    <p><b>Country:</b> {country_input} &nbsp; <b>Age:</b> {age} &nbsp; <b>Sex:</b> {sex} &nbsp; <b>Arrival:</b> {arrv}</p>
    <hr/>{summary_text.replace('\n','<br/>')}
    </body></html>
    """
    st.download_button(
        "⬇️ Download print-friendly HTML",
        data=html.encode("utf-8"),
        file_name=f"cultura_summary_{country_input}_{age}.html",
        mime="text/html",
        key="dl_html",
    )

# ------------------ Footer ------------------
st.markdown("---")
st.caption("© 2025 Cultura AI · Evidence. Equity. Insight.")
