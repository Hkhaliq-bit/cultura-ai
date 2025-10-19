# app.py — Cultura AI (polished, RAG-enabled, sidebar fixed)
import os
import logging
from datetime import datetime
from difflib import get_close_matches

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# ------------------ Page & Brand ------------------
st.set_page_config(
    page_title="Cultura AI — Culturally Intelligent Clinical Guidance",
    layout="wide",
)
logging.basicConfig(level=logging.INFO)

PRIMARY = "#0E6299"     # deep teal-blue
ACCENT  = "#0EA5E9"     # sky blue
MUTED   = "#6B7280"     # gray-500
BG_SOFT = "#F7FAFC"     # soft bg

# Minimal CSS polish
st.markdown(
    f"""
    <style>
      .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
      }}
      h1, h2, h3, h4, h5 {{
        color: {PRIMARY};
      }}
      .subtitle {{
        color: {MUTED}; font-size: 0.95rem; margin-top: -10px;
      }}
      .footer-note {{
        color: {MUTED}; font-size: 0.85rem; text-align:center; margin-top: 24px;
      }}
      .badge {{
        display:inline-block; padding: 6px 10px; border-radius: 10px;
        background: {BG_SOFT}; color: {PRIMARY}; font-weight: 600; font-size: 0.85rem;
        border: 1px solid #E5E7EB;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Config ------------------
DATA_PATH   = "AI Newcomer Navigator.xlsx"   # Excel with your rows
CHAT_MODEL  = "gpt-4o-mini"
EMB_MODEL   = "text-embedding-3-small"       # quality/cost sweet spot

# ------------------ Sidebar Branding ------------------
st.sidebar.markdown("###       Cultura AI")
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)
else:
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/5e/Globe_icon.svg", width=88)
st.sidebar.caption("Culturally intelligent, evidence-based care.")

# Sidebar status (FIXED: explicit if/else; no inline ternary)
OPENAI_OK = bool(os.getenv("OPENAI_API_KEY"))
if OPENAI_OK:
    st.sidebar.success("✅ OpenAI connected")
else:
    st.sidebar.warning("⚠️ OPENAI_API_KEY not set")

# Utility: file mtime for cache invalidation
def get_file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0

# ------------------ Load Excel (cached) ------------------
@st.cache_data
def load_data(path: str, _mtime: float) -> pd.DataFrame:
    return pd.read_excel(path)

# ------------------ Country helpers ------------------
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

# ------------------ RAG: docs & embeddings ------------------
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
        docs.append({
            "id": int(i),
            "region": str(row.get("Region", "")),
            "countries": str(row.get("Countries", "")),
            "text": text,
        })
    return docs

@st.cache_data(show_spinner=False)
def embed_texts_cached(texts: list[str], excel_mtime: float, api_key_present: bool) -> np.ndarray | None:
    if not api_key_present or not texts:
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

# ------------------ Rule modifiers & guardrails ------------------
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
    sc = screening.lower()
    ch = chronic.lower()
    if "tb" in sc or "tuberculosis" in sc:
        mi.append("Specify TB test modality (IGRA vs TST) and chest X-ray if indicated.")
    if any(k in sc for k in ["hepatitis", "hbv", "hcv"]) or any(k in ch for k in ["hepatitis", "hbv", "hcv"]):
        mi.append("For HBV include HBsAg, anti-HBc, +/- anti-HBs; for HCV include anti-HCV +/- RNA.")
    return mi

# ------------------ Header ------------------
st.title("🧭 Cultura AI — Newcomer Health Intelligence")
st.markdown('<div class="subtitle">Global evidence. Local insight. Culturally aware.</div>', unsafe_allow_html=True)
st.info("This tool supports - not replaces - clinical judgment. Follow provincial/local guidance and patient context.")

# ------------------ Top Controls ------------------
c1, c2, _ = st.columns([1, 1, 4])
with c1:
    if st.button("🔄 Reload data", key="reload"):
        st.cache_data.clear()
        st.toast("Data cache cleared. Reloading…", icon="🔄")
        st.rerun()
with c2:
    if st.button("🧠 Force re-embed", key="reembed"):
        st.cache_data.clear()  # clears embeddings cache too
        st.toast("Embeddings cache cleared. Rebuilding on next generate…", icon="🧠")

# ------------------ Load Excel ------------------
mtime = get_file_mtime(DATA_PATH)
try:
    df = load_data(DATA_PATH, mtime)
    st.success("✅ Dataset loaded.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Build country artifacts
all_countries = list_countries(df)
country_index = build_country_index(df)

# ------------------ Sidebar Patient Inputs (kept in sidebar) ------------------
st.sidebar.header("Patient")
country = st.sidebar.selectbox("🌍 Country of Origin", [""] + all_countries, index=0, key="country_select")
free_text = st.sidebar.text_input("…or type a country", key="country_free_text")
selected_country = free_text.strip() or country

if selected_country and selected_country not in all_countries:
    sug = suggest_country(selected_country, all_countries)
    if sug:
        st.sidebar.caption(f"Did you mean **{sug}**?")

age = st.sidebar.number_input("🎂 Age", min_value=0, max_value=120, value=35)
sex = st.sidebar.selectbox("⚧ Sex", ["Female", "Male", "Other"])
arrival = st.sidebar.number_input("📅 Year of Arrival", min_value=1980, max_value=2035, value=2023)
note_mode = st.sidebar.radio("Output style", ["Clinician note", "Patient handout", "Both"], index=2)

# ------------------ Build Docs & Embeddings (cached) ------------------
docs = rows_to_docs(df)
texts = [d["text"] for d in docs]
X = embed_texts_cached(texts, excel_mtime=mtime, api_key_present=OPENAI_OK)
KB_READY = (X is not None) and OPENAI_OK

# ------------------ Generate ------------------
if st.button("✨ Generate Cultura AI Summary", key="generate_btn_main"):
    row = resolve_row(selected_country, df, country_index)
    if row is None:
        st.warning("No match found. Check spelling or confirm the country exists in your Excel.")
        st.stop()

    # Extract fields
    region    = str(row.get("Region", ""))
    screening = str(row.get("Infectious Disease Screening", ""))
    chronic   = str(row.get("Chronic/Metabolic Screening", ""))
    immun     = str(row.get("Immunization Catch-Up", ""))
    culture   = str(row.get("Cultural/Contextual Considerations", ""))

    # Optional audit/source fields
    srcs  = str(row.get("Sources", "") or "").strip()
    rev   = str(row.get("Last_Reviewed", "") or "").strip()
    notes = str(row.get("Notes", "") or "").strip()

    # Guardrails
    modifiers = rules_modifiers(age, sex)
    if modifiers:
        st.info("Rule-based modifiers:\n- " + "\n- ".join(modifiers))

    must_include = must_include_from_text(screening, chronic)
    if must_include:
        st.info("Must-include tests:\n- " + "\n- ".join(must_include))

    # Retrieval context
    retrieval_context = ""
    top_docs = []
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if KB_READY:
            retrieval_query = (
                f"{selected_country} {region} age:{age} sex:{sex} arrival:{arrival} "
                f"screening:{screening} chronic:{chronic} immun:{immun} culture:{culture}"
            )
            top_docs = semantic_search(retrieval_query, docs, X, client, top_k=3)
            if top_docs:
                joined = "\n\n---\n\n".join([d["text"] for d in top_docs])
                retrieval_context = f"Relevant guideline snippets from knowledge base:\n\n{joined}"
        elif not OPENAI_OK:
            st.warning("OPENAI_API_KEY not set — semantic retrieval disabled.")
    except Exception as e:
        logging.exception("Semantic retrieval failed")
        st.warning(f"Semantic retrieval error: {e}")

    # Build prompt
    mode_text = {
        "Clinician note": "Return only the clinician summary in concise bullets.",
        "Patient handout": "Return only the patient-friendly summary in plain English.",
        "Both": "Return both sections."
    }[note_mode]

    user_prompt = f"""
You are an evidence-based Canadian family medicine assistant.

Patient:
- Country: {selected_country}
- Region: {region}
- Age: {age}, Sex: {sex}, Arrival year: {arrival}

Guideline context (verbatim snippets):
{retrieval_context if retrieval_context else "[No KB snippets available]"}

Columns from structured dataset:
- Infectious disease screening: {screening}
- Chronic/metabolic screening: {chronic}
- Immunizations: {immun}
- Cultural considerations: {culture}

Instructions:
1) Synthesize recommendations consistent with CMAJ, PHO, WHO, CPS, and CTFPHC.
2) Resolve any conflicts by preferring Canadian sources; be explicit if uncertainty exists.
3) Include specific tests (e.g., IGRA vs TST, HBsAg vs anti-HBc vs anti-HBs) and age/sex modifiers.
4) Flag red flags or referrals if indicated.
5) Keep the output succinct (≈250–400 words). {mode_text}
"""
    if modifiers:
        user_prompt += "\nAdditional rule-based modifiers to consider:\n- " + "\n- ".join(modifiers) + "\n"
    if must_include:
        user_prompt += "\nAlways include:\n- " + "\n- ".join(must_include) + "\n"

    # AI call with graceful fallback
    def fallback_summary() -> str:
        doc = [
            "**Clinical Summary (Offline Fallback)**",
            f"- Region: {region}",
            f"- Infectious disease screening: {screening}",
            f"- Chronic/metabolic screening: {chronic}",
            f"- Immunization catch-up: {immun}",
            f"- Cultural/contextual notes: {culture}",
            "",
            "**Patient-friendly explanation:**",
            f"You are {age} years old and arrived in {arrival} from {selected_country}.",
            f"Based on your region ({region}), we’ll review infection screening, chronic conditions, and vaccines, while respecting cultural preferences ({culture}).",
        ]
        return "\n".join(doc)

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        with st.spinner("🧠 Generating evidence-based guidance..."):
            messages = [
                {"role": "system", "content": "You are a concise, evidence-based Canadian family medicine assistant."},
                {"role": "user", "content": user_prompt}
            ]
            full_text = ""
            for _ in range(3):
                resp = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=900,
                )
                part = resp.choices[0].message.content or ""
                full_text += (("\n" if full_text else "") + part).strip()
                finish_reason = getattr(resp.choices[0], "finish_reason", None)
                if finish_reason != "length":
                    break
                messages += [
                    {"role": "assistant", "content": part},
                    {"role": "user", "content": "Continue where you left off. Finish succinctly."}
                ]
            summary_text = full_text if full_text.strip() else fallback_summary()
    except Exception as e:
        logging.exception("OpenAI chat failed")
        st.warning(f"⚠️ AI unavailable ({e}). Showing offline summary instead.")
        summary_text = fallback_summary()

    # Confidence heuristic (simple signal)
    score = 0
    if KB_READY: score += 40
    if top_docs: score += 30
    if srcs:     score += 20
    if modifiers or must_include: score += 10
    score = min(score, 100)

    # Display
    st.subheader("🧠 Cultura AI Summary")
    st.progress(score/100.0, text=f"Confidence (heuristic): {score}%")
    st.markdown(summary_text)

    # Sources & audit trail
    with st.expander("📚 Sources & audit trail"):
        st.markdown(f"**Sources:** {srcs or '—'}")
        st.markdown(f"**Last reviewed:** {rev or '—'}")
        if notes:
            st.markdown(f"**Notes:** {notes}")

    # Evidence snippets used
    if top_docs:
        with st.expander("🔎 Evidence snippets used (from your knowledge base)"):
            for j, d in enumerate(top_docs, start=1):
                st.markdown(f"**Snippet {j}**")
                st.markdown(d["text"])
                st.divider()

    # History
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "country": selected_country, "age": age, "sex": sex, "arrival": arrival,
        "summary": summary_text, "score": score
    })
    with st.expander("🕘 History this session"):
        for i, item in enumerate(reversed(st.session_state["history"]), start=1):
            st.markdown(
                f"**{i}. {item['country']} / {item['age']} / {item['sex']} / {item['arrival']}** "
                f"<span class='badge'>Confidence {item['score']}%</span>",
                unsafe_allow_html=True,
            )
            st.markdown(item["summary"])
            st.divider()

    # Copy & downloads
    st.divider()
    st.caption("📋 Copy or download for your EMR")
    st.text_area("Copy from here:", value=summary_text, height=240, key="copy_area")

    st.download_button(
        label="⬇️ Download summary (.txt)",
        data=summary_text.encode("utf-8"),
        file_name=f"cultura_summary_{selected_country}_{age}.txt",
        mime="text/plain",
        key="download_txt",
    )
    st.download_button(
        label="⬇️ Download summary (.md)",
        data=summary_text.encode("utf-8"),
        file_name=f"cultura_summary_{selected_country}_{age}.md",
        mime="text/markdown",
        key="download_md",
    )
    html = f"""
    <html><body>
    <h2>Cultura AI Summary</h2>
    <p><b>Country:</b> {selected_country} &nbsp; <b>Age:</b> {age} &nbsp; <b>Sex:</b> {sex} &nbsp; <b>Arrival:</b> {arrival}</p>
    <hr/>{summary_text.replace('\n','<br/>')}
    </body></html>
    """
    st.download_button(
        "⬇️ Download print-friendly HTML",
        data=html.encode("utf-8"),
        file_name=f"cultura_summary_{selected_country}_{age}.html",
        mime="text/html",
        key="download_html",
    )

# ------------------ Footer ------------------
st.markdown("<div class='footer-note'>© 2025 Cultura AI — Evidence. Equity. Insight.</div>", unsafe_allow_html=True)
