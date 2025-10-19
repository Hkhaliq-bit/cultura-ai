# app.py — Cultura Health (single-page, no sidebar, proxy-proof OpenAI client)

import os
from datetime import datetime
from difflib import get_close_matches
import logging

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
import httpx  # used to bypass env proxies

# ------------------ App / Style (set early) ------------------
st.set_page_config(page_title="Cultura Health", page_icon="🌍", layout="wide")
logging.basicConfig(level=logging.INFO)

PRIMARY = "#0E6299"
MUTED   = "#6B7280"
BG_SOFT = "#F7FAFC"

st.markdown(
    f"""
    <style>
      .block-container {{
        padding-top: 0.8rem;
        padding-bottom: 2rem;
        max-width: 1100px;
      }}
      h1, h2, h3, h4 {{ color: {PRIMARY}; }}
      .subtitle {{ color: {MUTED}; font-size: 0.95rem; margin-top: -6px; }}
      .badge {{
        display:inline-block; padding: 4px 8px; border-radius: 10px;
        background: {BG_SOFT}; color: {PRIMARY}; border:1px solid #E5E7EB; font-weight:600;
      }}
      .section {{
        padding: 0.6rem 0.8rem; border: 1px solid #E5E7EB; border-radius: 10px; background: #FFFFFF;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Landing Page / Splash Screen ------------------
if "started" not in st.session_state:
    st.session_state["started"] = False

if not st.session_state["started"]:
    st.markdown("<div style='text-align:center; margin-top:100px;'>", unsafe_allow_html=True)

    if os.path.exists("logo.png"):
        st.image("logo.png", width=260)
    else:
        st.markdown("<h1 style='font-size:60px;'>🌍</h1>", unsafe_allow_html=True)

    st.markdown("<h1 style='font-size:45px; color:#0E6299;'>Cultura Health</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6B7280; font-size:18px;'>Culturally intelligent, evidence-based care for newcomers</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- About Section ----
    st.markdown("""
    ### The Mission
    Cultura Health exists to bridge the gap between clinical evidence and cultural understanding. 
    
    It helps clinicians deliver equitable, culturally aware, and evidence-informed care for newcomers and refugees.  

    Healthcare is at its best when it sees the person behind the patient.  
    
    Cultura Health was created to bring empathy and evidence together, helping physicians make confident, informed, and culturally sensitive decisions within minutes.

    ---

    ### The Challenge
    Every day, primary care physicians meet patients whose health stories began elsewhere.  
    For newcomers and refugees, medical histories may be incomplete, vaccination records uncertain, and region-specific screening needs unclear.  

    Yet, the relevant guidance – from CMAJ, Public Health Ontario, WHO, and refugee-health frameworks – is scattered across multiple sources.  
    For a busy clinician, finding and interpreting that information within a short appointment is nearly impossible.  

    The result is inconsistency, missed opportunities for prevention, and inequities in care that should no longer exist in a system striving for inclusivity.

    ---

    ### The Solution: Cultura Health
    Cultura Health is an evidence-informed, culturally intelligent assistant for clinicians.  
    It consolidates regional screening guidance, immunization catch-up recommendations, and cultural context into one adaptive tool.  

    In seconds, it can provide the information that would otherwise take hours to gather from various guidelines.  
    Cultura Health supports physicians with concise, evidence-based insights tailored to a patient’s country of origin, health background, and cultural context.  

    This is not about replacing clinical judgment.  
    It is about enhancing it – giving clinicians a strong, evidence-based starting point so that every patient encounter begins with understanding.

    ---

    ### Why This Matters

    Every year, Canada welcomes more than half a million newcomers - each bringing unique health backgrounds, exposures, and needs. Yet studies from *CMAJ* and the Canadian Collaboration for Immigrant and Refugee Health (CCIRH) continue to highlight persistent gaps in screening, immunization, and chronic disease prevention for these populations. 

    A 2023 *BMC Family Practice* study found that over 70% of family physicians felt only somewhat confident identifying the appropriate screening and vaccination protocols for patients arriving from high-prevalence regions. Meanwhile, data from the Public Health Agency of Canada show that culturally adapted care models can improve patient engagement and follow-up by up to 40%.

    Cultura Health was designed to close these gaps. It brings scattered, complex guidance into one place - transforming it into concise, evidence-based insights that clinicians can trust. By reducing variability in care, saving time, and enhancing confidence, Cultura Health helps physicians deliver medicine that is not only smarter, but also fairer and more human - ensuring every patient feels seen, understood, and cared for as a person, not just a diagnosis.

    ---
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Enter", use_container_width=True):
        st.session_state["started"] = True
        st.rerun()

    st.stop()
# ------------------ Config ------------------
DATA_PATH  = "AI Newcomer Navigator.xlsx"
CHAT_MODEL = "gpt-4o-mini"
EMB_MODEL  = "text-embedding-3-small"

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_OK  = bool(OPENAI_KEY)

# ------------------ Header Branding (main area only) ------------------
cols = st.columns([1, 4])
with cols[0]:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_column_width=True)
    else:
        st.markdown("<div style='font-size:48px;line-height:1'>🌍</div>", unsafe_allow_html=True)
with cols[1]:
    st.title("Cultura Health")
    st.markdown('<div class="subtitle">Global evidence. Local insight. Culturally aware.</div>', unsafe_allow_html=True)
st.info("Decision support only - always apply clinical judgment and local/provincial guidance.")

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

# --- Proxy-proof OpenAI client (ignores env proxies) ---
def get_openai_client() -> OpenAI | None:
    """Create an OpenAI client that ignores HTTP(S) proxy env vars."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    http_client = httpx.Client(trust_env=False, timeout=30.0)  # ignore HTTP(S)_PROXY
    return OpenAI(api_key=key, http_client=http_client)

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
def embed_texts_cached(texts: list[str], excel_mtime: float, api_key_ok: bool) -> np.ndarray | None:
    """Returns normalized embeddings matrix or None on failure."""
    if not api_key_ok or not texts:
        return None
    try:
        client = get_openai_client()
        if client is None:
            return None
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

# ------------------ Data Load & Controls ------------------
top_controls = st.columns([1, 6])
with top_controls[0]:
    if st.button("🔄 Reload Data"):
        st.cache_data.clear()
        st.toast("Reloading dataset…", icon="🔄")
        st.rerun()

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

# ------------------ Patient Inputs (main screen) ------------------
st.markdown("### Patient")
box = st.container()
with box:
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        country_pick = st.selectbox("Country (from list)", [""] + all_countries, index=0)
        country_free = st.text_input("…or type a country")
        country_input = country_free.strip() or country_pick
        if country_input and country_input not in all_countries:
            sug = suggest_country(country_input, all_countries)
            if sug:
                st.caption(f"Did you mean **{sug}**?")
    with c2:
        age = st.number_input("Age", min_value=0, max_value=120, value=35)
    with c3:
        sex = st.selectbox("Sex", ["Female", "Male", "Other"])
    with c4:
        arrv = st.number_input("Year of Arrival", min_value=1980, max_value=2035, value=2023)

    mode = st.radio("Output style", ["Clinician note", "Patient handout", "Both"], index=2, horizontal=True)

# ------------------ Prepare KB / Embeddings ------------------
docs = rows_to_docs(df)
texts = [d["text"] for d in docs]
X = embed_texts_cached(texts, excel_mtime=mtime, api_key_ok=OPENAI_OK)
KB_READY = (X is not None) and OPENAI_OK

# ------------------ Generate ------------------
st.markdown("### Generate")
if st.button("✨ Generate Cultura Health Summary", key="generate"):
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
            client = get_openai_client()
            if client:
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
    # Offline fallback
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

    # AI call (proxy-proof)
    try:
        client = get_openai_client()
        if client is None:
            raise RuntimeError("OPENAI_API_KEY missing")
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

    # Display result
    st.markdown("### 🧠 Cultura Health Summary")
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

    # Download (TXT only)
    st.divider()
    st.caption("📋 Copy or download for your EMR")
    st.text_area("Copy from here:", value=summary_text, height=220, key="copybox")
    st.download_button(
        "⬇️ Download summary (.txt)",
        data=summary_text.encode("utf-8"),
        file_name=f"culturahealth_summary_{country_input}_{age}.txt",
        mime="text/plain",
        key="dl_txt",
    )

# ------------------ Footer ------------------
st.markdown("---")
st.caption("© 2025 Cultura Health · Evidence. Equity. Insight.")
