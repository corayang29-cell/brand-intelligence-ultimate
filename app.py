import os
import re
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import jieba
except Exception:
    jieba = None

# Optional Groq (for better sentiment/summary if user wants)
try:
    from groq import Groq
except Exception:
    Groq = None


# =========================
# UI: Apple-like minimal style (no emoji)
# =========================
def apply_minimal_ui():
    st.set_page_config(
        page_title="Aurora Campaign Risk OS",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    css = """
    <style>
      :root{
        --bg:#FFFFFF;
        --soft:#F5F5F7;
        --text:#111111;
        --muted:#6B7280;
        --border:#D1D1D6;
        --accent:#0A84FF;
      }

      .stApp { background: var(--bg); color: var(--text); }
      section[data-testid="stSidebar"] { background: var(--soft); border-right: 1px solid var(--border); }
      div[data-testid="stVerticalBlockBorderWrapper"]{ border: 1px solid var(--border); border-radius: 12px; padding: 14px; background: #fff; }
      .card { border: 1px solid var(--border); border-radius: 12px; padding: 14px; background: #fff; }
      .kpi-title { font-size: 12px; color: var(--muted); letter-spacing: .02em; text-transform: uppercase; }
      .kpi-value { font-size: 30px; font-weight: 700; line-height: 1.1; color: var(--text); }
      .kpi-sub { font-size: 12px; color: var(--muted); margin-top: 6px; }
      .badge { display:inline-block; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--border); font-size: 12px; }
      .badge.low{ background:#F5F5F7; color:#111; }
      .badge.med{ background:#EEF2FF; color:#1E3A8A; border-color:#C7D2FE; }
      .badge.high{ background:#FFF7ED; color:#9A3412; border-color:#FED7AA; }
      .badge.critical{ background:#FEF2F2; color:#991B1B; border-color:#FECACA; }
      hr { border: none; border-top: 1px solid var(--border); margin: 10px 0 12px 0; }
      .small-note { color: var(--muted); font-size: 12px; }
      .section-title { font-size: 16px; font-weight: 700; margin: 4px 0 6px 0; }
      .subtle { color: var(--muted); font-size: 13px; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =========================
# Constants & helpers
# =========================
TEXT_COL_CANDIDATES = ["comment_text", "content", "text", "评论", "评论正文", "comment", "内容"]
DATE_COL_CANDIDATES = ["created_at", "create_time", "time", "date", "发布时间", "时间", "日期"]
SENTIMENT_COL_CANDIDATES = ["sentiment_label", "sentiment", "情感", "情绪", "label"]
KOL_COL_CANDIDATES = ["kol_name", "kol", "达人", "博主", "KOL", "红人"]
BRAND_COL_CANDIDATES = ["brand", "品牌"]
CAMPAIGN_COL_CANDIDATES = ["campaign", "活动", "campaign_name"]
PLATFORM_COL_CANDIDATES = ["platform", "平台"]
POST_COL_CANDIDATES = ["post_id", "note_id", "作品id", "笔记id", "post"]

DEFAULT_RISK_TERMS = [
    # product safety / allergy
    "过敏", "刺痛", "红肿", "发炎", "起皮", "干裂", "爆皮", "灼烧", "痒", "不适", "敏感肌",
    # quality / authenticity
    "批次", "质量", "不稳定", "假货", "仿", "漏油", "断裂", "变质",
    # dissatisfaction / service
    "投诉", "退货", "维权", "失望", "踩雷", "翻车", "避雷", "不值", "太贵", "智商税",
    # brand response
    "不回应", "没回应", "未回应", "官方回应", "道歉",
]

POSITIVE_HINTS = ["好看", "显白", "高级", "喜欢", "值得", "温柔", "好用", "回购", "推荐", "惊艳", "好闻", "舒服", "顺滑"]
NEGATIVE_HINTS = ["过敏", "刺痛", "干裂", "起皮", "爆皮", "翻车", "避雷", "失望", "不值", "太贵", "智商税", "难用", "拔干", "掉色", "沾杯"]


def normalize_colname(s: str) -> str:
    return re.sub(r"\s+", "_", str(s).strip().lower())


def is_string_like_series(s: pd.Series) -> bool:
    # object dtype is typical for text
    if s.dtype == "object":
        return True
    # pandas string dtype
    if str(s.dtype).startswith("string"):
        return True
    return False


def safe_to_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# =========================
# Column mapping & validation
# =========================
def auto_match_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {normalize_colname(c): c for c in cols}
    for cand in candidates:
        key = normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    # fuzzy: contains
    for c in cols:
        cn = normalize_colname(c)
        for cand in candidates:
            if normalize_colname(cand) in cn:
                return c
    return None


def suggest_text_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if is_string_like_series(df[c]):
            cols.append(c)
    return cols


def validate_dataframe(df: pd.DataFrame) -> Dict[str, object]:
    report = {}
    report["rows"] = int(len(df))
    report["cols"] = int(len(df.columns))
    report["empty_rows"] = int(df.isna().all(axis=1).sum())
    report["duplicates"] = int(df.duplicated().sum())
    report["nulls_total"] = int(df.isna().sum().sum())
    return report


def map_required_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {
        "platform": auto_match_column(df, PLATFORM_COL_CANDIDATES),
        "campaign": auto_match_column(df, CAMPAIGN_COL_CANDIDATES),
        "brand": auto_match_column(df, BRAND_COL_CANDIDATES),
        "kol_name": auto_match_column(df, KOL_COL_CANDIDATES),
        "post_id": auto_match_column(df, POST_COL_CANDIDATES),
        "comment_text": auto_match_column(df, TEXT_COL_CANDIDATES),
        "created_at": auto_match_column(df, DATE_COL_CANDIDATES),
        "sentiment_label": auto_match_column(df, SENTIMENT_COL_CANDIDATES),
    }
    return mapping


def enforce_text_column(df: pd.DataFrame, text_col: str) -> pd.Series:
    s = df[text_col]
    if not is_string_like_series(s):
        raise ValueError("Selected text column is not a text/string column.")
    # Convert to string and strip
    return s.astype(str).fillna("").map(lambda x: x.strip())


def parse_datetime_series(df: pd.DataFrame, col: Optional[str]) -> Optional[pd.Series]:
    if col is None or col not in df.columns:
        return None
    try:
        s = pd.to_datetime(df[col], errors="coerce", utc=False)
        return s
    except Exception:
        return None


# =========================
# Sentiment engine
# =========================
def standardize_sentiment_label(x: str) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    # common variants
    if s in ["positive", "pos", "p", "正面", "正向", "好评", "好"]:
        return "positive"
    if s in ["neutral", "neu", "n", "中性", "一般", "还行"]:
        return "neutral"
    if s in ["negative", "neg", "bad", "负面", "负向", "差评", "差"]:
        return "negative"
    return None


def rule_based_sentiment(text: str) -> str:
    t = safe_to_str(text)
    if not t:
        return "neutral"
    neg_hits = sum(1 for w in NEGATIVE_HINTS if w in t)
    pos_hits = sum(1 for w in POSITIVE_HINTS if w in t)
    if neg_hits > pos_hits and neg_hits > 0:
        return "negative"
    if pos_hits > neg_hits and pos_hits > 0:
        return "positive"
    return "neutral"


@st.cache_data(show_spinner=False)
def compute_sentiments(df: pd.DataFrame, mapping: Dict[str, Optional[str]], use_llm: bool, groq_model: str) -> pd.DataFrame:
    out = df.copy()

    # If sentiment column exists and is usable, standardize it
    sent_col = mapping.get("sentiment_label")
    if sent_col and sent_col in out.columns:
        standardized = out[sent_col].map(standardize_sentiment_label)
        if standardized.notna().mean() > 0.6:
            out["_sentiment"] = standardized.fillna("neutral")
            return out

    # Otherwise: rule-based sentiment (fast, deterministic)
    text_col = mapping.get("comment_text")
    if not text_col:
        out["_sentiment"] = "neutral"
        return out

    texts = out[text_col].astype(str).fillna("").tolist()

    # Optional LLM sentiment (only if configured & requested)
    if use_llm and Groq is not None:
        api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if api_key:
            try:
                client = Groq(api_key=api_key)
                labels = []
                # batch in small chunks
                chunk_size = 25
                for i in range(0, len(texts), chunk_size):
                    chunk = texts[i:i+chunk_size]
                    prompt = (
                        "You are a strict sentiment classifier. "
                        "Classify each Chinese comment into one label: positive, neutral, negative. "
                        "Return ONLY a JSON array of strings, same length as input.\n\n"
                        f"Input comments:\n{chunk}"
                    )
                    resp = client.chat.completions.create(
                        model=groq_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                    )
                    raw = resp.choices[0].message.content.strip()
                    # try to extract json array
                    import json
                    arr = json.loads(raw)
                    for lab in arr:
                        lab2 = standardize_sentiment_label(lab) or "neutral"
                        labels.append(lab2)
                if len(labels) == len(texts):
                    out["_sentiment"] = labels
                    return out
            except Exception:
                pass  # fallback to rule-based

    out["_sentiment"] = [rule_based_sentiment(t) for t in texts]
    return out


# =========================
# Keyword extraction & stats
# =========================
def tokenize_cn(text: str) -> List[str]:
    t = safe_to_str(text)
    if not t:
        return []
    # keep Chinese/letters/numbers as tokens
    if jieba is not None:
        tokens = [w.strip() for w in jieba.lcut(t) if w.strip()]
    else:
        # fallback: split by non-word
        tokens = re.split(r"[^\w\u4e00-\u9fff]+", t)
        tokens = [w for w in tokens if w]
    # remove short tokens
    tokens = [w for w in tokens if len(w) >= 2]
    return tokens


@st.cache_data(show_spinner=False)
def build_keyword_stats(df: pd.DataFrame, mapping: Dict[str, Optional[str]], risk_terms: List[str]) -> Dict[str, object]:
    text_col = mapping.get("comment_text")
    if not text_col:
        return {"term_stats": pd.DataFrame(), "kol_term": pd.DataFrame()}

    kol_col = mapping.get("kol_name") or "_kol_fallback"
    if kol_col not in df.columns:
        df = df.copy()
        df[kol_col] = "Unknown"

    # Prepare rows
    rows = []
    for _, r in df.iterrows():
        txt = safe_to_str(r[text_col])
        sentiment = safe_to_str(r["_sentiment"])
        kol = safe_to_str(r[kol_col]) or "Unknown"
        tokens = tokenize_cn(txt)
        for tok in tokens:
            rows.append((tok, sentiment, kol, txt))

    if not rows:
        return {"term_stats": pd.DataFrame(), "kol_term": pd.DataFrame()}

    tmp = pd.DataFrame(rows, columns=["term", "sentiment", "kol_name", "source_text"])

    # Term stats with sentiment ratios
    pivot = tmp.pivot_table(index="term", columns="sentiment", values="source_text", aggfunc="count", fill_value=0)
    for c in ["positive", "neutral", "negative"]:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot["freq"] = pivot[["positive", "neutral", "negative"]].sum(axis=1)
    pivot["neg_ratio"] = np.where(pivot["freq"] > 0, pivot["negative"] / pivot["freq"], 0.0)
    pivot["pos_ratio"] = np.where(pivot["freq"] > 0, pivot["positive"] / pivot["freq"], 0.0)
    pivot = pivot.sort_values("freq", ascending=False).reset_index()

    # Mark risk terms (dictionary + computed)
    risk_set = set(risk_terms)
    pivot["is_risk_dictionary"] = pivot["term"].apply(lambda x: x in risk_set)

    # computed high risk term rule
    pivot["is_high_risk_term"] = (pivot["neg_ratio"] > 0.60) & (pivot["freq"] >= 3)

    # keyword x kol heatmap table (top N terms only to keep it readable)
    top_terms = pivot.head(30)["term"].tolist()
    sub = tmp[tmp["term"].isin(top_terms)]
    kol_term = sub.pivot_table(index="kol_name", columns="term", values="source_text", aggfunc="count", fill_value=0)

    return {"term_stats": pivot, "kol_term": kol_term}


# =========================
# Risk scoring
# =========================
def log_norm(x: float, max_x: float) -> float:
    if x <= 0:
        return 0.0
    return math.log1p(x) / math.log1p(max_x) if max_x > 0 else 0.0


@st.cache_data(show_spinner=False)
def compute_risk_scores(df: pd.DataFrame, mapping: Dict[str, Optional[str]], term_stats: pd.DataFrame, risk_terms: List[str]) -> Dict[str, object]:
    # Campaign level
    total = len(df)
    neg_ratio = float((df["_sentiment"] == "negative").mean()) if total else 0.0

    # Risk term score based on dictionary hits
    text_col = mapping.get("comment_text")
    if not text_col:
        risk_term_hits = 0
    else:
        risk_set = set(risk_terms)
        risk_term_hits = 0
        for txt in df[text_col].astype(str).fillna("").tolist():
            risk_term_hits += sum(1 for rt in risk_set if rt in txt)

    # Normalize risk term hits
    # Use a soft cap so it doesn't explode
    risk_term_score = clamp(risk_term_hits / 30.0, 0.0, 1.0)  # 30 hits ~ full scale
    volume_norm = log_norm(total, max_x=max(total, 50))  # normalize to itself or 50 baseline

    risk_score = 60 * neg_ratio + 25 * risk_term_score + 15 * volume_norm
    risk_score = float(clamp(risk_score, 0.0, 100.0))

    def level(score: float) -> str:
        if score <= 24:
            return "Low"
        if score <= 49:
            return "Medium"
        if score <= 74:
            return "High"
        return "Critical"

    campaign_level = level(risk_score)

    # KOL level scoring
    kol_col = mapping.get("kol_name")
    if not kol_col or kol_col not in df.columns:
        kol_stats = pd.DataFrame()
    else:
        g = df.groupby(kol_col, dropna=False)
        rows = []
        max_vol = max(g.size().max(), 1)
        risk_set = set(risk_terms)

        for kol, sub in g:
            vol = len(sub)
            neg_r = float((sub["_sentiment"] == "negative").mean()) if vol else 0.0
            # term hits in this kol
            hits = 0
            if text_col:
                for txt in sub[text_col].astype(str).fillna("").tolist():
                    hits += sum(1 for rt in risk_set if rt in txt)
            hits_norm = clamp(hits / 15.0, 0.0, 1.0)  # smaller cap per kol
            vol_n = log_norm(vol, max_vol)
            score = float(clamp(60 * neg_r + 25 * hits_norm + 15 * vol_n, 0.0, 100.0))
            rows.append([kol, vol, neg_r, hits, score, level(score)])

        kol_stats = pd.DataFrame(rows, columns=["kol_name", "comment_volume", "negative_ratio", "risk_term_hits", "risk_score", "risk_level"])
        kol_stats = kol_stats.sort_values("risk_score", ascending=False)

    return {
        "campaign": {
            "risk_score": risk_score,
            "risk_level": campaign_level,
            "negative_ratio": neg_ratio,
            "risk_term_hits": int(risk_term_hits),
            "total_comments": int(total),
        },
        "kol_stats": kol_stats,
    }


# =========================
# Competitor comparison
# =========================
@st.cache_data(show_spinner=False)
def build_brand_comparison(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    brand_col = mapping.get("brand")
    if not brand_col or brand_col not in df.columns:
        return pd.DataFrame()

    g = df.groupby(brand_col)
    rows = []
    for brand, sub in g:
        total = len(sub)
        neg = float((sub["_sentiment"] == "negative").mean()) if total else 0.0
        pos = float((sub["_sentiment"] == "positive").mean()) if total else 0.0
        neu = float((sub["_sentiment"] == "neutral").mean()) if total else 0.0
        rows.append([brand, total, pos, neu, neg])

    out = pd.DataFrame(rows, columns=["brand", "comments", "positive_ratio", "neutral_ratio", "negative_ratio"])
    out = out.sort_values("comments", ascending=False)
    return out


# =========================
# Rendering helpers
# =========================
def badge_html(level: str) -> str:
    level = (level or "").strip().lower()
    if level == "low":
        cls = "low"
    elif level == "medium":
        cls = "med"
    elif level == "high":
        cls = "high"
    else:
        cls = "critical"
    return f'<span class="badge {cls}">{level.title()}</span>'


def kpi_card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def chart_theme(fig):
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial", color="#111111"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(title=None),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False)
    return fig


# =========================
# App
# =========================
def main():
    apply_minimal_ui()

    st.title("Aurora Campaign Risk OS")
    st.caption("AI-driven Campaign Risk Governance Engine")

    with st.sidebar:
        st.markdown("<div class='section-title'>Configuration</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Risk Dictionary</div>", unsafe_allow_html=True)
        use_default_terms = st.checkbox("Use default risk terms", value=True)
        custom_terms_raw = st.text_area("Add custom risk terms (comma-separated)", value="", height=80)
        risk_terms = DEFAULT_RISK_TERMS[:] if use_default_terms else []
        if custom_terms_raw.strip():
            extra = [t.strip() for t in re.split(r"[，,]+", custom_terms_raw) if t.strip()]
            risk_terms.extend(extra)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Optional AI</div>", unsafe_allow_html=True)

        api_key_present = bool((hasattr(st, "secrets") and st.secrets.get("GROQ_API_KEY")) or os.environ.get("GROQ_API_KEY"))
        use_llm = st.checkbox("Use Groq for sentiment (optional)", value=False, disabled=not api_key_present)
        groq_model = st.selectbox("Groq model", options=["llama-3.1-70b-versatile", "llama-3.1-8b-instant"], index=0, disabled=not use_llm)

        if not api_key_present:
            st.markdown("<div class='small-note'>Groq key not detected. Sentiment will use a deterministic rule-based fallback.</div>", unsafe_allow_html=True)

    if not uploaded:
        st.info("Upload a CSV to start.")
        st.stop()

    # Load CSV safely
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception:
        # try utf-8-sig
        df_raw = pd.read_csv(uploaded, encoding="utf-8-sig")

    if df_raw.empty:
        st.error("CSV is empty.")
        st.stop()

    # Normalize column names? (keep original for display, but mapping uses fuzzy)
    mapping = map_required_columns(df_raw)

    tabs = st.tabs(["Overview", "Data", "KOL Monitor", "Insights", "Actions"])

    # -------------------------
    # Data tab: mapping & validation
    # -------------------------
    with tabs[1]:
        st.markdown("<div class='section-title'>Data Preview</div>", unsafe_allow_html=True)
        rep = validate_dataframe(df_raw)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Rows", str(rep["rows"]))
        with c2:
            kpi_card("Columns", str(rep["cols"]))
        with c3:
            kpi_card("Empty rows", str(rep["empty_rows"]))
        with c4:
            kpi_card("Duplicates", str(rep["duplicates"]))

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Field Mapping</div>", unsafe_allow_html=True)
        st.caption("Auto-detected fields. If any are wrong, adjust below.")

        # Build selectable options
        all_cols = df_raw.columns.tolist()
        text_candidates = suggest_text_columns(df_raw)

        # Text column must be string-like
        default_text = mapping.get("comment_text") if mapping.get("comment_text") in text_candidates else (text_candidates[0] if text_candidates else None)
        if default_text is None:
            st.error("No text-like columns found. Your CSV must contain a text comment column (e.g., comment_text/content).")
            st.stop()

        col_text = st.selectbox("Comment text column", options=text_candidates, index=text_candidates.index(default_text))
        st.info(f"Selected text column: {col_text}")

        # Other fields (optional)
        col_kol = st.selectbox("KOL column (optional but recommended)", options=["(none)"] + all_cols,
                               index=(["(none)"] + all_cols).index(mapping["kol_name"]) if mapping["kol_name"] in all_cols else 0)
        col_brand = st.selectbox("Brand column (optional, for competitor comparison)", options=["(none)"] + all_cols,
                                 index=(["(none)"] + all_cols).index(mapping["brand"]) if mapping["brand"] in all_cols else 0)
        col_campaign = st.selectbox("Campaign column (optional)", options=["(none)"] + all_cols,
                                    index=(["(none)"] + all_cols).index(mapping["campaign"]) if mapping["campaign"] in all_cols else 0)
        col_post = st.selectbox("Post ID column (optional)", options=["(none)"] + all_cols,
                                index=(["(none)"] + all_cols).index(mapping["post_id"]) if mapping["post_id"] in all_cols else 0)
        col_date = st.selectbox("Created time column (optional, for trend)", options=["(none)"] + all_cols,
                                index=(["(none)"] + all_cols).index(mapping["created_at"]) if mapping["created_at"] in all_cols else 0)
        col_sent = st.selectbox("Sentiment label column (optional)", options=["(none)"] + all_cols,
                                index=(["(none)"] + all_cols).index(mapping["sentiment_label"]) if mapping["sentiment_label"] in all_cols else 0)

        # Save mapping to session
        mapping_user = {
            "comment_text": col_text,
            "kol_name": None if col_kol == "(none)" else col_kol,
            "brand": None if col_brand == "(none)" else col_brand,
            "campaign": None if col_campaign == "(none)" else col_campaign,
            "post_id": None if col_post == "(none)" else col_post,
            "created_at": None if col_date == "(none)" else col_date,
            "sentiment_label": None if col_sent == "(none)" else col_sent,
            "platform": mapping.get("platform"),
        }
        st.session_state["mapping_user"] = mapping_user

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Sample Rows</div>", unsafe_allow_html=True)
        st.dataframe(df_raw.head(30), use_container_width=True)

    # Use mapping from session state
    mapping_user = st.session_state.get("mapping_user")
    if not mapping_user:
        # If user didn't open Data tab, fallback to auto mapping with enforced text col
        mapping_user = mapping
        mapping_user["comment_text"] = mapping.get("comment_text") or suggest_text_columns(df_raw)[0]
        st.session_state["mapping_user"] = mapping_user

    # Enforce and prepare analysis DF
    try:
        df = df_raw.copy()
        df[mapping_user["comment_text"]] = enforce_text_column(df, mapping_user["comment_text"])
    except Exception as e:
        st.error(f"Text column selection error: {e}")
        st.stop()

    # If missing kol, create placeholder
    if not mapping_user.get("kol_name") or mapping_user["kol_name"] not in df.columns:
        df["_kol_fallback"] = "Unknown"
        mapping_user["kol_name"] = "_kol_fallback"

    # Compute sentiment
    df = compute_sentiments(df, mapping_user, use_llm=bool(use_llm), groq_model=groq_model)

    # Keywords
    kw = build_keyword_stats(df, mapping_user, risk_terms=risk_terms)
    term_stats = kw["term_stats"]
    kol_term = kw["kol_term"]

    # Risk scores
    risk = compute_risk_scores(df, mapping_user, term_stats=term_stats, risk_terms=risk_terms)
    campaign = risk["campaign"]
    kol_stats = risk["kol_stats"]

    # Brand comparison (if brand exists)
    brand_cmp = build_brand_comparison(df, mapping_user)

    # -------------------------
    # Overview
    # -------------------------
    with tabs[0]:
        left, right = st.columns([1.2, 1])

        with left:
            st.markdown("<div class='section-title'>Campaign Health</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='subtle'>Risk Score</div>"
                f"<div style='font-size:44px;font-weight:800;margin-top:4px;'>{campaign['risk_score']:.1f}</div>"
                f"<div style='margin-top:8px;'>{badge_html(campaign['risk_level'])}</div>",
                unsafe_allow_html=True
            )
            st.markdown("<hr/>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                kpi_card("Comments", str(campaign["total_comments"]))
            with c2:
                kpi_card("Negative ratio", f"{campaign['negative_ratio']*100:.1f}%")
            with c3:
                kpi_card("Risk term hits", str(campaign["risk_term_hits"]))

        with right:
            st.markdown("<div class='section-title'>Sentiment Distribution</div>", unsafe_allow_html=True)
            dist = df["_sentiment"].value_counts(normalize=True).reindex(["positive", "neutral", "negative"]).fillna(0).reset_index()
            dist.columns = ["sentiment", "ratio"]
            fig = px.bar(dist, x="sentiment", y="ratio", text=dist["ratio"].map(lambda x: f"{x*100:.1f}%"))
            fig.update_traces(textposition="outside")
            fig = chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        colA, colB = st.columns([1, 1])

        with colA:
            st.markdown("<div class='section-title'>Top Risk KOL</div>", unsafe_allow_html=True)
            if isinstance(kol_stats, pd.DataFrame) and not kol_stats.empty:
                st.dataframe(kol_stats.head(10), use_container_width=True)
            else:
                st.caption("KOL column not available or insufficient data.")

        with colB:
            st.markdown("<div class='section-title'>Top Risk Terms</div>", unsafe_allow_html=True)
            if isinstance(term_stats, pd.DataFrame) and not term_stats.empty:
                risk_terms_df = term_stats[term_stats["is_high_risk_term"]].copy()
                risk_terms_df = risk_terms_df.sort_values(["neg_ratio", "freq"], ascending=[False, False]).head(15)
                show_cols = ["term", "freq", "neg_ratio", "is_risk_dictionary", "is_high_risk_term"]
                if risk_terms_df.empty:
                    st.caption("No high-risk terms detected with current thresholds (neg_ratio > 60% and freq >= 3).")
                    st.dataframe(term_stats.head(15)[["term","freq","neg_ratio","pos_ratio"]], use_container_width=True)
                else:
                    risk_terms_df["neg_ratio"] = (risk_terms_df["neg_ratio"]*100).round(1).astype(str) + "%"
                    st.dataframe(risk_terms_df[show_cols], use_container_width=True)
            else:
                st.caption("Keyword stats unavailable (jieba missing or no text).")

        # Competitor comparison
        if isinstance(brand_cmp, pd.DataFrame) and not brand_cmp.empty and brand_cmp["brand"].nunique() > 1:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Brand Comparison</div>", unsafe_allow_html=True)
            fig = px.bar(
                brand_cmp,
                x="brand",
                y="negative_ratio",
                text=brand_cmp["negative_ratio"].map(lambda x: f"{x*100:.1f}%"),
            )
            fig.update_traces(textposition="outside")
            fig = chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # KOL Monitor
    # -------------------------
    with tabs[2]:
        st.markdown("<div class='section-title'>KOL Monitor</div>", unsafe_allow_html=True)
        st.caption("Ranked by risk score (0–100). Drill down to see drivers and evidence.")

        if kol_stats is None or (isinstance(kol_stats, pd.DataFrame) and kol_stats.empty):
            st.info("KOL monitor is not available because KOL column is missing or empty.")
        else:
            st.dataframe(kol_stats, use_container_width=True)

            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>KOL Drill-down</div>", unsafe_allow_html=True)
            kol_list = kol_stats["kol_name"].tolist()
            selected_kol = st.selectbox("Select a KOL", options=kol_list)

            sub = df[df[mapping_user["kol_name"]] == selected_kol].copy()
            st.markdown(f"<div class='subtle'>Comments: {len(sub)}</div>", unsafe_allow_html=True)

            # Show sentiment distribution for this KOL
            dist = sub["_sentiment"].value_counts(normalize=True).reindex(["positive", "neutral", "negative"]).fillna(0).reset_index()
            dist.columns = ["sentiment", "ratio"]
            fig = px.bar(dist, x="sentiment", y="ratio", text=dist["ratio"].map(lambda x: f"{x*100:.1f}%"))
            fig.update_traces(textposition="outside")
            fig = chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            # Evidence comments: show negative first
            text_col = mapping_user["comment_text"]
            neg_comments = sub[sub["_sentiment"] == "negative"][text_col].astype(str).head(20).tolist()
            neu_comments = sub[sub["_sentiment"] == "neutral"][text_col].astype(str).head(10).tolist()

            st.markdown("<div class='section-title'>Evidence (Negative)</div>", unsafe_allow_html=True)
            if neg_comments:
                for i, c in enumerate(neg_comments, 1):
                    st.write(f"{i}. {c}")
            else:
                st.caption("No negative comments found for this KOL.")

            st.markdown("<div class='section-title'>Neutral Signals</div>", unsafe_allow_html=True)
            if neu_comments:
                for i, c in enumerate(neu_comments, 1):
                    st.write(f"{i}. {c}")
            else:
                st.caption("No neutral comments found for this KOL.")

    # -------------------------
    # Insights
    # -------------------------
    with tabs[3]:
        st.markdown("<div class='section-title'>Insights</div>", unsafe_allow_html=True)
        st.caption("Keyword intelligence and structured charts. Word cloud is intentionally not a primary module.")

        if term_stats is None or term_stats.empty:
            st.info("Keyword stats not available. Please ensure jieba is installed and comment_text contains Chinese text.")
        else:
            c1, c2 = st.columns([1, 1])

            with c1:
                st.markdown("<div class='section-title'>Negative Keywords</div>", unsafe_allow_html=True)
                neg_top = term_stats.sort_values(["neg_ratio", "freq"], ascending=[False, False]).head(20).copy()
                neg_top["neg_ratio_pct"] = neg_top["neg_ratio"] * 100
                fig = px.bar(neg_top, x="neg_ratio_pct", y="term", orientation="h")
                fig = chart_theme(fig)
                fig.update_layout(xaxis_title="Negative ratio (%)", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("<div class='section-title'>Opportunity Keywords</div>", unsafe_allow_html=True)
                pos_top = term_stats.sort_values(["pos_ratio", "freq"], ascending=[False, False]).head(20).copy()
                pos_top["pos_ratio_pct"] = pos_top["pos_ratio"] * 100
                fig = px.bar(pos_top, x="pos_ratio_pct", y="term", orientation="h")
                fig = chart_theme(fig)
                fig.update_layout(xaxis_title="Positive ratio (%)", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("<hr/>", unsafe_allow_html=True)

            # Keyword x KOL heatmap
            st.markdown("<div class='section-title'>Keyword × KOL Matrix</div>", unsafe_allow_html=True)
            if kol_term is None or kol_term.empty:
                st.caption("Not enough data to build heatmap.")
            else:
                # keep top 15 KOL to fit
                kol_term_view = kol_term.copy()
                if kol_term_view.shape[0] > 15:
                    kol_term_view = kol_term_view.loc[kol_term_view.sum(axis=1).sort_values(ascending=False).head(15).index]
                fig = px.imshow(
                    kol_term_view.values,
                    x=kol_term_view.columns.tolist(),
                    y=kol_term_view.index.tolist(),
                    aspect="auto",
                )
                fig = chart_theme(fig)
                fig.update_layout(xaxis_title="Keyword", yaxis_title="KOL")
                st.plotly_chart(fig, use_container_width=True)

            # Sentiment stacked chart by brand (if available)
            brand_col = mapping_user.get("brand")
            if brand_col and brand_col in df.columns and df[brand_col].nunique() > 1:
                st.markdown("<hr/>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Sentiment by Brand</div>", unsafe_allow_html=True)
                pivot = (
                    df.groupby([brand_col, "_sentiment"])
                    .size()
                    .reset_index(name="count")
                )
                total = pivot.groupby(brand_col)["count"].transform("sum")
                pivot["ratio"] = pivot["count"] / total
                fig = px.bar(pivot, x=brand_col, y="ratio", color="_sentiment", barmode="stack")
                fig = chart_theme(fig)
                fig.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Actions
    # -------------------------
    with tabs[4]:
        st.markdown("<div class='section-title'>Actions</div>", unsafe_allow_html=True)
        st.caption("Decision support recommendations based on risk level and drivers.")

        level = campaign["risk_level"].lower()
        drivers = []

        # drivers from term stats
        if isinstance(term_stats, pd.DataFrame) and not term_stats.empty:
            high_risk_terms = term_stats[term_stats["is_high_risk_term"]].sort_values(["neg_ratio", "freq"], ascending=[False, False]).head(5)
            drivers = high_risk_terms["term"].tolist()

        st.markdown(f"<div class='subtle'>Campaign risk level: {badge_html(campaign['risk_level'])}</div>", unsafe_allow_html=True)
        st.markdown("<hr/>", unsafe_allow_html=True)

        col1, col2 = st.columns([1.1, 1])

        with col1:
            st.markdown("<div class='section-title'>Recommended Actions</div>", unsafe_allow_html=True)

            if level in ["low"]:
                st.write("- Continue monitoring; no immediate intervention required.")
                st.write("- Capture positive terms as creative angles for next content wave.")
                st.write("- Prepare a lightweight FAQ for repeated neutral questions.")
            elif level in ["medium"]:
                st.write("- Increase monitoring frequency; review top risk terms daily.")
                st.write("- Coordinate with community managers on response consistency.")
                st.write("- Identify top 3 risk-contributing KOL and align messaging.")
            elif level in ["high"]:
                st.write("- Escalate to PR / Brand safety owner; prepare official statement draft.")
                st.write("- Contact high-risk KOL for clarification and provide guidance.")
                st.write("- Set up keyword watchlist; prioritize safety/quality concerns.")
                st.write("- Consider pausing paid amplification for risky posts.")
            else:
                st.write("- Activate crisis protocol immediately.")
                st.write("- Publish official response with clear facts and next steps.")
                st.write("- Offer customer support path (refund/exchange/consultation) if applicable.")
                st.write("- Freeze risky placements and prioritize containment messaging.")
                st.write("- Start internal root-cause review (batch/quality/ingredient).")

        with col2:
            st.markdown("<div class='section-title'>Risk Drivers</div>", unsafe_allow_html=True)
            if drivers:
                st.write("High-risk terms detected:")
                for d in drivers:
                    st.write(f"- {d}")
            else:
                st.write("No high-risk terms meet the strict thresholds yet.")
                st.write("Tip: add custom risk terms in the sidebar if your category has specific issues.")

            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>One-page Summary (Copy)</div>", unsafe_allow_html=True)

            summary_lines = [
                f"Campaign Risk Score: {campaign['risk_score']:.1f} ({campaign['risk_level']})",
                f"Comments: {campaign['total_comments']}",
                f"Negative ratio: {campaign['negative_ratio']*100:.1f}%",
                f"Risk term hits: {campaign['risk_term_hits']}",
            ]
            if drivers:
                summary_lines.append("Top risk drivers: " + ", ".join(drivers))

            if isinstance(kol_stats, pd.DataFrame) and not kol_stats.empty:
                top_kol = kol_stats.iloc[0]
                summary_lines.append(f"Top risk KOL: {top_kol['kol_name']} (score {top_kol['risk_score']:.1f}, neg {top_kol['negative_ratio']*100:.1f}%)")

            st.code("\n".join(summary_lines), language="text")


if __name__ == "__main__":
    main()
