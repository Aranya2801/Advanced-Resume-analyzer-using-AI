"""Streamlit front end.  Run with:  streamlit run app.py"""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from resume_analyzer import ResumeAnalyzer
from resume_analyzer.llm import get_llm
from resume_analyzer.parsing import UnsupportedFormatError

load_dotenv()

st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .chip {display:inline-block; padding:2px 10px; margin:2px; border-radius:12px; font-size:0.85rem;}
      .chip-ok {background:#e6f4ea; color:#1e6b34;}
      .chip-part {background:#fff4e0; color:#8a5a00;}
      .chip-miss {background:#fde8e8; color:#9b1c1c;}
      .finding {opacity:0.75; font-size:0.9rem; margin:0 0 0.2rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models...")
def load_analyzer(encoder: str) -> ResumeAnalyzer:
    return ResumeAnalyzer(encoder=encoder)


def chips(items: list[str], kind: str) -> str:
    if not items:
        return "<span class='finding'>none</span>"
    return "".join(f"<span class='chip chip-{kind}'>{i}</span>" for i in items)


def radar(components: dict) -> go.Figure:
    names = list(components)
    values = [components[n]["score"] for n in names]
    fig = go.Figure(go.Scatterpolar(r=values + values[:1], theta=names + names[:1], fill="toself"))
    fig.update_layout(polar={"radialaxis": {"range": [0, 100]}}, showlegend=False,
                      height=320, margin={"l": 40, "r": 40, "t": 20, "b": 20})
    return fig


# --- sidebar -----------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    encoder = st.selectbox(
        "Sentence encoder", ["auto", "sbert", "spacy", "none"],
        help="Used for requirement-level semantic matching. 'auto' picks the best one installed.",
    )
    st.divider()
    st.subheader("Optional: LLM feedback")
    st.caption("Scores never depend on this. It only adds written feedback, bullet rewrites and cover letters.")
    provider = st.selectbox("Provider", ["anthropic", "openai"])
    key = st.text_input("API key", type="password", value="",
                        help="Kept in this browser session only. Or set ANTHROPIC_API_KEY / OPENAI_API_KEY in .env")
    llm = get_llm(provider=provider, api_key=key or None) if (key or os.getenv(f"{provider.upper()}_API_KEY")) else None
    st.caption("LLM: " + ("on" if llm else "off"))

analyzer = load_analyzer(encoder)

st.title("Resume Analyzer")
st.caption("Explainable resume scoring and job matching. "
           f"Encoder: `{analyzer.encoder.name if analyzer.encoder else 'none'}`")

tab_one, tab_rank = st.tabs(["Analyze a resume", "Rank several resumes"])

# --- single resume -------------------------------------------------------------

with tab_one:
    c1, c2 = st.columns([1, 1])
    with c1:
        upload = st.file_uploader("Resume (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"], key="one")
    with c2:
        jd_text = st.text_area("Job description (optional)", height=170,
                               placeholder="Paste the job posting to get a match score and skill gaps")

    if upload:
        try:
            result = analyzer.analyze(upload.getvalue(), job_description=jd_text or None, filename=upload.name)
        except UnsupportedFormatError as err:
            st.error(str(err))
            st.stop()
        d = result.to_dict()
        q = d["quality"]

        cols = st.columns(4)
        cols[0].metric("Resume quality", f"{q['overall']:.0f}/100")
        if "match" in d:
            cols[1].metric("Job match", f"{d['match']['score']:.0f}/100")
        cols[2].metric("Bullets", d["stats"]["bullets"])
        yrs = d["stats"]["months_experience"] / 12
        cols[3].metric("Experience (from dates)", f"{yrs:.1f} yrs")

        sub = st.tabs(["Quality", "Job match", "Bullets", "Skills", "LLM feedback", "Raw JSON"])

        with sub[0]:
            left, right = st.columns([1, 1.3])
            left.plotly_chart(radar(q["components"]), width="stretch")
            with right:
                for name, comp in q["components"].items():
                    st.markdown(f"**{name.title()}** · {comp['score']:.0f}")
                    for f in comp["findings"]:
                        st.markdown(f"<p class='finding'>{f}</p>", unsafe_allow_html=True)

        with sub[1]:
            if "match" not in d:
                st.info("Paste a job description above to see how well this resume matches it.")
            else:
                m = d["match"]
                st.markdown("Component scores: " + " · ".join(f"**{k}** {v:.0f}" for k, v in m["components"].items()))
                st.markdown("**Matched skills**<br>" + chips(m["skills"]["matched"], "ok"), unsafe_allow_html=True)
                st.markdown("**Partial (related skill)**<br>" + chips(m["skills"]["partial"], "part"), unsafe_allow_html=True)
                st.markdown("**Missing**<br>" + chips(m["skills"]["missing"], "miss"), unsafe_allow_html=True)
                listed = [i["skill"] for i in m["skills"]["items"] if i["via"] == "listed-only"]
                if listed:
                    st.caption("Only in your skills list, never shown in a bullet (counts for less): " + ", ".join(listed))
                if m["requirements"]:
                    st.markdown("#### Requirement by requirement")
                    st.caption("Each JD line next to the resume line most similar to it.")
                    df = pd.DataFrame(m["requirements"]).sort_values("similarity")
                    df["weight"] = df["weight"].map({1.0: "required", 0.5: "nice to have"})
                    st.dataframe(df.rename(columns={"best_line": "closest line in resume"}),
                                 width="stretch", hide_index=True)

        with sub[2]:
            b = d["bullets"]
            st.write(f"{b['strong_openers']} strong openers, {b['weak_openers']} weak, "
                     f"{b['quantified']} of {b['total']} with a number.")
            if b["weak_examples"]:
                st.markdown("**Weak openers**")
                for i, bullet in enumerate(b["weak_examples"]):
                    st.markdown(f"- {bullet}")
                    if llm and st.button("Suggest a rewrite", key=f"rw{i}"):
                        with st.spinner("Asking the model..."):
                            try:
                                rw = llm.rewrite_bullet(bullet)
                                st.success(rw.get("rewrite", ""))
                                st.caption(rw.get("why", ""))
                            except Exception as err:
                                st.error(f"Rewrite failed: {err}")
            if b["unquantified_examples"]:
                st.markdown("**No numbers** - can you add scale or outcome?")
                for bullet in b["unquantified_examples"]:
                    st.markdown(f"- {bullet}")

        with sub[3]:
            for group, names in d["skills"].items():
                st.markdown(f"**{group}**<br>" + chips(names, "ok"), unsafe_allow_html=True)

        with sub[4]:
            if not llm:
                st.info("Add an API key in the sidebar to get written feedback. Everything else works without it.")
            else:
                if st.button("Get feedback"):
                    with st.spinner("Writing feedback..."):
                        try:
                            fb = llm.review(result.resume.text, result.summary_text(), jd_text)
                            st.markdown("**Strengths**")
                            for s in fb.get("strengths", []):
                                st.markdown(f"- {s}")
                            st.markdown("**Issues**")
                            for issue in fb.get("issues", []):
                                st.markdown(f"- *{issue.get('section', '')}*: {issue.get('issue', '')} → {issue.get('fix', '')}")
                            if fb.get("summary_rewrite"):
                                st.markdown("**Suggested summary**")
                                st.write(fb["summary_rewrite"])
                        except Exception as err:
                            st.error(f"LLM request failed: {err}")
                if jd_text:
                    company = st.text_input("Company name for a cover letter")
                    if company and st.button("Draft cover letter"):
                        with st.spinner("Drafting..."):
                            st.text_area("Cover letter", llm.cover_letter(result.resume.text, jd_text, company), height=300)

        with sub[5]:
            st.json(d, expanded=False)
            st.download_button("Download JSON", data=json.dumps(d, indent=2), file_name="resume_report.json")

# --- ranking --------------------------------------------------------------------

with tab_rank:
    st.caption("Ranks resumes for one job. With 3+ resumes, TF-IDF similarity across the pool is added to the score.")
    files = st.file_uploader("Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="many")
    jd_rank = st.text_area("Job description", height=170, key="jd_rank")
    if files and jd_rank:
        resumes, failed = [], []
        for f in files:
            try:
                resumes.append(analyzer.parse(f.getvalue(), filename=f.name))
            except Exception:
                failed.append(f.name)
        if failed:
            st.warning("Skipped: " + ", ".join(failed))
        ranked = analyzer.rank(resumes, jd_rank)
        rows = [{"rank": i + 1, "file": r.resume.source, "score": round(r.score * 100, 1),
                 **{k: round(v * 100) for k, v in r.components.items()},
                 "missing skills": ", ".join(r.match.skills.missing[:6])} for i, r in enumerate(ranked)]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
