"""
╔══════════════════════════════════════════════════════════════════╗
║        ADVANCED RESUME ANALYZER - Streamlit Application          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import tempfile
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Advanced Resume Analyzer · AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Aranya2801/Advanced-Resume-analyzer-using-AI",
        "Report a bug": "https://github.com/Aranya2801/Advanced-Resume-analyzer-using-AI/issues",
        "About": "Advanced Resume Analyzer powered by Claude AI",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
  --primary: #6C63FF;
  --secondary: #FF6584;
  --accent: #43E97B;
  --bg-dark: #0D1117;
  --card-bg: #161B22;
  --border: #30363D;
  --text: #E6EDF3;
  --muted: #8B949E;
}

html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif !important;
}

.stApp { background: var(--bg-dark); }

/* Gradient header */
.hero-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
  padding: 2.5rem 2rem;
  border-radius: 16px;
  margin-bottom: 2rem;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.hero-header::before {
  content: '';
  position: absolute;
  inset: 0;
  background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.hero-header h1 {
  font-size: 2.8rem;
  font-weight: 700;
  color: white;
  margin: 0;
  text-shadow: 0 2px 20px rgba(0,0,0,0.3);
}
.hero-header p {
  color: rgba(255,255,255,0.85);
  font-size: 1.1rem;
  margin-top: 0.5rem;
}

/* Score cards */
.score-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem;
  text-align: center;
  transition: transform 0.2s, border-color 0.2s;
}
.score-card:hover { transform: translateY(-3px); border-color: var(--primary); }
.score-value { font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg,#6C63FF,#f093fb); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.score-label { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* Tags */
.tag {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.78rem;
  margin: 3px;
  font-weight: 500;
}
.tag-matched { background: rgba(67,233,123,0.15); color: #43E97B; border: 1px solid rgba(67,233,123,0.3); }
.tag-missing { background: rgba(255,101,132,0.15); color: #FF6584; border: 1px solid rgba(255,101,132,0.3); }
.tag-skill { background: rgba(108,99,255,0.15); color: #6C63FF; border: 1px solid rgba(108,99,255,0.3); }

/* Recommendation cards */
.rec-card {
  background: var(--card-bg);
  border-left: 4px solid #6C63FF;
  border-radius: 0 8px 8px 0;
  padding: 1rem 1.25rem;
  margin: 0.5rem 0;
}
.rec-card.high { border-left-color: #FF6584; }
.rec-card.medium { border-left-color: #FFD93D; }
.rec-card.low { border-left-color: #43E97B; }
.rec-priority { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
.rec-issue { font-weight: 600; color: var(--text); margin: 4px 0; }
.rec-fix { color: var(--muted); font-size: 0.9rem; }
.rec-example { background: rgba(108,99,255,0.1); border-radius: 4px; padding: 6px 10px; margin-top: 8px; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; color: #b3b3ff; }

/* Metric highlight */
.metric-highlight {
  background: linear-gradient(135deg, rgba(108,99,255,0.1), rgba(240,147,251,0.1));
  border: 1px solid rgba(108,99,255,0.2);
  border-radius: 10px;
  padding: 1rem;
  margin: 0.5rem 0;
}

/* Better bullets */
.bullet-before {
  background: rgba(255,101,132,0.1);
  border: 1px solid rgba(255,101,132,0.2);
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 0.9rem;
  color: #FF6584;
}
.bullet-after {
  background: rgba(67,233,123,0.1);
  border: 1px solid rgba(67,233,123,0.2);
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 0.9rem;
  color: #43E97B;
  margin-top: 4px;
}

/* Section headers */
.section-header {
  font-size: 1.2rem;
  font-weight: 700;
  color: var(--text);
  border-bottom: 2px solid var(--primary);
  padding-bottom: 6px;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* Cover letter box */
.cover-letter-box {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  white-space: pre-wrap;
  line-height: 1.8;
  font-size: 0.95rem;
  color: var(--text);
}

/* Stagger animation */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
.animate-up { animation: fadeUp 0.5s ease forwards; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--card-bg) !important; border-right: 1px solid var(--border); }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────

def gauge_chart(value: float, title: str, color: str = "#6C63FF") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13, "color": "#8B949E", "family": "Space Grotesk"}},
        number={"font": {"size": 28, "color": color, "family": "Space Grotesk"}, "suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#30363D"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#161B22",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(255,101,132,0.15)"},
                {"range": [40, 70], "color": "rgba(255,217,61,0.1)"},
                {"range": [70, 100], "color": "rgba(67,233,123,0.1)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": value},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
        height=200,
        font=dict(family="Space Grotesk"),
    )
    return fig


def radar_chart(scores: dict) -> go.Figure:
    categories = list(scores.keys())
    values = list(scores.values())
    values += [values[0]]
    categories += [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories,
        fill="toself",
        fillcolor="rgba(108,99,255,0.15)",
        line=dict(color="#6C63FF", width=2),
        marker=dict(color="#6C63FF", size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#8B949E", size=9), gridcolor="#30363D"),
            angularaxis=dict(tickfont=dict(color="#E6EDF3", size=11), gridcolor="#30363D"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=320,
    )
    return fig


def score_color(score: float) -> str:
    if score >= 75: return "#43E97B"
    if score >= 50: return "#FFD93D"
    return "#FF6584"


def render_score_card(label: str, value: float, emoji: str = ""):
    color = score_color(value)
    st.markdown(f"""
    <div class="score-card animate-up">
        <div style="font-size:1.8rem">{emoji}</div>
        <div class="score-value" style="background:none;-webkit-text-fill-color:{color}">{int(value)}</div>
        <div class="score-label">{label}</div>
    </div>""", unsafe_allow_html=True)


def render_recommendation(rec: dict):
    priority = rec.get("priority", "MEDIUM").lower()
    colors = {"high": "#FF6584", "medium": "#FFD93D", "low": "#43E97B"}
    color = colors.get(priority, "#6C63FF")
    st.markdown(f"""
    <div class="rec-card {priority}">
      <div class="rec-priority" style="color:{color}">⚡ {rec.get('priority','').upper()} · {rec.get('section','')}</div>
      <div class="rec-issue">{rec.get('issue','')}</div>
      <div class="rec-fix">→ {rec.get('fix','')}</div>
      {'<div class="rec-example">💡 ' + rec.get('example','') + '</div>' if rec.get('example') else ''}
    </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0;">
        <div style="font-size:3rem">🧠</div>
        <div style="font-weight:700;font-size:1.1rem;color:#E6EDF3">Resume Analyzer</div>
        <div style="color:#8B949E;font-size:0.8rem">Powered by Claude AI</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get yours at console.anthropic.com",
    )
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")

    target_role = st.text_input("🎯 Target Role", placeholder="e.g. Senior ML Engineer")

    st.divider()
    st.markdown("### 📋 Analysis Options")
    run_bullet_improve = st.checkbox("✨ AI Bullet Improvements", value=True)
    run_cover_letter = st.checkbox("📝 Generate Cover Letter", value=False)
    run_career_path = st.checkbox("🗺️ Career Path Analysis", value=True)

    if run_cover_letter:
        company_name = st.text_input("🏢 Company Name", placeholder="e.g. Google")
    else:
        company_name = ""

    st.divider()
    st.markdown("""
    <div style="color:#8B949E;font-size:0.78rem;line-height:1.6">
    <b style="color:#E6EDF3">Supported Formats</b><br>
    📄 PDF · DOCX · TXT<br><br>
    <b style="color:#E6EDF3">Analysis Includes</b><br>
    🔍 ATS Scoring<br>
    🤖 Claude AI Review<br>
    📊 Keyword Matching<br>
    💡 Bullet Rewrites<br>
    🗺️ Career Paths<br>
    📝 Cover Letter Gen
    </div>""", unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <h1>🧠 Advanced Resume Analyzer</h1>
  <p>MIT-grade AI-powered resume intelligence · Claude AI · NLP · Semantic Search</p>
</div>""", unsafe_allow_html=True)


# ── Upload Area ───────────────────────────────────────────────────
col_upload, col_jd = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="section-header">📤 Upload Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your resume here",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.success(f"✅ **{uploaded_file.name}** ({uploaded_file.size // 1024} KB)")

with col_jd:
    st.markdown('<div class="section-header">💼 Job Description (Optional)</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste job description",
        height=160,
        placeholder="Paste the job description here to get keyword matching, ATS alignment score, and role-specific recommendations...",
        label_visibility="collapsed",
    )


# ── Analyse Button ────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    analyse_btn = st.button("🚀 Analyse Resume", use_container_width=True, type="primary")


# ── Analysis Logic ────────────────────────────────────────────────
if analyse_btn:
    if not uploaded_file:
        st.error("⚠️ Please upload a resume file first.")
        st.stop()
    if not api_key:
        st.error("⚠️ Please enter your Anthropic API key in the sidebar.")
        st.stop()

    # Save uploaded file to temp
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from analyzer import ResumeAnalyzer

        analyzer = ResumeAnalyzer(anthropic_api_key=api_key)

        with st.spinner("🔍 Parsing document..."):
            pass

        progress = st.progress(0, text="Starting analysis pipeline...")

        with st.spinner("🤖 Running AI deep analysis..."):
            progress.progress(20, "Parsing resume...")
            result = analyzer.full_analysis(
                tmp_path,
                job_description=job_description,
                target_role=target_role,
            )
            progress.progress(60, "Generating insights...")

        if run_bullet_improve:
            with st.spinner("✨ Improving bullet points..."):
                analyzer.improve_bullets(result, n=4)
                progress.progress(75, "Improving bullets...")

        cover_letter_text = ""
        if run_cover_letter and company_name:
            with st.spinner("📝 Writing cover letter..."):
                cover_letter_text = analyzer.generate_cover_letter(result, job_description, company_name)
                progress.progress(90, "Writing cover letter...")

        progress.progress(100, "✅ Analysis complete!")
        st.balloons()

        # ── RESULTS ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)

        # Top score cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: render_score_card("Overall", result.overall_score, "🏆")
        with c2: render_score_card("ATS Score", result.ats_score, "🤖")
        with c3: render_score_card("Content", result.content_score, "📝")
        with c4: render_score_card("Impact", result.impact_score, "⚡")
        with c5: render_score_card("Quantification", result.quantification_score, "📈")

        st.markdown("<br>", unsafe_allow_html=True)

        # Radar + Gauges
        col_radar, col_gauges = st.columns([1.2, 1])

        with col_radar:
            st.markdown('<div class="section-header">📡 Skill Radar</div>', unsafe_allow_html=True)
            radar_scores = {
                "ATS": result.ats_score,
                "Content": result.content_score,
                "Impact": result.impact_score,
                "Keywords": result.keyword_match_score or 50,
                "Quantification": result.quantification_score,
                "Action Verbs": result.action_verb_analysis.get("score", 50),
            }
            st.plotly_chart(radar_chart(radar_scores), use_container_width=True)

        with col_gauges:
            st.markdown('<div class="section-header">🎯 Key Metrics</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(result.overall_score, "Overall Score", score_color(result.overall_score)), use_container_width=True)
            if result.job_match_percentage:
                st.plotly_chart(gauge_chart(result.job_match_percentage, "Job Match %", "#f093fb"), use_container_width=True)

        # ── Tabs ─────────────────────────────────────────────────
        tabs = st.tabs([
            "💡 Recommendations",
            "🔑 Keywords",
            "🛠️ Skills",
            "✨ Bullet Rewrites",
            "🗺️ Career Path",
            "📊 Readability",
            "📝 Cover Letter",
            "👤 Candidate Info",
        ])

        # Tab 1 – Recommendations
        with tabs[0]:
            col_s, col_w = st.columns(2)
            with col_s:
                st.markdown('<div class="section-header">💪 Strengths</div>', unsafe_allow_html=True)
                for s in result.strengths:
                    st.markdown(f'<div class="metric-highlight">✅ {s}</div>', unsafe_allow_html=True)
            with col_w:
                st.markdown('<div class="section-header">⚠️ Weaknesses</div>', unsafe_allow_html=True)
                for w in result.weaknesses:
                    st.markdown(f'<div class="metric-highlight" style="border-color:rgba(255,101,132,0.3)">❌ {w}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">🔧 Action Items</div>', unsafe_allow_html=True)
            for rec in result.recommendations:
                render_recommendation(rec)

            if result.rewritten_summary:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">✍️ AI-Rewritten Summary</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="cover-letter-box">{result.rewritten_summary}</div>', unsafe_allow_html=True)

        # Tab 2 – Keywords
        with tabs[1]:
            if result.matched_keywords or result.missing_keywords:
                col_m, col_miss = st.columns(2)
                with col_m:
                    st.markdown(f'<div class="section-header">✅ Matched Keywords ({len(result.matched_keywords)})</div>', unsafe_allow_html=True)
                    tags = " ".join(f'<span class="tag tag-matched">{k}</span>' for k in result.matched_keywords[:25])
                    st.markdown(tags, unsafe_allow_html=True)
                with col_miss:
                    st.markdown(f'<div class="section-header">❌ Missing Keywords ({len(result.missing_keywords)})</div>', unsafe_allow_html=True)
                    tags = " ".join(f'<span class="tag tag-missing">{k}</span>' for k in result.missing_keywords[:25])
                    st.markdown(tags, unsafe_allow_html=True)

                # Bar chart
                kw_data = pd.DataFrame({
                    "Category": ["Matched", "Missing"],
                    "Count": [len(result.matched_keywords), len(result.missing_keywords)],
                    "Color": ["#43E97B", "#FF6584"],
                })
                fig = px.bar(kw_data, x="Category", y="Count", color="Category",
                             color_discrete_map={"Matched": "#43E97B", "Missing": "#FF6584"},
                             template="plotly_dark")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Upload a job description to see keyword matching analysis.")

        # Tab 3 – Skills
        with tabs[2]:
            skills = result.resume_data.skills
            if skills:
                for category, skill_list in skills.items():
                    st.markdown(f"**{category}**")
                    tags = " ".join(f'<span class="tag tag-skill">{s}</span>' for s in skill_list)
                    st.markdown(tags, unsafe_allow_html=True)
                    st.markdown("")

                # Skills sunburst
                rows = []
                for cat, slist in skills.items():
                    for s in slist:
                        rows.append({"Category": cat, "Skill": s, "Count": 1})
                if rows:
                    df = pd.DataFrame(rows)
                    fig = px.sunburst(df, path=["Category", "Skill"], values="Count",
                                      color_discrete_sequence=px.colors.qualitative.Pastel,
                                      template="plotly_dark")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=450)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No structured skills extracted. Ensure your resume has a clear Skills section.")

        # Tab 4 – Bullet Rewrites
        with tabs[3]:
            if result.bullet_improvements:
                for i, imp in enumerate(result.bullet_improvements):
                    st.markdown(f"**Bullet #{i+1}**")
                    st.markdown(f'<div class="bullet-before">❌ {imp.get("original","")}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="bullet-after">✅ {imp.get("improved","")}</div>', unsafe_allow_html=True)
                    st.caption(f"💡 {imp.get('explanation','')}")
                    st.markdown("")
            else:
                st.info("Enable 'AI Bullet Improvements' in the sidebar and re-run.")

        # Tab 5 – Career Path
        with tabs[4]:
            ci = result.career_insights
            if ci:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Current Trajectory", ci.get("current_trajectory", "N/A"))
                    st.metric("Market Demand", ci.get("market_demand", "N/A"))
                    if "salary_range" in ci:
                        sr = ci["salary_range"]
                        st.metric("Est. Salary Range", f"${sr.get('min',0):,} – ${sr.get('max',0):,}")
                with col_b:
                    st.markdown("**🎯 Skills to Invest In**")
                    for skill in ci.get("skill_investments", []):
                        st.markdown(f'<span class="tag tag-skill">📚 {skill}</span>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown('<div class="section-header">🗺️ Recommended Career Paths</div>', unsafe_allow_html=True)
                paths = ci.get("recommended_paths", [])
                for path in paths:
                    with st.expander(f"🚀 {path.get('path','')}  —  Fit: {path.get('fit_score',0)}/100"):
                        st.progress(path.get("fit_score", 0) / 100)
                        st.markdown(f"**Timeline:** {path.get('timeline','')}")
                        st.markdown("**Steps:**")
                        for step in path.get("steps", []):
                            st.markdown(f"• {step}")

                if ci.get("differentiation_advice"):
                    st.markdown('<div class="section-header">🌟 Differentiation Strategy</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="cover-letter-box">{ci["differentiation_advice"]}</div>', unsafe_allow_html=True)

        # Tab 6 – Readability
        with tabs[5]:
            rs = result.readability_stats
            va = result.action_verb_analysis

            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("Word Count", rs.get("word_count", 0))
            col_r2.metric("Page Count", rs.get("page_count", 0))
            col_r3.metric("Reading Ease", rs.get("flesch_reading_ease", 0))
            col_r4.metric("Grade Level", rs.get("grade_level", 0))

            st.markdown("---")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.metric("Power Verb Ratio", f"{va.get('power_verb_ratio',0)*100:.0f}%")
                st.metric("Strong Bullets", va.get("power_count", 0))
            with col_v2:
                st.metric("Weak Bullets", va.get("weak_count", 0))
                st.metric("Total Bullets", va.get("total_bullets", 0))

            if va.get("weak_bullets"):
                st.markdown("**⚠️ Weak Bullets to Fix:**")
                for b in va["weak_bullets"]:
                    st.markdown(f'<div class="bullet-before">— {b}</div>', unsafe_allow_html=True)

            # Action verb donut
            fig_v = go.Figure(go.Pie(
                labels=["Power Verbs", "Weak Verbs"],
                values=[va.get("power_count", 0), va.get("weak_count", 0)],
                hole=0.6,
                marker_colors=["#43E97B", "#FF6584"],
                textfont=dict(color="white"),
            ))
            fig_v.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300,
                                legend=dict(font=dict(color="#8B949E")))
            st.plotly_chart(fig_v, use_container_width=True)

        # Tab 7 – Cover Letter
        with tabs[6]:
            if cover_letter_text:
                st.markdown(f'<div class="cover-letter-box">{cover_letter_text}</div>', unsafe_allow_html=True)
                st.download_button("⬇️ Download Cover Letter", cover_letter_text, file_name="cover_letter.txt")
            else:
                st.info("Enable 'Generate Cover Letter' in the sidebar and enter a company name, then re-run.")

        # Tab 8 – Candidate Info
        with tabs[7]:
            pi = result.resume_data.personal_info
            col_i1, col_i2 = st.columns(2)
            with col_i1:
                st.markdown("**👤 Extracted Personal Info**")
                if pi.name: st.markdown(f"**Name:** {pi.name}")
                if pi.email: st.markdown(f"**Email:** {pi.email}")
                if pi.phone: st.markdown(f"**Phone:** {pi.phone}")
                if pi.linkedin: st.markdown(f"**LinkedIn:** {pi.linkedin}")
                if pi.github: st.markdown(f"**GitHub:** {pi.github}")
            with col_i2:
                if result.ai_feedback:
                    st.markdown("**🤖 AI Executive Summary**")
                    st.markdown(f'<div class="cover-letter-box">{result.ai_feedback}</div>', unsafe_allow_html=True)

        # Download full JSON report
        st.markdown("---")
        report_json = json.dumps(analyzer.to_dict(result), indent=2, default=str)
        st.download_button(
            "📥 Download Full Analysis Report (JSON)",
            report_json,
            file_name="resume_analysis_report.json",
            mime="application/json",
        )

    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        st.exception(e)
    finally:
        tmp_path.unlink(missing_ok=True)
