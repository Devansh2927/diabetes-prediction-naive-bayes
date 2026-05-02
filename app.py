import streamlit as st
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="DiabetesScan · AI Diagnostics",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Outfit:wght@200;300;400;500;600;700;900&display=swap');

:root {
    --bg:        #030508;
    --bg2:       #080d14;
    --bg3:       #0d1520;
    --card:      #0a1120;
    --border:    #1a2840;
    --border2:   #243450;
    --lime:      #aaff00;
    --lime2:     #88cc00;
    --cyan:      #00e5ff;
    --red:       #ff3b5c;
    --amber:     #ffb800;
    --text:      #e8f0ff;
    --muted:     #4a6080;
    --muted2:    #2a3a50;
}

*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 120% 40% at 50% -5%, rgba(0,229,255,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 80% 60% at 100% 100%, rgba(170,255,0,0.03) 0%, transparent 50%),
        var(--bg) !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"],
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── HEADER ── */
.app-header {
    position: relative;
    overflow: hidden;
    border-bottom: 1px solid var(--border);
}
.header-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 3rem;
    position: relative;
    z-index: 2;
}
.header-left { display: flex; align-items: center; gap: 1.2rem; }
.logo-mark {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--lime), var(--cyan));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; flex-shrink: 0;
    box-shadow: 0 0 20px rgba(170,255,0,0.3);
}
.brand-name {
    font-family: 'Outfit', sans-serif;
    font-weight: 900; font-size: 1.35rem;
    letter-spacing: -0.04em; color: var(--text); line-height: 1;
}
.brand-name span { color: var(--lime); }
.brand-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem; color: var(--muted);
    letter-spacing: 0.2em; text-transform: uppercase; margin-top: 0.25rem;
}
.header-badges { display: flex; gap: 0.6rem; align-items: center; }
.hbadge {
    font-family: 'Space Mono', monospace; font-size: 0.6rem;
    padding: 0.3rem 0.75rem; border-radius: 4px;
    letter-spacing: 0.1em; text-transform: uppercase; font-weight: 700;
}
.hbadge-green {
    background: rgba(170,255,0,0.1); border: 1px solid rgba(170,255,0,0.3); color: var(--lime);
}
.hbadge-blue {
    background: rgba(0,229,255,0.08); border: 1px solid rgba(0,229,255,0.2); color: var(--cyan);
}
.ecg-strip {
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 36px; overflow: hidden; opacity: 0.22;
}
.ecg-svg { width: 200%; height: 100%; animation: ecgScroll 3s linear infinite; }
@keyframes ecgScroll {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
}

/* ── SECTION TITLES ── */
.sec-title {
    display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;
}
.sec-num {
    font-family: 'Space Mono', monospace; font-size: 0.62rem; font-weight: 700;
    color: var(--lime); background: rgba(170,255,0,0.08);
    border: 1px solid rgba(170,255,0,0.2); border-radius: 4px;
    padding: 0.2rem 0.5rem; letter-spacing: 0.05em; flex-shrink: 0;
}
.sec-name { font-weight: 700; font-size: 0.95rem; color: var(--text); letter-spacing: -0.01em; }
.sec-line { flex: 1; height: 1px; background: var(--border); }

/* ── INPUT OVERRIDES ── */
[data-testid="stNumberInput"] > div > div {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stNumberInput"] > div > div:focus-within {
    border-color: var(--lime) !important;
    box-shadow: 0 0 0 2px rgba(170,255,0,0.1) !important;
}
[data-testid="stNumberInput"] input {
    background: transparent !important; color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important; font-weight: 700 !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label {
    color: var(--muted) !important; font-size: 0.72rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.1em !important; font-family: 'Space Mono', monospace !important;
}
[data-testid="stSlider"] > div > div > div > div { background: var(--lime) !important; }
[data-testid="stSlider"] > div > div > div { background: var(--border) !important; }
[data-testid="stNumberInput"] button {
    background: transparent !important; border: none !important; color: var(--muted) !important;
}
[data-testid="stNumberInput"] button:hover { color: var(--lime) !important; }
[data-testid="stSlider"] div[data-testid="stTickBarMin"],
[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: var(--muted) !important; font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    background: var(--lime) !important; color: #030508 !important;
    font-family: 'Outfit', sans-serif !important; font-size: 0.9rem !important;
    font-weight: 900 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; border: none !important;
    border-radius: 8px !important; height: 3rem !important; width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 30px rgba(170,255,0,0.2) !important;
}
.stButton > button:hover {
    background: #c8ff20 !important;
    box-shadow: 0 0 50px rgba(170,255,0,0.45), 0 4px 20px rgba(0,0,0,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── METRICS STRIP ── */
.metrics-strip {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: 10px;
    overflow: hidden; margin-bottom: 2rem;
}
.metric-cell { background: var(--bg2); padding: 1rem; text-align: center; }
.metric-val {
    font-family: 'Space Mono', monospace; font-size: 1.1rem;
    font-weight: 700; color: var(--lime); line-height: 1;
}
.metric-label {
    font-size: 0.62rem; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.1em; margin-top: 0.3rem; font-weight: 500;
}

/* ── RIGHT PANEL ── */
.vitals-header {
    display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem;
}
.vitals-title {
    font-family: 'Space Mono', monospace; font-size: 0.65rem;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.15em;
}
.live-dot {
    display: flex; align-items: center; gap: 0.4rem;
    font-family: 'Space Mono', monospace; font-size: 0.6rem;
    color: var(--lime); letter-spacing: 0.1em;
}
.dot {
    width: 6px; height: 6px; background: var(--lime);
    border-radius: 50%; animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px var(--lime); }
    50% { opacity: 0.4; box-shadow: none; }
}
.vital-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.85rem 1rem; background: var(--card);
    border: 1px solid var(--border); border-radius: 8px;
    margin-bottom: 0.5rem; transition: border-color 0.2s;
}
.vital-row:hover { border-color: var(--border2); }
.vital-left { display: flex; align-items: center; gap: 0.65rem; }
.vital-icon { font-size: 1rem; width: 28px; text-align: center; flex-shrink: 0; }
.vital-name { font-size: 0.72rem; color: var(--muted); font-weight: 500; line-height: 1.2; }
.vital-abbr {
    font-family: 'Space Mono', monospace; font-size: 0.58rem;
    color: var(--muted2); letter-spacing: 0.05em; margin-top: 0.1rem;
}
.vital-val {
    font-family: 'Space Mono', monospace; font-size: 1rem;
    font-weight: 700; color: var(--lime); text-align: right;
}
.vital-unit {
    font-family: 'Space Mono', monospace; font-size: 0.58rem;
    color: var(--muted); text-align: right; margin-top: 0.15rem;
}

/* ── RESULT ── */
.result-box-pos {
    background: linear-gradient(135deg, rgba(255,59,92,0.12), rgba(255,59,92,0.04));
    border: 1px solid rgba(255,59,92,0.4); border-radius: 12px;
    padding: 1.5rem; text-align: center;
    box-shadow: 0 0 40px rgba(255,59,92,0.08), inset 0 1px 0 rgba(255,255,255,0.04);
    animation: fadeUp 0.4s ease;
}
.result-box-neg {
    background: linear-gradient(135deg, rgba(170,255,0,0.1), rgba(0,229,255,0.04));
    border: 1px solid rgba(170,255,0,0.35); border-radius: 12px;
    padding: 1.5rem; text-align: center;
    box-shadow: 0 0 40px rgba(170,255,0,0.06), inset 0 1px 0 rgba(255,255,255,0.04);
    animation: fadeUp 0.4s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-emoji { font-size: 2.4rem; margin-bottom: 0.5rem; }
.result-label {
    font-family: 'Space Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0.4rem; opacity: 0.7;
}
.result-verdict {
    font-family: 'Outfit', sans-serif; font-size: 1.5rem;
    font-weight: 900; letter-spacing: -0.03em; line-height: 1; margin-bottom: 0.6rem;
}
.result-sub {
    font-size: 0.75rem; opacity: 0.55; font-weight: 400;
    line-height: 1.5; max-width: 220px; margin: 0 auto;
}
.prob-section {
    margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border);
}
.prob-item { margin-bottom: 0.7rem; }
.prob-header {
    display: flex; justify-content: space-between; margin-bottom: 0.3rem;
    font-family: 'Space Mono', monospace; font-size: 0.62rem;
}
.prob-track { height: 5px; background: var(--border); border-radius: 10px; overflow: hidden; }
.prob-fill-neg {
    height: 100%; border-radius: 10px;
    background: linear-gradient(90deg, var(--lime), var(--cyan));
}
.prob-fill-pos {
    height: 100%; border-radius: 10px;
    background: linear-gradient(90deg, var(--amber), var(--red));
}
.disclaimer-box {
    margin-top: 1rem; padding: 0.8rem 1rem;
    background: rgba(255,184,0,0.05); border: 1px solid rgba(255,184,0,0.15);
    border-left: 2px solid var(--amber); border-radius: 6px;
    font-size: 0.68rem; color: var(--muted); line-height: 1.5;
}

[data-testid="stHorizontalBlock"] { gap: 0.9rem !important; }
[data-testid="column"] { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load("diabetes_naive.pkl")

model = load_model()

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="header-inner">
    <div class="header-left">
      <div class="logo-mark">🫀</div>
      <div>
        <div class="brand-name">Diabetes<span>Scan</span></div>
        <div class="brand-sub">AI Diagnostic System · v2.0</div>
      </div>
    </div>
    <div class="header-badges">
      <div class="hbadge hbadge-green">● Model Ready</div>
      <div class="hbadge hbadge-blue">GNB · PIMA Dataset</div>
    </div>
  </div>
  <div class="ecg-strip">
    <svg class="ecg-svg" viewBox="0 0 1920 36" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
      <polyline fill="none" stroke="#aaff00" stroke-width="1.5"
        points="0,18 40,18 50,18 60,4 70,32 75,1 80,32 90,18 110,18
                120,18 160,18 170,18 180,4 190,32 195,1 200,32 210,18 230,18
                240,18 280,18 290,18 300,4 310,32 315,1 320,32 330,18 350,18
                360,18 400,18 410,18 420,4 430,32 435,1 440,32 450,18 470,18
                480,18 520,18 530,18 540,4 550,32 555,1 560,32 570,18 590,18
                600,18 640,18 650,18 660,4 670,32 675,1 680,32 690,18 710,18
                720,18 760,18 770,18 780,4 790,32 795,1 800,32 810,18 830,18
                840,18 880,18 890,18 900,4 910,32 915,1 920,32 930,18 950,18
                960,18 1000,18 1010,18 1020,4 1030,32 1035,1 1040,32 1050,18 1070,18
                1080,18 1120,18 1130,18 1140,4 1150,32 1155,1 1160,32 1170,18 1190,18
                1200,18 1240,18 1250,18 1260,4 1270,32 1275,1 1280,32 1290,18 1310,18
                1320,18 1360,18 1370,18 1380,4 1390,32 1395,1 1400,32 1410,18 1430,18
                1440,18 1480,18 1490,18 1500,4 1510,32 1515,1 1520,32 1530,18 1550,18
                1560,18 1600,18 1610,18 1620,4 1630,32 1635,1 1640,32 1650,18 1670,18
                1680,18 1720,18 1730,18 1740,4 1750,32 1755,1 1760,32 1770,18 1790,18
                1800,18 1840,18 1850,18 1860,4 1870,32 1875,1 1880,32 1890,18 1920,18"/>
    </svg>
  </div>
</div>
""", unsafe_allow_html=True)

# ── MAIN CONTENT ───────────────────────────────────────────────────────────────
left, right = st.columns([3, 1.3])

with left:
    st.markdown("<div style='padding: 2.5rem 0 0 3rem;'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="metrics-strip">
        <div class="metric-cell"><div class="metric-val">GNB</div><div class="metric-label">Algorithm</div></div>
        <div class="metric-cell"><div class="metric-val">8</div><div class="metric-label">Features</div></div>
        <div class="metric-cell"><div class="metric-val">768</div><div class="metric-label">Train Samples</div></div>
        <div class="metric-cell"><div class="metric-val">Binary</div><div class="metric-label">Output</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Section 01
    st.markdown("""
    <div class="sec-title">
        <span class="sec-num">01</span>
        <span class="sec-name">Metabolic Markers</span>
        <span class="sec-line"></span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=110, step=1)
    with c2:
        insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=79, step=1)
    with c3:
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=32.0, step=0.1, format="%.1f")

    # Section 02
    st.markdown("""
    <div class="sec-title" style="margin-top:1.5rem;">
        <span class="sec-num">02</span>
        <span class="sec-name">Physical Vitals</span>
        <span class="sec-line"></span>
    </div>
    """, unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=150, value=72, step=1)
    with c5:
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=23, step=1)
    with c6:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3, step=1)

    # Section 03
    st.markdown("""
    <div class="sec-title" style="margin-top:1.5rem;">
        <span class="sec-num">03</span>
        <span class="sec-name">Genetic & Demographic</span>
        <span class="sec-line"></span>
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        dpf = st.slider("Diabetes Pedigree Function", 0.050, 2.500, 0.470, 0.001, format="%.3f")
    with sc2:
        age = st.slider("Age (years)", 18, 90, 35)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    predict = st.button("⚡  ANALYZE RISK PROFILE", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── RIGHT PANEL ────────────────────────────────────────────────────────────────
with right:
    st.markdown("<div style='padding: 2.5rem 0 0 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="vitals-header">
        <div class="vitals-title">Live Parameters</div>
        <div class="live-dot"><div class="dot"></div>ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

    vitals = [
        ("🍬", "Glucose",        "GLU",  glucose,          "mg/dL"),
        ("💉", "Insulin",         "INS",  insulin,          "μU/mL"),
        ("⚖️", "BMI",             "BMI",  f"{bmi:.1f}",     "kg/m²"),
        ("💓", "Blood Pressure",  "DBP",  blood_pressure,   "mmHg"),
        ("📏", "Skin Thickness",  "SKN",  skin_thickness,   "mm"),
        ("🤰", "Pregnancies",     "PRG",  pregnancies,      "count"),
        ("🧬", "Pedigree Func.",  "DPF",  f"{dpf:.3f}",     "score"),
        ("📅", "Age",             "AGE",  age,              "yrs"),
    ]

    for icon, name, abbr, val, unit in vitals:
        st.markdown(f"""
        <div class="vital-row">
            <div class="vital-left">
                <div class="vital-icon">{icon}</div>
                <div>
                    <div class="vital-name">{name}</div>
                    <div class="vital-abbr">{abbr}</div>
                </div>
            </div>
            <div>
                <div class="vital-val">{val}</div>
                <div class="vital-unit">{unit}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if predict:
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                               insulin, bmi, dpf, age]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0]

        prob_neg = proba[0] * 100
        prob_pos = proba[1] * 100

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

        if pred == 1:
            st.markdown(f"""
            <div class="result-box-pos">
                <div class="result-emoji">⚠️</div>
                <div class="result-label" style="color:#ff6b8a;">Diagnostic Output</div>
                <div class="result-verdict" style="color:#ff3b5c;">POSITIVE</div>
                <div class="result-sub">Elevated diabetes risk detected. Clinical consultation strongly advised.</div>
                <div class="prob-section">
                    <div class="prob-item">
                        <div class="prob-header">
                            <span style="color:var(--muted)">No Diabetes</span>
                            <span style="color:var(--lime)">{prob_neg:.1f}%</span>
                        </div>
                        <div class="prob-track"><div class="prob-fill-neg" style="width:{prob_neg:.1f}%"></div></div>
                    </div>
                    <div class="prob-item">
                        <div class="prob-header">
                            <span style="color:var(--muted)">Diabetes</span>
                            <span style="color:var(--red)">{prob_pos:.1f}%</span>
                        </div>
                        <div class="prob-track"><div class="prob-fill-pos" style="width:{prob_pos:.1f}%"></div></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-neg">
                <div class="result-emoji">✅</div>
                <div class="result-label" style="color:var(--lime);">Diagnostic Output</div>
                <div class="result-verdict" style="color:var(--lime);">NEGATIVE</div>
                <div class="result-sub">No significant diabetes risk detected. Maintain regular check-ups.</div>
                <div class="prob-section">
                    <div class="prob-item">
                        <div class="prob-header">
                            <span style="color:var(--muted)">No Diabetes</span>
                            <span style="color:var(--lime)">{prob_neg:.1f}%</span>
                        </div>
                        <div class="prob-track"><div class="prob-fill-neg" style="width:{prob_neg:.1f}%"></div></div>
                    </div>
                    <div class="prob-item">
                        <div class="prob-header">
                            <span style="color:var(--muted)">Diabetes</span>
                            <span style="color:var(--red)">{prob_pos:.1f}%</span>
                        </div>
                        <div class="prob-track"><div class="prob-fill-pos" style="width:{prob_pos:.1f}%"></div></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-box">
            ⚠ <strong>Disclaimer:</strong> For research use only. Not a substitute for clinical diagnosis. Consult a licensed physician.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)






































































































































































































































































































