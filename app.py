import streamlit as st
import joblib
import re

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NewsGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* -------------------- MAIN BACKGROUND -------------------- */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(55, 65, 150, 0.18), transparent 30%),
        radial-gradient(circle at 90% 10%, rgba(0, 150, 180, 0.12), transparent 30%),
        linear-gradient(135deg, #080d20 0%, #0b1025 50%, #07111d 100%);
    color: #f8fafc;
}

/* -------------------- REMOVE DEFAULT PADDING -------------------- */

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* -------------------- HEADER -------------------- */

.top-badge {
    display: inline-block;
    padding: 10px 20px;
    border-radius: 30px;
    border: 1px solid rgba(120, 140, 255, 0.45);
    background: rgba(40, 45, 100, 0.35);
    color: #aebdff;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 1px;
    margin-bottom: 20px;
}

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    margin: 5px 0;
    background: linear-gradient(90deg, #c7d2fe, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #b6c0d4;
    font-size: 18px;
    margin-top: 12px;
}

.description {
    text-align: center;
    color: #a8b3c7;
    font-size: 16px;
    margin-top: 8px;
}

/* -------------------- CARDS -------------------- */

.info-card {
    background: rgba(20, 27, 55, 0.72);
    border: 1px solid rgba(130, 145, 190, 0.20);
    border-radius: 22px;
    padding: 28px 32px;
    margin-top: 55px;
    margin-bottom: 30px;
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.25);
}

.card-title {
    font-size: 22px;
    font-weight: 750;
    color: #f8fafc;
    margin-bottom: 15px;
}

.card-text {
    color: #9da9c0;
    font-size: 15px;
    line-height: 1.6;
}

/* -------------------- INPUT LABELS -------------------- */

label {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

/* -------------------- INPUT BOX -------------------- */

.stTextInput input,
.stTextArea textarea {
    color: #111827 !important;
    background-color: #f3f4f6 !important;
    border: 2px solid #9ca3af !important;
    border-radius: 12px !important;
    font-size: 16px !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.25) !important;
}

/* -------------------- BUTTON -------------------- */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    padding: 14px;
    font-size: 17px;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, #4f46e5, #2563eb);
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.25);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.40);
}

/* -------------------- RESULT CARDS -------------------- */

.result-real {
    background: rgba(16, 90, 70, 0.22);
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-radius: 20px;
    padding: 25px;
    margin-top: 30px;
    text-align: center;
}

.result-fake {
    background: rgba(110, 30, 50, 0.22);
    border: 1px solid rgba(248, 113, 113, 0.35);
    border-radius: 20px;
    padding: 25px;
    margin-top: 30px;
    text-align: center;
}

.result-title {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-subtitle {
    color: #b9c2d4;
    font-size: 15px;
}

/* -------------------- STAT CARDS -------------------- */

.stat-card {
    background: rgba(20, 27, 55, 0.72);
    border: 1px solid rgba(130, 145, 190, 0.18);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    margin-top: 20px;
}

.stat-number {
    font-size: 27px;
    font-weight: 800;
    color: #dbeafe;
}

.stat-label {
    color: #8995ac;
    font-size: 13px;
    margin-top: 5px;
}

/* -------------------- CUSTOM PROBABILITY BARS -------------------- */

.probability-label {
    color: #dbe4f5;
    font-size: 15px;
    font-weight: 600;
    margin-top: 18px;
    margin-bottom: 7px;
}

.progress-container {
    width: 100%;
    height: 13px;
    background: rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 12px;
}

.progress-fake {
    height: 100%;
    background: linear-gradient(90deg, #ef4444, #f87171);
    border-radius: 20px;
}

.progress-real {
    height: 100%;
    background: linear-gradient(90deg, #10b981, #34d399);
    border-radius: 20px;
}

/* -------------------- SECTION TITLE -------------------- */

.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #e5e7eb;
    margin-top: 55px;
    margin-bottom: 20px;
}

/* -------------------- FOOTER -------------------- */

.footer {
    text-align: center;
    margin-top: 60px;
    padding-top: 25px;
    border-top: 1px solid rgba(130, 145, 190, 0.15);
    color: #78849a;
    font-size: 13px;
}

.disclaimer {
    background: rgba(30, 35, 60, 0.55);
    border: 1px solid rgba(130, 145, 190, 0.15);
    border-radius: 14px;
    padding: 18px;
    margin-top: 25px;
    color: #929db2;
    font-size: 13px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div style="text-align:center;"><span class="top-badge">✦ AI POWERED • MACHINE LEARNING</span></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🛡️ NewsGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Fake News Detection & Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="description">Analyze news content using Natural Language Processing and Machine Learning</div>',
    unsafe_allow_html=True
)

# ============================================================
# INTRO CARD
# ============================================================

st.markdown("""
<div class="info-card">
    <div class="card-title">📰 Analyze a News Article</div>
    <div class="card-text">
        Enter the headline and article content below.
        Our trained ML model will analyze the text and classify it as
        <b>Fake</b> or <b>Real</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================

headline = st.text_input(
    "News Headline",
    placeholder="e.g. Scientists announce a major breakthrough..."
)

article = st.text_area(
    "News Article",
    placeholder="Paste the complete news article here...",
    height=220
)

# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button("🔍 Analyze News")

# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if headline.strip() == "" and article.strip() == "":
        st.warning("⚠️ Please enter a headline or news article first.")

    else:

        # Combine headline and article
        content = headline + " " + article

        # TF-IDF transformation
        vectorized_text = tfidf.transform([content])

        # Prediction
        prediction = model.predict(vectorized_text)[0]

        # Probability
        probabilities = model.predict_proba(vectorized_text)[0]

        fake_probability = probabilities[0] * 100
        real_probability = probabilities[1] * 100

        # ====================================================
        # RESULT
        # ====================================================

        if prediction == 1:

            st.markdown(f"""
            <div class="result-real">
                <div class="result-title">🟢 REAL NEWS</div>
                <div class="result-subtitle">
                    The model classified this article as likely real.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-fake">
                <div class="result-title">🔴 FAKE NEWS</div>
                <div class="result-subtitle">
                    The model classified this article as likely fake.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ====================================================
        # PROBABILITY
        # ====================================================

        st.markdown(
            '<div class="section-title">📊 Prediction Confidence</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{fake_probability:.2f}%</div>
                <div class="stat-label">Fake Probability</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{real_probability:.2f}%</div>
                <div class="stat-label">Real Probability</div>
            </div>
            """, unsafe_allow_html=True)

        # ====================================================
        # CUSTOM PROBABILITY BARS
        # ====================================================

        st.markdown(
            f"""
            <div class="probability-label">
                🔴 Fake — {fake_probability:.2f}%
            </div>

            <div class="progress-container">
                <div class="progress-fake"
                     style="width: {fake_probability}%;">
                </div>
            </div>

            <div class="probability-label">
                🟢 Real — {real_probability:.2f}%
            </div>

            <div class="progress-container">
                <div class="progress-real"
                     style="width: {real_probability}%;">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ====================================================
        # ARTICLE STATISTICS
        # ====================================================

        st.markdown(
            '<div class="section-title">📈 Article Analysis</div>',
            unsafe_allow_html=True
        )

        words = re.findall(r'\b\w+\b', content)

        sentences = re.split(r'[.!?]+', content)
        sentences = [s for s in sentences if s.strip()]

        word_count = len(words)
        character_count = len(content)
        sentence_count = len(sentences)

        stat1, stat2, stat3 = st.columns(3)

        with stat1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{word_count}</div>
                <div class="stat-label">Words</div>
            </div>
            """, unsafe_allow_html=True)

        with stat2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{character_count}</div>
                <div class="stat-label">Characters</div>
            </div>
            """, unsafe_allow_html=True)

        with stat3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{sentence_count}</div>
                <div class="stat-label">Sentences</div>
            </div>
            """, unsafe_allow_html=True)

        # ====================================================
        # HOW IT WORKS
        # ====================================================

        st.markdown(
            '<div class="section-title">🤖 How NewsGuard AI Works</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="info-card" style="margin-top:0;">
                <div class="card-title">1️⃣ Text Processing</div>
                <div class="card-text">
                    The headline and article are combined and converted
                    into numerical features using NLP techniques.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-card" style="margin-top:0;">
                <div class="card-title">2️⃣ TF-IDF</div>
                <div class="card-text">
                    TF-IDF identifies important words and represents
                    the article as numerical vectors.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="info-card" style="margin-top:0;">
                <div class="card-title">3️⃣ ML Prediction</div>
                <div class="card-text">
                    A Logistic Regression model analyzes the features
                    and predicts whether the article is Fake or Real.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <b>Disclaimer:</b>
            This application uses a machine learning model trained on
            a news dataset. The prediction indicates the model's learned
            classification pattern and should not be treated as a definitive
            verification of the truthfulness of a news article.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <b>NewsGuard AI</b> • Fake News Detection & Analyzer<br>
    Built using Python • Scikit-learn • TF-IDF • Logistic Regression • Streamlit
</div>
""", unsafe_allow_html=True)