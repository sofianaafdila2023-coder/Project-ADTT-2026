# -*- coding: utf-8 -*-
"""
Analisis Sentimen Masyarakat terhadap Kebijakan PPKM
Streamlit App — Proyek Akhir ADTT 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import re, os, warnings, io
warnings.filterwarnings('ignore')

import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen PPKM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — tema formal terang: putih & biru muda
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;400;500;600;700&display=swap');

/* ── Palet warna utama ── */
:root {
    --bg:          #F0F4F9;
    --bg2:         #FFFFFF;
    --bg3:         #E8EEF5;
    --primary:     #1A4F8A;
    --primary2:    #2563AB;
    --accent:      #3B9ED4;
    --accent2:     #5BB8EA;
    --border:      #C8D8EC;
    --border2:     #DDEAF5;
    --text:        #0F2744;
    --text2:       #3D5775;
    --muted:       #7A96B4;
    --green:       #1A8A4F;
    --green-bg:    #E6F5EE;
    --red:         #C0392B;
    --red-bg:      #FDECEA;
    --blue-soft:   #E8F3FA;
    --shadow:      0 2px 12px rgba(26,79,138,0.08);
    --shadow-md:   0 4px 20px rgba(26,79,138,0.12);
}

/* ── Reset global ── */
html, body, [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0D1F35 !important;
    border-right: 1px solid #1A3050 !important;
}
[data-testid="stSidebar"] * {
    color: #D6E6F5 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #D6E6F5 !important;
}
[data-testid="stSidebar"] .stSlider > div > div {
    background: rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: white !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1A3050 !important;
}
[data-testid="stSidebar"] .stFileUploader label {
    color: #A8C8E8 !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] code {
    background: rgba(59,158,212,0.2) !important;
    color: #7EC8EC !important;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 0.8rem !important;
}
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #7A9EBF !important;
}
[data-testid="stSidebar"] strong, [data-testid="stSidebar"] b {
    color: #FFFFFF !important;
}

/* ── Hero header ── */
.hero-wrap {
    background: linear-gradient(135deg, var(--primary) 0%, #1E5E9E 50%, var(--primary2) 100%);
    border-radius: 8px;
    padding: 2.2rem 2.8rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-md);
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -30px; right: -30px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent2), var(--accent), var(--accent2));
}
.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 0.55rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    font-weight: 400;
    color: #FFFFFF;
    line-height: 1.25;
    margin-bottom: 0.65rem;
}
.hero-sub {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.7);
    font-weight: 300;
    letter-spacing: 0.01em;
}
.hero-sub b {
    color: rgba(255,255,255,0.95);
    font-weight: 600;
}
.hero-divider {
    width: 40px;
    height: 2px;
    background: var(--accent2);
    margin: 0.9rem 0;
    border-radius: 2px;
}

/* ── Navigasi menu utama ── */
.nav-container {
    background: var(--bg2);
    border: 1.5px solid var(--border);
    border-radius: 8px;
    padding: 5px;
    margin-bottom: 2rem;
    display: flex;
    gap: 4px;
    box-shadow: var(--shadow);
}
/* override tombol Streamlit untuk nav */
div[data-testid="column"] .stButton > button {
    width: 100% !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.6rem 0.5rem !important;
    border-radius: 5px !important;
    transition: all 0.18s !important;
    text-align: center !important;
}
/* Tombol aktif (primary) */
div[data-testid="column"] .stButton > button[kind="primary"] {
    background: var(--primary) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(26,79,138,0.25) !important;
}
div[data-testid="column"] .stButton > button[kind="primary"]:hover {
    background: var(--primary2) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(26,79,138,0.3) !important;
}
/* Tombol non-aktif (secondary) */
div[data-testid="column"] .stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--text2) !important;
    border: 1px solid transparent !important;
}
div[data-testid="column"] .stButton > button[kind="secondary"]:hover {
    background: var(--bg3) !important;
    color: var(--primary) !important;
    border-color: var(--border2) !important;
}

/* ── Breadcrumb aktif page ── */
.breadcrumb {
    background: var(--blue-soft);
    border: 1px solid var(--border2);
    border-left: 3px solid var(--primary);
    border-radius: 4px;
    padding: 0.55rem 1rem;
    margin: 0 0 1.8rem;
    font-size: 0.82rem;
    color: var(--primary2);
    font-weight: 500;
    letter-spacing: 0.02em;
}

/* ── Section header ── */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    font-weight: 400;
    color: var(--primary);
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
    display: inline-block;
    letter-spacing: 0.01em;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--bg2);
    border: 1.5px solid var(--border2);
    border-top: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: var(--shadow);
    transition: box-shadow 0.2s;
}
.metric-card:hover {
    box-shadow: var(--shadow-md);
}
.metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.1rem;
    font-weight: 400;
    color: var(--primary);
    line-height: 1;
}
.metric-sub {
    font-size: 0.76rem;
    color: var(--accent);
    margin-top: 0.3rem;
    font-weight: 500;
}

/* ── Landing cards ── */
.landing-card {
    background: var(--bg2);
    border: 1.5px solid var(--border2);
    border-radius: 8px;
    padding: 1.5rem;
    height: 100%;
    box-shadow: var(--shadow);
}
.landing-card h4 {
    font-family: 'DM Serif Display', serif;
    color: var(--primary);
    margin-bottom: 0.6rem;
    font-size: 1rem;
    font-weight: 400;
}
.landing-card p {
    font-size: 0.87rem;
    color: var(--text2);
    line-height: 1.65;
}

/* ── Status banners ── */
.status-ok {
    background: rgba(26,138,79,0.15);
    border: 1px solid rgba(26,138,79,0.4);
    border-left: 3px solid #1A8A4F;
    border-radius: 5px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #5EC991;
    font-weight: 500;
}
.status-warn {
    background: rgba(230,168,23,0.12);
    border: 1px solid rgba(230,168,23,0.35);
    border-left: 3px solid #E6A817;
    border-radius: 5px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #F0C04A;
    font-weight: 500;
}

/* ── Result cards ── */
.result-pos {
    background: var(--green-bg);
    border: 1.5px solid #A8DFC2;
    border-left: 4px solid var(--green);
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
}
.result-neg {
    background: var(--red-bg);
    border: 1.5px solid #F1AFAA;
    border-left: 4px solid var(--red);
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
}
.result-net {
    background: var(--blue-soft);
    border: 1.5px solid var(--border2);
    border-left: 4px solid var(--accent);
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
}
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    font-weight: 400;
    color: var(--text);
}
.result-conf {
    font-size: 0.85rem;
    color: var(--text2);
    margin-top: 0.25rem;
}

/* ── Winner badge ── */
.winner-badge {
    display: inline-block;
    background: var(--primary);
    color: white;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.28rem 0.85rem;
    border-radius: 3px;
    box-shadow: 0 2px 6px rgba(26,79,138,0.2);
}

/* ── Sidebar branding ── */
.sidebar-brand {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    font-weight: 400;
    color: #FFFFFF !important;
    margin-bottom: 0.15rem;
}
.sidebar-sub {
    font-size: 0.7rem;
    color: #7A9EBF !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.sidebar-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5A8AB0 !important;
    margin-bottom: 0.4rem;
    margin-top: 0.6rem;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow) !important;
}

/* ── Streamlit metric default ── */
[data-testid="stMetric"] {
    background: var(--bg2);
    border: 1.5px solid var(--border2);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    box-shadow: var(--shadow);
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stMetricValue"] {
    color: var(--primary) !important;
    font-family: 'DM Serif Display', serif !important;
}

/* ── Buttons global (non-nav) ── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    border-radius: 5px !important;
}

/* ── Text area & input ── */
.stTextArea textarea, .stTextInput input {
    background: var(--bg2) !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 6px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59,158,212,0.12) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    box-shadow: var(--shadow) !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: var(--accent) !important;
}

/* ── Alert boxes ── */
.stSuccess  { border-radius: 5px !important; }
.stError    { border-radius: 5px !important; }
.stInfo     { border-radius: 5px !important; }
.stWarning  { border-radius: 5px !important; }

/* ── Divider ── */
hr { border-color: rgba(26,79,138,0.12) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg3); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Caption / small ── */
.stCaption, small, [data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
}

/* ── Tab hide (pakai custom nav) ── */
[data-testid="stTabs"] > div:first-child { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KONSTANTA — plot palette disesuaikan warna terang
# ─────────────────────────────────────────────
PALETTE = {'positif': '#1A8A4F', 'negatif': '#C0392B', 'netral': '#2563AB'}
BG      = '#F0F4F9'
BG2     = '#FFFFFF'
PRIMARY = '#1A4F8A'
ACCENT  = '#3B9ED4'
MUTED   = '#7A96B4'
TEXT    = '#0F2744'

# ─────────────────────────────────────────────
# INIT NLP
# ─────────────────────────────────────────────
@st.cache_resource
def init_nlp():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    sw_factory = StopWordRemoverFactory()
    stopword_list = set(sw_factory.get_stop_words())
    custom_sw = {
        'yang','ini','itu','dan','dengan','untuk','dari','ke','di',
        'pada','aja','juga','banget','nih','sih','deh','lah','kan',
        'ya','yg','udah','sudah','bisa','ada','lebih','masih','jadi',
        'https','http','rt','co','amp','ppkm','via','bit','ly',
        'tco','pic','twitter','com','www'
    }
    stopword_list.update(custom_sw)
    return stemmer, stopword_list

slang_dict = {
    'gak':'tidak','ga':'tidak','ngga':'tidak','nggak':'tidak',
    'yg':'yang','dgn':'dengan','utk':'untuk','krn':'karena',
    'gimana':'bagaimana','kalo':'kalau','udah':'sudah',
    'bnyk':'banyak','hrs':'harus','trs':'terus','jg':'juga',
    'dr':'dari','pd':'pada','sy':'saya','skrg':'sekarang',
    'kpd':'kepada','dlm':'dalam','sdh':'sudah','tsb':'tersebut',
    'thd':'terhadap','ttg':'tentang','krna':'karena',
    'emg':'memang','emang':'memang','bgt':'sangat','msh':'masih',
    'sm':'sama','lg':'lagi','ny':'nya','mk':'maka','spy':'supaya',
    'wkwk':'','haha':'','hehe':'','wkwkwk':''
}

def preprocess(text, stemmer, stopword_list):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [slang_dict.get(w, w) for w in text.split()]
    tokens = word_tokenize(' '.join(words))
    tokens = [t for t in tokens if t not in stopword_list and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def read_csv_auto(source):
    raw = open(source, 'rb').read() if isinstance(source, (str, os.PathLike)) else source
    for sep in ['\t', ',', ';']:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, on_bad_lines='skip')
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(raw), on_bad_lines='skip')

def load_dataframe(raw_source, labeled_source):
    df_raw     = read_csv_auto(raw_source)
    df_labeled = read_csv_auto(labeled_source)
    cols_lower = {c.lower(): c for c in df_labeled.columns}

    sentiment_col = next((cols_lower[c] for c in ['sentiment','sentimen','label','polarity','class'] if c in cols_lower), None)
    tweet_col     = next((cols_lower[c] for c in ['tweet','text','teks','content','full_text'] if c in cols_lower), None)
    date_col      = next((cols_lower[c] for c in ['date','tanggal','created_at','timestamp','time'] if c in cols_lower), None)

    if not sentiment_col: raise ValueError(f"Kolom label tidak ditemukan. Kolom: {list(df_labeled.columns)}")
    if not tweet_col:     raise ValueError(f"Kolom tweet tidak ditemukan. Kolom: {list(df_labeled.columns)}")

    rename_map = {tweet_col: 'Tweet', sentiment_col: 'sentiment_raw'}
    if date_col: rename_map[date_col] = 'Date'
    df_labeled = df_labeled.rename(columns=rename_map)
    if 'Date' not in df_labeled.columns: df_labeled['Date'] = pd.NaT

    label_map = {
        0:'negatif',1:'positif',2:'netral',
        '0':'negatif','1':'positif','2':'netral',
        'positif':'positif','negatif':'negatif','netral':'netral',
        'positive':'positif','negative':'negatif','neutral':'netral',
    }
    def map_label(v):
        if pd.isna(v): return None
        try: return label_map.get(int(float(str(v).strip())))
        except: return label_map.get(str(v).strip().lower())

    df_labeled['sentimen'] = df_labeled['sentiment_raw'].apply(map_label)
    df_labeled = df_labeled.dropna(subset=['sentimen'])
    return df_raw, df_labeled

def build_df(df_labeled, stemmer, stopword_list, max_per_class):
    df = df_labeled[['Date','Tweet','sentimen']].copy()
    df = df.dropna(subset=['Tweet','sentimen'])
    df = df[df['Tweet'].str.strip() != ''].reset_index(drop=True)
    parts = []
    for lbl in df['sentimen'].unique():
        sub = df[df['sentimen'] == lbl]
        parts.append(sub.sample(min(len(sub), max_per_class), random_state=42))
    df = pd.concat(parts).reset_index(drop=True)

    progress = st.progress(0, text="Memproses tweet...")
    total = len(df); results = []
    for i, row in df.iterrows():
        results.append(preprocess(row['Tweet'], stemmer, stopword_list))
        if i % 200 == 0:
            progress.progress(min(int(i/total*100), 99), text=f"Preprocessing {i}/{total} tweet...")
    progress.progress(100, text="Selesai!"); progress.empty()

    df['tweet_bersih'] = results
    df = df[df['tweet_bersih'].str.strip().str.len() > 3].reset_index(drop=True)
    df['panjang_sebelum'] = df['Tweet'].apply(lambda x: len(str(x).split()))
    df['panjang_sesudah'] = df['tweet_bersih'].apply(lambda x: len(str(x).split()))
    return df

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_models(df):
    if 'sentimen' not in df.columns:
        raise KeyError(f"Kolom 'sentimen' tidak ada. Kolom tersedia: {list(df.columns)}")
    le = LabelEncoder()
    dfc = df.copy()
    dfc['label'] = le.fit_transform(dfc['sentimen'])
    X, y = dfc['tweet_bersih'], dfc['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.90, sublinear_tf=True)
    X_tr = tfidf.fit_transform(X_train); X_te = tfidf.transform(X_test)
    nb = MultinomialNB(alpha=0.5); nb.fit(X_tr, y_train); y_nb = nb.predict(X_te)
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    lr.fit(X_tr, y_train); y_lr = lr.predict(X_te)
    def mtr(yt, yp): return {'Accuracy':accuracy_score(yt,yp),'Precision':precision_score(yt,yp,average='weighted',zero_division=0),'Recall':recall_score(yt,yp,average='weighted',zero_division=0),'F1-Score':f1_score(yt,yp,average='weighted',zero_division=0)}
    return {'le':le,'tfidf':tfidf,'nb':nb,'lr':lr,
            'acc_nb':accuracy_score(y_test,y_nb),'acc_lr':accuracy_score(y_test,y_lr),
            'y_test':y_test,'y_nb':y_nb,'y_lr':y_lr,
            'cm_nb':confusion_matrix(y_test,y_nb),'cm_lr':confusion_matrix(y_test,y_lr),
            'report_nb':classification_report(y_test,y_nb,target_names=le.classes_,digits=4,output_dict=True),
            'report_lr':classification_report(y_test,y_lr,target_names=le.classes_,digits=4,output_dict=True),
            'df_compare':pd.DataFrame({'Naïve Bayes':mtr(y_test,y_nb),'Logistic Regression':mtr(y_test,y_lr)}).T.round(4),
            'X':X,'y':y}

# ─────────────────────────────────────────────
# PLOT HELPERS — latar terang
# ─────────────────────────────────────────────
def style_ax(ax):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(PRIMARY)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color('#C8D8EC')

def make_fig(w, h, cols=1):
    if cols == 1:
        fig, ax = plt.subplots(figsize=(w, h))
        fig.patch.set_facecolor(BG2)
        return fig, ax
    else:
        fig, axes = plt.subplots(1, cols, figsize=(w, h))
        fig.patch.set_facecolor(BG2)
        return fig, axes

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo / brand area
    st.markdown("""
    <div style="background:rgba(59,158,212,0.12);border:1px solid #1A3A5C;border-left:3px solid #3B9ED4;border-radius:6px;padding:1rem 1.1rem;margin-bottom:0.5rem;">
      <div class="sidebar-brand">Analisis Sentimen PPKM</div>
      <div class="sidebar-sub">ADTT · Proyek Akhir 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    DATA_DIR      = os.path.join(os.path.dirname(__file__), 'data')
    LOCAL_RAW     = os.path.join(DATA_DIR, 'INA_TweetsPPKM_Raw.csv')
    LOCAL_LABELED = os.path.join(DATA_DIR, 'INA_TweetsPPKM_Labeled_Pure.csv')
    HAS_LOCAL     = os.path.isfile(LOCAL_RAW) and os.path.isfile(LOCAL_LABELED)

    if HAS_LOCAL:
        st.markdown('<div class="status-ok">✓ Dataset terdeteksi dari folder <code>data/</code></div>', unsafe_allow_html=True)
        data_source = 'local'; uploaded_raw = uploaded_labeled = None
    else:
        st.markdown('<div class="sidebar-section-label">Upload Dataset</div>', unsafe_allow_html=True)
        st.caption("Dua file CSV dari Kaggle PPKM Twitter")
        uploaded_raw     = st.file_uploader("File Raw CSV",     type=['csv','tsv'], key='raw')
        uploaded_labeled = st.file_uploader("File Labeled CSV", type=['csv','tsv'], key='labeled')
        data_source = 'upload'
        if not (uploaded_raw and uploaded_labeled):
            st.markdown('<div class="status-warn">⚠ Taruh file di folder <code>data/</code> untuk tampil otomatis</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-section-label">Ukuran Sample</div>', unsafe_allow_html=True)
    max_per_class = st.slider("Maks tweet per kelas", 500, 5000, 2000, 500,
                              help="Kurangi jika proses terlalu lama", label_visibility="collapsed")
    st.caption(f"Per kelas: **{max_per_class:,}** tweet · Total maks: **{max_per_class*3:,}**")

    st.divider()
    st.markdown('<div class="sidebar-section-label">Tentang Aplikasi</div>', unsafe_allow_html=True)
    st.caption("Mata Kuliah: Analisis Data Tak Terstruktur")
    st.caption("Dataset: Kaggle PPKM Twitter (CC0)")
    st.caption("Metode: Naïve Bayes · Logistic Regression")

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">Proyek Akhir &nbsp;·&nbsp; Analisis Data Tak Terstruktur &nbsp;·&nbsp; 2026</div>
  <div class="hero-title">Analisis Sentimen Masyarakat<br>terhadap Kebijakan PPKM</div>
  <div class="hero-divider"></div>
  <div class="hero-sub">
    Platform <b>X (Twitter)</b> &nbsp;·&nbsp;
    Metode <b>Naïve Bayes</b> &amp; <b>Logistic Regression</b> &nbsp;·&nbsp;
    Bahasa Indonesia &nbsp;·&nbsp; Dataset Kaggle CC0
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CEK DATA
# ─────────────────────────────────────────────
ready = HAS_LOCAL or (uploaded_raw is not None and uploaded_labeled is not None)

if not ready:
    st.markdown("""
    <div class="status-warn" style="margin-bottom:1.5rem">
      👈 Silakan upload dataset di sidebar, atau letakkan file CSV di folder <code>data/</code> pada repo untuk memuat otomatis.
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="landing-card"><h4>📌 Tujuan</h4><p>Mengklasifikasikan sentimen publik Twitter (positif, netral, negatif) terhadap kebijakan PPKM 2021–2022.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="landing-card"><h4>⚙ Metode</h4><p>Text preprocessing Sastrawi + TF-IDF + Naïve Bayes + Logistic Regression.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="landing-card"><h4>📊 Output</h4><p>EDA, Word Cloud, Evaluasi Model, Perbandingan Algoritma, Demo Prediksi Real-Time.</p></div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# LOAD & PROSES
# ─────────────────────────────────────────────
stemmer, stopword_list = init_nlp()

try:
    if data_source == 'local':
        df_raw, df_labeled = load_dataframe(LOCAL_RAW, LOCAL_LABELED)
    else:
        df_raw, df_labeled = load_dataframe(uploaded_raw.read(), uploaded_labeled.read())
except Exception as e:
    st.error(f"❌ Gagal membaca dataset: {e}")
    st.stop()

cache_key = f"df_{max_per_class}_{len(df_labeled)}"
if st.session_state.get('cache_key') != cache_key:
    with st.spinner("Memproses data..."):
        df = build_df(df_labeled, stemmer, stopword_list, max_per_class)
    st.session_state['df'] = df
    st.session_state['cache_key'] = cache_key
    st.session_state.pop('results', None)

df = st.session_state['df']

if 'results' not in st.session_state:
    with st.spinner("Melatih model klasifikasi..."):
        st.session_state['results'] = train_models(df)

R = st.session_state['results']
sentimen_list = ['positif','negatif','netral']

# ─────────────────────────────────────────────
# NAVIGASI KUSTOM — tombol tegas & formal
# ─────────────────────────────────────────────
PAGES = [
    ("📋", "Data Overview"),
    ("📊", "EDA & Visualisasi"),
    ("🤖", "Model & Evaluasi"),
    ("📈", "Perbandingan Model"),
    ("🔮", "Demo Prediksi"),
]
PAGE_LABELS = [f"{icon}  {label}" for icon, label in PAGES]

if 'page' not in st.session_state:
    st.session_state['page'] = 0

# Render tombol navigasi
cols = st.columns(len(PAGES))
for i, (col, label) in enumerate(zip(cols, PAGE_LABELS)):
    with col:
        is_active = st.session_state['page'] == i
        btn_type = "primary" if is_active else "secondary"
        if st.button(label, key=f"nav_{i}", use_container_width=True, type=btn_type):
            st.session_state['page'] = i
            st.rerun()

page = st.session_state['page']

# Breadcrumb
icon, label = PAGES[page]
st.markdown(f"""
<div class="breadcrumb">
  {icon} &nbsp; <strong>{label}</strong>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE 0 — DATA OVERVIEW
# ═══════════════════════════════════════════════
if page == 0:
    dist = df['sentimen'].value_counts()

    st.markdown('<div class="section-header">Ringkasan Dataset</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    def mcard(col, label, val, sub="", accent_color=None):
        top_color = accent_color or "#3B9ED4"
        col.markdown(
            f'<div class="metric-card" style="border-top-color:{top_color};">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{top_color};">{val}</div>'
            f'<div class="metric-sub" style="color:{top_color};">{sub}</div></div>',
            unsafe_allow_html=True
        )

    mcard(c1, "Total Tweet",  f"{len(df):,}",                     "dataset terproses",           ACCENT)
    mcard(c2, "😊 Positif",   f"{dist.get('positif',0):,}",       f"{dist.get('positif',0)/len(df)*100:.1f}% dari total", '#1A8A4F')
    mcard(c3, "😡 Negatif",   f"{dist.get('negatif',0):,}",       f"{dist.get('negatif',0)/len(df)*100:.1f}% dari total", '#C0392B')
    mcard(c4, "😐 Netral",    f"{dist.get('netral',0):,}",        f"{dist.get('netral',0)/len(df)*100:.1f}% dari total",  '#2563AB')

    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns([3, 2])
    with ca:
        st.markdown('<div class="section-header">Contoh Data Berlabel</div>', unsafe_allow_html=True)
        st.dataframe(df[['Date','Tweet','sentimen']].head(10), use_container_width=True, hide_index=True)
    with cb:
        st.markdown('<div class="section-header">Statistik Preprocessing</div>', unsafe_allow_html=True)
        stats = df[['panjang_sebelum','panjang_sesudah']].describe().round(2)
        stats.index = ['Count','Mean','Std','Min','25%','50%','75%','Max']
        st.dataframe(stats, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Sebelum vs Sesudah Preprocessing</div>', unsafe_allow_html=True)
    s = df[['Tweet','tweet_bersih','sentimen']].sample(5, random_state=42).copy()
    s.columns = ['Tweet Asli', 'Setelah Preprocessing', 'Sentimen']
    st.dataframe(s, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════
# PAGE 1 — EDA
# ═══════════════════════════════════════════════
elif page == 1:
    st.markdown('<div class="section-header">Distribusi Sentimen</div>', unsafe_allow_html=True)
    dv = df['sentimen'].value_counts()
    labels = list(dv.index); counts = list(dv.values)
    colors = [PALETTE[l] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(BG2)

    wedges, _, autotexts = axes[0].pie(
        counts, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140, explode=[0.03]*len(labels),
        textprops={'fontsize': 11, 'color': TEXT}
    )
    for at in autotexts:
        at.set_fontweight('bold'); at.set_color('#FFFFFF'); at.set_fontsize(10)
    axes[0].set_facecolor(BG2)
    axes[0].set_title('Proporsi Sentimen', color=PRIMARY, fontsize=11, pad=12, fontweight='600')

    bars = axes[1].bar(labels, counts, color=colors, width=0.5, edgecolor=BG2, linewidth=1.5)
    for bar, cnt in zip(bars, counts):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+12,
                     f'{cnt:,}', ha='center', fontsize=11, fontweight='bold', color=TEXT)
    axes[1].set_ylim(0, max(counts)*1.22)
    style_ax(axes[1])
    axes[1].set_title('Jumlah Tweet per Sentimen', color=PRIMARY, fontsize=11, fontweight='600')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Word Cloud per Sentimen</div>', unsafe_allow_html=True)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.patch.set_facecolor(BG2)
    cmap_list = ['YlGn', 'OrRd', 'Blues']
    for ax, sent, cmap, emoji in zip(axes2, sentimen_list, cmap_list, ['😊','😡','😐']):
        txt = ' '.join(df[df['sentimen']==sent]['tweet_bersih'])
        if not txt.strip(): continue
        wc = WordCloud(width=600, height=350, background_color=BG2,
                       colormap=cmap, max_words=80, random_state=42).generate(txt)
        ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
        ax.set_facecolor(BG2)
        ax.set_title(f'{emoji}  {sent.upper()}', fontsize=13, fontweight='bold', color=PRIMARY, pad=10)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Top 15 Kata per Sentimen</div>', unsafe_allow_html=True)
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
    fig3.patch.set_facecolor(BG2)
    for ax, sent, emoji in zip(axes3, sentimen_list, ['😊','😡','😐']):
        txt = ' '.join(df[df['sentimen']==sent]['tweet_bersih'])
        cnts = Counter(txt.split()).most_common(15)
        if not cnts: continue
        words, freqs = zip(*cnts)
        ax.barh(words[::-1], freqs[::-1], color=PALETTE[sent], alpha=0.85, edgecolor=BG2, linewidth=0.8)
        style_ax(ax)
        ax.set_title(f'{emoji}  {sent.upper()}', color=PRIMARY, fontweight='600', fontsize=11)
        ax.set_xlabel('Frekuensi')
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Distribusi Panjang Tweet</div>', unsafe_allow_html=True)
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    fig4.patch.set_facecolor(BG2)
    for sent in sentimen_list:
        sub = df[df['sentimen']==sent]
        axes4[0].hist(sub['panjang_sebelum'], bins=30, alpha=0.55, color=PALETTE[sent], label=sent.capitalize(), edgecolor=BG2)
        axes4[1].hist(sub['panjang_sesudah'], bins=30, alpha=0.55, color=PALETTE[sent], label=sent.capitalize(), edgecolor=BG2)
    for ax, ttl in zip(axes4, ['Sebelum Preprocessing', 'Sesudah Preprocessing']):
        style_ax(ax); ax.set_title(ttl, color=PRIMARY, fontsize=11, fontweight='600')
        ax.set_xlabel('Jumlah Kata'); ax.set_ylabel('Frekuensi')
        ax.legend(fontsize=9, facecolor=BG2, edgecolor='#C8D8EC')
    plt.tight_layout(); st.pyplot(fig4); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Trend Volume Tweet per Bulan</div>', unsafe_allow_html=True)
    df['tanggal'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
    dft = df.dropna(subset=['tanggal']).copy()
    dft['bulan'] = dft['tanggal'].dt.to_period('M')
    trend = dft.groupby(['bulan','sentimen']).size().unstack(fill_value=0)
    if not trend.empty:
        fig5, ax5 = plt.subplots(figsize=(13, 5))
        fig5.patch.set_facecolor(BG2)
        trend.plot(kind='bar', ax=ax5,
                   color={s: PALETTE[s] for s in sentimen_list if s in trend.columns},
                   width=0.75, edgecolor=BG2, linewidth=0.8)
        style_ax(ax5)
        ax5.set_title('Volume Tweet per Sentimen per Bulan', color=PRIMARY, fontsize=11, fontweight='600')
        ax5.set_xticklabels([str(p) for p in trend.index], rotation=45, ha='right', color=TEXT, fontsize=8)
        ax5.legend(title='Sentimen', fontsize=9, facecolor=BG2, edgecolor='#C8D8EC', title_fontsize=9)
        plt.tight_layout(); st.pyplot(fig5); plt.close()
    else:
        st.info("Data tanggal tidak tersedia untuk chart trend.")

# ═══════════════════════════════════════════════
# PAGE 2 — MODEL & EVALUASI
# ═══════════════════════════════════════════════
elif page == 2:
    st.markdown('<div class="section-header">Performa Model</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    def mcard2(col, label, val, sub="", color=ACCENT):
        col.markdown(
            f'<div class="metric-card" style="border-top-color:{color};">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{color};">{val}</div>'
            f'<div class="metric-sub" style="color:{color};">{sub}</div></div>',
            unsafe_allow_html=True
        )

    mcard2(c1, "Naïve Bayes — Akurasi",        f"{R['acc_nb']*100:.2f}%",
           "Multinomial NB · α=0.5", '#2563AB')
    mcard2(c2, "Logistic Regression — Akurasi", f"{R['acc_lr']*100:.2f}%",
           f"{'▲' if R['acc_lr']>=R['acc_nb'] else '▼'} {abs(R['acc_lr']-R['acc_nb'])*100:.2f}% vs Naïve Bayes",
           PRIMARY)

    st.markdown("<br>", unsafe_allow_html=True)
    cn, cl = st.columns(2)
    for col, key, title, cmap in [(cn,'nb','Naïve Bayes','Blues'),(cl,'lr','Logistic Regression','Greens')]:
        with col:
            st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(R[f'report_{key}']).T.round(4), use_container_width=True)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
            fig_cm.patch.set_facecolor(BG2); ax_cm.set_facecolor(BG2)
            ConfusionMatrixDisplay(R[f'cm_{key}'], display_labels=R['le'].classes_).plot(
                ax=ax_cm, cmap=cmap, colorbar=False)
            ax_cm.set_title(f'Confusion Matrix — {title}\nAkurasi: {R[f"acc_{key}"]*100:.2f}%',
                            fontsize=10, fontweight='bold', color=PRIMARY)
            ax_cm.tick_params(colors=TEXT)
            ax_cm.xaxis.label.set_color(TEXT); ax_cm.yaxis.label.set_color(TEXT)
            for sp in ax_cm.spines.values(): sp.set_color('#C8D8EC')
            plt.tight_layout(); st.pyplot(fig_cm); plt.close()

# ═══════════════════════════════════════════════
# PAGE 3 — PERBANDINGAN MODEL
# ═══════════════════════════════════════════════
elif page == 3:
    st.markdown('<div class="section-header">Tabel Perbandingan Performa</div>', unsafe_allow_html=True)
    st.dataframe(
        R['df_compare'].style
          .highlight_max(axis=0, color='rgba(59,158,212,0.18)')
          .format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Visualisasi Perbandingan</div>', unsafe_allow_html=True)
    m_names = list(R['df_compare'].columns)
    nb_v = R['df_compare'].loc['Naïve Bayes'].values
    lr_v = R['df_compare'].loc['Logistic Regression'].values
    x = np.arange(len(m_names)); w = 0.35

    fig_c, ax_c = plt.subplots(figsize=(11, 5))
    fig_c.patch.set_facecolor(BG2)
    b1 = ax_c.bar(x-w/2, nb_v, w, label='Naïve Bayes',        color='#2563AB', alpha=0.85, edgecolor=BG2)
    b2 = ax_c.bar(x+w/2, lr_v, w, label='Logistic Regression', color='#1A8A4F', alpha=0.85, edgecolor=BG2)
    for bar in list(b1)+list(b2):
        ax_c.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                  f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, color=TEXT)
    ax_c.set_xticks(x); ax_c.set_xticklabels(m_names, fontsize=11, color=TEXT)
    ax_c.set_ylim(0, 1.12)
    style_ax(ax_c)
    ax_c.set_title('Naïve Bayes vs Logistic Regression', color=PRIMARY, fontsize=12, fontweight='600')
    ax_c.legend(fontsize=10, facecolor=BG2, edgecolor='#C8D8EC')
    ax_c.axhline(y=0.8, color=ACCENT, linestyle='--', alpha=0.5, linewidth=1.2)
    ax_c.text(3.55, 0.81, 'threshold 0.8', color=ACCENT, fontsize=8, alpha=0.8)
    plt.tight_layout(); st.pyplot(fig_c); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Top Fitur — Logistic Regression</div>', unsafe_allow_html=True)
    feat_names = np.array(R['tfidf'].get_feature_names_out())
    fig_f, axes_f = plt.subplots(1, 3, figsize=(18, 5))
    fig_f.patch.set_facecolor(BG2)
    for i, sent in enumerate(R['le'].classes_):
        coef = R['lr'].coef_[i]
        top_idx = np.argsort(coef)[-12:][::-1]
        axes_f[i].barh(feat_names[top_idx][::-1], coef[top_idx][::-1],
                       color=PALETTE.get(sent, ACCENT), alpha=0.85, edgecolor=BG2)
        style_ax(axes_f[i])
        axes_f[i].set_title(sent.capitalize(), color=PRIMARY, fontsize=11, fontweight='600')
        axes_f[i].set_xlabel('Koefisien')
    plt.tight_layout(); st.pyplot(fig_f); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Ringkasan Hasil</div>', unsafe_allow_html=True)
    winner = 'Logistic Regression' if R['acc_lr'] >= R['acc_nb'] else 'Naïve Bayes'
    w_acc  = max(R['acc_lr'], R['acc_nb'])
    n_pos  = (df['sentimen']=='positif').sum()
    n_neg  = (df['sentimen']=='negatif').sum()
    n_net  = (df['sentimen']=='netral').sum()

    ca, cb = st.columns([2, 1])
    with ca:
        st.markdown(f"""
| Keterangan | Nilai |
|---|---|
| Total Tweet Dianalisis | **{len(df):,}** |
| 😊 Positif | **{n_pos:,}** ({n_pos/len(df)*100:.1f}%) |
| 😡 Negatif | **{n_neg:,}** ({n_neg/len(df)*100:.1f}%) |
| 😐 Netral  | **{n_net:,}** ({n_net/len(df)*100:.1f}%) |
| Akurasi Naïve Bayes | **{R['acc_nb']*100:.2f}%** |
| Akurasi Logistic Regression | **{R['acc_lr']*100:.2f}%** |
        """)
    with cb:
        st.markdown(f"""
<div class="metric-card" style="border-top-color:{PRIMARY};margin-top:0.5rem;text-align:center;">
  <div class="metric-label">🏆 Model Terbaik</div>
  <div style="margin:0.7rem 0"><span class="winner-badge">{winner}</span></div>
  <div class="metric-value" style="color:{PRIMARY};font-size:1.7rem;">{w_acc*100:.2f}%</div>
  <div class="metric-sub" style="color:{ACCENT};">akurasi test set</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE 4 — DEMO PREDIKSI
# ═══════════════════════════════════════════════
elif page == 4:
    model_name = 'Logistic Regression' if R['acc_lr'] >= R['acc_nb'] else 'Naïve Bayes'
    best_model = R['lr'] if R['acc_lr'] >= R['acc_nb'] else R['nb']

    st.markdown('<div class="section-header">Prediksi Real-Time</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="status-ok">Model aktif: <b>{model_name}</b> — akurasi {max(R["acc_nb"],R["acc_lr"])*100:.2f}%</div><br>',
        unsafe_allow_html=True
    )

    user_input = st.text_area(
        "Masukkan teks tweet tentang PPKM:",
        placeholder="Contoh: PPKM sangat membantu mengurangi penyebaran COVID di Indonesia",
        height=110
    )

    if st.button("🔍  Analisis Sentimen", type="primary"):
        if user_input.strip():
            clean = preprocess(user_input, stemmer, stopword_list)
            vec   = R['tfidf'].transform([clean])
            pred  = best_model.predict(vec)[0]
            proba = best_model.predict_proba(vec)[0]
            label = R['le'].inverse_transform([pred])[0]
            conf  = max(proba)
            emoji = '😊' if label=='positif' else '😡' if label=='negatif' else '😐'
            css   = 'result-pos' if label=='positif' else 'result-neg' if label=='negatif' else 'result-net'
            st.markdown(f"""
<div class="{css}" style="margin:1rem 0">
  <div class="result-label">{emoji} &nbsp; {label.upper()}</div>
  <div class="result-conf">Confidence: <b>{conf:.2%}</b></div>
</div>
""", unsafe_allow_html=True)
            proba_df = pd.DataFrame({
                'Sentimen': R['le'].classes_,
                'Probabilitas': proba
            }).sort_values('Probabilitas', ascending=False)
            st.dataframe(proba_df.style.format({'Probabilitas': '{:.4f}'}),
                         use_container_width=True, hide_index=True)
            st.caption(f"Teks setelah preprocessing: `{clean}`")
        else:
            st.warning("Masukkan teks terlebih dahulu.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Contoh Prediksi Batch</div>', unsafe_allow_html=True)
    examples = [
        "PPKM sangat membantu mengurangi penyebaran COVID di Indonesia",
        "PPKM membuat usaha saya bangkrut, pemerintah tidak peduli rakyat kecil",
        "Pemerintah resmi umumkan perpanjangan PPKM level 2 hingga akhir bulan",
        "Kebijakan PPKM sudah tepat, kasus COVID turun drastis akhirnya",
        "Susah cari makan gara-gara PPKM, warung dilarang buka malam",
    ]
    hasil = []
    for tw in examples:
        cl = preprocess(tw, stemmer, stopword_list)
        v  = R['tfidf'].transform([cl])
        lb = R['le'].inverse_transform([best_model.predict(v)[0]])[0]
        cf = max(best_model.predict_proba(v)[0])
        em = '😊' if lb=='positif' else '😡' if lb=='negatif' else '😐'
        hasil.append({'Tweet': tw, 'Prediksi': f'{em} {lb.upper()}', 'Confidence': f'{cf:.2%}'})
    st.dataframe(pd.DataFrame(hasil), use_container_width=True, hide_index=True)
