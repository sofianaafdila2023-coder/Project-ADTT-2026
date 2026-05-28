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
# CUSTOM CSS — Tema Slate-Charcoal & Warm Ivory
# Tipografi: DM Serif Display + DM Sans
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ══ Design Tokens ══ */
:root {
    --bg-base:      #141920;
    --bg-surface:   #1C2430;
    --bg-raised:    #232E3C;
    --bg-hover:     #2A3647;
    --border:       #2E3D50;
    --border-light: #3A4E65;

    --text-primary: #EEF1F5;
    --text-secondary:#98A8BB;
    --text-muted:   #5E7290;

    --accent:       #4F8EF7;
    --accent-soft:  rgba(79,142,247,0.12);
    --accent-glow:  rgba(79,142,247,0.25);

    --teal:         #2FBFB0;
    --teal-soft:    rgba(47,191,176,0.12);
    --teal-glow:    rgba(47,191,176,0.2);

    --amber:        #F5A623;
    --amber-soft:   rgba(245,166,35,0.1);
    --amber-glow:   rgba(245,166,35,0.2);

    --green:        #3DD68C;
    --red:          #F0616D;
    --blue:         #4F8EF7;

    --shadow-sm:    0 1px 3px rgba(0,0,0,0.3);
    --shadow-md:    0 4px 16px rgba(0,0,0,0.35);
    --shadow-lg:    0 8px 32px rgba(0,0,0,0.4);

    --radius-sm:    6px;
    --radius-md:    10px;
    --radius-lg:    14px;

    --font-display: 'DM Serif Display', Georgia, serif;
    --font-body:    'DM Sans', -apple-system, sans-serif;
}

/* ══ Base Reset ══ */
html, body, [data-testid="stApp"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 15px;
    line-height: 1.6;
}

/* ══ Scrollbar ══ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-surface); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 99px; }

/* ══ Sidebar ══ */
[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

/* ══ Hero Banner ══ */
.hero-wrap {
    position: relative;
    overflow: hidden;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: var(--radius-lg);
    padding: 2.4rem 2.8rem 2rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-md);
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(79,142,247,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40%;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(47,191,176,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-body);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid rgba(79,142,247,0.2);
    border-radius: 99px;
    padding: 0.25rem 0.85rem;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: var(--font-display);
    font-size: 2.1rem;
    font-weight: 400;
    color: var(--text-primary);
    line-height: 1.2;
    margin-bottom: 0.75rem;
    letter-spacing: -0.01em;
}
.hero-title em {
    font-style: italic;
    color: var(--teal);
}
.hero-divider {
    width: 48px;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--teal));
    border-radius: 2px;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-size: 0.88rem;
    color: var(--text-secondary);
    font-weight: 300;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1.2rem;
    align-items: center;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-secondary);
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 0.2rem 0.7rem;
}

/* ══ Navigation ══ */
.nav-row {
    display: flex;
    gap: 6px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 5px;
    margin-bottom: 2rem;
}

/* ══ Section Headers ══ */
.sec-header {
    font-family: var(--font-display);
    font-size: 1.2rem;
    font-weight: 400;
    color: var(--text-primary);
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.sec-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: 0.4rem;
}

/* ══ Metric Cards ══ */
.kpi-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: var(--border-light); }
.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: var(--font-display);
    font-size: 2.1rem;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-sub {
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-weight: 400;
}
.kpi-accent-bar {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 0 0 var(--radius-md) var(--radius-md);
}

/* ══ Status Banners ══ */
.banner-info {
    background: var(--accent-soft);
    border: 1px solid rgba(79,142,247,0.2);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #a8c4f5;
    margin-bottom: 1rem;
}
.banner-warn {
    background: var(--amber-soft);
    border: 1px solid rgba(245,166,35,0.2);
    border-left: 3px solid var(--amber);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #f5cc7a;
    margin-bottom: 1rem;
}
.banner-success {
    background: var(--teal-soft);
    border: 1px solid rgba(47,191,176,0.2);
    border-left: 3px solid var(--teal);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #88dbd5;
    margin-bottom: 1rem;
}

/* ══ Result Cards ══ */
.result-pos {
    background: rgba(61,214,140,0.08);
    border: 1px solid rgba(61,214,140,0.2);
    border-left: 4px solid var(--green);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.5rem;
}
.result-neg {
    background: rgba(240,97,109,0.08);
    border: 1px solid rgba(240,97,109,0.2);
    border-left: 4px solid var(--red);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.5rem;
}
.result-net {
    background: var(--accent-soft);
    border: 1px solid rgba(79,142,247,0.2);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.5rem;
}
.result-label {
    font-family: var(--font-display);
    font-size: 1.6rem;
    color: var(--text-primary);
    margin-bottom: 0.2rem;
}
.result-conf {
    font-size: 0.83rem;
    color: var(--text-secondary);
}

/* ══ Winner Badge ══ */
.winner-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: linear-gradient(135deg, var(--accent), var(--teal));
    color: #fff;
    font-family: var(--font-body);
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 99px;
    box-shadow: 0 2px 12px var(--accent-glow);
}

/* ══ Sidebar Brand ══ */
.sb-brand {
    font-family: var(--font-display);
    font-size: 1.05rem;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin-bottom: 0.15rem;
}
.sb-tagline {
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
}

/* ══ Landing Cards ══ */
.landing-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    height: 100%;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.landing-card:hover {
    border-color: var(--border-light);
    box-shadow: var(--shadow-sm);
}
.landing-card-icon {
    font-size: 1.5rem;
    margin-bottom: 0.6rem;
    display: block;
}
.landing-card h4 {
    font-family: var(--font-display);
    font-size: 1rem;
    font-weight: 400;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}
.landing-card p {
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.65;
    font-weight: 300;
}

/* ══ Streamlit Overrides ══ */
/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.55rem 1.6rem !important;
    box-shadow: 0 2px 8px var(--accent-glow) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #3a7ee8 !important;
    box-shadow: 0 4px 16px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--bg-raised) !important;
    color: var(--text-secondary) !important;
    box-shadow: none !important;
    border: 1px solid var(--border) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-light) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Text inputs */
.stTextArea textarea, .stTextInput input {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}

/* Progress bar */
.stProgress > div > div { background: var(--accent) !important; }

/* Slider */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: var(--text-primary) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-family: var(--font-display) !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}

/* Alerts */
.stSuccess, .stError, .stInfo, .stWarning {
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-body) !important;
}

/* Tabs (hide default — we use custom nav) */
[data-testid="stTabs"] > div:first-child { display: none !important; }

/* Captions & small text */
.stCaption { color: var(--text-muted) !important; font-size: 0.78rem !important; }

/* Info box override */
.stInfo { background: var(--accent-soft) !important; border-color: rgba(79,142,247,0.3) !important; }

/* Divider */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KONSTANTA PLOT
# ─────────────────────────────────────────────
PALETTE = {'positif': '#3DD68C', 'negatif': '#F0616D', 'netral': '#4F8EF7'}
BG      = '#141920'
BG2     = '#1C2430'
BG3     = '#232E3C'
ACCENT  = '#4F8EF7'
TEAL    = '#2FBFB0'
MUTED   = '#98A8BB'

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
# PLOT HELPERS
# ─────────────────────────────────────────────
def style_ax(ax, title=""):
    ax.set_facecolor(BG3)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title: ax.set_title(title, color=MUTED, fontsize=10.5, pad=10, fontweight='normal')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color('#2E3D50')
    ax.grid(axis='y', color='#2E3D50', linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

def make_fig(w, h, cols=1):
    if cols == 1:
        fig, ax = plt.subplots(figsize=(w, h))
        fig.patch.set_facecolor(BG2)
        return fig, ax
    fig, axes = plt.subplots(1, cols, figsize=(w, h))
    fig.patch.set_facecolor(BG2)
    return fig, axes

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">Sentimen PPKM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tagline">ADTT · 2026</div>', unsafe_allow_html=True)
    st.divider()

    DATA_DIR      = os.path.join(os.path.dirname(__file__), 'data')
    LOCAL_RAW     = os.path.join(DATA_DIR, 'INA_TweetsPPKM_Raw.csv')
    LOCAL_LABELED = os.path.join(DATA_DIR, 'INA_TweetsPPKM_Labeled_Pure.csv')
    HAS_LOCAL     = os.path.isfile(LOCAL_RAW) and os.path.isfile(LOCAL_LABELED)

    if HAS_LOCAL:
        st.markdown('<div class="banner-success">✓ Dataset terdeteksi dari <code>data/</code></div>', unsafe_allow_html=True)
        data_source = 'local'; uploaded_raw = uploaded_labeled = None
    else:
        st.markdown("**Upload Dataset**")
        st.caption("Dua file CSV dari Kaggle PPKM Twitter")
        uploaded_raw     = st.file_uploader("File Raw CSV",     type=['csv','tsv'], key='raw')
        uploaded_labeled = st.file_uploader("File Labeled CSV", type=['csv','tsv'], key='labeled')
        data_source = 'upload'
        if not (uploaded_raw and uploaded_labeled):
            st.markdown('<div class="banner-warn">⚠ Taruh file di folder <code>data/</code> untuk otomatis</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**Ukuran Sample**")
    max_per_class = st.slider("Maks tweet per kelas", 500, 5000, 2000, 500,
                              help="Kurangi jika proses terlalu lama", label_visibility="collapsed")
    st.caption(f"Per kelas: **{max_per_class:,}** · Maks total: **{max_per_class*3:,}**")
    st.divider()
    st.caption("Analisis Data Tak Terstruktur")
    st.caption("Dataset: Kaggle PPKM Twitter · CC0")

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">📋 Proyek Akhir · Analisis Data Tak Terstruktur · 2026</div>
  <div class="hero-title">Analisis Sentimen Masyarakat<br>terhadap <em>Kebijakan PPKM</em></div>
  <div class="hero-divider"></div>
  <div class="hero-sub">
    <span class="hero-pill">🐦 Platform X (Twitter)</span>
    <span class="hero-pill">🤖 Naïve Bayes &amp; Logistic Regression</span>
    <span class="hero-pill">🇮🇩 Bahasa Indonesia</span>
    <span class="hero-pill">📦 Dataset Kaggle CC0</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CEK DATA
# ─────────────────────────────────────────────
ready = HAS_LOCAL or (uploaded_raw is not None and uploaded_labeled is not None)

if not ready:
    st.markdown('<div class="banner-warn">👈 Upload dataset di sidebar, atau letakkan CSV di folder <code>data/</code> untuk memuat otomatis.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="landing-card">
          <span class="landing-card-icon">📌</span>
          <h4>Tujuan</h4>
          <p>Mengklasifikasikan sentimen publik Twitter (positif, netral, negatif) terhadap kebijakan PPKM 2021–2022.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="landing-card">
          <span class="landing-card-icon">⚙</span>
          <h4>Metode</h4>
          <p>Text preprocessing Sastrawi + TF-IDF + Naïve Bayes + Logistic Regression dengan validasi silang.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="landing-card">
          <span class="landing-card-icon">📊</span>
          <h4>Output</h4>
          <p>EDA, Word Cloud, Evaluasi Model, Perbandingan Algoritma, dan Demo Prediksi Real-Time.</p>
        </div>""", unsafe_allow_html=True)
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
# NAVIGASI
# ─────────────────────────────────────────────
PAGES = ["📋 Data Overview", "📊 EDA & Visualisasi", "🤖 Model & Evaluasi", "📈 Perbandingan", "🔮 Demo Prediksi"]
if 'page' not in st.session_state:
    st.session_state['page'] = 0

cols = st.columns(len(PAGES))
for i, (col, label) in enumerate(zip(cols, PAGES)):
    with col:
        if st.button(label, key=f"nav_{i}", use_container_width=True,
                     type="primary" if st.session_state['page'] == i else "secondary"):
            st.session_state['page'] = i
            st.rerun()

page = st.session_state['page']

# breadcrumb
page_name = PAGES[page].split(' ', 1)[1]
st.markdown(f"""
<div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--radius-sm);
            padding:0.45rem 1rem;margin:0.6rem 0 1.8rem;font-size:0.78rem;color:var(--text-muted);
            display:flex;align-items:center;gap:0.4rem;">
  <span style="color:var(--text-muted)">Dashboard</span>
  <span style="color:var(--border-light)">›</span>
  <span style="color:var(--accent)">{page_name}</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE 0 — DATA OVERVIEW
# ═══════════════════════════════════════════════
if page == 0:
    dist = df['sentimen'].value_counts()

    st.markdown('<div class="sec-header">Ringkasan Dataset</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    def kpi(col, label, val, sub, bar_color):
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-sub">{sub}</div>
          <div class="kpi-accent-bar" style="background:{bar_color}"></div>
        </div>""", unsafe_allow_html=True)

    kpi(c1, "Total Tweet",  f"{len(df):,}",
        "dataset terproses", ACCENT)
    kpi(c2, "😊 Positif",
        f"{dist.get('positif',0):,}",
        f"{dist.get('positif',0)/len(df)*100:.1f}% dari total",
        PALETTE['positif'])
    kpi(c3, "😡 Negatif",
        f"{dist.get('negatif',0):,}",
        f"{dist.get('negatif',0)/len(df)*100:.1f}% dari total",
        PALETTE['negatif'])
    kpi(c4, "😐 Netral",
        f"{dist.get('netral',0):,}",
        f"{dist.get('netral',0)/len(df)*100:.1f}% dari total",
        PALETTE['netral'])

    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns([3, 2])
    with ca:
        st.markdown('<div class="sec-header">Contoh Data Berlabel</div>', unsafe_allow_html=True)
        st.dataframe(df[['Date','Tweet','sentimen']].head(10), use_container_width=True, hide_index=True)
    with cb:
        st.markdown('<div class="sec-header">Statistik Preprocessing</div>', unsafe_allow_html=True)
        stats = df[['panjang_sebelum','panjang_sesudah']].describe().round(2)
        stats.index = ['Count','Mean','Std','Min','25%','50%','75%','Max']
        st.dataframe(stats, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Sebelum vs Sesudah Preprocessing</div>', unsafe_allow_html=True)
    s = df[['Tweet','tweet_bersih','sentimen']].sample(5, random_state=42).copy()
    s.columns = ['Tweet Asli', 'Setelah Preprocessing', 'Sentimen']
    st.dataframe(s, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════
# PAGE 1 — EDA
# ═══════════════════════════════════════════════
elif page == 1:
    # Distribusi
    st.markdown('<div class="sec-header">Distribusi Sentimen</div>', unsafe_allow_html=True)
    dv = df['sentimen'].value_counts()
    labels = list(dv.index); counts = list(dv.values)
    colors = [PALETTE[l] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(BG2)

    wedges, _, autotexts = axes[0].pie(
        counts, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140, explode=[0.03]*len(labels),
        textprops={'fontsize': 11, 'color': 'white'},
        wedgeprops={'edgecolor': BG2, 'linewidth': 2}
    )
    for at in autotexts: at.set_fontweight('bold'); at.set_color(BG)
    axes[0].set_facecolor(BG2)
    axes[0].set_title('Proporsi Sentimen', color=MUTED, fontsize=11, pad=10)

    bars = axes[1].bar(labels, counts, color=colors, width=0.45, edgecolor=BG2, linewidth=1.5, zorder=3)
    for bar, cnt in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                     f'{cnt:,}', ha='center', fontsize=11, fontweight='600', color='white', zorder=4)
    axes[1].set_ylim(0, max(counts)*1.22)
    style_ax(axes[1], 'Jumlah Tweet per Sentimen')

    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Word Cloud
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Word Cloud per Sentimen</div>', unsafe_allow_html=True)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.patch.set_facecolor(BG2)
    cmap_list = ['Greens', 'Reds', 'Blues']
    for ax, sent, cmap, emoji in zip(axes2, sentimen_list, cmap_list, ['😊', '😡', '😐']):
        txt = ' '.join(df[df['sentimen'] == sent]['tweet_bersih'])
        if not txt.strip(): continue
        wc = WordCloud(width=600, height=350, background_color=BG3,
                       colormap=cmap, max_words=80, random_state=42).generate(txt)
        ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
        ax.set_title(f'{emoji}  {sent.upper()}', fontsize=12, fontweight='600', color='white', pad=10)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    # Top Kata
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Top 15 Kata per Sentimen</div>', unsafe_allow_html=True)
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
    fig3.patch.set_facecolor(BG2)
    for ax, sent, emoji in zip(axes3, sentimen_list, ['😊', '😡', '😐']):
        txt = ' '.join(df[df['sentimen'] == sent]['tweet_bersih'])
        cnts = Counter(txt.split()).most_common(15)
        if not cnts: continue
        words, freqs = zip(*cnts)
        bars = ax.barh(words[::-1], freqs[::-1], color=PALETTE[sent], alpha=0.85,
                       edgecolor=BG2, linewidth=0.8, zorder=3)
        style_ax(ax, f'{emoji}  {sent.upper()}')
        ax.set_xlabel('Frekuensi')
        ax.grid(axis='x', color='#2E3D50', linewidth=0.6, alpha=0.5)
        ax.grid(axis='y', visible=False)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    # Panjang Tweet
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Distribusi Panjang Tweet</div>', unsafe_allow_html=True)
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    fig4.patch.set_facecolor(BG2)
    for sent in sentimen_list:
        sub = df[df['sentimen'] == sent]
        axes4[0].hist(sub['panjang_sebelum'], bins=30, alpha=0.5,
                      color=PALETTE[sent], label=sent.capitalize(), edgecolor=BG2, linewidth=0.5)
        axes4[1].hist(sub['panjang_sesudah'], bins=30, alpha=0.5,
                      color=PALETTE[sent], label=sent.capitalize(), edgecolor=BG2, linewidth=0.5)
    for ax, ttl in zip(axes4, ['Sebelum Preprocessing', 'Sesudah Preprocessing']):
        style_ax(ax, ttl)
        ax.set_xlabel('Jumlah Kata'); ax.set_ylabel('Frekuensi')
        ax.legend(fontsize=9, labelcolor='white', facecolor=BG3, edgecolor='#2E3D50')
    plt.tight_layout(); st.pyplot(fig4); plt.close()

    # Trend
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Trend Volume Tweet per Bulan</div>', unsafe_allow_html=True)
    df['tanggal'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
    dft = df.dropna(subset=['tanggal']).copy()
    dft['bulan'] = dft['tanggal'].dt.to_period('M')
    trend = dft.groupby(['bulan','sentimen']).size().unstack(fill_value=0)
    if not trend.empty:
        fig5, ax5 = plt.subplots(figsize=(13, 5))
        fig5.patch.set_facecolor(BG2)
        trend.plot(kind='bar', ax=ax5,
                   color={s: PALETTE[s] for s in sentimen_list if s in trend.columns},
                   width=0.72, edgecolor=BG2, linewidth=0.8, zorder=3)
        style_ax(ax5, 'Volume Tweet per Sentimen per Bulan')
        ax5.set_xticklabels([str(p) for p in trend.index], rotation=45, ha='right', color=MUTED, fontsize=8)
        ax5.legend(title='Sentimen', fontsize=9, labelcolor='white', facecolor=BG3,
                   edgecolor='#2E3D50', title_fontsize=9)
        plt.tight_layout(); st.pyplot(fig5); plt.close()
    else:
        st.markdown('<div class="banner-info">ℹ Data tanggal tidak tersedia untuk analisis tren.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE 2 — MODEL & EVALUASI
# ═══════════════════════════════════════════════
elif page == 2:
    st.markdown('<div class="sec-header">Performa Model</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    def kpi2(col, label, val, sub, bar_color):
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-sub">{sub}</div>
          <div class="kpi-accent-bar" style="background:{bar_color}"></div>
        </div>""", unsafe_allow_html=True)

    kpi2(c1, "Naïve Bayes — Akurasi", f"{R['acc_nb']*100:.2f}%",
         "Multinomial NB · α = 0.5", TEAL)
    delta = R['acc_lr'] - R['acc_nb']
    arrow = '▲' if delta >= 0 else '▼'
    kpi2(c2, "Logistic Regression — Akurasi", f"{R['acc_lr']*100:.2f}%",
         f"{arrow} {abs(delta)*100:.2f}% vs Naïve Bayes", ACCENT)

    st.markdown("<br>", unsafe_allow_html=True)
    cn, cl = st.columns(2)
    for col, key, title, cmap_name in [
        (cn, 'nb', 'Naïve Bayes', 'BuGn'),
        (cl, 'lr', 'Logistic Regression', 'Blues')
    ]:
        with col:
            st.markdown(f'<div class="sec-header">{title}</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(R[f'report_{key}']).T.round(4), use_container_width=True)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
            fig_cm.patch.set_facecolor(BG2); ax_cm.set_facecolor(BG3)
            ConfusionMatrixDisplay(R[f'cm_{key}'], display_labels=R['le'].classes_).plot(
                ax=ax_cm, cmap=cmap_name, colorbar=False)
            ax_cm.set_title(f'Confusion Matrix — {title}  |  Akurasi: {R[f"acc_{key}"]*100:.2f}%',
                            fontsize=10, color=MUTED, pad=10)
            ax_cm.tick_params(colors=MUTED)
            ax_cm.xaxis.label.set_color(MUTED); ax_cm.yaxis.label.set_color(MUTED)
            for sp in ax_cm.spines.values(): sp.set_color('#2E3D50')
            plt.tight_layout(); st.pyplot(fig_cm); plt.close()

# ═══════════════════════════════════════════════
# PAGE 3 — PERBANDINGAN
# ═══════════════════════════════════════════════
elif page == 3:
    st.markdown('<div class="sec-header">Tabel Perbandingan Performa</div>', unsafe_allow_html=True)
    st.dataframe(
        R['df_compare'].style
            .highlight_max(axis=0, color='rgba(79,142,247,0.18)')
            .format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Visualisasi Perbandingan</div>', unsafe_allow_html=True)
    m_names = list(R['df_compare'].columns)
    nb_v = R['df_compare'].loc['Naïve Bayes'].values
    lr_v = R['df_compare'].loc['Logistic Regression'].values
    x = np.arange(len(m_names)); w = 0.35

    fig_c, ax_c = plt.subplots(figsize=(11, 5.5))
    fig_c.patch.set_facecolor(BG2)
    b1 = ax_c.bar(x - w/2, nb_v, w, label='Naïve Bayes',
                  color=TEAL, alpha=0.9, edgecolor=BG2, linewidth=1, zorder=3)
    b2 = ax_c.bar(x + w/2, lr_v, w, label='Logistic Regression',
                  color=ACCENT, alpha=0.9, edgecolor=BG2, linewidth=1, zorder=3)
    for bar in list(b1) + list(b2):
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                  f'{bar.get_height():.3f}', ha='center', va='bottom',
                  fontsize=9.5, color='white', fontweight='500')
    ax_c.set_xticks(x); ax_c.set_xticklabels(m_names, fontsize=11, color=MUTED)
    ax_c.set_ylim(0, 1.15)
    style_ax(ax_c, 'Naïve Bayes vs Logistic Regression')
    ax_c.legend(fontsize=10, labelcolor='white', facecolor=BG3, edgecolor='#2E3D50')
    ax_c.axhline(y=0.8, color=ACCENT, linestyle='--', alpha=0.3, linewidth=1.2)
    ax_c.text(3.6, 0.81, 'threshold 0.8', color=ACCENT, fontsize=8.5, alpha=0.6)
    plt.tight_layout(); st.pyplot(fig_c); plt.close()

    # Top Fitur LR
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Top Fitur — Logistic Regression</div>', unsafe_allow_html=True)
    feat_names = np.array(R['tfidf'].get_feature_names_out())
    fig_f, axes_f = plt.subplots(1, 3, figsize=(18, 5.5))
    fig_f.patch.set_facecolor(BG2)
    for i, sent in enumerate(R['le'].classes_):
        coef = R['lr'].coef_[i]
        top_idx = np.argsort(coef)[-12:][::-1]
        axes_f[i].barh(feat_names[top_idx][::-1], coef[top_idx][::-1],
                       color=PALETTE.get(sent, '#888'), alpha=0.85, edgecolor=BG2, linewidth=0.8, zorder=3)
        style_ax(axes_f[i], sent.capitalize())
        axes_f[i].set_xlabel('Koefisien')
        axes_f[i].grid(axis='x', color='#2E3D50', linewidth=0.6, alpha=0.5)
        axes_f[i].grid(axis='y', visible=False)
    plt.tight_layout(); st.pyplot(fig_f); plt.close()

    # Ringkasan
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Ringkasan Hasil</div>', unsafe_allow_html=True)
    winner = 'Logistic Regression' if R['acc_lr'] >= R['acc_nb'] else 'Naïve Bayes'
    w_acc  = max(R['acc_lr'], R['acc_nb'])
    n_pos  = (df['sentimen'] == 'positif').sum()
    n_neg  = (df['sentimen'] == 'negatif').sum()
    n_net  = (df['sentimen'] == 'netral').sum()

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
<div class="kpi-card" style="text-align:center;padding:1.8rem 1.4rem">
  <div class="kpi-label">🏆 Model Terbaik</div>
  <div style="margin:0.7rem 0">
    <span class="winner-badge">✦ {winner}</span>
  </div>
  <div class="kpi-value">{w_acc*100:.2f}%</div>
  <div class="kpi-sub">akurasi pada test set</div>
  <div class="kpi-accent-bar" style="background:linear-gradient(90deg,{ACCENT},{TEAL})"></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE 4 — DEMO PREDIKSI
# ═══════════════════════════════════════════════
elif page == 4:
    model_name = 'Logistic Regression' if R['acc_lr'] >= R['acc_nb'] else 'Naïve Bayes'
    best_model = R['lr'] if R['acc_lr'] >= R['acc_nb'] else R['nb']

    st.markdown('<div class="sec-header">Prediksi Real-Time</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="banner-success">Model aktif: <b>{model_name}</b> — akurasi {max(R["acc_nb"],R["acc_lr"])*100:.2f}%</div>',
                unsafe_allow_html=True)

    user_input = st.text_area(
        "Masukkan teks tweet tentang PPKM:",
        placeholder="Contoh: PPKM sangat membantu mengurangi penyebaran COVID di Indonesia",
        height=110
    )

    if st.button("🔍 Analisis Sentimen", type="primary"):
        if user_input.strip():
            clean = preprocess(user_input, stemmer, stopword_list)
            vec   = R['tfidf'].transform([clean])
            pred  = best_model.predict(vec)[0]
            proba = best_model.predict_proba(vec)[0]
            label = R['le'].inverse_transform([pred])[0]
            conf  = max(proba)
            emoji = '😊' if label == 'positif' else '😡' if label == 'negatif' else '😐'
            css   = 'result-pos' if label == 'positif' else 'result-neg' if label == 'negatif' else 'result-net'
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
            st.markdown('<div class="banner-warn">⚠ Masukkan teks terlebih dahulu.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Contoh Prediksi Batch</div>', unsafe_allow_html=True)
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
        em = '😊' if lb == 'positif' else '😡' if lb == 'negatif' else '😐'
        hasil.append({'Tweet': tw, 'Prediksi': f'{em} {lb.upper()}', 'Confidence': f'{cf:.2%}'})
    st.dataframe(pd.DataFrame(hasil), use_container_width=True, hide_index=True)
