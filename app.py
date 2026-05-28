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
# CSS TEMA — Deep Ocean Blue, terang & formal
# Font: Lora (display) + IBM Plex Sans (body)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;0,700;1,500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0F1923;
    --surface:   #162232;
    --raised:    #1D2E42;
    --hover:     #243550;
    --border:    #263D58;
    --border-hi: #3A5A7A;

    --txt1:   #E8EEF5;
    --txt2:   #8FAFC8;
    --txt3:   #4D7090;

    --blue:   #3B9EFF;
    --blue2:  #5BB8FF;
    --cyan:   #22D3C8;
    --green:  #34D399;
    --red:    #F87171;
    --amber:  #FBBF24;

    --green-sent: #34D399;
    --red-sent:   #F87171;
    --blue-sent:  #60A5FA;

    --r1: 8px;
    --r2: 12px;
    --r3: 16px;

    --ff-head: 'Lora', Georgia, serif;
    --ff-body: 'IBM Plex Sans', system-ui, sans-serif;
}

html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--txt1) !important;
    font-family: var(--ff-body) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--surface)}
::-webkit-scrollbar-thumb{background:var(--border-hi);border-radius:99px}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
    background:var(--surface) !important;
    border-right:1px solid var(--border) !important;
}
[data-testid="stSidebar"] *{color:var(--txt1)!important;font-family:var(--ff-body)!important}

/* ── Main content padding ── */
.block-container{padding-top:1.8rem!important}

/* ── Hero ── */
.hero{
    background:linear-gradient(135deg,#162232 0%,#1a3050 55%,#12263d 100%);
    border:1px solid var(--border);
    border-top:3px solid var(--blue);
    border-radius:var(--r3);
    padding:2.2rem 2.8rem 2rem;
    margin-bottom:1.8rem;
    position:relative;
    overflow:hidden;
}
.hero::after{
    content:'';position:absolute;top:-80px;right:-80px;
    width:300px;height:300px;border-radius:50%;
    background:radial-gradient(circle,rgba(59,158,255,.09) 0%,transparent 70%);
    pointer-events:none;
}
.hero-chip{
    display:inline-block;
    background:rgba(59,158,255,.15);
    border:1px solid rgba(59,158,255,.3);
    color:var(--blue2);
    font-size:.7rem;font-weight:600;
    letter-spacing:.16em;text-transform:uppercase;
    padding:.28rem .85rem;border-radius:99px;
    margin-bottom:.9rem;
}
.hero-title{
    font-family:var(--ff-head);
    font-size:2rem;font-weight:600;
    color:var(--txt1);line-height:1.25;
    margin-bottom:.65rem;
}
.hero-title span{color:var(--cyan);font-style:italic}
.hero-bar{
    width:44px;height:3px;
    background:linear-gradient(90deg,var(--blue),var(--cyan));
    border-radius:99px;margin-bottom:.75rem;
}
.hero-pills{display:flex;flex-wrap:wrap;gap:.45rem}
.hero-pill{
    background:var(--raised);border:1px solid var(--border);
    color:var(--txt2);font-size:.75rem;font-weight:500;
    padding:.22rem .75rem;border-radius:99px;
}

/* ── NAV BAR ── */
.nav-bar{
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:var(--r2);
    padding:5px;
    display:flex;gap:4px;
    margin-bottom:1.6rem;
}
/* Semua tombol nav — default (nonaktif) */
.nav-bar .stButton > button{
    background:transparent !important;
    color:var(--txt2) !important;
    border:none !important;
    border-radius:var(--r1) !important;
    font-family:var(--ff-body) !important;
    font-size:.82rem !important;
    font-weight:500 !important;
    padding:.55rem .6rem !important;
    width:100% !important;
    box-shadow:none !important;
    transition:background .15s,color .15s !important;
}
.nav-bar .stButton > button:hover{
    background:var(--hover) !important;
    color:var(--txt1) !important;
    transform:none !important;
    box-shadow:none !important;
}
/* Tombol AKTIF — pakai class .nav-active yang kita inject */
.nav-active .stButton > button{
    background:var(--blue) !important;
    color:#fff !important;
    box-shadow:0 2px 12px rgba(59,158,255,.35) !important;
}
.nav-active .stButton > button:hover{
    background:var(--blue2) !important;
}

/* ── KPI Card ── */
.kpi{
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:var(--r2);
    padding:1.2rem 1.4rem 1.1rem;
    position:relative;overflow:hidden;
}
.kpi-label{
    font-size:.68rem;font-weight:600;
    letter-spacing:.14em;text-transform:uppercase;
    color:var(--txt3);margin-bottom:.5rem;
}
.kpi-val{
    font-family:var(--ff-head);
    font-size:2rem;color:var(--txt1);
    line-height:1;margin-bottom:.3rem;
}
.kpi-sub{font-size:.78rem;color:var(--txt2)}
.kpi-line{
    position:absolute;bottom:0;left:0;right:0;
    height:3px;border-radius:0 0 var(--r2) var(--r2);
}

/* ── Section header ── */
.sh{
    display:flex;align-items:center;gap:.65rem;
    font-family:var(--ff-head);
    font-size:1.15rem;font-weight:500;
    color:var(--txt1);margin-bottom:1.1rem;
}
.sh::after{content:'';flex:1;height:1px;background:var(--border)}

/* ── Banners ── */
.b-info{background:rgba(59,158,255,.1);border:1px solid rgba(59,158,255,.2);border-left:3px solid var(--blue);border-radius:var(--r1);padding:.7rem 1rem;font-size:.84rem;color:#93c5fd;margin-bottom:1rem}
.b-warn{background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.2);border-left:3px solid var(--amber);border-radius:var(--r1);padding:.7rem 1rem;font-size:.84rem;color:#fcd34d;margin-bottom:1rem}
.b-ok{background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.2);border-left:3px solid var(--green);border-radius:var(--r1);padding:.7rem 1rem;font-size:.84rem;color:#6ee7b7;margin-bottom:1rem}

/* ── Result cards ── */
.r-pos{background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.2);border-left:4px solid var(--green);border-radius:var(--r2);padding:1.2rem 1.5rem;margin:1rem 0}
.r-neg{background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.2);border-left:4px solid var(--red);border-radius:var(--r2);padding:1.2rem 1.5rem;margin:1rem 0}
.r-net{background:rgba(59,158,255,.08);border:1px solid rgba(59,158,255,.2);border-left:4px solid var(--blue);border-radius:var(--r2);padding:1.2rem 1.5rem;margin:1rem 0}
.r-label{font-family:var(--ff-head);font-size:1.55rem;color:var(--txt1)}
.r-conf{font-size:.83rem;color:var(--txt2);margin-top:.25rem}

/* ── Winner badge ── */
.w-badge{
    display:inline-block;
    background:linear-gradient(135deg,var(--blue),var(--cyan));
    color:#fff;font-weight:600;font-size:.78rem;
    letter-spacing:.06em;text-transform:uppercase;
    padding:.3rem .9rem;border-radius:99px;
    box-shadow:0 2px 14px rgba(59,158,255,.3);
}

/* ── Landing cards ── */
.lc{background:var(--surface);border:1px solid var(--border);border-radius:var(--r2);padding:1.5rem;height:100%}
.lc h4{font-family:var(--ff-head);font-size:.98rem;font-weight:500;color:var(--txt1);margin-bottom:.5rem}
.lc p{font-size:.85rem;color:var(--txt2);line-height:1.65;font-weight:300}

/* ── Sidebar brand ── */
.sb-brand{font-family:var(--ff-head);font-size:1.05rem;color:var(--txt1);margin-bottom:.15rem}
.sb-tag{font-size:.68rem;color:var(--txt3);letter-spacing:.12em;text-transform:uppercase;font-weight:500}

/* ── Streamlit widget overrides ── */
/* Tombol biasa (non-nav) */
.stButton > button{
    background:var(--blue) !important;
    color:#fff !important;
    font-family:var(--ff-body) !important;
    font-weight:500 !important;
    border:none !important;
    border-radius:var(--r1) !important;
    padding:.5rem 1.6rem !important;
    box-shadow:0 2px 8px rgba(59,158,255,.28) !important;
    transition:all .2s !important;
}
.stButton > button:hover{
    background:var(--blue2) !important;
    box-shadow:0 4px 16px rgba(59,158,255,.4) !important;
    transform:translateY(-1px) !important;
}

/* Text inputs */
.stTextArea textarea,.stTextInput input{
    background:var(--raised) !important;
    border:1px solid var(--border) !important;
    border-radius:var(--r1) !important;
    color:var(--txt1) !important;
    font-family:var(--ff-body) !important;
}
.stTextArea textarea:focus,.stTextInput input:focus{
    border-color:var(--blue) !important;
    box-shadow:0 0 0 2px rgba(59,158,255,.2) !important;
}

[data-testid="stDataFrame"]{border:1px solid var(--border) !important;border-radius:var(--r2) !important}
.stProgress > div > div{background:var(--blue) !important}
[data-testid="stMetric"]{background:var(--surface) !important;border:1px solid var(--border) !important;border-radius:var(--r2) !important;padding:.9rem 1.1rem !important}
[data-testid="stMetricLabel"]{color:var(--txt3) !important;font-size:.75rem !important}
[data-testid="stMetricValue"]{color:var(--txt1) !important;font-family:var(--ff-head) !important}
[data-testid="stExpander"]{background:var(--surface) !important;border:1px solid var(--border) !important;border-radius:var(--r2) !important}
hr{border-color:var(--border) !important}
.stCaption{color:var(--txt3) !important;font-size:.77rem !important}
[data-testid="stTabs"] > div:first-child{display:none !important}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KONSTANTA PLOT
# ─────────────────────────────────────────────
PALETTE = {'positif': '#34D399', 'negatif': '#F87171', 'netral': '#60A5FA'}
BG      = '#0F1923'
BG2     = '#162232'
BG3     = '#1D2E42'
BLUE    = '#3B9EFF'
CYAN    = '#22D3C8'
MUTED   = '#8FAFC8'

# ─────────────────────────────────────────────
# NLP INIT
# ─────────────────────────────────────────────
@st.cache_resource
def init_nlp():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    sw_factory = StopWordRemoverFactory()
    sw = set(sw_factory.get_stop_words())
    sw.update({'yang','ini','itu','dan','dengan','untuk','dari','ke','di',
               'pada','aja','juga','banget','nih','sih','deh','lah','kan',
               'ya','yg','udah','sudah','bisa','ada','lebih','masih','jadi',
               'https','http','rt','co','amp','ppkm','via','bit','ly',
               'tco','pic','twitter','com','www'})
    return stemmer, sw

slang = {
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

def preprocess(text, stemmer, sw):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+','',text)
    text = re.sub(r'@\w+|#\w+','',text)
    text = re.sub(r'[^a-z\s]','',text)
    text = re.sub(r'\b\w{1,2}\b','',text)
    text = re.sub(r'\s+',' ',text).strip()
    words = [slang.get(w,w) for w in text.split()]
    tokens = word_tokenize(' '.join(words))
    tokens = [t for t in tokens if t not in sw and len(t)>2]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def read_csv_auto(source):
    raw = open(source,'rb').read() if isinstance(source,(str,os.PathLike)) else source
    for sep in ['\t',',',';']:
        try:
            df = pd.read_csv(io.BytesIO(raw),sep=sep,on_bad_lines='skip')
            if df.shape[1]>1: return df
        except: continue
    return pd.read_csv(io.BytesIO(raw),on_bad_lines='skip')

def load_dataframe(raw_src, lbl_src):
    df_raw = read_csv_auto(raw_src)
    df_lbl = read_csv_auto(lbl_src)
    cl = {c.lower():c for c in df_lbl.columns}
    sc = next((cl[c] for c in ['sentiment','sentimen','label','polarity','class'] if c in cl),None)
    tc = next((cl[c] for c in ['tweet','text','teks','content','full_text'] if c in cl),None)
    dc = next((cl[c] for c in ['date','tanggal','created_at','timestamp','time'] if c in cl),None)
    if not sc: raise ValueError(f"Kolom label tidak ditemukan. Kolom: {list(df_lbl.columns)}")
    if not tc: raise ValueError(f"Kolom tweet tidak ditemukan. Kolom: {list(df_lbl.columns)}")
    rm = {tc:'Tweet',sc:'sentiment_raw'}
    if dc: rm[dc]='Date'
    df_lbl = df_lbl.rename(columns=rm)
    if 'Date' not in df_lbl.columns: df_lbl['Date']=pd.NaT
    lmap = {0:'negatif',1:'positif',2:'netral','0':'negatif','1':'positif','2':'netral',
            'positif':'positif','negatif':'negatif','netral':'netral',
            'positive':'positif','negative':'negatif','neutral':'netral'}
    def ml(v):
        if pd.isna(v): return None
        try: return lmap.get(int(float(str(v).strip())))
        except: return lmap.get(str(v).strip().lower())
    df_lbl['sentimen'] = df_lbl['sentiment_raw'].apply(ml)
    df_lbl = df_lbl.dropna(subset=['sentimen'])
    return df_raw, df_lbl

def build_df(df_lbl, stemmer, sw, max_per):
    df = df_lbl[['Date','Tweet','sentimen']].copy()
    df = df.dropna(subset=['Tweet','sentimen'])
    df = df[df['Tweet'].str.strip()!=''].reset_index(drop=True)
    parts = []
    for lbl in df['sentimen'].unique():
        sub = df[df['sentimen']==lbl]
        parts.append(sub.sample(min(len(sub),max_per),random_state=42))
    df = pd.concat(parts).reset_index(drop=True)
    prog = st.progress(0,text="Memproses tweet...")
    total=len(df); res=[]
    for i,row in df.iterrows():
        res.append(preprocess(row['Tweet'],stemmer,sw))
        if i%200==0: prog.progress(min(int(i/total*100),99),text=f"Preprocessing {i}/{total}...")
    prog.progress(100,text="Selesai!"); prog.empty()
    df['tweet_bersih']=res
    df = df[df['tweet_bersih'].str.strip().str.len()>3].reset_index(drop=True)
    df['panjang_sebelum']=df['Tweet'].apply(lambda x:len(str(x).split()))
    df['panjang_sesudah']=df['tweet_bersih'].apply(lambda x:len(str(x).split()))
    return df

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_models(df):
    le = LabelEncoder()
    dfc = df.copy()
    dfc['label'] = le.fit_transform(dfc['sentimen'])
    X,y = dfc['tweet_bersih'],dfc['label']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    tfidf = TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=2,max_df=.9,sublinear_tf=True)
    Xtr_=tfidf.fit_transform(Xtr); Xte_=tfidf.transform(Xte)
    nb=MultinomialNB(alpha=.5); nb.fit(Xtr_,ytr); ynb=nb.predict(Xte_)
    lr=LogisticRegression(max_iter=1000,C=1.,solver='lbfgs',random_state=42)
    lr.fit(Xtr_,ytr); ylr=lr.predict(Xte_)
    def mtr(yt,yp):return{'Accuracy':accuracy_score(yt,yp),'Precision':precision_score(yt,yp,average='weighted',zero_division=0),'Recall':recall_score(yt,yp,average='weighted',zero_division=0),'F1-Score':f1_score(yt,yp,average='weighted',zero_division=0)}
    return dict(le=le,tfidf=tfidf,nb=nb,lr=lr,
                acc_nb=accuracy_score(yte,ynb),acc_lr=accuracy_score(yte,ylr),
                y_test=yte,y_nb=ynb,y_lr=ylr,
                cm_nb=confusion_matrix(yte,ynb),cm_lr=confusion_matrix(yte,ylr),
                report_nb=classification_report(yte,ynb,target_names=le.classes_,digits=4,output_dict=True),
                report_lr=classification_report(yte,ylr,target_names=le.classes_,digits=4,output_dict=True),
                df_compare=pd.DataFrame({'Naïve Bayes':mtr(yte,ynb),'Logistic Regression':mtr(yte,ylr)}).T.round(4),
                X=X,y=y)

# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────
def style_ax(ax, title=""):
    ax.set_facecolor(BG3)
    ax.tick_params(colors=MUTED,labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    if title: ax.set_title(title,color=MUTED,fontsize=10.5,pad=10)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color('#263D58')
    ax.grid(axis='y',color='#263D58',linewidth=.6,alpha=.5); ax.set_axisbelow(True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">Sentimen PPKM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tag">ADTT · 2026</div>', unsafe_allow_html=True)
    st.divider()

    DATA_DIR      = os.path.join(os.path.dirname(__file__),'data')
    LOCAL_RAW     = os.path.join(DATA_DIR,'INA_TweetsPPKM_Raw.csv')
    LOCAL_LABELED = os.path.join(DATA_DIR,'INA_TweetsPPKM_Labeled_Pure.csv')
    HAS_LOCAL     = os.path.isfile(LOCAL_RAW) and os.path.isfile(LOCAL_LABELED)

    if HAS_LOCAL:
        st.markdown('<div class="b-ok">✓ Dataset terdeteksi dari <code>data/</code></div>', unsafe_allow_html=True)
        data_source='local'; uploaded_raw=uploaded_labeled=None
    else:
        st.markdown("**Upload Dataset**")
        st.caption("Dua file CSV dari Kaggle PPKM Twitter")
        uploaded_raw     = st.file_uploader("File Raw CSV",    type=['csv','tsv'],key='raw')
        uploaded_labeled = st.file_uploader("File Labeled CSV",type=['csv','tsv'],key='labeled')
        data_source='upload'
        if not (uploaded_raw and uploaded_labeled):
            st.markdown('<div class="b-warn">⚠ Taruh file di <code>data/</code> untuk otomatis</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**Ukuran Sample**")
    max_per_class = st.slider("Maks tweet per kelas",500,5000,2000,500,
                              help="Kurangi jika proses lambat",label_visibility="collapsed")
    st.caption(f"Per kelas: **{max_per_class:,}** · Total maks: **{max_per_class*3:,}**")
    st.divider()
    st.caption("Analisis Data Tak Terstruktur")
    st.caption("Dataset: Kaggle PPKM Twitter · CC0")

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-chip">📋 Proyek Akhir · ADTT · 2026</div>
  <div class="hero-title">Analisis Sentimen Masyarakat<br>terhadap <span>Kebijakan PPKM</span></div>
  <div class="hero-bar"></div>
  <div class="hero-pills">
    <span class="hero-pill">🐦 Platform X / Twitter</span>
    <span class="hero-pill">🤖 Naïve Bayes &amp; Logistic Regression</span>
    <span class="hero-pill">🇮🇩 Bahasa Indonesia</span>
    <span class="hero-pill">📦 Kaggle CC0</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CEK DATA
# ─────────────────────────────────────────────
ready = HAS_LOCAL or (uploaded_raw is not None and uploaded_labeled is not None)

if not ready:
    st.markdown('<div class="b-warn">👈 Upload dataset di sidebar, atau letakkan CSV di folder <code>data/</code>.</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="lc"><h4>📌 Tujuan</h4><p>Mengklasifikasikan sentimen publik Twitter (positif, netral, negatif) terhadap kebijakan PPKM 2021–2022.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="lc"><h4>⚙ Metode</h4><p>Text preprocessing Sastrawi + TF-IDF + Naïve Bayes + Logistic Regression.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="lc"><h4>📊 Output</h4><p>EDA, Word Cloud, Evaluasi Model, Perbandingan Algoritma, Demo Prediksi Real-Time.</p></div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# LOAD & PROSES
# ─────────────────────────────────────────────
stemmer, sw = init_nlp()

try:
    if data_source=='local':
        df_raw,df_labeled = load_dataframe(LOCAL_RAW,LOCAL_LABELED)
    else:
        df_raw,df_labeled = load_dataframe(uploaded_raw.read(),uploaded_labeled.read())
except Exception as e:
    st.error(f"❌ Gagal membaca dataset: {e}"); st.stop()

ck = f"df_{max_per_class}_{len(df_labeled)}"
if st.session_state.get('cache_key')!=ck:
    with st.spinner("Memproses data..."):
        df = build_df(df_labeled,stemmer,sw,max_per_class)
    st.session_state['df']=df; st.session_state['cache_key']=ck
    st.session_state.pop('results',None)
df = st.session_state['df']

if 'results' not in st.session_state:
    with st.spinner("Melatih model..."):
        st.session_state['results'] = train_models(df)
R = st.session_state['results']
slist = ['positif','negatif','netral']

# ─────────────────────────────────────────────
# NAVIGASI — pakai div wrapper .nav-active untuk highlight
# ─────────────────────────────────────────────
PAGES = ["📋 Data Overview","📊 EDA & Visualisasi","🤖 Model & Evaluasi","📈 Perbandingan","🔮 Demo Prediksi"]
if 'page' not in st.session_state: st.session_state['page']=0
page = st.session_state['page']

nav_cols = st.columns(len(PAGES))
for i,(col,lbl) in enumerate(zip(nav_cols,PAGES)):
    with col:
        active = (i == page)
        # Bungkus dengan div berclass nav-active jika aktif
        if active:
            st.markdown('<div class="nav-bar-wrap nav-active">', unsafe_allow_html=True)
        else:
            st.markdown('<div class="nav-bar-wrap">', unsafe_allow_html=True)
        if st.button(lbl, key=f"nav_{i}", use_container_width=True):
            st.session_state['page']=i; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Inject CSS khusus untuk nav wrapper aktif — lebih reliable dari class selector
st.markdown(f"""
<style>
/* Highlight tombol nav yang aktif berdasarkan posisi ke-{page+1} */
div[data-testid="stHorizontalBlock"] > div:nth-child({page+1}) .stButton > button {{
    background: var(--blue) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 14px rgba(59,158,255,.38) !important;
    font-weight: 600 !important;
}}
/* Semua tombol nav: reset ke style transparan dulu */
div[data-testid="stHorizontalBlock"] > div .stButton > button {{
    background: transparent !important;
    color: var(--txt2) !important;
    border: none !important;
    border-radius: var(--r1) !important;
    font-family: var(--ff-body) !important;
    font-size: .82rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    padding: .55rem .6rem !important;
    width: 100% !important;
    transition: background .15s, color .15s !important;
}}
div[data-testid="stHorizontalBlock"] > div .stButton > button:hover {{
    background: var(--hover) !important;
    color: var(--txt1) !important;
    transform: none !important;
}}
/* Override aktif (harus di bawah agar lebih spesifik) */
div[data-testid="stHorizontalBlock"] > div:nth-child({page+1}) .stButton > button {{
    background: var(--blue) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 14px rgba(59,158,255,.38) !important;
    font-weight: 600 !important;
}}
</style>
""", unsafe_allow_html=True)

# Breadcrumb
st.markdown(f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r1);
     padding:.42rem 1rem;margin:.5rem 0 1.6rem;font-size:.77rem;color:var(--txt3);
     display:flex;align-items:center;gap:.4rem">
  <span>Dashboard</span>
  <span style="color:var(--border-hi)">›</span>
  <span style="color:var(--blue)">{PAGES[page].split(' ',1)[1]}</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE 0 — DATA OVERVIEW
# ═══════════════════════════════════════════════
if page==0:
    dist = df['sentimen'].value_counts()
    st.markdown('<div class="sh">Ringkasan Dataset</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    def kpi(col,lbl,val,sub,color):
        col.markdown(f'<div class="kpi"><div class="kpi-label">{lbl}</div><div class="kpi-val">{val}</div>'
                     f'<div class="kpi-sub">{sub}</div><div class="kpi-line" style="background:{color}"></div></div>',
                     unsafe_allow_html=True)
    kpi(c1,"Total Tweet",f"{len(df):,}","dataset terproses",BLUE)
    kpi(c2,"😊 Positif",f"{dist.get('positif',0):,}",f"{dist.get('positif',0)/len(df)*100:.1f}% dari total",PALETTE['positif'])
    kpi(c3,"😡 Negatif",f"{dist.get('negatif',0):,}",f"{dist.get('negatif',0)/len(df)*100:.1f}% dari total",PALETTE['negatif'])
    kpi(c4,"😐 Netral",f"{dist.get('netral',0):,}",f"{dist.get('netral',0)/len(df)*100:.1f}% dari total",PALETTE['netral'])

    st.markdown("<br>", unsafe_allow_html=True)
    ca,cb = st.columns([3,2])
    with ca:
        st.markdown('<div class="sh">Contoh Data Berlabel</div>', unsafe_allow_html=True)
        st.dataframe(df[['Date','Tweet','sentimen']].head(10),use_container_width=True,hide_index=True)
    with cb:
        st.markdown('<div class="sh">Statistik Preprocessing</div>', unsafe_allow_html=True)
        stats = df[['panjang_sebelum','panjang_sesudah']].describe().round(2)
        stats.index=['Count','Mean','Std','Min','25%','50%','75%','Max']
        st.dataframe(stats,use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Sebelum vs Sesudah Preprocessing</div>', unsafe_allow_html=True)
    s = df[['Tweet','tweet_bersih','sentimen']].sample(5,random_state=42).copy()
    s.columns=['Tweet Asli','Setelah Preprocessing','Sentimen']
    st.dataframe(s,use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════
# PAGE 1 — EDA
# ═══════════════════════════════════════════════
elif page==1:
    st.markdown('<div class="sh">Distribusi Sentimen</div>', unsafe_allow_html=True)
    dv=df['sentimen'].value_counts()
    labels=list(dv.index); counts=list(dv.values)
    colors=[PALETTE[l] for l in labels]

    fig,axes=plt.subplots(1,2,figsize=(12,4.5)); fig.patch.set_facecolor(BG2)
    wedges,_,autotexts=axes[0].pie(counts,labels=labels,colors=colors,
        autopct='%1.1f%%',startangle=140,explode=[.03]*len(labels),
        textprops={'fontsize':11,'color':'white'},
        wedgeprops={'edgecolor':BG2,'linewidth':2})
    for at in autotexts: at.set_fontweight('bold'); at.set_color(BG)
    axes[0].set_facecolor(BG2); axes[0].set_title('Proporsi Sentimen',color=MUTED,fontsize=11,pad=10)
    bars=axes[1].bar(labels,counts,color=colors,width=.45,edgecolor=BG2,linewidth=1.5,zorder=3)
    for bar,cnt in zip(bars,counts):
        axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+8,
                     f'{cnt:,}',ha='center',fontsize=11,fontweight='600',color='white',zorder=4)
    axes[1].set_ylim(0,max(counts)*1.22)
    style_ax(axes[1],'Jumlah Tweet per Sentimen')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Word Cloud per Sentimen</div>', unsafe_allow_html=True)
    fig2,axes2=plt.subplots(1,3,figsize=(18,5)); fig2.patch.set_facecolor(BG2)
    for ax,sent,cmap,emoji in zip(axes2,slist,['Greens','Reds','Blues'],['😊','😡','😐']):
        txt=' '.join(df[df['sentimen']==sent]['tweet_bersih'])
        if not txt.strip(): continue
        wc=WordCloud(width=600,height=350,background_color=BG3,
                     colormap=cmap,max_words=80,random_state=42).generate(txt)
        ax.imshow(wc,interpolation='bilinear'); ax.axis('off')
        ax.set_title(f'{emoji}  {sent.upper()}',fontsize=12,fontweight='600',color='white',pad=10)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Top 15 Kata per Sentimen</div>', unsafe_allow_html=True)
    fig3,axes3=plt.subplots(1,3,figsize=(18,6)); fig3.patch.set_facecolor(BG2)
    for ax,sent,emoji in zip(axes3,slist,['😊','😡','😐']):
        txt=' '.join(df[df['sentimen']==sent]['tweet_bersih'])
        cnts=Counter(txt.split()).most_common(15)
        if not cnts: continue
        words,freqs=zip(*cnts)
        ax.barh(words[::-1],freqs[::-1],color=PALETTE[sent],alpha=.85,edgecolor=BG2,linewidth=.8,zorder=3)
        style_ax(ax,f'{emoji}  {sent.upper()}'); ax.set_xlabel('Frekuensi')
        ax.grid(axis='x',color='#263D58',linewidth=.6,alpha=.5); ax.grid(axis='y',visible=False)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Distribusi Panjang Tweet</div>', unsafe_allow_html=True)
    fig4,axes4=plt.subplots(1,2,figsize=(14,5)); fig4.patch.set_facecolor(BG2)
    for sent in slist:
        sub=df[df['sentimen']==sent]
        axes4[0].hist(sub['panjang_sebelum'],bins=30,alpha=.5,color=PALETTE[sent],label=sent.capitalize(),edgecolor=BG2,linewidth=.5)
        axes4[1].hist(sub['panjang_sesudah'],bins=30,alpha=.5,color=PALETTE[sent],label=sent.capitalize(),edgecolor=BG2,linewidth=.5)
    for ax,ttl in zip(axes4,['Sebelum Preprocessing','Sesudah Preprocessing']):
        style_ax(ax,ttl); ax.set_xlabel('Jumlah Kata'); ax.set_ylabel('Frekuensi')
        ax.legend(fontsize=9,labelcolor='white',facecolor=BG3,edgecolor='#263D58')
    plt.tight_layout(); st.pyplot(fig4); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Trend Volume Tweet per Bulan</div>', unsafe_allow_html=True)
    df['tanggal']=pd.to_datetime(df['Date'],utc=True,errors='coerce')
    dft=df.dropna(subset=['tanggal']).copy(); dft['bulan']=dft['tanggal'].dt.to_period('M')
    trend=dft.groupby(['bulan','sentimen']).size().unstack(fill_value=0)
    if not trend.empty:
        fig5,ax5=plt.subplots(figsize=(13,5)); fig5.patch.set_facecolor(BG2)
        trend.plot(kind='bar',ax=ax5,
                   color={s:PALETTE[s] for s in slist if s in trend.columns},
                   width=.72,edgecolor=BG2,linewidth=.8,zorder=3)
        style_ax(ax5,'Volume Tweet per Sentimen per Bulan')
        ax5.set_xticklabels([str(p) for p in trend.index],rotation=45,ha='right',color=MUTED,fontsize=8)
        ax5.legend(title='Sentimen',fontsize=9,labelcolor='white',facecolor=BG3,edgecolor='#263D58',title_fontsize=9)
        plt.tight_layout(); st.pyplot(fig5); plt.close()
    else:
        st.markdown('<div class="b-info">ℹ Data tanggal tidak tersedia untuk analisis tren.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE 2 — MODEL & EVALUASI
# ═══════════════════════════════════════════════
elif page==2:
    st.markdown('<div class="sh">Performa Model</div>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    def kpi2(col,lbl,val,sub,color):
        col.markdown(f'<div class="kpi"><div class="kpi-label">{lbl}</div><div class="kpi-val">{val}</div>'
                     f'<div class="kpi-sub">{sub}</div><div class="kpi-line" style="background:{color}"></div></div>',
                     unsafe_allow_html=True)
    kpi2(c1,"Naïve Bayes — Akurasi",f"{R['acc_nb']*100:.2f}%","Multinomial NB · α = 0.5",CYAN)
    delta=R['acc_lr']-R['acc_nb']
    kpi2(c2,"Logistic Regression — Akurasi",f"{R['acc_lr']*100:.2f}%",
         f"{'▲' if delta>=0 else '▼'} {abs(delta)*100:.2f}% vs Naïve Bayes",BLUE)

    st.markdown("<br>", unsafe_allow_html=True)
    cn,cl=st.columns(2)
    for col,key,title,cmap in [(cn,'nb','Naïve Bayes','BuGn'),(cl,'lr','Logistic Regression','Blues')]:
        with col:
            st.markdown(f'<div class="sh">{title}</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(R[f'report_{key}']).T.round(4),use_container_width=True)
            fig_cm,ax_cm=plt.subplots(figsize=(6,4.5))
            fig_cm.patch.set_facecolor(BG2); ax_cm.set_facecolor(BG3)
            ConfusionMatrixDisplay(R[f'cm_{key}'],display_labels=R['le'].classes_).plot(
                ax=ax_cm,cmap=cmap,colorbar=False)
            ax_cm.set_title(f'Confusion Matrix — {title}  |  Akurasi: {R[f"acc_{key}"]*100:.2f}%',
                            fontsize=10,color=MUTED,pad=10)
            ax_cm.tick_params(colors=MUTED)
            ax_cm.xaxis.label.set_color(MUTED); ax_cm.yaxis.label.set_color(MUTED)
            for sp in ax_cm.spines.values(): sp.set_color('#263D58')
            plt.tight_layout(); st.pyplot(fig_cm); plt.close()

# ═══════════════════════════════════════════════
# PAGE 3 — PERBANDINGAN
# ═══════════════════════════════════════════════
elif page==3:
    st.markdown('<div class="sh">Tabel Perbandingan Performa</div>', unsafe_allow_html=True)
    st.dataframe(R['df_compare'].style.highlight_max(axis=0,color='rgba(59,158,255,.18)').format("{:.4f}"),
                 use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Visualisasi Perbandingan</div>', unsafe_allow_html=True)
    mn=list(R['df_compare'].columns)
    nb_v=R['df_compare'].loc['Naïve Bayes'].values
    lr_v=R['df_compare'].loc['Logistic Regression'].values
    x=np.arange(len(mn)); w=.35

    fig_c,ax_c=plt.subplots(figsize=(11,5.5)); fig_c.patch.set_facecolor(BG2)
    b1=ax_c.bar(x-w/2,nb_v,w,label='Naïve Bayes',color=CYAN,alpha=.9,edgecolor=BG2,linewidth=1,zorder=3)
    b2=ax_c.bar(x+w/2,lr_v,w,label='Logistic Regression',color=BLUE,alpha=.9,edgecolor=BG2,linewidth=1,zorder=3)
    for bar in list(b1)+list(b2):
        ax_c.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.006,
                  f'{bar.get_height():.3f}',ha='center',va='bottom',fontsize=9.5,color='white',fontweight='500')
    ax_c.set_xticks(x); ax_c.set_xticklabels(mn,fontsize=11,color=MUTED)
    ax_c.set_ylim(0,1.15); style_ax(ax_c,'Naïve Bayes vs Logistic Regression')
    ax_c.legend(fontsize=10,labelcolor='white',facecolor=BG3,edgecolor='#263D58')
    ax_c.axhline(y=.8,color=BLUE,linestyle='--',alpha=.3,linewidth=1.2)
    ax_c.text(3.6,.81,'threshold 0.8',color=BLUE,fontsize=8.5,alpha=.6)
    plt.tight_layout(); st.pyplot(fig_c); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Top Fitur — Logistic Regression</div>', unsafe_allow_html=True)
    fn=np.array(R['tfidf'].get_feature_names_out())
    fig_f,axes_f=plt.subplots(1,3,figsize=(18,5.5)); fig_f.patch.set_facecolor(BG2)
    for i,sent in enumerate(R['le'].classes_):
        coef=R['lr'].coef_[i]; top=np.argsort(coef)[-12:][::-1]
        axes_f[i].barh(fn[top][::-1],coef[top][::-1],
                       color=PALETTE.get(sent,'#888'),alpha=.85,edgecolor=BG2,linewidth=.8,zorder=3)
        style_ax(axes_f[i],sent.capitalize()); axes_f[i].set_xlabel('Koefisien')
        axes_f[i].grid(axis='x',color='#263D58',linewidth=.6,alpha=.5); axes_f[i].grid(axis='y',visible=False)
    plt.tight_layout(); st.pyplot(fig_f); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Ringkasan Hasil</div>', unsafe_allow_html=True)
    winner='Logistic Regression' if R['acc_lr']>=R['acc_nb'] else 'Naïve Bayes'
    wacc=max(R['acc_lr'],R['acc_nb'])
    n_pos=(df['sentimen']=='positif').sum()
    n_neg=(df['sentimen']=='negatif').sum()
    n_net=(df['sentimen']=='netral').sum()

    ca,cb=st.columns([2,1])
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
<div class="kpi" style="text-align:center;padding:1.8rem 1.4rem">
  <div class="kpi-label">🏆 Model Terbaik</div>
  <div style="margin:.7rem 0"><span class="w-badge">✦ {winner}</span></div>
  <div class="kpi-val">{wacc*100:.2f}%</div>
  <div class="kpi-sub">akurasi pada test set</div>
  <div class="kpi-line" style="background:linear-gradient(90deg,{BLUE},{CYAN})"></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE 4 — DEMO PREDIKSI
# ═══════════════════════════════════════════════
elif page==4:
    mn = 'Logistic Regression' if R['acc_lr']>=R['acc_nb'] else 'Naïve Bayes'
    bm = R['lr'] if R['acc_lr']>=R['acc_nb'] else R['nb']

    st.markdown('<div class="sh">Prediksi Real-Time</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="b-ok">Model aktif: <b>{mn}</b> — akurasi {max(R["acc_nb"],R["acc_lr"])*100:.2f}%</div>', unsafe_allow_html=True)

    user_input = st.text_area("Masukkan teks tweet tentang PPKM:",
        placeholder="Contoh: PPKM sangat membantu mengurangi penyebaran COVID di Indonesia",height=110)

    if st.button("🔍 Analisis Sentimen",type="primary"):
        if user_input.strip():
            clean=preprocess(user_input,stemmer,sw)
            vec=R['tfidf'].transform([clean])
            pred=bm.predict(vec)[0]
            proba=bm.predict_proba(vec)[0]
            label=R['le'].inverse_transform([pred])[0]
            conf=max(proba)
            emoji='😊' if label=='positif' else '😡' if label=='negatif' else '😐'
            css='r-pos' if label=='positif' else 'r-neg' if label=='negatif' else 'r-net'
            st.markdown(f'<div class="{css}"><div class="r-label">{emoji} &nbsp; {label.upper()}</div>'
                        f'<div class="r-conf">Confidence: <b>{conf:.2%}</b></div></div>',unsafe_allow_html=True)
            pdf=pd.DataFrame({'Sentimen':R['le'].classes_,'Probabilitas':proba}).sort_values('Probabilitas',ascending=False)
            st.dataframe(pdf.style.format({'Probabilitas':'{:.4f}'}),use_container_width=True,hide_index=True)
            st.caption(f"Teks setelah preprocessing: `{clean}`")
        else:
            st.markdown('<div class="b-warn">⚠ Masukkan teks terlebih dahulu.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Contoh Prediksi Batch</div>', unsafe_allow_html=True)
    examples=[
        "PPKM sangat membantu mengurangi penyebaran COVID di Indonesia",
        "PPKM membuat usaha saya bangkrut, pemerintah tidak peduli rakyat kecil",
        "Pemerintah resmi umumkan perpanjangan PPKM level 2 hingga akhir bulan",
        "Kebijakan PPKM sudah tepat, kasus COVID turun drastis akhirnya",
        "Susah cari makan gara-gara PPKM, warung dilarang buka malam",
    ]
    hasil=[]
    for tw in examples:
        cl=preprocess(tw,stemmer,sw)
        v=R['tfidf'].transform([cl])
        lb=R['le'].inverse_transform([bm.predict(v)[0]])[0]
        cf=max(bm.predict_proba(v)[0])
        em='😊' if lb=='positif' else '😡' if lb=='negatif' else '😐'
        hasil.append({'Tweet':tw,'Prediksi':f'{em} {lb.upper()}','Confidence':f'{cf:.2%}'})
    st.dataframe(pd.DataFrame(hasil),use_container_width=True,hide_index=True)
