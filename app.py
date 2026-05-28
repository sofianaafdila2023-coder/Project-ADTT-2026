# -*- coding: utf-8 -*-
"""
Analisis Sentimen Masyarakat terhadap Kebijakan PPKM
pada Platform X (Twitter) — Naïve Bayes & Logistic Regression
Streamlit App — Proyek Akhir ADTT 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import io

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
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen PPKM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PALETTE = {'positif': '#2ECC71', 'negatif': '#E74C3C', 'netral': '#3498DB'}
BG = '#0E1117'

# ─────────────────────────────────────────────
# INIT NLP — pakai cache_resource agar TIDAK di-hash
# ─────────────────────────────────────────────
@st.cache_resource
def init_nlp():
    nltk.download('punkt',      quiet=True)
    nltk.download('punkt_tab',  quiet=True)
    nltk.download('stopwords',  quiet=True)

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

# ─────────────────────────────────────────────
# FUNGSI PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(text, stemmer, stopword_list):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    tokens = word_tokenize(' '.join(words))
    tokens = [t for t in tokens if t not in stopword_list and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

# ─────────────────────────────────────────────
# LOAD DATA
# FIX: tidak passing stemmer/stopword_list sebagai param cache_data.
# Gunakan session_state untuk menyimpan hasil, dan @st.cache_resource
# yang sudah terbukti bisa handle objek non-hashable.
# ─────────────────────────────────────────────
def load_dataframe(raw_source, labeled_source):
    """raw_source / labeled_source: path (str) atau bytes."""
    if isinstance(raw_source, (str, os.PathLike)):
        df_raw     = pd.read_csv(raw_source,     sep='\t', on_bad_lines='skip')
        df_labeled = pd.read_csv(labeled_source, sep='\t', on_bad_lines='skip')
    else:
        df_raw     = pd.read_csv(io.BytesIO(raw_source),     sep='\t', on_bad_lines='skip')
        df_labeled = pd.read_csv(io.BytesIO(labeled_source), sep='\t', on_bad_lines='skip')

    label_map = {0: 'negatif', 1: 'positif', 2: 'netral'}
    df_labeled['sentimen'] = df_labeled['sentiment'].map(label_map)
    return df_raw, df_labeled

def build_df(df_labeled, stemmer, stopword_list, max_per_class):
    df = df_labeled[['Date','Tweet','sentimen']].copy()
    df = df.dropna(subset=['Tweet','sentimen'])
    df = df[df['Tweet'].str.strip() != ''].reset_index(drop=True)
    df = df.groupby('sentimen').apply(
        lambda x: x.sample(min(len(x), max_per_class), random_state=42)
    ).reset_index(drop=True)

    progress = st.progress(0, text="Memproses tweet...")
    total = len(df)
    results = []
    for i, row in df.iterrows():
        results.append(preprocess(row['Tweet'], stemmer, stopword_list))
        if i % 200 == 0:
            progress.progress(min(int(i/total*100), 99), text=f"Preprocessing... {i}/{total}")
    progress.progress(100, text="✅ Selesai!")
    progress.empty()

    df['tweet_bersih'] = results
    df = df[df['tweet_bersih'].str.strip().str.len() > 3].reset_index(drop=True)
    df['panjang_sebelum'] = df['Tweet'].apply(lambda x: len(str(x).split()))
    df['panjang_sesudah'] = df['tweet_bersih'].apply(lambda x: len(str(x).split()))
    return df

# ─────────────────────────────────────────────
# TRAINING — simpan di session_state
# ─────────────────────────────────────────────
def train_models(df):
    le = LabelEncoder()
    dfc = df.copy()
    dfc['label'] = le.fit_transform(dfc['sentimen'])
    X, y = dfc['tweet_bersih'], dfc['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                             min_df=2, max_df=0.90, sublinear_tf=True)
    X_tr = tfidf.fit_transform(X_train)
    X_te = tfidf.transform(X_test)

    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_tr, y_train)
    y_nb = nb.predict(X_te)

    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                             multi_class='multinomial', random_state=42)
    lr.fit(X_tr, y_train)
    y_lr = lr.predict(X_te)

    def metrics(yt, yp):
        return {
            'Accuracy' : accuracy_score(yt, yp),
            'Precision': precision_score(yt, yp, average='weighted', zero_division=0),
            'Recall'   : recall_score(yt, yp, average='weighted', zero_division=0),
            'F1-Score' : f1_score(yt, yp, average='weighted', zero_division=0),
        }

    return {
        'le':le,'tfidf':tfidf,'nb':nb,'lr':lr,
        'acc_nb':accuracy_score(y_test,y_nb),
        'acc_lr':accuracy_score(y_test,y_lr),
        'y_test':y_test,'y_nb':y_nb,'y_lr':y_lr,
        'cm_nb':confusion_matrix(y_test,y_nb),
        'cm_lr':confusion_matrix(y_test,y_lr),
        'report_nb':classification_report(y_test,y_nb,target_names=le.classes_,digits=4,output_dict=True),
        'report_lr':classification_report(y_test,y_lr,target_names=le.classes_,digits=4,output_dict=True),
        'df_compare':pd.DataFrame({'Naïve Bayes':metrics(y_test,y_nb),
                                   'Logistic Regression':metrics(y_test,y_lr)}).T.round(4),
        'X':X,'y':y,
    }

# ─────────────────────────────────────────────
# HELPERS PLOT
# ─────────────────────────────────────────────
def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for sp in ['top','right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']:
        ax.spines[sp].set_color('#444')

def fig_bg():
    plt.rcParams['figure.facecolor'] = BG

# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.title("📊 Analisis Sentimen PPKM")
    st.caption("Naïve Bayes & Logistic Regression")
    st.divider()

    # ── Cek apakah dataset tersedia di repo (data/)
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    LOCAL_RAW     = os.path.join(DATA_DIR, 'INA_TweetsPPKM_Raw.csv')
    LOCAL_LABELED = os.path.join(DATA_DIR, 'INA_TweetsPPKM_Labeled_Pure.csv')
    HAS_LOCAL = os.path.isfile(LOCAL_RAW) and os.path.isfile(LOCAL_LABELED)

    if HAS_LOCAL:
        st.success("✅ Dataset terdeteksi otomatis dari folder `data/`")
        data_source = 'local'
        uploaded_raw = uploaded_labeled = None
    else:
        st.subheader("📂 Upload Dataset")
        st.markdown("""Upload **dua file CSV** (TSV):
- `INA_TweetsPPKM_Raw.csv`
- `INA_TweetsPPKM_Labeled_Pure.csv`

💡 *Tip: taruh file di folder `data/` di repo agar tidak perlu upload manual.*
        """)
        uploaded_raw     = st.file_uploader("File Raw CSV",     type=['csv','tsv'], key='raw')
        uploaded_labeled = st.file_uploader("File Labeled CSV", type=['csv','tsv'], key='labeled')
        data_source = 'upload'

    st.divider()
    max_per_class = st.slider("Maks tweet per kelas", 500, 5000, 2000, 500,
                              help="Kurangi jika lambat")
    st.divider()
    st.caption("Mata Kuliah: Analisis Data Tak Terstruktur")
    st.caption("Sumber: Kaggle — PPKM Twitter (CC0)")

# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
st.title("🔍 Analisis Sentimen Masyarakat terhadap Kebijakan PPKM")
st.markdown("**Platform X (Twitter)** · *Naïve Bayes* & *Logistic Regression* · Bahasa Indonesia")

# Cek ketersediaan data
ready = HAS_LOCAL or (uploaded_raw is not None and uploaded_labeled is not None)

if not ready:
    st.info("👈 Silakan upload kedua file dataset di sidebar, atau taruh di folder `data/` di repo untuk tampil otomatis.")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown("**📌 Tujuan**"); st.markdown("Klasifikasi sentimen publik (positif, netral, negatif) terhadap PPKM.")
    with col2:
        st.markdown("**⚙️ Metode**"); st.markdown("Sastrawi preprocessing + TF-IDF + Naïve Bayes + Logistic Regression")
    with col3:
        st.markdown("**📊 Output**"); st.markdown("EDA, Word Cloud, Evaluasi Model, Demo Prediksi Real-Time")
    st.stop()

# ── Inisialisasi NLP (cache_resource, TIDAK di-hash)
stemmer, stopword_list = init_nlp()

# ── Load data
if data_source == 'local':
    df_raw, df_labeled = load_dataframe(LOCAL_RAW, LOCAL_LABELED)
else:
    df_raw, df_labeled = load_dataframe(uploaded_raw.read(), uploaded_labeled.read())

# ── Preprocessing (simpan di session_state agar tidak re-run saat slider tidak berubah)
cache_key = f"df_{max_per_class}_{len(df_labeled)}"
if st.session_state.get('cache_key') != cache_key:
    with st.spinner("⏳ Preprocessing data..."):
        df = build_df(df_labeled, stemmer, stopword_list, max_per_class)
    st.session_state['df'] = df
    st.session_state['cache_key'] = cache_key
    st.session_state.pop('results', None)   # hapus model lama jika data berubah

df = st.session_state['df']
st.success(f"✅ Data siap! Total **{len(df):,}** tweet dianalisis.")

# ── Training (simpan di session_state)
if 'results' not in st.session_state:
    with st.spinner("🤖 Melatih model..."):
        st.session_state['results'] = train_models(df)

R = st.session_state['results']
sentimen_list = ['positif','negatif','netral']

# ═══════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📋 Data Overview","📊 EDA & Visualisasi",
    "🤖 Model & Evaluasi","📈 Perbandingan Model","🔮 Demo Prediksi"
])

# ────────────────────────────
# TAB 1
# ────────────────────────────
with tab1:
    st.subheader("📋 Ringkasan Dataset")
    dist = df['sentimen'].value_counts()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Tweet", f"{len(df):,}")
    c2.metric("😊 Positif", f"{dist.get('positif',0):,}", f"{dist.get('positif',0)/len(df)*100:.1f}%")
    c3.metric("😡 Negatif", f"{dist.get('negatif',0):,}", f"{dist.get('negatif',0)/len(df)*100:.1f}%")
    c4.metric("😐 Netral",  f"{dist.get('netral',0):,}",  f"{dist.get('netral',0)/len(df)*100:.1f}%")
    st.divider()
    ca,cb = st.columns(2)
    with ca:
        st.subheader("Contoh Data Berlabel")
        st.dataframe(df[['Date','Tweet','sentimen']].head(10), use_container_width=True, hide_index=True)
    with cb:
        st.subheader("Statistik Preprocessing")
        stats = df[['panjang_sebelum','panjang_sesudah']].describe().round(2)
        stats.index = ['Count','Mean','Std','Min','25%','50%','75%','Max']
        st.dataframe(stats, use_container_width=True)
    st.divider()
    st.subheader("Sebelum vs Sesudah Preprocessing")
    s = df[['Tweet','tweet_bersih','sentimen']].sample(5, random_state=42)
    s.columns = ['Tweet Asli','Setelah Preprocessing','Sentimen']
    st.dataframe(s, use_container_width=True, hide_index=True)

# ────────────────────────────
# TAB 2
# ────────────────────────────
with tab2:
    st.subheader("📊 Exploratory Data Analysis")

    # Viz 1 — Distribusi Sentimen
    st.markdown("#### Distribusi Sentimen")
    dv = df['sentimen'].value_counts()
    labels = list(dv.index); counts = list(dv.values)
    colors = [PALETTE[l] for l in labels]

    fig_bg()
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    fig.patch.set_facecolor(BG)

    wedges,texts,autotexts = axes[0].pie(
        counts, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=140, explode=[0.04]*len(labels),
        textprops={'fontsize':11,'color':'white'}
    )
    for at in autotexts: at.set_fontweight('bold')
    axes[0].set_title('Proporsi Sentimen', color='white')

    bars = axes[1].bar(labels, counts, color=colors, width=0.5, edgecolor='white')
    for bar,cnt in zip(bars,counts):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                     f'{cnt:,}', ha='center', fontsize=11, fontweight='bold', color='white')
    axes[1].set_ylim(0, max(counts)*1.2)
    style_ax(axes[1]); axes[1].set_title('Jumlah Tweet per Sentimen', color='white')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Viz 2 — Word Cloud
    st.markdown("#### Word Cloud per Sentimen")
    fig2, axes2 = plt.subplots(1, 3, figsize=(18,5))
    fig2.patch.set_facecolor(BG)
    for ax,sent,cmap,emoji in zip(axes2, sentimen_list,
                                   ['Greens','Reds','Blues'],['😊','😡','😐']):
        txt = ' '.join(df[df['sentimen']==sent]['tweet_bersih'])
        if not txt.strip(): continue
        wc = WordCloud(width=600,height=350,background_color=BG,
                       colormap=cmap,max_words=80,random_state=42).generate(txt)
        ax.imshow(wc,interpolation='bilinear'); ax.axis('off')
        ax.set_title(f'{emoji} {sent.upper()}', fontsize=13, fontweight='bold', color='white')
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    # Viz 3 — Top 15 Kata
    st.markdown("#### Top 15 Kata per Sentimen")
    fig3, axes3 = plt.subplots(1, 3, figsize=(18,6))
    fig3.patch.set_facecolor(BG)
    for ax,sent,emoji in zip(axes3, sentimen_list, ['😊','😡','😐']):
        txt = ' '.join(df[df['sentimen']==sent]['tweet_bersih'])
        cnts = Counter(txt.split()).most_common(15)
        if not cnts: continue
        words,freqs = zip(*cnts)
        ax.barh(words[::-1], freqs[::-1], color=PALETTE[sent], alpha=0.85, edgecolor='white')
        style_ax(ax); ax.set_title(f'{emoji} {sent.upper()}', color='white', fontweight='bold')
        ax.set_xlabel('Frekuensi')
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    # Viz 4 — Panjang Tweet
    st.markdown("#### Distribusi Panjang Tweet")
    fig4, axes4 = plt.subplots(1, 2, figsize=(14,5))
    fig4.patch.set_facecolor(BG)
    for sent in sentimen_list:
        sub = df[df['sentimen']==sent]
        axes4[0].hist(sub['panjang_sebelum'], bins=30, alpha=0.5,
                      color=PALETTE[sent], label=sent.capitalize(), edgecolor='white')
        axes4[1].hist(sub['panjang_sesudah'], bins=30, alpha=0.5,
                      color=PALETTE[sent], label=sent.capitalize(), edgecolor='white')
    for ax,title in zip(axes4,['Sebelum Preprocessing','Sesudah Preprocessing']):
        style_ax(ax); ax.set_title(title, color='white')
        ax.set_xlabel('Jumlah Kata'); ax.set_ylabel('Frekuensi')
        ax.legend(fontsize=9, labelcolor='white', facecolor='#1a1a2e')
    plt.tight_layout(); st.pyplot(fig4); plt.close()

    # Viz 5 — Trend Waktu
    st.markdown("#### Trend Volume Tweet per Bulan")
    df['tanggal'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
    dft = df.dropna(subset=['tanggal']).copy()
    dft['bulan'] = dft['tanggal'].dt.to_period('M')
    trend = dft.groupby(['bulan','sentimen']).size().unstack(fill_value=0)
    if not trend.empty:
        fig5, ax5 = plt.subplots(figsize=(13,5))
        fig5.patch.set_facecolor(BG)
        trend.plot(kind='bar', ax=ax5,
                   color={s:PALETTE[s] for s in sentimen_list if s in trend.columns},
                   width=0.75, edgecolor='white')
        style_ax(ax5)
        ax5.set_title('Volume Tweet per Sentimen per Bulan', fontsize=13, fontweight='bold', color='white')
        ax5.set_xticklabels([str(p) for p in trend.index], rotation=45, ha='right', color='white')
        ax5.legend(title='Sentimen', fontsize=9, labelcolor='white',
                   facecolor='#1a1a2e', title_fontsize=9)
        plt.tight_layout(); st.pyplot(fig5); plt.close()
    else:
        st.info("Data tanggal tidak tersedia.")

# ────────────────────────────
# TAB 3
# ────────────────────────────
with tab3:
    st.subheader("🤖 Evaluasi Model")
    c1,c2 = st.columns(2)
    c1.metric("🟢 Naïve Bayes", f"{R['acc_nb']*100:.2f}%")
    c2.metric("🔵 Logistic Regression", f"{R['acc_lr']*100:.2f}%",
              delta=f"{(R['acc_lr']-R['acc_nb'])*100:+.2f}% vs NB")
    st.divider()

    cn, cl = st.columns(2)
    for col, key, title, cmap in [
        (cn, 'nb', 'Naïve Bayes', 'Greens'),
        (cl, 'lr', 'Logistic Regression', 'Blues')
    ]:
        with col:
            st.markdown(f"#### {title}")
            st.dataframe(pd.DataFrame(R[f'report_{key}']).T.round(4), use_container_width=True)
            fig_cm, ax_cm = plt.subplots(figsize=(6,4))
            fig_cm.patch.set_facecolor(BG); ax_cm.set_facecolor(BG)
            ConfusionMatrixDisplay(R[f'cm_{key}'], display_labels=R['le'].classes_).plot(
                ax=ax_cm, cmap=cmap, colorbar=False)
            ax_cm.set_title(f'Confusion Matrix — {title}\nAkurasi: {R[f"acc_{key}"]*100:.2f}%',
                            fontsize=11, fontweight='bold', color='white')
            ax_cm.tick_params(colors='white')
            ax_cm.xaxis.label.set_color('white'); ax_cm.yaxis.label.set_color('white')
            plt.tight_layout(); st.pyplot(fig_cm); plt.close()

# ────────────────────────────
# TAB 4
# ────────────────────────────
with tab4:
    st.subheader("📈 Perbandingan Performa Model")
    st.dataframe(R['df_compare'].style.highlight_max(axis=0, color='#1a4731').format("{:.4f}"),
                 use_container_width=True)
    st.divider()

    # Bar chart perbandingan
    metrics_names = list(R['df_compare'].columns)
    nb_vals = R['df_compare'].loc['Naïve Bayes'].values
    lr_vals = R['df_compare'].loc['Logistic Regression'].values
    x = np.arange(len(metrics_names)); w = 0.35

    fig_c, ax_c = plt.subplots(figsize=(11,5))
    fig_c.patch.set_facecolor(BG)
    b1 = ax_c.bar(x-w/2, nb_vals, w, label='Naïve Bayes',        color='#27AE60', alpha=0.85, edgecolor='white')
    b2 = ax_c.bar(x+w/2, lr_vals, w, label='Logistic Regression', color='#2980B9', alpha=0.85, edgecolor='white')
    for bar in list(b1)+list(b2):
        ax_c.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                  f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, color='white')
    ax_c.set_xticks(x); ax_c.set_xticklabels(metrics_names, fontsize=11, color='white')
    ax_c.set_ylim(0,1.12); style_ax(ax_c)
    ax_c.set_title('Perbandingan Performa: Naïve Bayes vs Logistic Regression',
                   fontsize=13, fontweight='bold', color='white')
    ax_c.legend(fontsize=10, labelcolor='white', facecolor='#1a1a2e')
    ax_c.axhline(y=0.8, color='gray', linestyle='--', alpha=0.4)
    plt.tight_layout(); st.pyplot(fig_c); plt.close()

    # Top Fitur LR
    st.divider()
    st.markdown("#### Top Fitur Berpengaruh — Logistic Regression")
    feat_names = np.array(R['tfidf'].get_feature_names_out())
    fig_f, axes_f = plt.subplots(1, 3, figsize=(18,5))
    fig_f.patch.set_facecolor(BG)
    fig_f.suptitle('Top Fitur per Kelas — Logistic Regression', fontsize=13,
                   fontweight='bold', color='white')
    for i, sent in enumerate(R['le'].classes_):
        coef    = R['lr'].coef_[i]
        top_idx = np.argsort(coef)[-12:][::-1]
        axes_f[i].barh(feat_names[top_idx][::-1], coef[top_idx][::-1],
                       color=PALETTE.get(sent,'#888'), alpha=0.85, edgecolor='white')
        style_ax(axes_f[i])
        axes_f[i].set_title(sent.capitalize(), fontsize=11, fontweight='bold', color='white')
        axes_f[i].set_xlabel('Koefisien')
    plt.tight_layout(); st.pyplot(fig_f); plt.close()

    # Ringkasan tabel
    st.divider()
    n_pos = (df['sentimen']=='positif').sum()
    n_neg = (df['sentimen']=='negatif').sum()
    n_net = (df['sentimen']=='netral').sum()
    winner = 'Logistic Regression' if R['acc_lr'] >= R['acc_nb'] else 'Naïve Bayes'
    w_acc  = max(R['acc_lr'], R['acc_nb'])
    st.markdown(f"""
| Keterangan | Nilai |
|---|---|
| Total Tweet | **{len(df):,}** |
| 😊 Positif | **{n_pos:,}** ({n_pos/len(df)*100:.1f}%) |
| 😡 Negatif | **{n_neg:,}** ({n_neg/len(df)*100:.1f}%) |
| 😐 Netral  | **{n_net:,}** ({n_net/len(df)*100:.1f}%) |
| Akurasi Naïve Bayes | **{R['acc_nb']*100:.2f}%** |
| Akurasi Logistic Regression | **{R['acc_lr']*100:.2f}%** |
| 🏆 Model Terbaik | **{winner}** ({w_acc*100:.2f}%) |
    """)

# ────────────────────────────
# TAB 5
# ────────────────────────────
with tab5:
    st.subheader("🔮 Demo Prediksi Real-Time")
    model_name = 'Logistic Regression' if R['acc_lr'] >= R['acc_nb'] else 'Naïve Bayes'
    best_model = R['lr'] if R['acc_lr'] >= R['acc_nb'] else R['nb']
    st.info(f"🏆 Model terbaik: **{model_name}** ({max(R['acc_nb'],R['acc_lr'])*100:.2f}%)")

    user_input = st.text_area("Masukkan teks tweet tentang PPKM:",
        placeholder="Contoh: PPKM sangat membantu mengurangi penyebaran COVID",
        height=100)

    if st.button("🔍 Prediksi Sentimen", type="primary"):
        if user_input.strip():
            clean = preprocess(user_input, stemmer, stopword_list)
            vec   = R['tfidf'].transform([clean])
            pred  = best_model.predict(vec)[0]
            proba = best_model.predict_proba(vec)[0]
            label = R['le'].inverse_transform([pred])[0]
            conf  = max(proba)
            emoji = '😊' if label=='positif' else '😡' if label=='negatif' else '😐'
            msg_fn = {'positif':st.success,'negatif':st.error,'netral':st.info}[label]
            msg_fn(f"{emoji} **Sentimen: {label.upper()}** — Confidence: {conf:.2%}")
            proba_df = pd.DataFrame({'Sentimen':R['le'].classes_,'Probabilitas':proba})\
                         .sort_values('Probabilitas',ascending=False)
            st.dataframe(proba_df.style.format({'Probabilitas':'{:.4f}'}),
                         use_container_width=True, hide_index=True)
            st.caption(f"Teks setelah preprocessing: `{clean}`")
        else:
            st.warning("⚠️ Masukkan teks terlebih dahulu.")

    st.divider()
    st.markdown("#### Contoh Prediksi Batch")
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
        hasil.append({'Tweet':tw,'Prediksi':f'{em} {lb.upper()}','Confidence':f'{cf:.2%}'})
    st.dataframe(pd.DataFrame(hasil), use_container_width=True, hide_index=True)
