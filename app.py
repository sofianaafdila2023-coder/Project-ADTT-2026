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
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import io

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ── Download NLTK data (sekali saja)
@st.cache_resource
def download_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk()

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

# ─────────────────────────────────────────────
# INISIALISASI NLP (cache agar tidak reload)
# ─────────────────────────────────────────────
@st.cache_resource
def init_nlp():
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()

    sw_factory = StopWordRemoverFactory()
    stopword_list = set(sw_factory.get_stop_words())
    custom_sw = {
        'yang', 'ini', 'itu', 'dan', 'dengan', 'untuk', 'dari', 'ke', 'di',
        'pada', 'aja', 'juga', 'banget', 'nih', 'sih', 'deh', 'lah', 'kan',
        'ya', 'yg', 'udah', 'sudah', 'bisa', 'ada', 'lebih', 'masih', 'jadi',
        'https', 'http', 'rt', 'co', 'amp', 'ppkm', 'via', 'bit', 'ly',
        'tco', 'pic', 'twitter', 'com', 'www'
    }
    stopword_list.update(custom_sw)
    return stemmer, stopword_list

slang_dict = {
    'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'krn': 'karena',
    'gimana': 'bagaimana', 'kalo': 'kalau', 'udah': 'sudah',
    'bnyk': 'banyak', 'hrs': 'harus', 'trs': 'terus', 'jg': 'juga',
    'dr': 'dari', 'pd': 'pada', 'sy': 'saya', 'skrg': 'sekarang',
    'kpd': 'kepada', 'dlm': 'dalam', 'sdh': 'sudah', 'tsb': 'tersebut',
    'thd': 'terhadap', 'ttg': 'tentang', 'krna': 'karena',
    'emg': 'memang', 'emang': 'memang', 'bgt': 'sangat', 'msh': 'masih',
    'sm': 'sama', 'lg': 'lagi', 'ny': 'nya', 'mk': 'maka', 'spy': 'supaya',
    'wkwk': '', 'haha': '', 'hehe': '', 'wkwkwk': ''
}

# ─────────────────────────────────────────────
# FUNGSI PREPROCESSING
# ─────────────────────────────────────────────
def case_folding(text):
    return str(text).lower()

def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_slang(text):
    words = text.split()
    return ' '.join([slang_dict.get(w, w) for w in words])

def remove_stopwords_and_stem(text, stemmer, stopword_list):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopword_list and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

def full_preprocess(text, stemmer, stopword_list):
    text = case_folding(text)
    text = clean_text(text)
    text = normalize_slang(text)
    text = remove_stopwords_and_stem(text, stemmer, stopword_list)
    return text

# ─────────────────────────────────────────────
# FUNGSI LOAD & PROSES DATA (cache by hash)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(raw_bytes, labeled_bytes, stemmer, stopword_list, max_per_class=2000):
    df_raw     = pd.read_csv(io.BytesIO(raw_bytes),     sep='\t', on_bad_lines='skip')
    df_labeled = pd.read_csv(io.BytesIO(labeled_bytes), sep='\t', on_bad_lines='skip')

    label_map = {0: 'negatif', 1: 'positif', 2: 'netral'}
    df_labeled['sentimen'] = df_labeled['sentiment'].map(label_map)

    df = df_labeled[['Date', 'Tweet', 'sentimen']].copy()
    df = df.dropna(subset=['Tweet', 'sentimen'])
    df = df[df['Tweet'].str.strip() != '']
    df = df.reset_index(drop=True)

    # Sampling per kelas
    df = df.groupby('sentimen').apply(
        lambda x: x.sample(min(len(x), max_per_class), random_state=42)
    ).reset_index(drop=True)

    df['tweet_bersih'] = df['Tweet'].apply(lambda t: full_preprocess(t, stemmer, stopword_list))
    df = df[df['tweet_bersih'].str.strip().str.len() > 3].reset_index(drop=True)

    df['panjang_sebelum'] = df['Tweet'].apply(lambda x: len(str(x).split()))
    df['panjang_sesudah'] = df['tweet_bersih'].apply(lambda x: len(str(x).split()))
    df['reduksi_kata']    = df['panjang_sebelum'] - df['panjang_sesudah']

    return df, df_raw

@st.cache_data(show_spinner=False)
def train_models(_df):
    le = LabelEncoder()
    df = _df.copy()
    df['label'] = le.fit_transform(df['sentimen'])

    X = df['tweet_bersih']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        min_df=2, max_df=0.90, sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    nb_model = MultinomialNB(alpha=0.5)
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)

    lr_model = LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs',
        multi_class='multinomial', random_state=42
    )
    lr_model.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)

    acc_nb = accuracy_score(y_test, y_pred_nb)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    def get_metrics(y_true, y_pred):
        return {
            'Accuracy' : accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall'   : recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1-Score' : f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

    df_compare = pd.DataFrame({
        'Naïve Bayes'        : get_metrics(y_test, y_pred_nb),
        'Logistic Regression': get_metrics(y_test, y_pred_lr),
    }).T.round(4)

    cm_nb = confusion_matrix(y_test, y_pred_nb)
    cm_lr = confusion_matrix(y_test, y_pred_lr)

    report_nb = classification_report(y_test, y_pred_nb, target_names=le.classes_, digits=4, output_dict=True)
    report_lr = classification_report(y_test, y_pred_lr, target_names=le.classes_, digits=4, output_dict=True)

    return {
        'le': le, 'tfidf': tfidf,
        'nb_model': nb_model, 'lr_model': lr_model,
        'acc_nb': acc_nb, 'acc_lr': acc_lr,
        'y_test': y_test, 'y_pred_nb': y_pred_nb, 'y_pred_lr': y_pred_lr,
        'df_compare': df_compare,
        'cm_nb': cm_nb, 'cm_lr': cm_lr,
        'report_nb': report_nb, 'report_lr': report_lr,
        'X': X, 'y': y,
    }

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Analisis Sentimen PPKM")
    st.caption("Naïve Bayes & Logistic Regression")
    st.divider()

    st.subheader("📂 Upload Dataset")
    st.markdown("""
    Upload **dua file CSV** (format TSV):
    - `INA_TweetsPPKM_Raw.csv`
    - `INA_TweetsPPKM_Labeled_Pure.csv`
    """)

    uploaded_raw     = st.file_uploader("File Raw CSV", type=['csv', 'tsv'], key='raw')
    uploaded_labeled = st.file_uploader("File Labeled CSV", type=['csv', 'tsv'], key='labeled')

    st.divider()
    max_per_class = st.slider("Maks tweet per kelas", 500, 5000, 2000, 500,
                              help="Kurangi jika lambat, tambah untuk hasil lebih akurat")

    st.divider()
    st.caption("Mata Kuliah: Analisis Data Tak Terstruktur")
    st.caption("Sumber Dataset: Kaggle — PPKM Twitter")

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.title("🔍 Analisis Sentimen Masyarakat terhadap Kebijakan PPKM")
st.markdown("**Platform X (Twitter)** · Metode: *Naïve Bayes* & *Logistic Regression* · Bahasa Indonesia")

if uploaded_raw is None or uploaded_labeled is None:
    st.info("👈 Silakan upload kedua file dataset di sidebar untuk memulai analisis.")

    st.markdown("---")
    st.subheader("ℹ️ Tentang Aplikasi")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📌 Tujuan**")
        st.markdown("Mengklasifikasikan sentimen publik (positif, netral, negatif) terhadap kebijakan PPKM di Twitter.")
    with col2:
        st.markdown("**⚙️ Metode**")
        st.markdown("Text preprocessing (Sastrawi) + TF-IDF + Naïve Bayes + Logistic Regression")
    with col3:
        st.markdown("**📊 Output**")
        st.markdown("EDA, Word Cloud, Perbandingan Model, Demo Prediksi Real-Time")

    st.stop()

# ── Inisialisasi NLP
stemmer, stopword_list = init_nlp()

# ── Load & Preprocess
with st.spinner("⏳ Memuat dan memproses data... (beberapa menit pertama kali)"):
    raw_bytes     = uploaded_raw.read()
    labeled_bytes = uploaded_labeled.read()
    df, df_raw = load_and_preprocess(raw_bytes, labeled_bytes, stemmer, stopword_list, max_per_class)

st.success(f"✅ Data siap! Total **{len(df):,}** tweet dianalisis.")

# ─────────────────────────────────────────────
# TAB NAVIGASI
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Overview",
    "📊 EDA & Visualisasi",
    "🤖 Model & Evaluasi",
    "📈 Perbandingan Model",
    "🔮 Demo Prediksi"
])

sentimen_list = ['positif', 'negatif', 'netral']

# ══════════════════════════════════════════════
# TAB 1: DATA OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.subheader("📋 Ringkasan Dataset")

    dist = df['sentimen'].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tweet", f"{len(df):,}")
    col2.metric("😊 Positif",  f"{dist.get('positif', 0):,}",
                f"{dist.get('positif',0)/len(df)*100:.1f}%")
    col3.metric("😡 Negatif",  f"{dist.get('negatif', 0):,}",
                f"{dist.get('negatif',0)/len(df)*100:.1f}%")
    col4.metric("😐 Netral",   f"{dist.get('netral', 0):,}",
                f"{dist.get('netral',0)/len(df)*100:.1f}%")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Contoh Data Berlabel")
        st.dataframe(
            df[['Date', 'Tweet', 'sentimen']].head(10),
            use_container_width=True, hide_index=True
        )

    with col_b:
        st.subheader("Statistik Preprocessing")
        stats = df[['panjang_sebelum', 'panjang_sesudah', 'reduksi_kata']].describe().round(2)
        stats.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
        st.dataframe(stats, use_container_width=True)

    st.divider()
    st.subheader("Perbandingan Sebelum vs Sesudah Preprocessing")
    sample_df = df[['Tweet', 'tweet_bersih', 'sentimen']].sample(5, random_state=42)
    sample_df.columns = ['Tweet Asli', 'Setelah Preprocessing', 'Sentimen']
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 2: EDA & VISUALISASI
# ══════════════════════════════════════════════
with tab2:
    st.subheader("📊 Exploratory Data Analysis")

    # Viz 1: Distribusi Sentimen
    st.markdown("#### Distribusi Sentimen")
    dist_vals = df['sentimen'].value_counts()
    labels = list(dist_vals.index)
    counts = list(dist_vals.values)
    colors = [PALETTE[l] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0E1117')
    for ax in axes:
        ax.set_facecolor('#0E1117')

    wedges, texts, autotexts = axes[0].pie(
        counts, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140,
        explode=[0.04]*len(labels),
        textprops={'fontsize': 11, 'color': 'white'}
    )
    for at in autotexts:
        at.set_fontweight('bold')
    axes[0].set_title('Proporsi Sentimen', fontsize=12, color='white')

    bars = axes[1].bar(labels, counts, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 10,
                     f'{count:,}', ha='center', fontsize=11,
                     fontweight='bold', color='white')
    axes[1].set_title('Jumlah Tweet per Sentimen', fontsize=12, color='white')
    axes[1].set_xlabel('Sentimen', color='white')
    axes[1].set_ylabel('Jumlah Tweet', color='white')
    axes[1].tick_params(colors='white')
    axes[1].set_ylim(0, max(counts) * 1.2)
    axes[1].spines[['top', 'right']].set_visible(False)
    for sp in ['left', 'bottom']:
        axes[1].spines[sp].set_color('#444')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Viz 2: Word Cloud
    st.markdown("#### Word Cloud per Sentimen")
    cmap_list  = ['Greens', 'Reds', 'Blues']
    emoji_list = ['😊', '😡', '😐']
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.patch.set_facecolor('#0E1117')

    for ax, sent, cmap, emoji in zip(axes2, sentimen_list, cmap_list, emoji_list):
        text = ' '.join(df[df['sentimen'] == sent]['tweet_bersih'])
        if len(text.strip()) == 0:
            ax.text(0.5, 0.5, 'Data kosong', ha='center', va='center', color='white')
            continue
        wc = WordCloud(
            width=600, height=350, background_color='#0E1117',
            colormap=cmap, max_words=80, prefer_horizontal=0.7, random_state=42
        ).generate(text)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{emoji} {sent.upper()}', fontsize=13, fontweight='bold', color='white', pad=10)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Viz 3: Top 15 Kata
    st.markdown("#### Top 15 Kata per Sentimen")
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
    fig3.patch.set_facecolor('#0E1117')

    for ax, sent, emoji in zip(axes3, sentimen_list, emoji_list):
        ax.set_facecolor('#0E1117')
        text   = ' '.join(df[df['sentimen'] == sent]['tweet_bersih'])
        cnts   = Counter(text.split()).most_common(15)
        if not cnts:
            continue
        words, freqs = zip(*cnts)
        color = PALETTE[sent]
        ax.barh(words[::-1], freqs[::-1], color=color, alpha=0.85, edgecolor='white')
        ax.set_title(f'{emoji} {sent.upper()}', fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel('Frekuensi', color='white')
        ax.tick_params(colors='white')
        ax.spines[['top', 'right']].set_visible(False)
        for sp in ['left', 'bottom']:
            ax.spines[sp].set_color('#444')

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Viz 4: Panjang Tweet
    st.markdown("#### Distribusi Panjang Tweet: Sebelum vs Sesudah Preprocessing")
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    fig4.patch.set_facecolor('#0E1117')

    for sent in sentimen_list:
        sub = df[df['sentimen'] == sent]
        axes4[0].hist(sub['panjang_sebelum'], bins=30, alpha=0.5,
                      color=PALETTE[sent], label=sent.capitalize(), edgecolor='white')
        axes4[1].hist(sub['panjang_sesudah'], bins=30, alpha=0.5,
                      color=PALETTE[sent], label=sent.capitalize(), edgecolor='white')

    for ax, title in zip(axes4, ['Sebelum Preprocessing', 'Sesudah Preprocessing']):
        ax.set_facecolor('#0E1117')
        ax.set_title(title, fontsize=11, color='white')
        ax.set_xlabel('Jumlah Kata', color='white')
        ax.set_ylabel('Frekuensi', color='white')
        ax.tick_params(colors='white')
        ax.legend(fontsize=9, labelcolor='white', facecolor='#1a1a2e')
        ax.spines[['top', 'right']].set_visible(False)
        for sp in ['left', 'bottom']:
            ax.spines[sp].set_color('#444')

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    # Viz 5: Trend Waktu
    st.markdown("#### Trend Volume Tweet per Bulan")
    df['tanggal'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
    df_time = df.dropna(subset=['tanggal']).copy()
    df_time['bulan'] = df_time['tanggal'].dt.to_period('M')
    trend = df_time.groupby(['bulan', 'sentimen']).size().unstack(fill_value=0)

    if not trend.empty:
        fig5, ax5 = plt.subplots(figsize=(13, 5))
        fig5.patch.set_facecolor('#0E1117')
        ax5.set_facecolor('#0E1117')
        color_map = {s: PALETTE[s] for s in sentimen_list if s in trend.columns}
        trend.plot(kind='bar', ax=ax5, color=color_map, width=0.75, edgecolor='white')
        ax5.set_title('Volume Tweet per Sentimen per Bulan', fontsize=13, fontweight='bold', color='white')
        ax5.set_xlabel('Bulan', color='white')
        ax5.set_ylabel('Jumlah Tweet', color='white')
        ax5.tick_params(colors='white')
        ax5.legend(title='Sentimen', fontsize=9, labelcolor='white',
                   facecolor='#1a1a2e', title_fontsize=9)
        ax5.set_xticklabels([str(p) for p in trend.index], rotation=45, ha='right', color='white')
        ax5.spines[['top', 'right']].set_visible(False)
        for sp in ['left', 'bottom']:
            ax5.spines[sp].set_color('#444')
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()
    else:
        st.info("Data tanggal tidak tersedia untuk visualisasi trend.")

# ══════════════════════════════════════════════
# TAB 3: MODEL & EVALUASI
# ══════════════════════════════════════════════
with tab3:
    st.subheader("🤖 Training & Evaluasi Model")

    with st.spinner("🔄 Melatih model... (pertama kali butuh beberapa saat)"):
        results = train_models(df)

    acc_nb = results['acc_nb']
    acc_lr = results['acc_lr']
    le     = results['le']

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🟢 Naïve Bayes — Akurasi", f"{acc_nb*100:.2f}%")
    with col2:
        st.metric("🔵 Logistic Regression — Akurasi", f"{acc_lr*100:.2f}%",
                  delta=f"{(acc_lr-acc_nb)*100:+.2f}% vs NB")

    st.divider()

    col_nb, col_lr = st.columns(2)

    with col_nb:
        st.markdown("#### Naïve Bayes — Classification Report")
        report_nb_df = pd.DataFrame(results['report_nb']).T.round(4)
        st.dataframe(report_nb_df, use_container_width=True)

        fig_cm1, ax_cm1 = plt.subplots(figsize=(6, 4))
        fig_cm1.patch.set_facecolor('#0E1117')
        ax_cm1.set_facecolor('#0E1117')
        ConfusionMatrixDisplay(results['cm_nb'], display_labels=le.classes_).plot(
            ax=ax_cm1, cmap='Greens', colorbar=False
        )
        ax_cm1.set_title(f'Confusion Matrix — Naïve Bayes\nAkurasi: {acc_nb*100:.2f}%',
                         fontsize=11, fontweight='bold', color='white')
        ax_cm1.tick_params(colors='white')
        ax_cm1.xaxis.label.set_color('white')
        ax_cm1.yaxis.label.set_color('white')
        plt.tight_layout()
        st.pyplot(fig_cm1)
        plt.close()

    with col_lr:
        st.markdown("#### Logistic Regression — Classification Report")
        report_lr_df = pd.DataFrame(results['report_lr']).T.round(4)
        st.dataframe(report_lr_df, use_container_width=True)

        fig_cm2, ax_cm2 = plt.subplots(figsize=(6, 4))
        fig_cm2.patch.set_facecolor('#0E1117')
        ax_cm2.set_facecolor('#0E1117')
        ConfusionMatrixDisplay(results['cm_lr'], display_labels=le.classes_).plot(
            ax=ax_cm2, cmap='Blues', colorbar=False
        )
        ax_cm2.set_title(f'Confusion Matrix — Logistic Regression\nAkurasi: {acc_lr*100:.2f}%',
                         fontsize=11, fontweight='bold', color='white')
        ax_cm2.tick_params(colors='white')
        ax_cm2.xaxis.label.set_color('white')
        ax_cm2.yaxis.label.set_color('white')
        plt.tight_layout()
        st.pyplot(fig_cm2)
        plt.close()

# ══════════════════════════════════════════════
# TAB 4: PERBANDINGAN MODEL
# ══════════════════════════════════════════════
with tab4:
    if 'results' not in dir():
        with st.spinner("🔄 Melatih model..."):
            results = train_models(df)

    st.subheader("📈 Perbandingan Performa Model")

    df_compare = results['df_compare']
    st.dataframe(
        df_compare.style.highlight_max(axis=0, color='#1a4731').format("{:.4f}"),
        use_container_width=True
    )

    st.divider()

    # Bar Chart Perbandingan
    metrics_names = list(df_compare.columns)
    nb_vals = df_compare.loc['Naïve Bayes'].values
    lr_vals = df_compare.loc['Logistic Regression'].values
    x = np.arange(len(metrics_names))
    width = 0.35

    fig_cmp, ax_cmp = plt.subplots(figsize=(11, 5))
    fig_cmp.patch.set_facecolor('#0E1117')
    ax_cmp.set_facecolor('#0E1117')

    bars1 = ax_cmp.bar(x - width/2, nb_vals, width, label='Naïve Bayes',
                       color='#27AE60', alpha=0.85, edgecolor='white')
    bars2 = ax_cmp.bar(x + width/2, lr_vals, width, label='Logistic Regression',
                       color='#2980B9', alpha=0.85, edgecolor='white')

    for bar in list(bars1) + list(bars2):
        ax_cmp.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=9, color='white')

    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels(metrics_names, fontsize=11, color='white')
    ax_cmp.set_ylim(0, 1.12)
    ax_cmp.set_ylabel('Score', color='white')
    ax_cmp.tick_params(colors='white')
    ax_cmp.set_title('Perbandingan Performa: Naïve Bayes vs Logistic Regression',
                     fontsize=13, fontweight='bold', color='white')
    ax_cmp.legend(fontsize=10, labelcolor='white', facecolor='#1a1a2e')
    ax_cmp.axhline(y=0.8, color='gray', linestyle='--', alpha=0.4)
    ax_cmp.spines[['top', 'right']].set_visible(False)
    for sp in ['left', 'bottom']:
        ax_cmp.spines[sp].set_color('#444')

    plt.tight_layout()
    st.pyplot(fig_cmp)
    plt.close()

    # Top Fitur Logistic Regression
    st.divider()
    st.markdown("#### Top Fitur Berpengaruh — Logistic Regression")
    tfidf     = results['tfidf']
    lr_model  = results['lr_model']
    le        = results['le']
    feature_names = np.array(tfidf.get_feature_names_out())
    n_top = 12

    fig_feat, axes_feat = plt.subplots(1, 3, figsize=(18, 5))
    fig_feat.patch.set_facecolor('#0E1117')
    fig_feat.suptitle('Top Fitur Paling Berpengaruh per Kelas — Logistic Regression',
                      fontsize=13, fontweight='bold', color='white')

    for i, (sent, color) in enumerate(zip(le.classes_, [PALETTE.get(s, '#888') for s in le.classes_])):
        axes_feat[i].set_facecolor('#0E1117')
        coef    = lr_model.coef_[i]
        top_idx = np.argsort(coef)[-n_top:][::-1]
        axes_feat[i].barh(feature_names[top_idx][::-1], coef[top_idx][::-1],
                          color=color, alpha=0.85, edgecolor='white')
        axes_feat[i].set_title(sent.capitalize(), fontsize=11, fontweight='bold', color='white')
        axes_feat[i].set_xlabel('Koefisien', color='white')
        axes_feat[i].tick_params(colors='white')
        axes_feat[i].spines[['top', 'right']].set_visible(False)
        for sp in ['left', 'bottom']:
            axes_feat[i].spines[sp].set_color('#444')

    plt.tight_layout()
    st.pyplot(fig_feat)
    plt.close()

    # Ringkasan
    st.divider()
    acc_nb  = results['acc_nb']
    acc_lr  = results['acc_lr']
    winner  = 'Logistic Regression' if acc_lr >= acc_nb else 'Naïve Bayes'
    winner_acc = max(acc_lr, acc_nb)

    n_pos = (df['sentimen'] == 'positif').sum()
    n_neg = (df['sentimen'] == 'negatif').sum()
    n_net = (df['sentimen'] == 'netral').sum()

    st.markdown(f"""
    ### 📊 Ringkasan Hasil Analisis Sentimen PPKM

    | Keterangan | Nilai |
    |---|---|
    | Total Tweet Dianalisis | **{len(df):,}** |
    | 😊 Positif | **{n_pos:,}** ({n_pos/len(df)*100:.1f}%) |
    | 😡 Negatif | **{n_neg:,}** ({n_neg/len(df)*100:.1f}%) |
    | 😐 Netral  | **{n_net:,}** ({n_net/len(df)*100:.1f}%) |
    | Akurasi Naïve Bayes | **{acc_nb*100:.2f}%** |
    | Akurasi Logistic Regression | **{acc_lr*100:.2f}%** |
    | 🏆 Model Terbaik | **{winner}** ({winner_acc*100:.2f}%) |
    """)

# ══════════════════════════════════════════════
# TAB 5: DEMO PREDIKSI
# ══════════════════════════════════════════════
with tab5:
    if 'results' not in dir():
        with st.spinner("🔄 Melatih model..."):
            results = train_models(df)

    st.subheader("🔮 Demo Prediksi Real-Time")

    acc_nb  = results['acc_nb']
    acc_lr  = results['acc_lr']
    tfidf   = results['tfidf']
    le      = results['le']
    nb_model = results['nb_model']
    lr_model = results['lr_model']

    best_model = lr_model if acc_lr >= acc_nb else nb_model
    model_name = 'Logistic Regression' if acc_lr >= acc_nb else 'Naïve Bayes'
    st.info(f"🏆 Menggunakan model terbaik: **{model_name}** (akurasi {max(acc_nb, acc_lr)*100:.2f}%)")

    user_input = st.text_area(
        "Masukkan teks tweet tentang PPKM:",
        placeholder="Contoh: PPKM sangat membantu mengurangi penyebaran COVID di Indonesia",
        height=100
    )

    if st.button("🔍 Prediksi Sentimen", type="primary"):
        if user_input.strip():
            clean = full_preprocess(user_input, stemmer, stopword_list)
            vec   = tfidf.transform([clean])
            pred  = best_model.predict(vec)[0]
            proba = best_model.predict_proba(vec)[0]
            label = le.inverse_transform([pred])[0]
            conf  = max(proba)
            emoji = '😊' if label == 'positif' else '😡' if label == 'negatif' else '😐'

            color_map_result = {'positif': 'success', 'negatif': 'error', 'netral': 'info'}
            getattr(st, color_map_result[label])(
                f"{emoji} **Sentimen: {label.upper()}** — Confidence: {conf:.2%}"
            )

            st.markdown("**Detail probabilitas:**")
            proba_df = pd.DataFrame({
                'Sentimen': le.classes_,
                'Probabilitas': proba
            }).sort_values('Probabilitas', ascending=False)
            st.dataframe(proba_df.style.format({'Probabilitas': '{:.4f}'}),
                         use_container_width=True, hide_index=True)

            st.caption(f"Teks setelah preprocessing: `{clean}`")
        else:
            st.warning("⚠️ Masukkan teks terlebih dahulu.")

    st.divider()
    st.markdown("#### Contoh Prediksi Batch")
    test_inputs = [
        "PPKM sangat membantu mengurangi penyebaran COVID di Indonesia",
        "PPKM membuat usaha saya bangkrut, pemerintah tidak peduli rakyat kecil",
        "Pemerintah resmi umumkan perpanjangan PPKM level 2 hingga akhir bulan",
        "Kebijakan PPKM sudah tepat, kasus COVID turun drastis akhirnya",
        "Susah cari makan gara-gara PPKM, warung dilarang buka malam",
    ]

    hasil = []
    for tweet in test_inputs:
        clean = full_preprocess(tweet, stemmer, stopword_list)
        vec   = tfidf.transform([clean])
        pred  = best_model.predict(vec)[0]
        proba = best_model.predict_proba(vec)[0]
        label = le.inverse_transform([pred])[0]
        conf  = max(proba)
        emoji = '😊' if label == 'positif' else '😡' if label == 'negatif' else '😐'
        hasil.append({'Tweet': tweet, 'Prediksi': f"{emoji} {label.upper()}", 'Confidence': f"{conf:.2%}"})

    st.dataframe(pd.DataFrame(hasil), use_container_width=True, hide_index=True)
