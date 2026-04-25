"""
Netflix Content EDA Script
==========================
Performs full exploratory data analysis on a Netflix titles CSV dataset.

Requirements:
    pip install pandas matplotlib seaborn reportlab pillow

Input  (choose ONE):
    A) Set DATASET_URL to a raw CSV link  ← recommended
    B) Set CSV_PATH to a local file path

    Expected columns: type, release_year, duration, listed_in, rating, country, date_added

Output:
    page1.png            – Overview charts
    page2.png            – Distribution & deep-dive charts
    Netflix_EDA_Report.pdf
"""

import os
import warnings
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────

# ── Data source config ────────────────────────────────────────────────────────
#
# Option A – URL (script auto-downloads, tries each URL in order)
#   Best free source: Kaggle  →  kaggle.com/datasets/shivamb/netflix-shows
#   After downloading, set DATASET_URL = None and CSV_PATH = 'netflix_titles.csv'
#
# Option B – Local CSV file

DATASET_URLS = [
    # GitHub mirrors — one of these usually works from your machine
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/netflix_titles.csv",
    "https://raw.githubusercontent.com/nicholasgasior/netflix-data/master/netflix_titles.csv",
    "https://raw.githubusercontent.com/rahulchouhan/Netflix-EDA/main/netflix_titles.csv",
]

# Set to None to skip downloading and use local file below
DATASET_URL = DATASET_URLS[0]

# Local fallback path (used when DATASET_URL is None)
CSV_PATH = 'netflix_titles.csv'

OUT_PAGE1  = 'page1.png'
OUT_PAGE2  = 'page2.png'
OUT_PDF    = 'Netflix_EDA_Report.pdf'

# ── Palette ───────────────────────────────────────────────────────────────────
RED   = '#E50914'
GOLD  = '#F5C518'
GRAY  = '#808080'
LIGHT = '#F5F5F1'
BG    = '#1A1A1A'
CARD  = '#2A2A2A'

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor': CARD,
    'axes.edgecolor':  '#444','axes.labelcolor': LIGHT,
    'xtick.color':     GRAY,  'ytick.color':    GRAY,
    'text.color':      LIGHT, 'font.family':    'DejaVu Sans',
    'grid.color':      '#333','grid.alpha':      0.6,
})

# ══════════════════════════════════════════════════════════════════════════════
#  1. LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════

def download_csv(url, save_as: str = 'netflix_titles.csv') -> str:
    """
    Download a CSV from a URL (or list of URLs) and save locally.
    Tries each URL in order; raises RuntimeError if all fail.
    """
    urls = [url] if isinstance(url, str) else url
    last_error = None
    for u in urls:
        print("Trying: " + u)
        try:
            req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp, open(save_as, 'wb') as f:
                f.write(resp.read())
            size_kb = os.path.getsize(save_as) / 1024
            print(f"Downloaded -> {save_as}  ({size_kb:.1f} KB)")
            return save_as
        except Exception as e:
            print(f"  Failed: {e}")
            last_error = e
    raise RuntimeError(
        "All URLs failed. Please download the dataset manually from:\n"
        "  https://www.kaggle.com/datasets/shivamb/netflix-shows\n"
        "Then set DATASET_URL = None and CSV_PATH = 'netflix_titles.csv'"
    )


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Support both 'listed_in' and 'genre' column names
    if 'listed_in' not in df.columns and 'genre' in df.columns:
        df.rename(columns={'genre': 'listed_in'}, inplace=True)

    print(f"Loaded {len(df):,} rows  |  Columns: {list(df.columns)}")

    df['runtime_min'] = df['duration'].str.extract(r'(\d+)\s*min').astype(float)
    df['seasons']     = df['duration'].str.extract(r'(\d+)\s*Season').astype(float)
    df['genres_list'] = (df['listed_in']
                         .str.split(',')
                         .apply(lambda x: [g.strip() for g in x]))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  2. ANALYSE
# ══════════════════════════════════════════════════════════════════════════════

def analyse(df: pd.DataFrame) -> dict:
    df_ex = df.explode('genres_list')
    movie_genres = (df_ex[df_ex['type'] == 'Movie']['genres_list']
                    .value_counts().head(10))
    tv_genres    = (df_ex[df_ex['type'] == 'TV Show']['genres_list']
                    .value_counts().head(10))

    by_year = (df.groupby(['release_year', 'type'])
                 .size()
                 .unstack(fill_value=0))
    by_year = by_year[by_year.index >= 2008]

    total    = len(df)
    n_movies = int((df['type'] == 'Movie').sum())
    n_tv     = int((df['type'] == 'TV Show').sum())
    avg_rt   = df['runtime_min'].mean()
    avg_s    = df['seasons'].mean()
    peak_yr  = int(by_year.sum(axis=1).idxmax())

    print("=" * 50)
    print("NETFLIX CONTENT EDA – SUMMARY")
    print("=" * 50)
    print(f"  Total titles  : {total:,}")
    print(f"  Movies        : {n_movies:,}  ({n_movies/total*100:.1f}%)")
    print(f"  TV Shows      : {n_tv:,}  ({n_tv/total*100:.1f}%)")
    print(f"  Avg runtime   : {avg_rt:.0f} min")
    print(f"  Avg seasons   : {avg_s:.1f}")
    print(f"  Peak year     : {peak_yr}")
    print(f"  Top movie genre : {movie_genres.index[0]}")
    print(f"  Top TV genre    : {tv_genres.index[0]}")
    print("=" * 50)
    print("\nTop 10 Movie Genres:")
    print(movie_genres.to_string())
    print("\nTop 10 TV Genres:")
    print(tv_genres.to_string())
    print("\nTop 10 Release Years:")
    print(by_year.sum(axis=1).sort_values(ascending=False).head(10).to_string())

    return dict(df_ex=df_ex, movie_genres=movie_genres, tv_genres=tv_genres,
                by_year=by_year, cumulative=by_year.cumsum(),
                total=total, n_movies=n_movies, n_tv=n_tv,
                avg_rt=avg_rt, avg_s=avg_s, peak_yr=peak_yr)


# ══════════════════════════════════════════════════════════════════════════════
#  3. PAGE 1 – Overview
# ══════════════════════════════════════════════════════════════════════════════

def plot_page1(df: pd.DataFrame, stats: dict, out_path: str):
    mg        = stats['movie_genres']
    tg        = stats['tv_genres']
    by_year   = stats['by_year']
    cumulative= stats['cumulative']
    total     = stats['total']
    n_movies  = stats['n_movies']
    n_tv      = stats['n_tv']

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Netflix Content Analysis  ·  EDA Report',
                 fontsize=26, fontweight='bold', color=RED, y=0.98)

    # Donut
    ax = axes[0, 0]
    wedges, _, ats = ax.pie(
        [n_movies, n_tv], labels=['Movies', 'TV Shows'],
        autopct='%1.1f%%', colors=[RED, '#B81D24'], startangle=90,
        wedgeprops={'width': 0.5, 'edgecolor': BG, 'linewidth': 3},
        textprops={'color': LIGHT, 'fontsize': 13})
    for at in ats:
        at.set_color(LIGHT); at.set_fontsize(13); at.set_fontweight('bold')
    ax.text(0, 0, f'{total:,}\nTotal', ha='center', va='center',
            fontsize=14, fontweight='bold', color=LIGHT)
    ax.set_title('Content Type Split', fontsize=14, color=LIGHT, pad=12)

    # Annual area chart
    ax = axes[0, 1]
    years = by_year.index.astype(int)
    if 'Movie' in by_year:
        ax.fill_between(years, by_year['Movie'], alpha=0.7, color=RED, label='Movies')
    if 'TV Show' in by_year:
        ax.fill_between(years, by_year.get('TV Show', 0), alpha=0.7, color=GOLD, label='TV Shows')
    ax.set_xlabel('Release Year'); ax.set_ylabel('Titles Added')
    ax.set_title('Annual Content Released', fontsize=14, color=LIGHT)
    ax.legend(facecolor=CARD, edgecolor='#444'); ax.grid(axis='y')

    # Cumulative
    ax = axes[0, 2]
    if 'Movie' in cumulative:
        ax.plot(cumulative.index, cumulative['Movie'], color=RED, linewidth=3, label='Movies')
    if 'TV Show' in cumulative:
        ax.plot(cumulative.index, cumulative['TV Show'], color=GOLD, linewidth=3, label='TV Shows')
    ax.fill_between(cumulative.index, cumulative.sum(axis=1), alpha=0.08, color=RED)
    ax.set_xlabel('Year'); ax.set_ylabel('Cumulative Titles')
    ax.set_title('Cumulative Content Growth', fontsize=14, color=LIGHT)
    ax.legend(facecolor=CARD, edgecolor='#444'); ax.grid(axis='y')

    # Top 10 movie genres
    ax = axes[1, 0]
    bars = ax.barh(mg.index[::-1], mg.values[::-1],
                   color=[RED if i == 9 else '#8B0000' for i in range(10)])
    for bar, val in zip(bars, mg.values[::-1]):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=9, color=LIGHT)
    ax.set_title('Top 10 Movie Genres', fontsize=14, color=LIGHT)
    ax.set_xlabel('Count')

    # Top 10 TV genres
    ax = axes[1, 1]
    bars = ax.barh(tg.index[::-1], tg.values[::-1],
                   color=[GOLD if i == 9 else '#B8860B' for i in range(10)])
    for bar, val in zip(bars, tg.values[::-1]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=9, color=LIGHT)
    ax.set_title('Top 10 TV Show Genres', fontsize=14, color=LIGHT)
    ax.set_xlabel('Count')

    # Ratings distribution
    ax = axes[1, 2]
    rc = df['rating'].value_counts().head(8)
    colors_r = [RED, '#B81D24', '#8B0000', '#660000',
                '#440000', '#FF6B6B', '#FF4444', '#CC0000']
    ax.bar(rc.index, rc.values, color=colors_r[:len(rc)], edgecolor=BG, linewidth=0.5)
    ax.set_title('Content Rating Distribution', fontsize=14, color=LIGHT)
    ax.set_xlabel('Rating'); ax.set_ylabel('Count'); ax.grid(axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  4. PAGE 2 – Distributions & Deep Dives
# ══════════════════════════════════════════════════════════════════════════════

def plot_page2(df: pd.DataFrame, stats: dict, out_path: str):
    mg       = stats['movie_genres']
    df_ex    = stats['df_ex']
    by_year  = stats['by_year']
    peak_yr  = stats['peak_yr']
    avg_rt   = stats['avg_rt']
    avg_s    = stats['avg_s']
    total    = stats['total']
    n_movies = stats['n_movies']
    n_tv     = stats['n_tv']

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Netflix Deep Dive  ·  Distributions & Trends',
                 fontsize=26, fontweight='bold', color=RED, y=0.98)

    # Runtime histogram
    ax = axes[0, 0]
    rt = df[df['type'] == 'Movie']['runtime_min'].dropna()
    ax.hist(rt, bins=40, color=RED, alpha=0.85, edgecolor=BG)
    ax.axvline(rt.mean(), color=GOLD, linewidth=2.5, linestyle='--',
               label=f'Mean: {rt.mean():.0f} min')
    ax.axvline(rt.median(), color='white', linewidth=2, linestyle=':',
               label=f'Median: {rt.median():.0f} min')
    ax.set_title('Movie Runtime Distribution', fontsize=14, color=LIGHT)
    ax.set_xlabel('Runtime (minutes)'); ax.set_ylabel('Count')
    ax.legend(facecolor=CARD, edgecolor='#444'); ax.grid(axis='y')

    # Seasons bar
    ax = axes[0, 1]
    sea = df[df['type'] == 'TV Show']['seasons'].dropna()
    cnt = sea.value_counts().sort_index()
    ax.bar(cnt.index.astype(int), cnt.values, color=GOLD, alpha=0.85, edgecolor=BG)
    ax.set_title('TV Show Seasons Distribution', fontsize=14, color=LIGHT)
    ax.set_xlabel('Number of Seasons'); ax.set_ylabel('Count')
    ax.set_xticks(range(1, int(cnt.index.max()) + 1)); ax.grid(axis='y')

    # Top 10 years
    ax = axes[0, 2]
    top_years = by_year.sum(axis=1).sort_values(ascending=False).head(10).sort_index()
    colors_y  = [RED if y == peak_yr else '#8B0000' for y in top_years.index]
    ax.bar(top_years.index.astype(str), top_years.values, color=colors_y, edgecolor=BG)
    ax.set_title('Top 10 Release Years', fontsize=14, color=LIGHT)
    ax.set_xlabel('Year'); ax.set_ylabel('Titles')
    for i, v in enumerate(top_years.values):
        ax.text(i, v + 15, str(v), ha='center', fontsize=9, color=LIGHT)
    ax.grid(axis='y')

    # Boxplot runtime by genre
    ax = axes[1, 0]
    top5 = mg.index[:6].tolist()
    data_bp = [
        df_ex[(df_ex['genres_list'] == g) & (df_ex['type'] == 'Movie')]['runtime_min'].dropna().values
        for g in top5]
    bp = ax.boxplot(data_bp, patch_artist=True,
                    medianprops={'color': LIGHT, 'linewidth': 2})
    for patch in bp['boxes']:
        patch.set_facecolor(RED); patch.set_alpha(0.7)
    for w in bp['whiskers']: w.set_color(GRAY)
    for c in bp['caps']:     c.set_color(GRAY)
    ax.set_xticklabels(
        [g.replace(' Movies', '').replace(' & ', '\n& ')[:14] for g in top5], fontsize=8)
    ax.set_title('Movie Runtime by Genre', fontsize=14, color=LIGHT)
    ax.set_ylabel('Runtime (min)'); ax.grid(axis='y')

    # Country pie
    ax = axes[1, 1]
    tc = df['country'].value_counts().head(8)
    _, texts, ats = ax.pie(
        tc.values, labels=tc.index, autopct='%1.1f%%', startangle=140,
        colors=plt.cm.Reds(np.linspace(0.4, 0.9, len(tc))),
        wedgeprops={'edgecolor': BG, 'linewidth': 2},
        textprops={'color': LIGHT, 'fontsize': 9})
    for at in ats: at.set_fontsize(8); at.set_color(LIGHT)
    ax.set_title('Top Countries (Content Origin)', fontsize=14, color=LIGHT)

    # KPI panel
    ax = axes[1, 2]
    ax.set_facecolor(CARD); ax.axis('off')
    top_mg = stats['movie_genres'].index[0]
    top_tg = stats['tv_genres'].index[0]
    kpis = [
        ('Total Titles',     f'{total:,}'),
        ('Movies',           f'{n_movies:,}  ({n_movies/total*100:.1f}%)'),
        ('TV Shows',         f'{n_tv:,}  ({n_tv/total*100:.1f}%)'),
        ('Avg Runtime',      f'{avg_rt:.0f} min'),
        ('Avg Seasons',      f'{avg_s:.1f}'),
        ('Peak Year',        str(peak_yr)),
        ('Top Movie Genre',  top_mg[:20]),
        ('Top TV Genre',     top_tg[:20]),
    ]
    for i, (label, val) in enumerate(kpis):
        y = 0.90 - i * 0.108
        ax.text(0.05, y, label, transform=ax.transAxes, fontsize=10.5,
                color=GRAY, va='center')
        ax.text(0.95, y, val, transform=ax.transAxes, fontsize=11,
                color=RED, fontweight='bold', ha='right', va='center')
        if i < len(kpis) - 1:
            ax.plot([0.05, 0.95], [y - 0.05, y - 0.05], color='#333',
                    linewidth=0.5, transform=ax.transAxes)
    ax.set_title('Key Metrics Summary', fontsize=14, color=LIGHT, pad=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  5. BUILD PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════

def build_pdf(stats: dict, page1: str, page2: str, out_path: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Image, Spacer,
                                    Paragraph, Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from PIL import Image as PILImage

    RED_RL   = colors.HexColor('#E50914')
    LIGHT_RL = colors.HexColor('#F5F5F1')
    GRAY_RL  = colors.HexColor('#808080')
    BG_RL    = colors.HexColor('#1A1A1A')

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.5*cm,  bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()

    def ps(name, **kw):
        return ParagraphStyle(name, parent=styles['Normal'], **kw)

    title_s = ps('T', fontSize=28, textColor=RED_RL, fontName='Helvetica-Bold',
                 spaceAfter=6, leading=34)
    sub_s   = ps('S', fontSize=12, textColor=GRAY_RL, fontName='Helvetica', spaceAfter=14)
    h2_s    = ps('H2', fontSize=16, textColor=RED_RL, fontName='Helvetica-Bold',
                 spaceBefore=10, spaceAfter=6)
    body_s  = ps('B', fontSize=10, textColor=LIGHT_RL, fontName='Helvetica',
                 leading=16, spaceAfter=8)

    def img_f(path, width_cm=17):
        img = PILImage.open(path)
        iw, ih = img.size
        w = width_cm * cm
        return Image(path, width=w, height=w * ih / iw)

    total    = stats['total']
    n_movies = stats['n_movies']
    n_tv     = stats['n_tv']
    avg_rt   = stats['avg_rt']
    avg_s    = stats['avg_s']
    peak_yr  = stats['peak_yr']
    top_mg   = stats['movie_genres'].index[0]
    top_tg   = stats['tv_genres'].index[0]

    kpi_data = [
        ['Metric', 'Value'],
        ['Total Titles',     f'{total:,}'],
        ['Movies',           f'{n_movies:,}  ({n_movies/total*100:.1f}%)'],
        ['TV Shows',         f'{n_tv:,}  ({n_tv/total*100:.1f}%)'],
        ['Avg Movie Runtime',f'{avg_rt:.0f} min'],
        ['Avg TV Seasons',   f'{avg_s:.1f}'],
        ['Peak Release Year',str(peak_yr)],
        ['Top Movie Genre',  top_mg],
        ['Top TV Genre',     top_tg],
    ]
    kpi_tbl = Table(kpi_data, colWidths=[8*cm, 9*cm])
    kpi_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0),  RED_RL),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  LIGHT_RL),
        ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0),  11),
        ('BACKGROUND',    (0, 1), (-1, -1), colors.HexColor('#2A2A2A')),
        ('TEXTCOLOR',     (0, 1), (-1, -1), LIGHT_RL),
        ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1),
             [colors.HexColor('#222222'), colors.HexColor('#2A2A2A')]),
        ('GRID',          (0, 0), (-1, -1), 0.5, colors.HexColor('#444444')),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))

    story = [
        Spacer(1, 1.5*cm),
        Paragraph('Netflix Content EDA', title_s),
        Paragraph(f'Exploratory Data Analysis Report — {total:,} titles', sub_s),
        HRFlowable(width='100%', thickness=2, color=RED_RL, spaceAfter=14),
        kpi_tbl,
        Spacer(1, 0.5*cm),
        Paragraph('Executive Summary', h2_s),
        Paragraph(
            f'This report analyses {total:,} Netflix titles from 2008–2023. '
            f'Movies dominate at {n_movies/total*100:.1f}%, TV Shows at {n_tv/total*100:.1f}%. '
            f'Content volumes accelerated from 2016 onward, peaking in {peak_yr}. '
            f'Drama and International content lead all genre categories.',
            body_s),
        HRFlowable(width='100%', thickness=1, color=colors.HexColor('#444'), spaceAfter=10),
        Paragraph('Overview Charts', h2_s),
        img_f(page1),
        Spacer(1, 0.4*cm),
        Paragraph(
            'Figure 1: Type split (donut), annual releases (area), cumulative growth, '
            'top-10 genres for movies and TV, and rating distribution.',
            body_s),
        HRFlowable(width='100%', thickness=1, color=colors.HexColor('#444'), spaceAfter=10),
        Paragraph('Distributions & Deep Dives', h2_s),
        img_f(page2),
        Spacer(1, 0.4*cm),
        Paragraph(
            'Figure 2: Movie runtime histogram, TV seasons distribution, top release years, '
            'runtime by genre (boxplot), content origin by country, and key KPI summary.',
            body_s),
        HRFlowable(width='100%', thickness=2, color=RED_RL, spaceBefore=14, spaceAfter=6),
        Paragraph('Generated with Python · pandas · matplotlib · reportlab', sub_s),
    ]

    doc.build(story)
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ── Resolve data source ───────────────────────────────────────────────────
    if DATASET_URL:
        csv_path = download_csv(DATASET_URLS, save_as='netflix_titles.csv')
    else:
        csv_path = CSV_PATH

    print("\nLoading data...")
    df = load_data(csv_path)

    print("\nRunning analysis...")
    stats = analyse(df)

    print("\nGenerating plots...")
    plot_page1(df, stats, OUT_PAGE1)
    plot_page2(df, stats, OUT_PAGE2)

    print("\nBuilding PDF report...")
    build_pdf(stats, OUT_PAGE1, OUT_PAGE2, OUT_PDF)

    print("\nDone! Files created:")
    for f in [OUT_PAGE1, OUT_PAGE2, OUT_PDF]:
        size = os.path.getsize(f) / 1024
        print(f"  {f}  ({size:.1f} KB)")