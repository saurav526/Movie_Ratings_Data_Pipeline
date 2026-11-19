"""
CINEPULSE // Movie Ratings Data Pipeline & Audience Analytics Studio
A creative Peach & Magenta themed Streamlit Analytics Dashboard powered by SQL, Pandas, and Plotly.
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.etl_pipeline import run_etl_pipeline
from src.data_generator import generate_raw_datasets
from src.database import (
    ensure_database,
    get_pipeline_audit,
    get_filter_bounds,
    query_filtered_movies,
    query_genre_aggregation,
    query_ratings_for_movies,
    get_raw_datasets_preview,
    execute_custom_sql,
    PREBUILT_SQL_QUERIES,
)

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CinePulse | Movie Ratings Data Pipeline & Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette Constants (Peach & Magenta Studio Theme)
PEACH_LIGHT = "#FFD1B3"
PEACH_CORE = "#FF9E7D"
PEACH_CORAL = "#FF6B8B"
MAGENTA_BRIGHT = "#FF2A85"
MAGENTA_DEEP = "#C81D77"
VIOLET_DARK = "#1D102B"
BG_DARK = "#11081A"
PEACH_CREAM = "#FFE5D9"

PEACH_MAGENTA_SCALE = [
    "#2E1235",
    "#6B1858",
    "#A61B6E",
    "#D9237E",
    "#FF2A85",
    "#FF5E7E",
    "#FF8E77",
    "#FFB88C",
    "#FFE0B5",
]

# ------------------------------------------------------------------------------
# CUSTOM CSS (PEACH & MAGENTA GLASSMORPHISM + NEON ACCENTS)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main App Background with subtle Peach-Magenta radial glow */
    .stApp {
        background: radial-gradient(circle at 12% 15%, rgba(255, 42, 133, 0.16) 0%, transparent 42%),
                    radial-gradient(circle at 88% 20%, rgba(255, 158, 125, 0.15) 0%, transparent 45%),
                    radial-gradient(circle at 50% 90%, rgba(200, 29, 119, 0.14) 0%, transparent 50%),
                    #11081A;
        color: #FFE5D9;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B0D29 0%, #13091E 100%) !important;
        border-right: 1px solid rgba(255, 158, 125, 0.25);
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(255, 42, 133, 0.22) 0%, rgba(255, 158, 125, 0.20) 55%, rgba(29, 16, 43, 0.90) 100%);
        border: 1px solid rgba(255, 158, 125, 0.35);
        border-radius: 20px;
        padding: 26px 32px;
        margin-bottom: 22px;
        box-shadow: 0 12px 35px rgba(255, 42, 133, 0.15);
        position: relative;
        overflow: hidden;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(90deg, #FF2A85 0%, #FF8E77 100%);
        color: #11081A;
        font-weight: 800;
        font-size: 0.76rem;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: 999px;
        margin-bottom: 10px;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.35rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        background: linear-gradient(90deg, #FFE5D9 0%, #FFAE8F 45%, #FF2A85 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #F4CBB8;
        font-size: 1.02rem;
        margin: 0;
        opacity: 0.92;
    }

    /* Glassmorphic KPI Cards */
    .kpi-card {
        background: linear-gradient(145deg, rgba(35, 17, 52, 0.88) 0%, rgba(24, 11, 36, 0.92) 100%);
        border: 1px solid rgba(255, 158, 125, 0.28);
        border-top: 3px solid #FF2A85;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.32);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: #FF9E7D;
    }
    .kpi-label {
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        color: #FFB699;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.95rem;
        font-weight: 800;
        color: #FFF0E8;
        line-height: 1.15;
    }
    .kpi-sub {
        font-size: 0.80rem;
        color: #FF75A0;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Section Header */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: #FFD1B3;
        border-left: 4px solid #FF2A85;
        padding-left: 12px;
        margin: 18px 0 12px 0;
    }

    /* Cinema Poster Cards for Top Ranked Movies */
    .movie-card {
        background: linear-gradient(155deg, rgba(43, 19, 63, 0.92) 0%, rgba(23, 11, 36, 0.96) 100%);
        border: 1px solid rgba(255, 158, 125, 0.30);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
        position: relative;
    }
    .movie-rank-badge {
        display: inline-block;
        background: linear-gradient(90deg, #FF2A85, #FF8E77);
        color: #11081A;
        font-weight: 800;
        font-size: 0.82rem;
        padding: 3px 11px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .movie-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.22rem;
        font-weight: 700;
        color: #FFE5D9;
        margin: 2px 0 4px 0;
    }
    .movie-meta {
        font-size: 0.83rem;
        color: #FFB899;
        margin-bottom: 8px;
    }
    .genre-pill {
        display: inline-block;
        background: rgba(255, 42, 133, 0.18);
        border: 1px solid rgba(255, 42, 133, 0.45);
        color: #FFC4B0;
        font-size: 0.73rem;
        font-weight: 600;
        padding: 2px 9px;
        border-radius: 999px;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .score-box {
        background: rgba(255, 158, 125, 0.14);
        border: 1px solid rgba(255, 158, 125, 0.4);
        border-radius: 10px;
        padding: 8px 12px;
        text-align: center;
    }

    /* Pipeline Step Card */
    .pipeline-step {
        background: rgba(33, 16, 49, 0.85);
        border: 1px solid rgba(255, 42, 133, 0.35);
        border-radius: 14px;
        padding: 15px 18px;
        margin-bottom: 12px;
    }
    .pipeline-step-title {
        font-weight: 700;
        color: #FF9E7D;
        font-size: 1.0rem;
        margin-bottom: 4px;
    }

    /* Tabs Customization */
    button[data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        color: #FFD1B3 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FF2A85 !important;
        border-bottom-color: #FF2A85 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_plotly_theme(fig, title: str = "", height: int = 410):
    """Applies a cohesive Peach & Magenta dark studio theme to all Plotly figures."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Space Grotesk, sans-serif", size=17, color="#FFE5D9"),
            x=0.01,
        ),
        height=height,
        paper_bgcolor="rgba(26, 13, 38, 0.65)",
        plot_bgcolor="rgba(17, 8, 26, 0.45)",
        font=dict(family="Outfit, sans-serif", color="#FFD1B3", size=12),
        margin=dict(l=24, r=24, t=54, b=28),
        legend=dict(
            bgcolor="rgba(29, 16, 43, 0.7)",
            bordercolor="rgba(255, 158, 125, 0.25)",
            borderwidth=1,
            font=dict(color="#FFE5D9"),
        ),
        xaxis=dict(
            gridcolor="rgba(255, 158, 125, 0.10)",
            zerolinecolor="rgba(255, 42, 133, 0.25)",
        ),
        yaxis=dict(
            gridcolor="rgba(255, 158, 125, 0.10)",
            zerolinecolor="rgba(255, 42, 133, 0.25)",
        ),
    )
    return fig


# ------------------------------------------------------------------------------
# INITIALIZE DATABASE & LOAD METADATA
# ------------------------------------------------------------------------------
ensure_database()
bounds = get_filter_bounds()
audit_stats = get_pipeline_audit()

# ------------------------------------------------------------------------------
# SIDEBAR: GLOBAL FILTERS & PIPELINE CONTROLS
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 8px 0 14px 0;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:800;
                        background: linear-gradient(90deg, #FF9E7D, #FF2A85);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🎬 CINEPULSE STUDIO
            </div>
            <div style="font-size:0.78rem; color:#FFB899; letter-spacing:1px;">
                Movie Ratings Data Pipeline & Analytics Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎛️ Audience & Catalog Filters")

    search_text = st.text_input(
        "🔍 Search Movie or Director",
        value="",
        placeholder="e.g. Dark Knight, Nolan, Godfather...",
    )

    selected_genres = st.multiselect(
        "🎭 Filter by Genre(s)",
        options=bounds["genres"],
        default=[],
        help="Leave empty to include all genres, or pick specific genres to slice the SQL warehouse.",
    )

    year_range = st.slider(
        "📅 Release Year Window",
        min_value=bounds["min_year"],
        max_value=bounds["max_year"],
        value=(bounds["min_year"], bounds["max_year"]),
    )

    min_rating = st.slider(
        "⭐ Minimum Weighted Rating",
        min_value=1.0,
        max_value=9.5,
        value=5.0,
        step=0.1,
        help="Filters by IMDB-style Bayesian Weighted Rating (1–10 scale).",
    )

    min_popularity = st.slider(
        "🔥 Minimum Popularity Index",
        min_value=0.0,
        max_value=90.0,
        value=0.0,
        step=2.5,
        help="Composite index (0-100) blending audience vote volume, rating score, and box office reach.",
    )

    min_votes = st.slider(
        "🗳️ Minimum Audience Votes",
        min_value=0,
        max_value=int(bounds["max_votes"]),
        value=10,
        step=5,
    )

    with st.expander("🌐 Advanced Filters (Language & Rating)"):
        selected_languages = st.multiselect(
            "Language",
            options=bounds["languages"],
            default=[],
        )
        selected_certificates = st.multiselect(
            "Content Certificate",
            options=bounds["certificates"],
            default=[],
        )

    st.markdown("---")
    st.markdown("### ⚙️ Data Pipeline Control")
    st.caption(
        f"**Warehouse:** `cinema_analytics.db` (SQLite)\n\n"
        f"**Last ETL Run:** `{audit_stats.get('run_timestamp', 'N/A')}`"
    )
    if st.button("🔄 Re-Run Full ETL Data Pipeline", use_container_width=True):
        with st.spinner("Regenerating raw datasets, deduplicating titles, imputing missing values, and rebuilding SQL tables..."):
            generate_raw_datasets()
            audit_stats = run_etl_pipeline()
        st.success("ETL Pipeline & SQL Database refreshed!")
        st.rerun()

# ------------------------------------------------------------------------------
# QUERY FILTERED DATASETS FROM SQL DATABASE
# ------------------------------------------------------------------------------
df_movies = query_filtered_movies(
    genres=selected_genres if selected_genres else None,
    year_range=year_range,
    min_rating=min_rating,
    min_popularity=min_popularity,
    min_votes=min_votes,
    languages=selected_languages if selected_languages else None,
    certificates=selected_certificates if selected_certificates else None,
    search_query=search_text,
)

filtered_movie_ids = df_movies["movie_id"].tolist()
df_genre_agg = query_genre_aggregation(filtered_movie_ids)
df_ratings_filtered = query_ratings_for_movies(filtered_movie_ids)

# ------------------------------------------------------------------------------
# HERO BANNER
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">PROJECT #7 • MOVIE RATINGS DATA PIPELINE & SQL ANALYTICS</div>
        <h1 class="hero-title">CinePulse: Audience Preference & Ratings Intelligence</h1>
        <p class="hero-subtitle">
                Streamlit dashboard for exploring movie ratings, audience preferences, and box office performance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# TOP-LEVEL KPI SUMMARY ROW
# ------------------------------------------------------------------------------
if df_movies.empty:
    st.warning("No movies match your current filter combination. Try loosening the sidebar filters.")
    st.stop()

total_movies = len(df_movies)
total_ratings = int(df_movies["rating_count"].sum())
mean_weighted = round(float(df_movies["weighted_score"].mean()), 2)
mean_pop = round(float(df_movies["popularity_index"].mean()), 1)
top_film = df_movies.iloc[0]
top_genre_name = df_genre_agg.iloc[0]["genre"] if not df_genre_agg.empty else "N/A"
top_genre_score = df_genre_agg.iloc[0]["avg_weighted_score"] if not df_genre_agg.empty else 0.0

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">🎬 Deduplicated Films</div>
            <div class="kpi-value">{total_movies:,}</div>
            <div class="kpi-sub">Merged {audit_stats['duplicate_movie_titles_merged']} duplicate raw titles</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">🗳️ Cleaned Audience Votes</div>
            <div class="kpi-value">{total_ratings:,}</div>
            <div class="kpi-sub">{audit_stats['ratings_remapped_from_duplicate_titles']} rescued via ID remapping</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">⭐ Mean Weighted Score</div>
            <div class="kpi-value">{mean_weighted:.2f} <span style="font-size:1rem;color:#FF9E7D;">/ 10</span></div>
            <div class="kpi-sub">Bayesian IMDB-Weighted Avg</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">🔥 Avg Popularity Index</div>
            <div class="kpi-value">{mean_pop:.1f} <span style="font-size:1rem;color:#FF9E7D;">pts</span></div>
            <div class="kpi-sub">Top Film: {top_film['title'][:18]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k5:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">🏆 Highest Rated Genre</div>
            <div class="kpi-value">{top_genre_name}</div>
            <div class="kpi-sub">Avg Score: {top_genre_score:.2f} ★ across genre</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# ------------------------------------------------------------------------------
tab_pulse, tab_top, tab_genre, tab_etl, tab_sql, tab_rec = st.tabs([
    "📊 Audience Pulse & Distributions",
    "🏆 Top-Rated Leaderboards",
    "🎭 Genre & Era Intelligence",
    "🛠️ Data Pipeline & Cleaning Inspector",
    "🗄️ SQL Warehouse Workbench",
    "✨ Peach-Magenta Discovery Engine",
])

# ==============================================================================
# TAB 1: AUDIENCE PULSE & DISTRIBUTIONS (Histograms, Bar Charts, Scatter)
# ==============================================================================
with tab_pulse:
    st.markdown('<div class="section-header">Histograms & Bar Charts of Audience Preferences</div>', unsafe_allow_html=True)

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        # Histogram 1: Distribution of Individual Audience Rating Scores
        fig_hist_ratings = px.histogram(
            df_ratings_filtered,
            x="rating",
            nbins=28,
            color="is_verified_viewer",
            barmode="overlay",
            opacity=0.85,
            color_discrete_map={1: MAGENTA_BRIGHT, 0: PEACH_CORE, True: MAGENTA_BRIGHT, False: PEACH_CORE},
            labels={"rating": "Audience Rating Score (1.0 - 10.0)", "is_verified_viewer": "Verified Viewer"},
        )
        mean_r = df_ratings_filtered["rating"].mean() if not df_ratings_filtered.empty else 0
        fig_hist_ratings.add_vline(
            x=mean_r,
            line_dash="dash",
            line_color=PEACH_LIGHT,
            annotation_text=f"Mean: {mean_r:.2f}★",
            annotation_font_color=PEACH_LIGHT,
        )
        apply_plotly_theme(fig_hist_ratings, "📊 Histogram: Audience Rating Score Distribution (All Logs)", height=400)
        st.plotly_chart(fig_hist_ratings, use_container_width=True)

    with col_h2:
        # Bar Chart 1: Average Weighted Score & Vote Volume by Genre
        if not df_genre_agg.empty:
            df_g_sorted = df_genre_agg.sort_values("avg_weighted_score", ascending=True)
            fig_bar_genre = go.Figure()
            fig_bar_genre.add_trace(
                go.Bar(
                    y=df_g_sorted["genre"],
                    x=df_g_sorted["avg_weighted_score"],
                    orientation="h",
                    marker=dict(
                        color=df_g_sorted["avg_weighted_score"],
                        colorscale=PEACH_MAGENTA_SCALE,
                        line=dict(color="rgba(255, 209, 179, 0.5)", width=1),
                    ),
                    text=df_g_sorted.apply(
                        lambda r: f" {r['avg_weighted_score']:.2f} ★ ({int(r['movie_count'])} films, {int(r['total_ratings']):,} votes)",
                        axis=1,
                    ),
                    textposition="inside",
                    insidetextanchor="start",
                    textfont=dict(color="#FFF5EE", size=11),
                    name="Weighted Score",
                )
            )
            fig_bar_genre.update_xaxes(range=[0, 10.2], title="Bayesian Weighted Rating (1-10)")
            apply_plotly_theme(fig_bar_genre, "🎭 Bar Chart: Audience Rating Preference by Genre", height=400)
            st.plotly_chart(fig_bar_genre, use_container_width=True)

    col_h3, col_h4 = st.columns(2)

    with col_h3:
        # Histogram 2: Movie Runtime & Popularity Distribution
        hist_metric = st.radio(
            "Select Movie Attribute for Histogram:",
            ["weighted_score", "popularity_index", "runtime_min", "roi_pct"],
            format_func=lambda x: {
                "weighted_score": "Movie Weighted Rating (★)",
                "popularity_index": "Popularity Index (0-100)",
                "runtime_min": "Movie Runtime (Minutes)",
                "roi_pct": "Return on Investment (ROI %)",
            }[x],
            horizontal=True,
        )
        fig_hist_movies = px.histogram(
            df_movies,
            x=hist_metric,
            nbins=20,
            color="audience_tier",
            color_discrete_sequence=[MAGENTA_BRIGHT, PEACH_CORAL, PEACH_CORE, "#9D4EDD"],
            marginal="box",
        )
        apply_plotly_theme(fig_hist_movies, f"📈 Histogram & Box Summary: {hist_metric.replace('_', ' ').title()}", height=410)
        st.plotly_chart(fig_hist_movies, use_container_width=True)

    with col_h4:
        # Scatter Plot: Popularity vs Weighted Score Bubble Matrix
        fig_scatter = px.scatter(
            df_movies,
            x="popularity_index",
            y="weighted_score",
            size="box_office_musd",
            color="primary_genre",
            hover_name="title",
            hover_data=["release_year", "director", "rating_count", "box_office_musd"],
            color_discrete_sequence=[
                "#FF2A85", "#FF9E7D", "#FF5E7E", "#FFD1B3", "#C81D77",
                "#FF75A0", "#FFA384", "#E040FB", "#FF80AB", "#FFCCBC"
            ],
            labels={
                "popularity_index": "Audience Popularity Index (0-100)",
                "weighted_score": "Bayesian Weighted Rating (★)",
                "primary_genre": "Primary Genre",
            },
        )
        apply_plotly_theme(fig_scatter, "✨ Audience Quadrant: Popularity vs. Weighted Score (Bubble = Box Office)", height=460)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Bar Chart 2: Ratings Breakdown by Viewer Age Group & Review Platform
    st.markdown('<div class="section-header">Audience Demographics & Platform Engagement</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        if not df_ratings_filtered.empty:
            age_grp = (
                df_ratings_filtered.groupby("viewer_age_group", as_index=False)
                .agg(avg_rating=("rating", "mean"), review_count=("rating_id", "count"))
                .sort_values("viewer_age_group")
            )
            fig_age = px.bar(
                age_grp,
                x="viewer_age_group",
                y="review_count",
                color="avg_rating",
                color_continuous_scale=PEACH_MAGENTA_SCALE,
                text=age_grp.apply(lambda r: f"{int(r['review_count']):,}<br>({r['avg_rating']:.2f}★)", axis=1),
                labels={"viewer_age_group": "Viewer Age Bracket", "review_count": "Total Ratings Submitted"},
            )
            fig_age.update_traces(textposition="outside")
            apply_plotly_theme(fig_age, "👥 Bar Chart: Ratings Volume & Avg Score by Viewer Age Group", height=370)
            st.plotly_chart(fig_age, use_container_width=True)

    with col_d2:
        if not df_ratings_filtered.empty:
            plat_grp = (
                df_ratings_filtered.groupby("review_platform", as_index=False)
                .agg(
                    avg_rating=("rating", "mean"),
                    review_count=("rating_id", "count"),
                    helpful_sum=("helpful_votes", "sum"),
                )
                .sort_values("review_count", ascending=False)
            )
            fig_plat = px.bar(
                plat_grp,
                x="review_platform",
                y="review_count",
                color="avg_rating",
                color_continuous_scale=PEACH_MAGENTA_SCALE,
                text=plat_grp.apply(lambda r: f"{r['avg_rating']:.2f} ★", axis=1),
                labels={"review_platform": "Review Source Platform", "review_count": "Ratings Count"},
            )
            fig_plat.update_traces(textposition="inside")
            apply_plotly_theme(fig_plat, "📱 Bar Chart: Rating Activity Across Streaming & Review Platforms", height=370)
            st.plotly_chart(fig_plat, use_container_width=True)


# ==============================================================================
# TAB 2: TOP-RATED MOVIE LEADERBOARDS & HALL OF FAME
# ==============================================================================
with tab_top:
    st.markdown('<div class="section-header">Top-Rated Cinema Leaderboard & Spotlight Cards</div>', unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        ranking_metric = st.selectbox(
            "Rank Leaderboard By:",
            options=["weighted_score", "popularity_index", "avg_rating", "rating_count", "box_office_musd", "roi_pct"],
            format_func=lambda x: {
                "weighted_score": "⭐ Bayesian Weighted Score (IMDB Standard)",
                "popularity_index": "🔥 Composite Popularity Index",
                "avg_rating": "📈 Raw Average Rating",
                "rating_count": "🗳️ Total Audience Vote Count",
                "box_office_musd": "💰 Global Box Office ($M USD)",
                "roi_pct": "🚀 Return on Investment (ROI %)",
            }[x],
        )
    with ctrl2:
        top_n = st.slider("Number of Top Movies to Display", min_value=5, max_value=50, value=12, step=1)
    with ctrl3:
        csv_export = df_movies.sort_values(ranking_metric, ascending=False).head(top_n).to_csv(index=False).encode("utf-8")
        st.write("")
        st.download_button(
            "📥 Export Top List (CSV)",
            data=csv_export,
            file_name="cinepulse_top_rated_movies.csv",
            mime="text/csv",
            use_container_width=True,
        )

    df_top = df_movies.sort_values(by=[ranking_metric, "weighted_score"], ascending=False).head(top_n).reset_index(drop=True)

    # Spotlight Top 6 Cards in a 3x2 Grid
    st.markdown("#### 🌟 Featured Top-Ranked Releases")
    card_cols = st.columns(3)
    for idx, row in df_top.head(6).iterrows():
        with card_cols[idx % 3]:
            genres_html = "".join([f'<span class="genre-pill">{g}</span>' for g in str(row["genres"]).split("|")])
            st.markdown(
                f"""
                <div class="movie-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="movie-rank-badge">RANK #{idx + 1} • {row['movie_id']}</span>
                        <span style="font-size:0.78rem; color:#FF9E7D; font-weight:700;">{row['content_rating']} • {row['runtime_min']}m</span>
                    </div>
                    <div class="movie-title">{row['title']} ({row['release_year']})</div>
                    <div class="movie-meta">🎬 Directed by <b>{row['director']}</b> • {row['language']}</div>
                    <div style="margin-bottom:10px;">{genres_html}</div>
                    <p style="font-size:0.82rem; color:#F2D0C2; min-height:38px; margin-bottom:12px;">
                        {row['synopsis'][:115]}...
                    </p>
                    <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:8px;">
                        <div class="score-box">
                            <div style="font-size:0.70rem; color:#FFB699;">WEIGHTED</div>
                            <div style="font-size:1.15rem; font-weight:800; color:#FF2A85;">{row['weighted_score']:.2f} ★</div>
                        </div>
                        <div class="score-box">
                            <div style="font-size:0.70rem; color:#FFB699;">POPULARITY</div>
                            <div style="font-size:1.15rem; font-weight:800; color:#FF9E7D;">{row['popularity_index']:.1f}</div>
                        </div>
                        <div class="score-box">
                            <div style="font-size:0.70rem; color:#FFB699;">BOX OFFICE</div>
                            <div style="font-size:1.05rem; font-weight:800; color:#FFE5D9;">${row['box_office_musd']:.0f}M</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Horizontal Bar Chart of Top N Movies
    col_t1, col_t2 = st.columns([5, 7])
    with col_t1:
        df_plot_top = df_top.head(15).iloc[::-1]
        fig_top_bar = px.bar(
            df_plot_top,
            x=ranking_metric,
            y="title",
            orientation="h",
            color="weighted_score",
            color_continuous_scale=PEACH_MAGENTA_SCALE,
            text=df_plot_top[ranking_metric].apply(lambda v: f" {v:.2f}"),
            labels={"title": "Movie Title", ranking_metric: ranking_metric.replace("_", " ").title()},
        )
        fig_top_bar.update_traces(textposition="inside")
        apply_plotly_theme(fig_top_bar, f"🏆 Top 15 Movies by {ranking_metric.replace('_', ' ').title()}", height=500)
        st.plotly_chart(fig_top_bar, use_container_width=True)

    with col_t2:
        st.markdown("#### 📋 Interactive Top-Rated Leaderboard Table")
        display_cols = [
            "movie_id", "title", "release_year", "genres", "director",
            "weighted_score", "avg_rating", "rating_count", "popularity_index",
            "box_office_musd", "roi_pct", "audience_tier"
        ]
        st.dataframe(
            df_top[display_cols],
            use_container_width=True,
            height=465,
            column_config={
                "movie_id": st.column_config.TextColumn("ID", width="small"),
                "title": st.column_config.TextColumn("Movie Title", width="medium"),
                "release_year": st.column_config.NumberColumn("Year", format="%d"),
                "weighted_score": st.column_config.ProgressColumn(
                    "Weighted Score ★", min_value=1.0, max_value=10.0, format="%.2f ★"
                ),
                "avg_rating": st.column_config.NumberColumn("Raw Avg", format="%.2f"),
                "rating_count": st.column_config.NumberColumn("Votes", format="%d"),
                "popularity_index": st.column_config.ProgressColumn(
                    "Popularity", min_value=0.0, max_value=100.0, format="%.1f"
                ),
                "box_office_musd": st.column_config.NumberColumn("Box Office ($M)", format="$%.1fM"),
                "roi_pct": st.column_config.NumberColumn("ROI %", format="%.1f%%"),
            },
            hide_index=True,
        )


# ==============================================================================
# TAB 3: GENRE & ERA INTELLIGENCE
# ==============================================================================
with tab_genre:
    st.markdown('<div class="section-header">Multi-Genre Co-Occurrence, Era Trends & Sunburst Hierarchy</div>', unsafe_allow_html=True)

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        # Sunburst Chart: Decade -> Primary Genre -> Audience Sentiment Tier
        fig_sun = px.sunburst(
            df_movies,
            path=["decade", "primary_genre", "audience_tier"],
            values="rating_count",
            color="weighted_score",
            color_continuous_scale=PEACH_MAGENTA_SCALE,
            hover_data=["avg_rating"],
        )
        apply_plotly_theme(fig_sun, "🌅 Sunburst: Decade → Primary Genre → Audience Tier (Sized by Votes)", height=450)
        st.plotly_chart(fig_sun, use_container_width=True)

    with g_col2:
        # Release Year Timeline Trend (Bar + Line Dual View)
        yearly_df = (
            df_movies.groupby("release_year", as_index=False)
            .agg(
                movie_count=("movie_id", "count"),
                avg_weighted=("weighted_score", "mean"),
                avg_pop=("popularity_index", "mean"),
                total_votes=("rating_count", "sum"),
            )
            .sort_values("release_year")
        )
        fig_year = go.Figure()
        fig_year.add_trace(
            go.Bar(
                x=yearly_df["release_year"],
                y=yearly_df["movie_count"],
                name="Movies Released",
                marker_color="rgba(255, 158, 125, 0.45)",
                yaxis="y",
            )
        )
        fig_year.add_trace(
            go.Scatter(
                x=yearly_df["release_year"],
                y=yearly_df["avg_weighted"],
                name="Avg Weighted Score (★)",
                mode="lines+markers",
                line=dict(color=MAGENTA_BRIGHT, width=3),
                marker=dict(size=8, color=PEACH_LIGHT, line=dict(color=MAGENTA_BRIGHT, width=2)),
                yaxis="y2",
            )
        )
        fig_year.update_layout(
            yaxis=dict(title="Number of Releases"),
            yaxis2=dict(title="Weighted Score (★)", overlaying="y", side="right", range=[5.0, 9.5]),
        )
        apply_plotly_theme(fig_year, "📅 Release Year Timeline: Catalog Volume vs. Audience Rating Trend", height=450)
        st.plotly_chart(fig_year, use_container_width=True)

    # Heatmap: Genre vs Decade Average Weighted Rating
    st.markdown('<div class="section-header">Genre × Decade Audience Preference Heatmap</div>', unsafe_allow_html=True)
    # Explode genres for accurate multi-genre representation
    exploded_list = []
    for _, r in df_movies.iterrows():
        for g in str(r["genres"]).split("|"):
            exploded_list.append({
                "genre": g.strip(),
                "decade": r["decade"],
                "weighted_score": r["weighted_score"],
                "popularity_index": r["popularity_index"],
            })
    df_exploded = pd.DataFrame(exploded_list)
    if not df_exploded.empty:
        piv = df_exploded.pivot_table(
            index="genre", columns="decade", values="weighted_score", aggfunc="mean"
        ).round(2)
        fig_heat = px.imshow(
            piv,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=PEACH_MAGENTA_SCALE,
            labels=dict(x="Release Decade", y="Genre", color="Weighted ★"),
        )
        apply_plotly_theme(fig_heat, "🔥 Heatmap: Mean Weighted Rating Across Genres & Release Decades", height=430)
        st.plotly_chart(fig_heat, use_container_width=True)


# ==============================================================================
# TAB 4: DATA PIPELINE, DEDUPLICATION & CLEANING INSPECTOR
# ==============================================================================
with tab_etl:
    st.markdown('<div class="section-header">ETL Pipeline Architecture & Data Hygiene Audit</div>', unsafe_allow_html=True)

    df_raw_movies, df_raw_ratings = get_raw_datasets_preview()

    # Pipeline Stage Cards
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            f"""
            <div class="pipeline-step">
                <div class="pipeline-step-title">1️⃣ Raw Ingestion (2 Datasets)</div>
                <div style="font-size:0.86rem; color:#FFE5D9;">
                    • <code>raw_movie_details.csv</code>: <b>{audit_stats['raw_movies_rows']}</b> rows<br>
                    • <code>raw_movie_ratings.csv</code>: <b>{audit_stats['raw_ratings_rows']:,}</b> rows<br>
                    • Stored as separate unjoined files
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"""
            <div class="pipeline-step">
                <div class="pipeline-step-title">2️⃣ Title Deduplication & ID Map</div>
                <div style="font-size:0.86rem; color:#FFE5D9;">
                    • Normalized whitespace & casing<br>
                    • Merged <b>{audit_stats['duplicate_movie_titles_merged']}</b> duplicate title records<br>
                    • Remapped <b>{audit_stats['ratings_remapped_from_duplicate_titles']}</b> ratings to canonical IDs
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        total_missing_fixed = sum(audit_stats["missing_values_before"].values())
        st.markdown(
            f"""
            <div class="pipeline-step">
                <div class="pipeline-step-title">3️⃣ Imputation & Outlier Hygiene</div>
                <div style="font-size:0.86rem; color:#FFE5D9;">
                    • Imputed <b>{total_missing_fixed}</b> missing movie fields<br>
                    • Removed <b>{audit_stats['orphan_ratings_removed']}</b> orphan ratings<br>
                    • Filtered <b>{audit_stats['null_ratings_removed'] + audit_stats['out_of_bounds_ratings_removed']}</b> null/invalid scores
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            f"""
            <div class="pipeline-step">
                <div class="pipeline-step-title">4️⃣ SQL Joins & Aggregation</div>
                <div style="font-size:0.86rem; color:#FFE5D9;">
                    • Deduplicated <b>{audit_stats['duplicate_user_ratings_removed']}</b> repeat user votes<br>
                    • Loaded <b>{audit_stats['clean_movies_rows']}</b> canonical films & <b>{audit_stats['clean_ratings_rows']:,}</b> ratings into SQLite
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        # Bar Chart: Missing Values Before vs After ETL Imputation
        mb = audit_stats["missing_values_before"]
        ma = audit_stats["missing_values_after"]
        df_miss = pd.DataFrame({
            "Field": list(mb.keys()),
            "Before Cleaning (Raw)": list(mb.values()),
            "After ETL Imputation (Clean)": list(ma.values()),
        })
        fig_miss = go.Figure()
        fig_miss.add_trace(
            go.Bar(
                x=df_miss["Field"],
                y=df_miss["Before Cleaning (Raw)"],
                name="Raw Missing Count",
                marker_color=MAGENTA_BRIGHT,
                text=df_miss["Before Cleaning (Raw)"],
                textposition="outside",
            )
        )
        fig_miss.add_trace(
            go.Bar(
                x=df_miss["Field"],
                y=df_miss["After ETL Imputation (Clean)"],
                name="Post-ETL Missing Count",
                marker_color=PEACH_CORE,
                text=df_miss["After ETL Imputation (Clean)"],
                textposition="outside",
            )
        )
        apply_plotly_theme(fig_miss, "🧹 Missing Values Resolved Across Movie Metadata Columns", height=380)
        st.plotly_chart(fig_miss, use_container_width=True)

    with col_e2:
        # Funnel / Waterfall Chart of Ratings Record Hygiene
        stages = [
            "Raw Rating Rows",
            "After Orphan ID Removal",
            "After Null Rating Removal",
            "After Out-of-Bounds Filter",
            "Final Deduplicated Ratings",
        ]
        r_raw = audit_stats["raw_ratings_rows"]
        r_s1 = r_raw - audit_stats["orphan_ratings_removed"]
        r_s2 = r_s1 - audit_stats["null_ratings_removed"]
        r_s3 = r_s2 - audit_stats["out_of_bounds_ratings_removed"]
        r_s4 = audit_stats["clean_ratings_rows"]

        fig_funnel = go.Figure(
            go.Funnel(
                y=stages,
                x=[r_raw, r_s1, r_s2, r_s3, r_s4],
                textinfo="value+percent initial",
                marker=dict(color=[MAGENTA_DEEP, MAGENTA_BRIGHT, PEACH_CORAL, PEACH_CORE, PEACH_LIGHT]),
            )
        )
        apply_plotly_theme(fig_funnel, "🔬 Audience Ratings Deduplication & Validation Funnel", height=380)
        st.plotly_chart(fig_funnel, use_container_width=True)

    # Show Actual Duplicate Titles Caught in Raw Data
    st.markdown("#### 🔍 Live Proof: Duplicate Movie Titles Detected & Coalesced by ETL")
    df_raw_movies_copy = df_raw_movies.copy()
    df_raw_movies_copy["norm_key"] = df_raw_movies_copy["title"].astype(str).str.strip().str.lower()
    dup_keys = df_raw_movies_copy["norm_key"].value_counts()
    dup_keys = dup_keys[dup_keys > 1].index.tolist()
    df_dups_showcase = df_raw_movies_copy[df_raw_movies_copy["norm_key"].isin(dup_keys)].sort_values("norm_key")
    st.dataframe(
        df_dups_showcase[["movie_id", "title", "release_year", "genres", "director", "runtime_min", "budget_musd", "box_office_musd"]],
        use_container_width=True,
        height=280,
        hide_index=True,
    )


# ==============================================================================
# TAB 5: SQL WAREHOUSE WORKBENCH
# ==============================================================================
with tab_sql:
    st.markdown('<div class="section-header">Interactive SQL Query Workbench (SQLite Relational Warehouse)</div>', unsafe_allow_html=True)
    st.caption(
        "Explore the normalized SQL schema (`movies`, `movie_genres`, `ratings`, `pipeline_audit`) "
        "using pre-built analytical joins or write your own custom `SELECT` query."
    )

    selected_preset = st.selectbox(
        "📚 Choose a Pre-Built Analytical SQL Query:",
        options=list(PREBUILT_SQL_QUERIES.keys()),
    )

    sql_input = st.text_area(
        "SQL Query Editor (Read-Only SELECT / CTE Mode):",
        value=PREBUILT_SQL_QUERIES[selected_preset],
        height=190,
    )

    run_col1, run_col2 = st.columns([1, 5])
    with run_col1:
        run_sql_btn = st.button("▶️ Execute SQL Query", type="primary", use_container_width=True)

    if run_sql_btn or sql_input:
        try:
            sql_result_df = execute_custom_sql(sql_input)
            st.success(f"Query executed successfully — returned **{len(sql_result_df):,}** rows.")
            st.dataframe(sql_result_df, use_container_width=True, height=340)
        except Exception as e:
            st.error(f"SQL Execution Error: {e}")


# ==============================================================================
# TAB 6: PEACH-MAGENTA DISCOVERY & SIMILARITY ENGINE
# ==============================================================================
with tab_rec:
    st.markdown('<div class="section-header">Content & Audience Preference Similarity Matcher</div>', unsafe_allow_html=True)
    st.caption(
        "Select any film in the cleaned warehouse to discover similar top-rated movies matched by "
        "multi-genre overlap, weighted audience score, popularity profile, and directorial style."
    )

    movie_titles_list = df_movies["title"].tolist()
    chosen_title = st.selectbox("🎯 Choose a Reference Movie:", options=movie_titles_list, index=0)

    ref_row = df_movies[df_movies["title"] == chosen_title].iloc[0]
    ref_genres = set(str(ref_row["genres"]).split("|"))

    def compute_similarity(candidate):
        if candidate["movie_id"] == ref_row["movie_id"]:
            return -1.0
        cand_genres = set(str(candidate["genres"]).split("|"))
        jaccard_genre = len(ref_genres & cand_genres) / max(len(ref_genres | cand_genres), 1)
        rating_prox = max(0.0, 1.0 - abs(candidate["weighted_score"] - ref_row["weighted_score"]) / 5.0)
        pop_prox = max(0.0, 1.0 - abs(candidate["popularity_index"] - ref_row["popularity_index"]) / 100.0)
        director_bonus = 0.15 if candidate["director"] == ref_row["director"] else 0.0
        return round((0.50 * jaccard_genre + 0.30 * rating_prox + 0.20 * pop_prox + director_bonus) * 100, 1)

    df_recs = df_movies.copy()
    df_recs["match_score_pct"] = df_recs.apply(compute_similarity, axis=1)
    df_recs = df_recs[df_recs["match_score_pct"] >= 0].sort_values(
        by=["match_score_pct", "weighted_score"], ascending=False
    ).head(6)

    r_col1, r_col2 = st.columns([4, 8])

    with r_col1:
        st.markdown("#### 🎬 Reference Film Profile")
        ref_genres_html = "".join([f'<span class="genre-pill">{g}</span>' for g in str(ref_row["genres"]).split("|")])
        st.markdown(
            f"""
            <div class="movie-card" style="border-color:#FF2A85; border-width:2px;">
                <span class="movie-rank-badge">SELECTED FILM • {ref_row['movie_id']}</span>
                <div class="movie-title">{ref_row['title']} ({ref_row['release_year']})</div>
                <div class="movie-meta">Directed by <b>{ref_row['director']}</b> • {ref_row['runtime_min']} min</div>
                <div style="margin-bottom:10px;">{ref_genres_html}</div>
                <p style="font-size:0.86rem; color:#FFE5D9;">{ref_row['synopsis']}</p>
                <hr style="border-color:rgba(255,158,125,0.2);">
                <div style="font-size:0.85rem; color:#FFB899;">
                    ⭐ <b>Weighted Score:</b> {ref_row['weighted_score']:.2f} / 10 ({ref_row['rating_count']} votes)<br>
                    🔥 <b>Popularity Index:</b> {ref_row['popularity_index']:.1f} / 100<br>
                    💰 <b>Box Office:</b> ${ref_row['box_office_musd']:.1f}M (ROI: {ref_row['roi_pct']:.1f}%)<br>
                    🏛️ <b>Studio:</b> {ref_row['production_studio']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r_col2:
        st.markdown("#### ✨ Top 6 Audience-Matched Recommendations")
        rec_cols = st.columns(2)
        for idx, (_, rec) in enumerate(df_recs.iterrows()):
            with rec_cols[idx % 2]:
                g_html = "".join([f'<span class="genre-pill">{g}</span>' for g in str(rec["genres"]).split("|")])
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div style="display:flex; justify-content:space-between;">
                            <span class="movie-rank-badge">{rec['match_score_pct']}% MATCH</span>
                            <span style="color:#FF9E7D; font-weight:700;">{rec['weighted_score']:.2f} ★</span>
                        </div>
                        <div class="movie-title">{rec['title']} ({rec['release_year']})</div>
                        <div class="movie-meta">🎬 {rec['director']} • Pop: {rec['popularity_index']:.1f}</div>
                        <div>{g_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
