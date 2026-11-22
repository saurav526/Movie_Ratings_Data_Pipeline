# 🎬 CinePulse: Movie Ratings Data Pipeline & Audience Analytics Studio

A complete end-to-end **Movie Ratings Data Pipeline** and **Peach & Magenta Streamlit Analytics Dashboard** built with **Python, Pandas, SQL, PostgreSQL, Plotly, and Streamlit**.

---

## 📌 Project Overview & Requirements Covered

| Requirement | Implementation in This Project |
| :--- | :--- |
| **Separate Datasets** | `data/raw/raw_movie_details.csv` (movie metadata) and `data/raw/raw_movie_ratings.csv` (7,900+ audience rating logs) generated & stored separately. |
| **Data Integration & Joins** | Merges raw movie metadata with aggregated & individual rating logs, plus a normalized 1-to-many `movie_genres` bridge table for multi-genre PostgreSQL joins. |
| **Clean Duplicate Titles** | Normalizes messy title casing/whitespace, merges duplicate movie entries into canonical records, and remaps duplicate `movie_id` values in ratings so zero votes are lost. |
| **Clean Missing & Invalid Values** | Coalesces partial duplicate rows, imputes missing numeric fields (`budget_musd`, `box_office_musd`, `runtime_min`) using genre-grouped medians, and removes null, orphan, out-of-bounds, and duplicate `(user_id, movie_id)` ratings. |
| **PostgreSQL Database Layer** | Normalized relational **PostgreSQL** warehouse (`movie_ratings_db`) with 4 indexed tables (`movies`, `movie_genres`, `ratings`, `pipeline_audit`) and an interactive SQL Workbench tab. |
| **Visual Summaries & Filters** | Interactive **Genre, Release Year, Weighted Rating, Vote Count, Language, Certificate, and Popularity** sidebar filters powering Plotly histograms, horizontal/vertical bar charts, bubble matrices, sunbursts, heatmaps, and top-rated movie leaderboards. |
| **Peach & Magenta Theme** | Custom `.streamlit/config.toml` and glassmorphic CSS styling in a signature **Peach (`#FF9E7D`, `#FFD1B3`) & Electric Magenta (`#FF2A85`, `#C81D77`)** dark cinema palette. |

---

## 🗂️ Project Structure

```text
Movie_Ratings_Data_Pipeline/
├── .streamlit/
│   └── config.toml                 # Peach & Magenta Streamlit UI theme configuration
├── data/
│   ├── raw/
│   │   ├── raw_movie_details.csv   # Raw movie catalog (with duplicates & missing values)
│   │   └── raw_movie_ratings.csv   # Raw audience ratings stream (with duplicates, nulls & orphans)
│   └── processed/
│       ├── cleaned_movies_integrated.csv
│       ├── cleaned_ratings.csv
│       ├── movie_genres_bridge.csv
│       └── pipeline_audit.json     # Detailed ETL data quality & lineage metrics
├── src/
│   ├── __init__.py
│   ├── data_generator.py           # Generates realistic raw datasets with intentional flaws
│   ├── etl_pipeline.py             # Deduplication, missing-value imputation, aggregation & PostgreSQL loader
│   └── database.py                 # Parameterized PostgreSQL queries, aggregations & SQL Workbench engine
├── app.py                          # Main Peach & Magenta Streamlit Analytics Dashboard
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

---

## 🚀 Quick Start

### 1. Run the ETL Data Pipeline (Optional — Auto-runs on first dashboard launch)
```bash
cd "Movie_Ratings_Data_Pipeline"
python -m src.etl_pipeline
```

### 2. Launch the Peach & Magenta Streamlit Dashboard
```bash
cd "Movie_Ratings_Data_Pipeline"
streamlit run app.py
```
