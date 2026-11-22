"""
PostgreSQL Analytics & Query Engine for Movie Ratings Data Pipeline
Provides parameterized SQL queries against the PostgreSQL relational warehouse (`movie_ratings_db`)
for filtering, aggregations, genre matrices, top-rated leaderboards, and interactive SQL exploration.
"""

import os
import json
import pandas as pd
import psycopg2
from src.etl_pipeline import PG_CONFIG, run_etl_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def ensure_database():
    """Ensures the PostgreSQL database `movie_ratings_db` and tables exist; runs ETL if not present."""
    try:
        conn = psycopg2.connect(
            host=PG_CONFIG["host"],
            port=PG_CONFIG["port"],
            database=PG_CONFIG["database"],
            user=PG_CONFIG["user"],
            password=PG_CONFIG["password"],
        )
        cur = conn.cursor()
        cur.execute("SELECT to_regclass('public.movies');")
        exists = cur.fetchone()[0]
        cur.close()
        conn.close()
        if not exists:
            run_etl_pipeline()
    except psycopg2.OperationalError:
        run_etl_pipeline()


def get_connection():
    ensure_database()
    return psycopg2.connect(
        host=PG_CONFIG["host"],
        port=PG_CONFIG["port"],
        database=PG_CONFIG["database"],
        user=PG_CONFIG["user"],
        password=PG_CONFIG["password"],
    )


def _fetch_df(sql: str, params=None) -> pd.DataFrame:
    """Executes a parameterized query on PostgreSQL via psycopg2 and returns a DataFrame with float types normalized."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        df = pd.DataFrame(rows, columns=cols)
        # Convert Decimal columns to float for seamless Pandas/Plotly compatibility
        for col in df.columns:
            if df[col].dtype == "object" and len(df) > 0:
                first_non_null = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if hasattr(first_non_null, "as_tuple"):  # decimal.Decimal
                    df[col] = df[col].astype(float)
        return df
    finally:
        conn.close()


def get_pipeline_audit() -> dict:
    ensure_database()
    audit_path = os.path.join(PROCESSED_DIR, "pipeline_audit.json")
    if not os.path.exists(audit_path):
        return run_etl_pipeline()
    with open(audit_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_filter_bounds() -> dict:
    genres_df = _fetch_df("SELECT DISTINCT genre FROM movie_genres ORDER BY genre;")
    bounds_df = _fetch_df("""
        SELECT
            MIN(release_year) AS min_year,
            MAX(release_year) AS max_year,
            MIN(weighted_score) AS min_rating,
            MAX(weighted_score) AS max_rating,
            MIN(popularity_index) AS min_pop,
            MAX(popularity_index) AS max_pop,
            MIN(rating_count) AS min_votes,
            MAX(rating_count) AS max_votes
        FROM movies;
    """)
    langs_df = _fetch_df("SELECT DISTINCT language FROM movies ORDER BY language;")
    certs_df = _fetch_df("SELECT DISTINCT content_rating FROM movies ORDER BY content_rating;")

    row = bounds_df.iloc[0]
    return {
        "genres": genres_df["genre"].tolist(),
        "min_year": int(row["min_year"]),
        "max_year": int(row["max_year"]),
        "min_rating": float(row["min_rating"]),
        "max_rating": float(row["max_rating"]),
        "min_pop": float(row["min_pop"]),
        "max_pop": float(row["max_pop"]),
        "min_votes": int(row["min_votes"]),
        "max_votes": int(row["max_votes"]),
        "languages": langs_df["language"].tolist(),
        "certificates": certs_df["content_rating"].tolist(),
    }


def query_filtered_movies(
    genres: list = None,
    year_range: tuple = (1950, 2025),
    min_rating: float = 1.0,
    min_popularity: float = 0.0,
    min_votes: int = 0,
    languages: list = None,
    certificates: list = None,
    search_query: str = ""
) -> pd.DataFrame:
    """Executes a PostgreSQL query joining `movies` and `movie_genres` with dynamic filters."""
    params = [year_range[0], year_range[1], min_rating, min_popularity, min_votes]

    where_clauses = [
        "m.release_year BETWEEN %s AND %s",
        "m.weighted_score >= %s",
        "m.popularity_index >= %s",
        "m.rating_count >= %s"
    ]

    if genres:
        placeholders = ",".join(["%s"] * len(genres))
        where_clauses.append(f"m.movie_id IN (SELECT movie_id FROM movie_genres WHERE genre IN ({placeholders}))")
        params.extend(genres)

    if languages:
        placeholders = ",".join(["%s"] * len(languages))
        where_clauses.append(f"m.language IN ({placeholders})")
        params.extend(languages)

    if certificates:
        placeholders = ",".join(["%s"] * len(certificates))
        where_clauses.append(f"m.content_rating IN ({placeholders})")
        params.extend(certificates)

    if search_query and search_query.strip():
        where_clauses.append("(LOWER(m.title) LIKE %s OR LOWER(m.director) LIKE %s)")
        q = f"%{search_query.strip().lower()}%"
        params.extend([q, q])

    sql = f"""
        SELECT m.*
        FROM movies m
        WHERE {' AND '.join(where_clauses)}
        ORDER BY m.weighted_score DESC, m.popularity_index DESC;
    """
    return _fetch_df(sql, params)


def query_genre_aggregation(movie_ids: list) -> pd.DataFrame:
    """PostgreSQL aggregation across the normalized `movie_genres` bridge table for filtered movies."""
    if not movie_ids:
        return pd.DataFrame()
    placeholders = ",".join(["%s"] * len(movie_ids))
    sql = f"""
        SELECT
            mg.genre AS genre,
            COUNT(DISTINCT m.movie_id) AS movie_count,
            ROUND(AVG(m.weighted_score), 2) AS avg_weighted_score,
            ROUND(AVG(m.avg_rating), 2) AS avg_raw_rating,
            ROUND(AVG(m.popularity_index), 1) AS avg_popularity,
            SUM(m.rating_count) AS total_ratings,
            ROUND(AVG(m.box_office_musd), 1) AS avg_box_office_musd,
            ROUND(AVG(m.roi_pct), 1) AS avg_roi_pct
        FROM movie_genres mg
        INNER JOIN movies m ON mg.movie_id = m.movie_id
        WHERE m.movie_id IN ({placeholders})
        GROUP BY mg.genre
        ORDER BY avg_weighted_score DESC, movie_count DESC;
    """
    return _fetch_df(sql, movie_ids)


def query_ratings_for_movies(movie_ids: list) -> pd.DataFrame:
    """Fetches individual cleaned audience ratings joined with movie details from PostgreSQL."""
    if not movie_ids:
        return pd.DataFrame()
    placeholders = ",".join(["%s"] * len(movie_ids))
    sql = f"""
        SELECT
            r.rating_id,
            r.movie_id,
            m.title,
            m.primary_genre,
            m.release_year,
            r.user_id,
            r.rating,
            r.review_platform,
            r.viewer_age_group,
            r.is_verified_viewer,
            r.helpful_votes,
            r.rated_at
        FROM ratings r
        INNER JOIN movies m ON r.movie_id = m.movie_id
        WHERE r.movie_id IN ({placeholders});
    """
    return _fetch_df(sql, movie_ids)


def get_raw_datasets_preview() -> tuple:
    ensure_database()
    df_raw_m = pd.read_csv(os.path.join(RAW_DIR, "raw_movie_details.csv"))
    df_raw_r = pd.read_csv(os.path.join(RAW_DIR, "raw_movie_ratings.csv"))
    return df_raw_m, df_raw_r


def execute_custom_sql(sql_query: str) -> pd.DataFrame:
    """Executes a read-only PostgreSQL SELECT query for the SQL Workbench tab."""
    cleaned = sql_query.strip()
    forbidden = ["DROP ", "DELETE ", "UPDATE ", "INSERT ", "ALTER ", "TRUNCATE ", "CREATE ", "REPLACE ", "GRANT ", "REVOKE "]
    upper_q = cleaned.upper()
    if not (upper_q.startswith("SELECT") or upper_q.startswith("WITH")):
        raise ValueError("Only read-only SELECT or WITH (CTE) queries are permitted in the PostgreSQL Explorer.")
    for word in forbidden:
        if word in upper_q:
            raise ValueError(f"Operation '{word.strip()}' is disabled in read-only mode.")
    return _fetch_df(cleaned)


PREBUILT_SQL_QUERIES = {
    "1. Top 15 Highest Weighted Movies (PostgreSQL Multi-Table Query)": """SELECT 
    m.movie_id,
    m.title,
    m.release_year,
    m.director,
    m.genres,
    m.weighted_score,
    m.avg_rating,
    m.rating_count,
    m.popularity_index,
    m.box_office_musd
FROM movies m
WHERE m.rating_count >= 25
ORDER BY m.weighted_score DESC, m.popularity_index DESC
LIMIT 15;""",

    "2. Genre Performance & Audience Preference Summary (Normalized Bridge JOIN)": """SELECT 
    mg.genre AS Genre,
    COUNT(DISTINCT m.movie_id) AS Total_Movies,
    SUM(m.rating_count) AS Total_Audience_Votes,
    ROUND(AVG(m.weighted_score), 2) AS Avg_Weighted_Rating,
    ROUND(AVG(m.popularity_index), 1) AS Avg_Popularity_Index,
    ROUND(AVG(m.box_office_musd), 1) AS Avg_Box_Office_MUSD,
    ROUND(AVG(m.roi_pct), 1) AS Avg_ROI_Pct
FROM movie_genres mg
JOIN movies m ON mg.movie_id = m.movie_id
GROUP BY mg.genre
ORDER BY Avg_Weighted_Rating DESC;""",

    "3. Audience Demographic & Platform Rating Breakdown (Ratings JOIN Movies)": """SELECT 
    r.viewer_age_group AS Viewer_Age_Bracket,
    r.review_platform AS Platform,
    COUNT(r.rating_id) AS Total_Reviews,
    ROUND(AVG(r.rating), 2) AS Mean_Rating,
    SUM(r.helpful_votes) AS Total_Helpful_Votes
FROM ratings r
INNER JOIN movies m ON r.movie_id = m.movie_id
GROUP BY r.viewer_age_group, r.review_platform
ORDER BY Viewer_Age_Bracket, Mean_Rating DESC;""",

    "4. Director Hall of Fame (Min 3 Movies Directed)": """SELECT 
    m.director AS Director,
    COUNT(m.movie_id) AS Movies_Directed,
    ROUND(AVG(m.weighted_score), 2) AS Career_Weighted_Rating,
    SUM(m.rating_count) AS Total_Ratings_Received,
    ROUND(SUM(m.box_office_musd), 1) AS Total_Box_Office_MUSD,
    ROUND(AVG(m.roi_pct), 1) AS Avg_ROI_Pct
FROM movies m
GROUP BY m.director
HAVING COUNT(m.movie_id) >= 3
ORDER BY Career_Weighted_Rating DESC;""",

    "5. Release Year & Decade Cinema Trends": """SELECT 
    m.decade AS Decade,
    COUNT(m.movie_id) AS Releases,
    ROUND(AVG(m.weighted_score), 2) AS Avg_Rating,
    ROUND(AVG(m.runtime_min), 0) AS Avg_Runtime_Min,
    ROUND(AVG(m.budget_musd), 1) AS Avg_Budget_MUSD,
    ROUND(AVG(m.box_office_musd), 1) AS Avg_Box_Office_MUSD
FROM movies m
GROUP BY m.decade
ORDER BY m.decade;""",

    "6. Data Pipeline ETL Audit Log (PostgreSQL Audit Table)": """SELECT 
    run_timestamp,
    raw_movies_rows,
    clean_movies_rows,
    duplicate_titles_merged,
    raw_ratings_rows,
    clean_ratings_rows,
    duplicate_user_ratings_removed,
    orphan_ratings_removed,
    null_or_oob_ratings_removed
FROM pipeline_audit;"""
}
