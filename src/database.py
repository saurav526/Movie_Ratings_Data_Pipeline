"""
SQL Analytics & Query Engine for Movie Ratings Data Pipeline
Provides parameterized SQL queries against the SQLite / PostgreSQL-compatible database
for filtering, aggregations, genre matrices, top-rated leaderboards, and interactive SQL exploration.
"""
import os
import sqlite3
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "cinema_analytics.db")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
# the database use for the ETL pipeline and SQL queries is SQLite, but the schema and queries are compatible with PostgreSQL for future scalability

def ensure_database():
    """Ensures the database and processed datasets exist; runs ETL if not present."""
    if not os.path.exists(DB_PATH):
        from src.etl_pipeline import run_etl_pipeline
        run_etl_pipeline()


def get_connection():
    ensure_database()
    return sqlite3.connect(DB_PATH)


def get_pipeline_audit() -> dict:
    audit_path = os.path.join(PROCESSED_DIR, "pipeline_audit.json")
    if not os.path.exists(audit_path):
        from src.etl_pipeline import run_etl_pipeline
        return run_etl_pipeline()
    with open(audit_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_filter_bounds() -> dict:
    conn = get_connection()
    genres_df = pd.read_sql_query("SELECT DISTINCT genre FROM movie_genres ORDER BY genre;", conn)
    bounds_df = pd.read_sql_query("""
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
    """, conn)
    langs_df = pd.read_sql_query("SELECT DISTINCT language FROM movies ORDER BY language;", conn)
    certs_df = pd.read_sql_query("SELECT DISTINCT content_rating FROM movies ORDER BY content_rating;", conn)
    conn.close()

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
    """Executes an SQL query joining `movies` and `movie_genres` with dynamic filters."""
    conn = get_connection()
    params = [year_range[0], year_range[1], min_rating, min_popularity, min_votes]

    where_clauses = [
        "m.release_year BETWEEN ? AND ?",
        "m.weighted_score >= ?",
        "m.popularity_index >= ?",
        "m.rating_count >= ?"
    ]

    if genres:
        placeholders = ",".join(["?"] * len(genres))
        where_clauses.append(f"m.movie_id IN (SELECT movie_id FROM movie_genres WHERE genre IN ({placeholders}))")
        params.extend(genres)

    if languages:
        placeholders = ",".join(["?"] * len(languages))
        where_clauses.append(f"m.language IN ({placeholders})")
        params.extend(languages)

    if certificates:
        placeholders = ",".join(["?"] * len(certificates))
        where_clauses.append(f"m.content_rating IN ({placeholders})")
        params.extend(certificates)

    if search_query and search_query.strip():
        where_clauses.append("(LOWER(m.title) LIKE ? OR LOWER(m.director) LIKE ?)")
        q = f"%{search_query.strip().lower()}%"
        params.extend([q, q])

    sql = f"""
        SELECT m.*
        FROM movies m
        WHERE {' AND '.join(where_clauses)}
        ORDER BY m.weighted_score DESC, m.popularity_index DESC;
    """
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def query_genre_aggregation(movie_ids: list) -> pd.DataFrame:
    """SQL aggregation across the normalized `movie_genres` bridge table for filtered movies."""
    if not movie_ids:
        return pd.DataFrame()
    conn = get_connection()
    placeholders = ",".join(["?"] * len(movie_ids))
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
    df = pd.read_sql_query(sql, conn, params=movie_ids)
    conn.close()
    return df


def query_ratings_for_movies(movie_ids: list) -> pd.DataFrame:
    """Fetches individual cleaned audience ratings joined with movie details."""
    if not movie_ids:
        return pd.DataFrame()
    conn = get_connection()
    placeholders = ",".join(["?"] * len(movie_ids))
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
    df = pd.read_sql_query(sql, conn, params=movie_ids)
    conn.close()
    return df


def get_raw_datasets_preview() -> tuple:
    ensure_database()
    df_raw_m = pd.read_csv(os.path.join(RAW_DIR, "raw_movie_details.csv"))
    df_raw_r = pd.read_csv(os.path.join(RAW_DIR, "raw_movie_ratings.csv"))
    return df_raw_m, df_raw_r


def execute_custom_sql(sql_query: str) -> pd.DataFrame:
    """Executes a read-only SQL SELECT query for the SQL Workbench tab."""
    cleaned = sql_query.strip()
    forbidden = ["DROP ", "DELETE ", "UPDATE ", "INSERT ", "ALTER ", "TRUNCATE ", "CREATE ", "REPLACE ", "ATTACH ", "DETACH ", "PRAGMA "]
    upper_q = cleaned.upper()
    if not (upper_q.startswith("SELECT") or upper_q.startswith("WITH")):
        raise ValueError("Only read-only SELECT or WITH (CTE) queries are permitted in the SQL Explorer.")
    for word in forbidden:
        if word in upper_q:
            raise ValueError(f"Operation '{word.strip()}' is disabled in read-only mode.")
    conn = get_connection()
    try:
        df = pd.read_sql_query(cleaned, conn)
        return df
    finally:
        conn.close()


PREBUILT_SQL_QUERIES = {
    "1. Top 15 Highest Weighted Movies (Multi-Table Join & Aggregation)": """SELECT 
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

    "2. Genre Performance & Audience Preference Summary (Normalized Bridge Join)": """SELECT 
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

    "6. Data Pipeline ETL Audit Log": """SELECT 
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
