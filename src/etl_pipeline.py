"""
Movie Ratings Data Pipeline & PostgreSQL Warehouse Engine
Combines raw_movie_details.csv and raw_movie_ratings.csv:
1. Cleans and normalizes movie titles (casing, whitespace, punctuation).
2. Deduplicates duplicate movie titles while preserving a mapping from duplicate movie_ids
   to the canonical movie_id so no audience ratings are lost during joins!
3. Coalesces and imputes missing values (budget, box office, runtime, director, genres, content rating).
4. Cleans ratings dataset: remaps duplicate movie_ids, removes null/out-of-bounds ratings,
   deduplicates repeat (user_id, movie_id) ratings, and filters out orphan movie_ids.
5. Computes IMDB-style Bayesian Weighted Ratings (WR) & Audience Popularity Indices.
6. Loads normalized tables into a PostgreSQL Database (`movie_ratings_db`) via psycopg2 & SQLAlchemy:
   - movies (canonical cleaned movies + aggregated metrics)
   - movie_genres (normalized 1-to-many genre bridge table for fast SQL genre analytics)
   - ratings (cleaned individual rating logs)
   - pipeline_audit (ETL lineage & data quality audit metrics)
"""

import os
import json
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

PG_CONFIG = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": int(os.environ.get("PG_PORT", 5432)),
    "database": os.environ.get("PG_DATABASE", "movie_ratings_db"),
    "user": os.environ.get("PG_USER", "postgres"),
    "password": os.environ.get("PG_PASSWORD", "3748"),
}


def create_postgres_database():
    """Creates the PostgreSQL database `movie_ratings_db` if it does not already exist."""
    conn = psycopg2.connect(
        host=PG_CONFIG["host"],
        port=PG_CONFIG["port"],
        database="postgres",
        user=PG_CONFIG["user"],
        password=PG_CONFIG["password"],
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (PG_CONFIG["database"],))
    exists = cur.fetchone()
    if not exists:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(PG_CONFIG["database"])))
    cur.close()
    conn.close()


def normalize_title(t: str) -> str:
    if pd.isna(t):
        return "Unknown Title"
    cleaned = " ".join(str(t).strip().split())
    words = cleaned.split(" ")
    small_words = {"of", "in", "the", "and", "to", "a", "an", "on", "at", "for", "v"}
    result = []
    for i, w in enumerate(words):
        wl = w.lower()
        if wl == "wall-e":
            result.append("WALL-E")
        elif wl in {"ii", "iii", "iv", "vi"}:
            result.append(w.upper())
        elif i > 0 and wl in small_words:
            result.append(wl)
        else:
            if "-" in w:
                result.append("-".join(part.capitalize() for part in w.split("-")))
            else:
                result.append(w.capitalize())
    return " ".join(result)


def run_etl_pipeline() -> dict:
    """Executes the full extract-transform-load pipeline into PostgreSQL and returns audit statistics."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    raw_movies_path = os.path.join(RAW_DIR, "raw_movie_details.csv")
    raw_ratings_path = os.path.join(RAW_DIR, "raw_movie_ratings.csv")

    if not os.path.exists(raw_movies_path) or not os.path.exists(raw_ratings_path):
        from src.data_generator import generate_raw_datasets
        generate_raw_datasets()

    df_movies_raw = pd.read_csv(raw_movies_path)
    df_ratings_raw = pd.read_csv(raw_ratings_path)

    audit = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_movies_rows": int(len(df_movies_raw)),
        "raw_ratings_rows": int(len(df_ratings_raw)),
    }

    # --- STEP 1: Clean & Deduplicate Movie Details ---
    df_movies = df_movies_raw.copy()
    df_movies["clean_title"] = df_movies["title"].apply(normalize_title)
    df_movies["title_key"] = df_movies["clean_title"].str.lower()

    audit["missing_values_before"] = {
        col: int(df_movies[col].isna().sum())
        for col in ["genres", "director", "runtime_min", "budget_musd", "box_office_musd", "content_rating"]
    }

    df_movies["non_null_count"] = df_movies.notna().sum(axis=1)
    df_movies = df_movies.sort_values(by=["title_key", "non_null_count", "movie_id"], ascending=[True, False, True])

    id_mapping = {}
    canonical_rows = []
    duplicate_titles_found = 0

    for title_key, group in df_movies.groupby("title_key", sort=False):
        canonical_id = group.iloc[0]["movie_id"]
        if len(group) > 1:
            duplicate_titles_found += len(group) - 1
        for _, row in group.iterrows():
            id_mapping[row["movie_id"]] = canonical_id

        first_valid = {}
        for col in group.columns:
            non_null_vals = group[col].dropna()
            first_valid[col] = non_null_vals.iloc[0] if len(non_null_vals) > 0 else np.nan
        first_valid["movie_id"] = canonical_id
        first_valid["title"] = group.iloc[0]["clean_title"]
        canonical_rows.append(first_valid)

    df_clean_movies = pd.DataFrame(canonical_rows)
    df_clean_movies.drop(columns=["clean_title", "title_key", "non_null_count"], inplace=True, errors="ignore")

    audit["duplicate_movie_titles_merged"] = int(duplicate_titles_found)
    audit["clean_movies_rows"] = int(len(df_clean_movies))

    # --- STEP 2: Impute Missing Values in Movie Details ---
    df_clean_movies["genres"] = df_clean_movies["genres"].fillna("Drama|Mystery")
    df_clean_movies["primary_genre"] = df_clean_movies["genres"].apply(lambda g: str(g).split("|")[0].strip())
    df_clean_movies["director"] = df_clean_movies["director"].fillna("Independent Collective")
    df_clean_movies["content_rating"] = df_clean_movies["content_rating"].fillna("PG-13")
    df_clean_movies["language"] = df_clean_movies["language"].fillna("English")

    for num_col in ["runtime_min", "budget_musd", "box_office_musd"]:
        genre_medians = df_clean_movies.groupby("primary_genre")[num_col].transform("median")
        global_median = df_clean_movies[num_col].median()
        df_clean_movies[num_col] = df_clean_movies[num_col].fillna(genre_medians).fillna(global_median)

    df_clean_movies["runtime_min"] = df_clean_movies["runtime_min"].round().astype(int)
    df_clean_movies["budget_musd"] = df_clean_movies["budget_musd"].round(2)
    df_clean_movies["box_office_musd"] = df_clean_movies["box_office_musd"].round(2)
    df_clean_movies["roi_pct"] = (
        ((df_clean_movies["box_office_musd"] - df_clean_movies["budget_musd"]) / df_clean_movies["budget_musd"]) * 100
    ).round(1)
    df_clean_movies["decade"] = ((df_clean_movies["release_year"] // 10) * 10).astype(str) + "s"

    audit["missing_values_after"] = {
        col: int(df_clean_movies[col].isna().sum())
        for col in ["genres", "director", "runtime_min", "budget_musd", "box_office_musd", "content_rating"]
    }

    # --- STEP 3: Clean, Remap, Validate & Deduplicate Ratings ---
    df_ratings = df_ratings_raw.copy()
    df_ratings.rename(columns={"review_Platform": "review_platform"}, inplace=True)

    remapped_count = int(df_ratings["movie_id"].isin(
        {k for k, v in id_mapping.items() if k != v}
    ).sum())
    df_ratings["movie_id"] = df_ratings["movie_id"].map(lambda x: id_mapping.get(x, x))
    audit["ratings_remapped_from_duplicate_titles"] = remapped_count

    valid_movie_ids = set(df_clean_movies["movie_id"])
    orphan_mask = ~df_ratings["movie_id"].isin(valid_movie_ids)
    audit["orphan_ratings_removed"] = int(orphan_mask.sum())
    df_ratings = df_ratings[~orphan_mask].copy()

    null_rating_mask = df_ratings["rating"].isna()
    audit["null_ratings_removed"] = int(null_rating_mask.sum())
    df_ratings = df_ratings[~null_rating_mask].copy()

    oob_mask = (df_ratings["rating"] < 1.0) | (df_ratings["rating"] > 10.0)
    audit["out_of_bounds_ratings_removed"] = int(oob_mask.sum())
    df_ratings = df_ratings[~oob_mask].copy()

    df_ratings = df_ratings.sort_values("rated_at", ascending=False)
    before_dedup = len(df_ratings)
    df_ratings = df_ratings.drop_duplicates(subset=["user_id", "movie_id"], keep="first").copy()
    audit["duplicate_user_ratings_removed"] = int(before_dedup - len(df_ratings))
    audit["clean_ratings_rows"] = int(len(df_ratings))

    # --- STEP 4: Join & Aggregate Ratings per Movie ---
    agg_ratings = df_ratings.groupby("movie_id").agg(
        rating_count=("rating", "count"),
        avg_rating=("rating", "mean"),
        median_rating=("rating", "median"),
        rating_std=("rating", "std"),
        total_helpful_votes=("helpful_votes", "sum"),
        verified_ratio=("is_verified_viewer", "mean")
    ).reset_index()

    agg_ratings["avg_rating"] = agg_ratings["avg_rating"].round(2)
    agg_ratings["median_rating"] = agg_ratings["median_rating"].round(2)
    agg_ratings["rating_std"] = agg_ratings["rating_std"].fillna(0.0).round(2)
    agg_ratings["verified_ratio"] = (agg_ratings["verified_ratio"] * 100).round(1)

    df_integrated = pd.merge(df_clean_movies, agg_ratings, on="movie_id", how="left")
    df_integrated["rating_count"] = df_integrated["rating_count"].fillna(0).astype(int)
    global_C = df_ratings["rating"].mean()
    df_integrated["avg_rating"] = df_integrated["avg_rating"].fillna(round(global_C, 2))
    df_integrated["median_rating"] = df_integrated["median_rating"].fillna(round(global_C, 2))
    df_integrated["rating_std"] = df_integrated["rating_std"].fillna(0.0)
    df_integrated["total_helpful_votes"] = df_integrated["total_helpful_votes"].fillna(0).astype(int)
    df_integrated["verified_ratio"] = df_integrated["verified_ratio"].fillna(75.0)

    m_quantile = float(np.percentile(df_integrated["rating_count"], 40))
    v = df_integrated["rating_count"]
    R = df_integrated["avg_rating"]
    df_integrated["weighted_score"] = (
        (v / (v + m_quantile)) * R + (m_quantile / (v + m_quantile)) * global_C
    ).round(2)

    v_norm = (df_integrated["rating_count"] - df_integrated["rating_count"].min()) / (
        df_integrated["rating_count"].max() - df_integrated["rating_count"].min() + 1e-6
    )
    bo_norm = (df_integrated["box_office_musd"] - df_integrated["box_office_musd"].min()) / (
        df_integrated["box_office_musd"].max() - df_integrated["box_office_musd"].min() + 1e-6
    )
    w_norm = (df_integrated["weighted_score"] - 1.0) / 9.0
    df_integrated["popularity_index"] = ((0.50 * v_norm + 0.30 * w_norm + 0.20 * bo_norm) * 100).round(1)

    def assign_tier(score):
        if score >= 8.2:
            return "Masterpiece (8.2+)"
        elif score >= 7.4:
            return "Critically Acclaimed (7.4-8.1)"
        elif score >= 6.5:
            return "Crowd Pleaser (6.5-7.3)"
        else:
            return "Mixed Reception (<6.5)"

    df_integrated["audience_tier"] = df_integrated["weighted_score"].apply(assign_tier)

    # --- STEP 5: Build Normalized Bridge Table for Genres (movie_genres) ---
    genre_records = []
    for _, row in df_integrated.iterrows():
        g_list = [g.strip() for g in str(row["genres"]).split("|") if g.strip()]
        for g in g_list:
            genre_records.append({
                "movie_id": row["movie_id"],
                "genre": g
            })
    df_movie_genres = pd.DataFrame(genre_records).drop_duplicates()
    audit["unique_genres_count"] = int(df_movie_genres["genre"].nunique())

    # Save cleaned CSV artifacts
    df_integrated.to_csv(os.path.join(PROCESSED_DIR, "cleaned_movies_integrated.csv"), index=False)
    df_ratings.to_csv(os.path.join(PROCESSED_DIR, "cleaned_ratings.csv"), index=False)
    df_movie_genres.to_csv(os.path.join(PROCESSED_DIR, "movie_genres_bridge.csv"), index=False)
    with open(os.path.join(PROCESSED_DIR, "pipeline_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    # --- STEP 6: Load into PostgreSQL Relational Warehouse ---
    create_postgres_database()

    conn = psycopg2.connect(
        host=PG_CONFIG["host"],
        port=PG_CONFIG["port"],
        database=PG_CONFIG["database"],
        user=PG_CONFIG["user"],
        password=PG_CONFIG["password"],
    )
    cursor = conn.cursor()

    cursor.execute("""
        DROP TABLE IF EXISTS movie_genres CASCADE;
        DROP TABLE IF EXISTS ratings CASCADE;
        DROP TABLE IF EXISTS movies CASCADE;
        DROP TABLE IF EXISTS pipeline_audit CASCADE;

        CREATE TABLE movies (
            movie_id VARCHAR(20) PRIMARY KEY,
            title TEXT NOT NULL,
            release_year INTEGER,
            decade VARCHAR(10),
            genres TEXT,
            primary_genre VARCHAR(50),
            director TEXT,
            runtime_min INTEGER,
            budget_musd NUMERIC(10, 2),
            box_office_musd NUMERIC(10, 2),
            roi_pct NUMERIC(10, 1),
            language VARCHAR(50),
            content_rating VARCHAR(20),
            production_studio TEXT,
            synopsis TEXT,
            rating_count INTEGER,
            avg_rating NUMERIC(4, 2),
            median_rating NUMERIC(4, 2),
            rating_std NUMERIC(4, 2),
            weighted_score NUMERIC(4, 2),
            popularity_index NUMERIC(5, 1),
            total_helpful_votes INTEGER,
            verified_ratio NUMERIC(5, 1),
            audience_tier VARCHAR(50)
        );

        CREATE TABLE movie_genres (
            movie_id VARCHAR(20) REFERENCES movies(movie_id) ON DELETE CASCADE,
            genre VARCHAR(50),
            PRIMARY KEY (movie_id, genre)
        );

        CREATE TABLE ratings (
            rating_id VARCHAR(25) PRIMARY KEY,
            movie_id VARCHAR(20) REFERENCES movies(movie_id) ON DELETE CASCADE,
            user_id VARCHAR(25),
            rating NUMERIC(4, 1),
            review_platform VARCHAR(50),
            viewer_age_group VARCHAR(20),
            is_verified_viewer BOOLEAN,
            helpful_votes INTEGER,
            rated_at TIMESTAMP
        );

        CREATE TABLE pipeline_audit (
            run_timestamp TEXT,
            raw_movies_rows INTEGER,
            clean_movies_rows INTEGER,
            duplicate_titles_merged INTEGER,
            raw_ratings_rows INTEGER,
            clean_ratings_rows INTEGER,
            duplicate_user_ratings_removed INTEGER,
            orphan_ratings_removed INTEGER,
            null_or_oob_ratings_removed INTEGER,
            audit_json TEXT
        );

        CREATE INDEX idx_movies_year ON movies(release_year);
        CREATE INDEX idx_movies_weighted ON movies(weighted_score DESC);
        CREATE INDEX idx_movies_popularity ON movies(popularity_index DESC);
        CREATE INDEX idx_movie_genres_genre ON movie_genres(genre);
        CREATE INDEX idx_ratings_movie_id ON ratings(movie_id);
    """)

    movie_cols = [
        "movie_id", "title", "release_year", "decade", "genres", "primary_genre",
        "director", "runtime_min", "budget_musd", "box_office_musd", "roi_pct",
        "language", "content_rating", "production_studio", "synopsis",
        "rating_count", "avg_rating", "median_rating", "rating_std",
        "weighted_score", "popularity_index", "total_helpful_votes",
        "verified_ratio", "audience_tier"
    ]
    movies_tuples = [
        tuple(None if pd.isna(val) else (val.item() if hasattr(val, "item") else val) for val in row)
        for row in df_integrated[movie_cols].to_numpy()
    ]
    execute_values(
        cursor,
        f"INSERT INTO movies ({', '.join(movie_cols)}) VALUES %s",
        movies_tuples
    )

    genres_tuples = [tuple(x) for x in df_movie_genres[["movie_id", "genre"]].to_numpy()]
    execute_values(
        cursor,
        "INSERT INTO movie_genres (movie_id, genre) VALUES %s",
        genres_tuples
    )

    rating_cols = [
        "rating_id", "movie_id", "user_id", "rating", "review_platform",
        "viewer_age_group", "is_verified_viewer", "helpful_votes", "rated_at"
    ]
    ratings_tuples = [
        tuple(
            bool(val) if col == "is_verified_viewer"
            else (val.item() if hasattr(val, "item") else val)
            for col, val in zip(rating_cols, row)
        )
        for row in df_ratings[rating_cols].to_numpy()
    ]
    execute_values(
        cursor,
        f"INSERT INTO ratings ({', '.join(rating_cols)}) VALUES %s",
        ratings_tuples
    )

    cursor.execute(
        """
        INSERT INTO pipeline_audit (
            run_timestamp, raw_movies_rows, clean_movies_rows, duplicate_titles_merged,
            raw_ratings_rows, clean_ratings_rows, duplicate_user_ratings_removed,
            orphan_ratings_removed, null_or_oob_ratings_removed, audit_json
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """,
        (
            audit["run_timestamp"],
            audit["raw_movies_rows"],
            audit["clean_movies_rows"],
            audit["duplicate_movie_titles_merged"],
            audit["raw_ratings_rows"],
            audit["clean_ratings_rows"],
            audit["duplicate_user_ratings_removed"],
            audit["orphan_ratings_removed"],
            audit["null_ratings_removed"] + audit["out_of_bounds_ratings_removed"],
            json.dumps(audit),
        ),
    )

    conn.commit()
    cursor.close()
    conn.close()

    return audit


if __name__ == "__main__":
    stats = run_etl_pipeline()
    print("ETL Pipeline completed successfully into PostgreSQL:")
    print(json.dumps(stats, indent=2))
