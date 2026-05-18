"""
Rekomendacje filmów oparte na drzewach decyzyjnych uczonych preferencji użytkownika.

Architektura:
  - Ray Data  → wczytanie Parquet
  - ray.put() → jeden egzemplarz cech filmowych w Object Store
  - @ray.remote (train_user_model)     → równoległe trenowanie per użytkownik
  - @ray.remote (recommend_for_user)   → równoległe generowanie rekomendacji
  - sklearn DecisionTreeClassifier     → model preferencji
  - wyniki zapisywane do data/recommendations.csv
"""

import ray
import ray.data
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
import os

warnings.filterwarnings("ignore")

# ── Stałe ─────────────────────────────────────────────────────────────────────

RATINGS_PATH = "data/processed/ratings"
MOVIES_PATH  = "data/processed/movies"
OUTPUT_PATH  = "data/recommendations.csv"
MIN_RATINGS  = 10   # minimalna liczba ocen, by trenować model
LIKE_THRESH  = 3.5  # ocena >= próg → "podoba się"
TOP_N        = 10   # liczba rekomendacji na użytkownika
MAX_DEPTH    = 6    # głębokość drzewa


# ── Ray remote: trenowanie modelu dla jednego użytkownika ─────────────────────

@ray.remote
def train_user_model(
    user_id: int,
    user_ratings: pd.DataFrame,
    movies_features: pd.DataFrame,
    min_ratings: int = MIN_RATINGS,
):
    """
    Trenuje drzewo decyzyjne uczące się, jakie filmy użytkownik lubi.
    Cechy: one-hot encoded gatunki filmowe.
    Etykieta: 1 = lubi (rating >= LIKE_THRESH), 0 = nie lubi.

    Zwraca (user_id, model_or_None).
    """
    if len(user_ratings) < min_ratings:
        return user_id, None

    data = user_ratings.merge(movies_features, on="movieId", how="inner")
    genre_cols = [c for c in movies_features.columns if c not in ("movieId", "title")]

    X = data[genre_cols].values.astype(np.float32)
    y = (data["rating"] >= LIKE_THRESH).astype(int).values

    # Potrzebujemy obu klas, żeby model miał sens
    if len(np.unique(y)) < 2:
        return user_id, None

    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X, y)
    return user_id, clf


# ── Ray remote: generowanie rekomendacji dla jednego użytkownika ──────────────

@ray.remote
def recommend_for_user(
    user_id: int,
    model,
    rated_movie_ids: set,
    movies_features: pd.DataFrame,
    top_n: int = TOP_N,
):
    """
    Na podstawie wytrenowanego modelu ocenia nieobejrzane filmy i zwraca top-N.
    Wynik sortowany malejąco wg P(podoba_się).

    Zwraca (user_id, DataFrame z kolumnami [movieId, title, score]).
    """
    empty = pd.DataFrame(columns=["movieId", "title", "score"])

    if model is None:
        return user_id, empty

    genre_cols = [c for c in movies_features.columns if c not in ("movieId", "title")]
    unrated = movies_features[~movies_features["movieId"].isin(rated_movie_ids)].copy()

    if unrated.empty:
        return user_id, empty

    X = unrated[genre_cols].values.astype(np.float32)
    proba = model.predict_proba(X)[:, 1]   # P(lubi)

    top_idx = np.argsort(proba)[::-1][:top_n]
    result = unrated.iloc[top_idx][["movieId", "title"]].copy()
    result["score"] = np.round(proba[top_idx], 4)
    return user_id, result.reset_index(drop=True)


# ── Feature engineering ───────────────────────────────────────────────────────

def build_movie_features(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy macierz cech filmów: one-hot encoding gatunków.
    Kolumny: movieId, title, <gatunek_1>, <gatunek_2>, ...
    """
    mlb = MultiLabelBinarizer()
    genres_split = movies_df["genres"].str.split("|")
    genre_matrix = mlb.fit_transform(genres_split)
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, dtype=np.float32)
    return pd.concat(
        [movies_df[["movieId", "title"]].reset_index(drop=True), genre_df],
        axis=1,
    )


# ── Główna logika ─────────────────────────────────────────────────────────────

def main():
    ray.init(ignore_reinit_error=True)

    # 1. Wczytanie przetworzonych danych z Parquet (przez Ray Data)
    print("► Wczytywanie danych z Parquet...")
    ratings_df = ray.data.read_parquet(RATINGS_PATH).to_pandas()
    movies_df  = ray.data.read_parquet(MOVIES_PATH).to_pandas()

    print(f"  Oceny:  {len(ratings_df):,} wierszy, {ratings_df['userId'].nunique():,} użytkowników")
    print(f"  Filmy:  {len(movies_df):,} wierszy")

    # 2. One-hot encoding gatunków
    print("\n► Budowanie cech filmowych (one-hot encoding gatunków)...")
    movies_features = build_movie_features(movies_df)
    genre_cols = [c for c in movies_features.columns if c not in ("movieId", "title")]
    print(f"  Gatunki: {genre_cols}")

    # 3. Umieszczenie DataFrame w Object Store Ray → jeden egzemplarz dla wszystkich tasków
    movies_features_ref = ray.put(movies_features)

    # 4. Grupowanie ocen per użytkownik
    user_ids = ratings_df["userId"].unique()
    ratings_by_user = {uid: grp.reset_index(drop=True) for uid, grp in ratings_df.groupby("userId")}

    # 5. Równoległe trenowanie drzew decyzyjnych (jedno na użytkownika)
    print(f"\n► Trenowanie drzew decyzyjnych dla {len(user_ids)} użytkowników (równolegle w Ray)...")
    train_futures = [
        train_user_model.remote(uid, ratings_by_user[uid], movies_features_ref)
        for uid in user_ids
    ]
    train_results = ray.get(train_futures)

    models = {uid: model for uid, model in train_results}
    trained_count = sum(1 for m in models.values() if m is not None)
    print(f"  Wytrenowano modele dla {trained_count}/{len(user_ids)} użytkowników "
          f"(pozostałe pominięte – za mało ocen lub brak kontrastu).")

    # 6. Równoległe generowanie rekomendacji
    print(f"\n► Generowanie top-{TOP_N} rekomendacji (równolegle w Ray)...")
    rec_futures = [
        recommend_for_user.remote(
            uid,
            models[uid],
            set(ratings_by_user[uid]["movieId"].tolist()),
            movies_features_ref,
        )
        for uid in user_ids
    ]
    rec_results = ray.get(rec_futures)

    # 7. Zapis do CSV
    os.makedirs("data", exist_ok=True)
    all_rows = []
    recommendations = {}
    for uid, recs_df in rec_results:
        if not recs_df.empty:
            recs_df.insert(0, "userId", uid)
            all_rows.append(recs_df)
            recommendations[uid] = recs_df

    if all_rows:
        output_df = pd.concat(all_rows, ignore_index=True)
        output_df.to_csv(OUTPUT_PATH, index=False)
        print(f"  Zapisano rekomendacje dla {len(recommendations)} użytkowników → {OUTPUT_PATH}")
    else:
        print("  Brak rekomendacji do zapisania.")

    # 8. Podgląd wyników dla kilku użytkowników
    print(f"\n{'═'*60}")
    print(f"  Przykładowe rekomendacje (pierwsze 5 użytkowników)")
    print(f"{'═'*60}")
    for uid in list(recommendations.keys())[:5]:
        rated_count = len(ratings_by_user[uid])
        like_count  = (ratings_by_user[uid]["rating"] >= LIKE_THRESH).sum()
        print(f"\n  Użytkownik {uid}  (ocenił {rated_count} filmów, lubił {like_count})")
        print(recommendations[uid][["title", "score"]].to_string(index=False))

    ray.shutdown()


if __name__ == "__main__":
    main()
