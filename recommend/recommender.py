"""
Silnik rekomendacji: trening modeli, generowanie rekomendacji, inicjalizacja.

Trening:
  - _train_remote()   : Ray remote task — DecisionTreeClassifier per użytkownik
  - rate_and_retrain(): zapisuje ocenę + odpala trening w Ray asynchronicznie

Rekomendacje:
  - get_recommendations(): model (predict_proba) lub fallback (cosinus podobieństwa)
  - compute_popular_movies(): Bayesian ranking dla cold-start (brak modelu)

Inicjalizacja:
  - init_system(): uruchamiany raz przy starcie serwera — ładuje dane, trenuje modele
"""

import joblib
import numpy as np
import pandas as pd
import ray
import ray.data
from sklearn.tree import DecisionTreeClassifier

from .config import (
    LIKE_THRESH, MAX_DEPTH, MIN_RATINGS, MODELS_DIR, MOVIES_PARQUET,
    NUM_USERS, RATINGS_PARQUET, TAGS_CSV, TOP_N,
)
from .features import build_movie_features, genre_cols
from .registry import get_registry
from .storage import (
    list_managed_users, load_model, load_user_ratings, model_path,
    save_model, save_user_ratings,
)


# ── Ray remote task: trening jednego drzewa decyzyjnego ───────────────────────

@ray.remote
def _train_remote(
    user_id: int,
    user_ratings: pd.DataFrame,
    movies_features: pd.DataFrame,
) -> tuple:
    """
    Trenuje DecisionTreeClassifier dla jednego użytkownika.
      Cechy   = gatunki one-hot + tagvec PCA
      Etykieta = rating >= LIKE_THRESH  →  1 (lubi) / 0 (nie lubi)

    Zwraca (user_id, model_lub_None).
    Zwraca None gdy: za mało ocen LUB brak kontrastu (tylko jedna klasa).
    """
    gcols = [c for c in movies_features.columns if c not in ("movieId", "title")]

    if len(user_ratings) < MIN_RATINGS:
        return user_id, None

    data = user_ratings.merge(movies_features, on="movieId", how="inner")
    X = data[gcols].values.astype(np.float32)
    y = (data["rating"] >= LIKE_THRESH).astype(int).values

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


# ── Rekomendacje ──────────────────────────────────────────────────────────────

def compute_popular_movies(
    ratings_df: pd.DataFrame,
    movies_features: pd.DataFrame,
    min_votes: int = 50,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Rekomendacje cold-start: Bayesian-weighted średnia ocen.
    Bayesian average przesuwa filmy z małą liczbą ocen w kierunku średniej globalnej.
    Score znormalizowany do 0–1 dla spójności z wyjściem modelu.
    """
    global_mean = ratings_df["rating"].mean()
    stats = (
        ratings_df.groupby("movieId")["rating"]
        .agg(n="count", mean="mean")
        .reset_index()
    )
    stats["score"] = (
        (stats["n"] * stats["mean"] + min_votes * global_mean)
        / (stats["n"] + min_votes)
    )
    top = (
        stats[stats["n"] >= min_votes]
        .nlargest(top_n, "score")
        .merge(movies_features[["movieId", "title"]], on="movieId", how="left")
    )
    top["score"] = (top["score"] / 5.0).round(4)
    return top[["movieId", "title", "score"]].reset_index(drop=True)


def get_recommendations(
    user_id: int,
    movies_features: pd.DataFrame,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """
    Generuje top-N rekomendacji dla użytkownika.

    Jeśli model istnieje → predict_proba() na DecisionTree (gatunki + tagvec PCA).
    Fallback (brak modelu) → cosinus podobieństwa do profilu lubianych filmów,
      tylko po kolumnach gatunków (tagvec PCA ma wartości ujemne — cosinus nie działa).
    """
    model = load_model(user_id)
    gcols = genre_cols(movies_features)
    rated_ids = set(load_user_ratings(user_id)["movieId"].tolist())
    unrated = movies_features[~movies_features["movieId"].isin(rated_ids)].copy()

    if model is None:
        genre_only = [c for c in gcols if not c.startswith("tagvec_")]
        user_ratings = load_user_ratings(user_id)
        liked = user_ratings[user_ratings["rating"] >= LIKE_THRESH]
        if liked.empty or unrated.empty:
            return pd.DataFrame(columns=["movieId", "title", "score"])
        liked_features = liked.merge(movies_features, on="movieId", how="inner")
        if liked_features.empty:
            return pd.DataFrame(columns=["movieId", "title", "score"])
        profile = liked_features[genre_only].values.astype(np.float32).mean(axis=0)
        profile_norm = np.linalg.norm(profile)
        if profile_norm == 0:
            return pd.DataFrame(columns=["movieId", "title", "score"])
        profile = profile / profile_norm
        X = unrated[genre_only].values.astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        scores = (X / norms) @ profile
        unrated = unrated.copy()
        unrated["score"] = np.round(scores, 4)
        return unrated.nlargest(top_n, "score")[["movieId", "title", "score"]].reset_index(drop=True)

    if unrated.empty:
        return pd.DataFrame(columns=["movieId", "title", "score"])

    proba = model.predict_proba(unrated[gcols].values.astype(np.float32))[:, 1]
    top_idx = np.argsort(proba)[::-1][:top_n]
    result = unrated.iloc[top_idx][["movieId", "title"]].copy()
    result["score"] = np.round(proba[top_idx], 4)
    return result.reset_index(drop=True)


def rate_and_retrain(
    user_id: int,
    movie_id: int,
    rating: float,
    movies_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Dodaje ocenę użytkownika do jego historii, a następnie retrenuje model w Ray.
    Zwraca zaktualizowane rekomendacje.
    """
    if user_id not in list_managed_users():
        print(f"  [!] Użytkownik {user_id} nie jest zarządzany.")
        return pd.DataFrame()

    title_series = movies_features.loc[movies_features["movieId"] == movie_id, "title"]
    title = title_series.values[0] if not title_series.empty else f"movieId={movie_id}"
    print(f"\n  Ocena: użytkownik {user_id}  →  '{title}'  ({rating}/5.0)")

    existing = load_user_ratings(user_id)
    existing = existing[existing["movieId"] != movie_id]
    new_row = pd.DataFrame([{"userId": user_id, "movieId": movie_id, "rating": rating}])
    updated = pd.concat([existing, new_row], ignore_index=True)
    save_user_ratings(user_id, updated)
    print(f"  Historia: {len(updated)} ocen  "
          f"({int((updated['rating'] >= LIKE_THRESH).sum())} lubianych / "
          f"{int((updated['rating'] < LIKE_THRESH).sum())} nielubianych)")

    print("  Douczanie modelu (Ray remote)...")
    ray.init(ignore_reinit_error=True, namespace="movielens")
    movies_ref = ray.put(movies_features)
    uid, model = ray.get(_train_remote.remote(user_id, updated, movies_ref))

    if model is not None:
        save_model(uid, model)
        print(f"  Model zapisany → {model_path(uid)}")
    else:
        n_liked = int((updated["rating"] >= LIKE_THRESH).sum())
        n_disliked = int((updated["rating"] < LIKE_THRESH).sum())
        if len(updated) < MIN_RATINGS:
            print(f"  Brak modelu – za mało ocen ({len(updated)}/{MIN_RATINGS}). Rekomendacje przez podobieństwo.")
        elif n_liked == 0 or n_disliked == 0:
            print(f"  Brak kontrastu ({n_liked} lubianych / {n_disliked} nielubianych). Rekomendacje przez podobieństwo do profilu.")

    return get_recommendations(user_id, movies_features)


# ── Inicjalizacja systemu ─────────────────────────────────────────────────────

def init_system(force: bool = False) -> pd.DataFrame:
    """
    Uruchamiany raz przy starcie serwera.
    Wczytuje dane, trenuje modele równolegle w Ray i zwraca movies_features.

    Tryb szybki (domyślny): jeśli modele już są na dysku, wczytuje je do
    ModelRegistry i zwraca features z cache — start w ~5s zamiast ~60s.
    Tryb pełny (force=True): pełna inicjalizacja od zera.
    """
    already_trained = [uid for uid in list_managed_users() if model_path(uid).exists()]

    if already_trained and not force:
        print(f"  Modele już istnieją dla użytkowników: {already_trained}")
        print("  Pomijam inicjalizację (użyj force=True aby wymusić).")
        ray.init(ignore_reinit_error=True, namespace="movielens")

        registry = get_registry()
        restored = 0
        for uid in already_trained:
            p = model_path(uid)
            if p.exists():
                ray.get(registry.save.remote(uid, joblib.load(p)))
                restored += 1
        print(f"  Przywrócono {restored} modeli do ModelRegistry.")

        movies_df = ray.data.read_parquet(MOVIES_PARQUET).to_pandas()
        tags_df = pd.read_csv(TAGS_CSV, usecols=["movieId", "tag"])
        return build_movie_features(movies_df, tags_df)

    ray.init(ignore_reinit_error=True, namespace="movielens")

    print("► Wczytywanie danych z Parquet...")
    ratings_df = ray.data.read_parquet(RATINGS_PARQUET).to_pandas()
    movies_df  = ray.data.read_parquet(MOVIES_PARQUET).to_pandas()
    print(f"  Oceny: {len(ratings_df):,}  |  Filmy: {len(movies_df):,}")

    print("► Budowanie cech filmowych (one-hot gatunki + tagi)...")
    tags_df = pd.read_csv(TAGS_CSV, usecols=["movieId", "tag"])
    movies_features = build_movie_features(movies_df, tags_df)
    print(f"  Łączna liczba cech: {len(genre_cols(movies_features))}")

    top_users = (
        ratings_df.groupby("userId")["movieId"]
        .count()
        .nlargest(NUM_USERS)
        .index.tolist()
    )
    print(f"\n► Wybrani użytkownicy (top {NUM_USERS} aktywnych): {top_users}")

    movies_ref = ray.put(movies_features)
    ratings_by_user = {
        uid: grp.reset_index(drop=True)
        for uid, grp in ratings_df[ratings_df["userId"].isin(top_users)].groupby("userId")
    }

    print(f"► Równoległe trenowanie {NUM_USERS} drzew decyzyjnych w Ray...")
    futures = [
        _train_remote.remote(uid, ratings_by_user[uid], movies_ref)
        for uid in top_users
    ]
    results = ray.get(futures)

    registry = get_registry()
    for uid, model in results:
        save_user_ratings(uid, ratings_by_user[uid])
        if model is not None:
            joblib.dump(model, model_path(uid))
            ray.get(registry.save.remote(uid, model))

    trained = sum(1 for _, m in results if m is not None)
    print(f"  Wytrenowano {trained}/{NUM_USERS} modeli. Zapisano do {MODELS_DIR}/ i ModelRegistry.")

    return movies_features
