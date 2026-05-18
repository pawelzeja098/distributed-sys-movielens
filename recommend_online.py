"""
System rekomendacji filmów z douczaniem modelu po nowej ocenie użytkownika.

Architektura:
  - 10 wybranych użytkowników (top-10 aktywnych)
  - Modele zapisywane na dysku (joblib) → data/models/user_{id}.pkl
  - Oceny użytkownika zapisywane na dysku → data/user_ratings/user_{id}.csv
  - Po nowej ocenie: Ray retrenuje model tylko tego użytkownika
  - Widać zmianę rekomendacji przed/po ocenieniu filmu

Uruchomienie:
  python recommend_online.py              # pełna inicjalizacja + demo douczania
  python recommend_online.py --recs 42   # pokaż rekomendacje dla użytkownika 42
  python recommend_online.py --rate 42 318 5.0  # oceń film 318 na 5.0 i douucz
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import ray
import ray.data
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")

# ── Ścieżki i hiperparametry ──────────────────────────────────────────────────

RATINGS_PARQUET  = "data/processed/ratings"
MOVIES_PARQUET   = "data/processed/movies"
MODELS_DIR       = Path("data/models")
USER_RATINGS_DIR = Path("data/user_ratings")

NUM_USERS   = 10    # ilu użytkowników obsługujemy
MIN_RATINGS = 5     # min. ocen do trenowania
LIKE_THRESH = 3.5   # próg „podoba mi się"
MAX_DEPTH   = 6     # głębokość drzewa
TOP_N       = 10    # liczba rekomendacji


# ── Inicjalizacja katalogów ───────────────────────────────────────────────────

MODELS_DIR.mkdir(parents=True, exist_ok=True)
USER_RATINGS_DIR.mkdir(parents=True, exist_ok=True)


# ── Pomocnicze I/O ────────────────────────────────────────────────────────────

def model_path(user_id: int) -> Path:
    return MODELS_DIR / f"user_{user_id}.pkl"


def ratings_path(user_id: int) -> Path:
    return USER_RATINGS_DIR / f"user_{user_id}.csv"


def load_user_ratings(user_id: int) -> pd.DataFrame:
    p = ratings_path(user_id)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=["userId", "movieId", "rating"])


def save_user_ratings(user_id: int, df: pd.DataFrame) -> None:
    df.to_csv(ratings_path(user_id), index=False)


def load_model(user_id: int):
    p = model_path(user_id)
    return joblib.load(p) if p.exists() else None


def save_model(user_id: int, model) -> None:
    joblib.dump(model, model_path(user_id))


def list_managed_users() -> list[int]:
    """Zwraca user_id dla których mamy zapisane pliki ocen."""
    return sorted(
        int(p.stem.split("_")[1])
        for p in USER_RATINGS_DIR.glob("user_*.csv")
    )


def create_user(user_id: int | None = None) -> int:
    """
    Tworzy nowego użytkownika z pustą historią ocen i zwraca jego user_id.
    Jeśli user_id nie podano, generuje kolejny dostępny (>= 1000,
    żeby nie kolidować z ID użytkowników MovieLens 1–610).
    """
    existing = list_managed_users()
    if user_id is None:
        user_id = max(1000, max(existing, default=999) + 1)
    if user_id not in existing:
        save_user_ratings(
            user_id,
            pd.DataFrame(columns=["userId", "movieId", "rating"]),
        )
    return user_id


def compute_popular_movies(
    ratings_df: pd.DataFrame,
    movies_features: pd.DataFrame,
    min_votes: int = 50,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Rekomendacje cold-start: Bayesian-weighted średnia ocen.
    Normalizuje score do 0–1 dla spójności z modelem.
    """
    global_mean = ratings_df["rating"].mean()
    stats = (
        ratings_df.groupby("movieId")["rating"]
        .agg(n="count", mean="mean")
        .reset_index()
    )
    # Bayesian average: przesuwa filmy z małą liczbą ocen w dół
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


# ── Feature engineering ───────────────────────────────────────────────────────

def build_movie_features(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encoding gatunków. Kolumny: movieId, title, <gatunek_1>, ...
    MLBinarizer jest deterministyczny dla tych samych danych → spójna kolejność.
    """
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies_df["genres"].str.split("|"))
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, dtype=np.float32)
    return pd.concat(
        [movies_df[["movieId", "title"]].reset_index(drop=True), genre_df],
        axis=1,
    )


def genre_cols(movies_features: pd.DataFrame) -> list[str]:
    return [c for c in movies_features.columns if c not in ("movieId", "title")]


# ── Ray remote: trening modelu jednego użytkownika ────────────────────────────

@ray.remote
def _train_remote(
    user_id: int,
    user_ratings: pd.DataFrame,
    movies_features: pd.DataFrame,
) -> tuple:
    """
    Trenuje DecisionTreeClassifier dla jednego użytkownika.
    Cechy = gatunki filmowe (one-hot), etykieta = rating >= LIKE_THRESH.
    Zwraca (user_id, model_lub_None).
    """
    gcols = [c for c in movies_features.columns if c not in ("movieId", "title")]

    if len(user_ratings) < MIN_RATINGS:
        return user_id, None

    data = user_ratings.merge(movies_features, on="movieId", how="inner")
    X = data[gcols].values.astype(np.float32)
    y = (data["rating"] >= LIKE_THRESH).astype(int).values

    if len(np.unique(y)) < 2:
        # Użytkownik oceniał tylko pozytywnie LUB tylko negatywnie – brak kontrastu
        return user_id, None

    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X, y)
    return user_id, clf


# ── Publiczne API ─────────────────────────────────────────────────────────────

def get_recommendations(
    user_id: int,
    movies_features: pd.DataFrame,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Generuje top-N rekomendacji dla użytkownika na podstawie zapisanego modelu."""
    model = load_model(user_id)
    if model is None:
        return pd.DataFrame(columns=["movieId", "title", "score"])

    gcols = genre_cols(movies_features)
    rated_ids = set(load_user_ratings(user_id)["movieId"].tolist())
    unrated = movies_features[~movies_features["movieId"].isin(rated_ids)].copy()

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
    Dodaje ocenę użytkownika do jego historii, a następnie
    retrenuje model w Ray i zapisuje go na dysku.

    Zwraca zaktualizowane rekomendacje.
    """
    if user_id not in list_managed_users():
        print(f"  [!] Użytkownik {user_id} nie jest zarządzany. Dostępni: {list_managed_users()}")
        return pd.DataFrame()

    # Nazwa filmu dla czytelniejszego logu
    title_series = movies_features.loc[movies_features["movieId"] == movie_id, "title"]
    title = title_series.values[0] if not title_series.empty else f"movieId={movie_id}"

    print(f"\n  Ocena: użytkownik {user_id}  →  '{title}'  ({rating}/5.0)")

    # Wczytaj istniejące oceny, dopisz nową (nadpisz jeśli już oceniał)
    existing = load_user_ratings(user_id)
    existing = existing[existing["movieId"] != movie_id]
    new_row = pd.DataFrame([{"userId": user_id, "movieId": movie_id, "rating": rating}])
    updated = pd.concat([existing, new_row], ignore_index=True)
    save_user_ratings(user_id, updated)
    print(f"  Historia: {len(updated)} ocen  "
          f"({int((updated['rating'] >= LIKE_THRESH).sum())} lubianych / "
          f"{int((updated['rating'] < LIKE_THRESH).sum())} nielubianych)")

    # Retrenuj model przez Ray
    print("  Douczanie modelu (Ray remote)...")
    ray.init(ignore_reinit_error=True)
    movies_ref = ray.put(movies_features)
    uid, model = ray.get(_train_remote.remote(user_id, updated, movies_ref))

    if model is not None:
        save_model(uid, model)
        print(f"  Model zapisany → {model_path(uid)}")
    else:
        print("  Nie można wytrenować modelu (za mało danych / brak kontrastu ocen).")

    recs = get_recommendations(user_id, movies_features)
    return recs


# ── Inicjalizacja systemu (pierwsze uruchomienie) ─────────────────────────────

def init_system(force: bool = False) -> pd.DataFrame:
    """
    Wczytuje dane, wybiera top-10 aktywnych użytkowników,
    trenuje modele równolegle w Ray i zapisuje je na dysku.
    Zwraca movies_features do dalszego użycia.
    """
    already_trained = list_managed_users()
    if already_trained and not force:
        print(f"  Modele już istnieją dla użytkowników: {already_trained}")
        print("  Pomijam inicjalizację (użyj force=True aby wymusić).")
        ray.init(ignore_reinit_error=True)
        # Wczytaj movies_features
        movies_df = ray.data.read_parquet(MOVIES_PARQUET).to_pandas()
        return build_movie_features(movies_df)

    ray.init(ignore_reinit_error=True)

    print("► Wczytywanie danych z Parquet...")
    ratings_df = ray.data.read_parquet(RATINGS_PARQUET).to_pandas()
    movies_df  = ray.data.read_parquet(MOVIES_PARQUET).to_pandas()
    print(f"  Oceny: {len(ratings_df):,}  |  Filmy: {len(movies_df):,}")

    print("► Budowanie cech filmowych (one-hot gatunki)...")
    movies_features = build_movie_features(movies_df)
    print(f"  Cechy gatunkowe: {genre_cols(movies_features)}")

    # Wybierz top-N aktywnych użytkowników
    top_users = (
        ratings_df.groupby("userId")["movieId"]
        .count()
        .nlargest(NUM_USERS)
        .index.tolist()
    )
    print(f"\n► Wybrani użytkownicy (top {NUM_USERS} aktywnych): {top_users}")

    # Zapisz oceny + trenuj w Ray
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

    for uid, model in results:
        save_user_ratings(uid, ratings_by_user[uid])
        if model is not None:
            save_model(uid, model)

    trained = sum(1 for _, m in results if m is not None)
    print(f"  Wytrenowano {trained}/{NUM_USERS} modeli. Zapisano do {MODELS_DIR}/")

    return movies_features


# ── Demo douczania ────────────────────────────────────────────────────────────

def run_demo(movies_features: pd.DataFrame) -> None:
    """
    Wybiera pierwszego zarządzanego użytkownika, pokazuje rekomendacje,
    każe mu ocenić film z gatunku Drama (wysoko) i prezentuje zmianę.
    """
    users = list_managed_users()
    if not users:
        print("Brak zainicjalizowanych użytkowników.")
        return

    uid = users[0]
    print(f"\n{'═'*62}")
    print(f"  DEMO DOUCZANIA – użytkownik {uid}")
    print(f"{'═'*62}")

    # Rekomendacje PRZED
    recs_before = get_recommendations(uid, movies_features)
    print(f"\n  Rekomendacje PRZED nową oceną:")
    print(recs_before[["title", "score"]].to_string(index=False))

    # Znajdź nieobejrzany film z gatunku Drama i oceń go wysoko
    rated_ids = set(load_user_ratings(uid)["movieId"].tolist())
    unrated = movies_features[~movies_features["movieId"].isin(rated_ids)]

    drama_col = "Drama" if "Drama" in unrated.columns else None
    candidate = (
        unrated[unrated[drama_col] == 1].iloc[0]
        if drama_col and not unrated[unrated[drama_col] == 1].empty
        else unrated.iloc[0]
    )

    print(f"\n{'─'*62}")
    print(f"  Użytkownik ogląda i ocenia film:")
    recs_after = rate_and_retrain(uid, int(candidate["movieId"]), 5.0, movies_features)

    print(f"\n  Rekomendacje PO nowej ocenie:")
    if not recs_after.empty:
        print(recs_after[["title", "score"]].to_string(index=False))
    else:
        print("  (brak)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MovieLens – douczanie drzew decyzyjnych")
    parser.add_argument("--recs",  type=int, metavar="USER_ID",
                        help="Pokaż rekomendacje dla użytkownika")
    parser.add_argument("--rate",  nargs=3, metavar=("USER_ID", "MOVIE_ID", "RATING"),
                        help="Oceń film i douucz model: --rate 42 318 5.0")
    parser.add_argument("--force-init", action="store_true",
                        help="Wymuś ponowną inicjalizację (nadpisz modele)")
    args = parser.parse_args()

    movies_features = init_system(force=args.force_init)

    if args.recs:
        uid = args.recs
        recs = get_recommendations(uid, movies_features)
        if recs.empty:
            print(f"Brak rekomendacji dla użytkownika {uid} "
                  f"(nie zarządzany lub brak modelu).")
        else:
            print(f"\nRekomendacje dla użytkownika {uid}:")
            print(recs[["movieId", "title", "score"]].to_string(index=False))

    elif args.rate:
        uid, mid, rat = int(args.rate[0]), int(args.rate[1]), float(args.rate[2])
        recs = rate_and_retrain(uid, mid, rat, movies_features)
        if not recs.empty:
            print(f"\nNowe rekomendacje dla użytkownika {uid}:")
            print(recs[["movieId", "title", "score"]].to_string(index=False))

    else:
        # Domyślnie: inicjalizacja + demo
        run_demo(movies_features)

    ray.shutdown()


if __name__ == "__main__":
    main()
