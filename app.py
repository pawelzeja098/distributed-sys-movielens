"""
Flask + Bootstrap – panel rekomendacji filmowych z douczaniem modelu.

Uruchomienie:
  python app.py
Następnie otwórz: http://localhost:5000
"""

import os
import sys

# Ustaw katalog roboczy na lokalizację tego skryptu
# (ważne – ścieżki do data/ w recommend_online są względne)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import ray
import ray.data as _ray_data
from flask import Flask, flash, redirect, render_template, request, url_for
from recommend_online import (
    MIN_RATINGS,
    RATINGS_PARQUET,
    compute_popular_movies,
    create_user,
    get_recommendations,
    init_system,
    list_managed_users,
    load_model,
    load_user_ratings,
    rate_and_retrain,
)

app = Flask(__name__)
app.secret_key = "movielens-dev-2026"

# ── Inicjalizacja Ray + danych raz przy starcie serwera ───────────────────────
print("► Inicjalizacja systemu rekomendacji (Ray + modele)…")
_movies_features = init_system()
print("► Wczytywanie ocen do obliczenia popularnych filmów…")
_ratings_df = _ray_data.read_parquet(RATINGS_PARQUET).to_pandas()
_popular_movies = compute_popular_movies(_ratings_df, _movies_features, top_n=20)
_movies_genres: dict[int, str] = (
    pd.read_csv("data/ml-latest-small/movies.csv", usecols=["movieId", "genres"])
    .set_index("movieId")["genres"]
    .to_dict()
)
print(f"► System gotowy. Popularne filmy: {len(_popular_movies)} pozycji.\n")


# ── Helpery ───────────────────────────────────────────────────────────────────

def _user_stats(uid: int) -> dict:
    ratings = load_user_ratings(uid)
    has_model = load_model(uid) is not None
    liked = int((ratings["rating"] >= 3.5).sum())
    rated_count = len(ratings)
    recs = get_recommendations(uid, _movies_features)
    return {
        "id": uid,
        "rated_count": rated_count,
        "liked_count": liked,
        "disliked_count": rated_count - liked,
        "has_model": has_model,
        "rec_count": len(recs),
        "ratings_progress": min(rated_count, MIN_RATINGS),
        "ratings_needed": MIN_RATINGS,
    }


# ── Widoki ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    users = [_user_stats(uid) for uid in list_managed_users()]
    return render_template("index.html", users=users)


@app.route("/user/<int:user_id>")
def user_page(user_id: int):
    if user_id not in list_managed_users():
        flash(f"Użytkownik {user_id} nie jest zarządzany przez system.", "warning")
        return redirect(url_for("index"))

    stats = _user_stats(user_id)
    is_cold_start = not stats["has_model"]

    rated = (
        load_user_ratings(user_id)
        .merge(_movies_features[["movieId", "title"]], on="movieId", how="left")
        .sort_values("rating", ascending=False)
    )
    rated_ids = set(rated["movieId"].tolist())

    if is_cold_start:
        # Cold-start: pokaż popularne filmy z pominięciem już ocenionych
        recs = _popular_movies[~_popular_movies["movieId"].isin(rated_ids)].head(10)
    else:
        recs = get_recommendations(user_id, _movies_features)

    unrated = (
        _movies_features[~_movies_features["movieId"].isin(rated_ids)][["movieId", "title"]]
        .sort_values("title")
    )

    def _add_genres(records: list[dict]) -> list[dict]:
        for r in records:
            r["genres"] = _movies_genres.get(r["movieId"], "")
        return records

    return render_template(
        "user.html",
        user_id=user_id,
        stats=stats,
        is_cold_start=is_cold_start,
        recommendations=_add_genres(recs.to_dict("records")),
        rated_movies=_add_genres(rated.to_dict("records")),
        unrated_movies=unrated.to_dict("records"),
    )


@app.route("/user/new", methods=["POST"])
def new_user():
    uid = create_user()
    flash(
        f"Utworzono nowego użytkownika <strong>#{uid}</strong>. "
        f"Oceń co najmniej {MIN_RATINGS} filmów, żeby zbudować swój model preferencji!",
        "info",
    )
    return redirect(url_for("user_page", user_id=uid))


@app.route("/user/<int:user_id>/rate", methods=["POST"])
def rate_movie(user_id: int):
    if user_id not in list_managed_users():
        flash("Nieznany użytkownik.", "danger")
        return redirect(url_for("index"))

    movie_id = int(request.form["movie_id"])
    rating = float(request.form["rating"])

    title_col = _movies_features.loc[_movies_features["movieId"] == movie_id, "title"]
    title = title_col.values[0] if not title_col.empty else f"Film #{movie_id}"

    rate_and_retrain(user_id, movie_id, rating, _movies_features)

    sentiment = "Lubię to! 👍" if rating >= 3.5 else "Nie podobało się 👎"
    flash(
        f"Oceniono <strong>{title}</strong> na {rating}/5.0 — {sentiment} "
        f"Model został douczony.",
        "success",
    )
    return redirect(url_for("user_page", user_id=user_id))


# ── Start ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # debug=False żeby uniknąć podwójnego ładowania Ray przez reloader
    app.run(host="0.0.0.0", port=5000, debug=False)
