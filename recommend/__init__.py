"""
Pakiet recommend — publiczne API systemu rekomendacji MovieLens.

Re-eksportuje wszystko czego potrzebuje app.py, dzięki czemu:
  from recommend import init_system, get_recommendations, ...
działa tak samo jak wcześniej z recommend_online.py.
"""

from .config import LIKE_THRESH, MIN_RATINGS, MOVIES_PARQUET, RATINGS_PARQUET, TOP_N
from .features import build_movie_features, genre_cols
from .recommender import (
    compute_popular_movies,
    get_recommendations,
    init_system,
    rate_and_retrain,
)
from .registry import ModelRegistry, get_registry
from .storage import (
    create_user,
    list_managed_users,
    load_model,
    load_user_ratings,
    model_path,
    save_model,
    save_user_ratings,
)

__all__ = [
    # config
    "LIKE_THRESH", "MIN_RATINGS", "MOVIES_PARQUET", "RATINGS_PARQUET", "TOP_N",
    # features
    "build_movie_features", "genre_cols",
    # recommender
    "compute_popular_movies", "get_recommendations", "init_system", "rate_and_retrain",
    # registry
    "ModelRegistry", "get_registry",
    # storage
    "create_user", "list_managed_users", "load_model", "load_user_ratings",
    "model_path", "save_model", "save_user_ratings",
]
