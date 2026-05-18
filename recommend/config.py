"""
Stałe konfiguracyjne: ścieżki do danych, hiperparametry modelu i systemu.
"""

from pathlib import Path

# ── Ścieżki ───────────────────────────────────────────────────────────────────

RATINGS_PARQUET  = "data/processed/ratings"
MOVIES_PARQUET   = "data/processed/movies"
TAGS_CSV         = "data/ml-latest-small/tags.csv"
FEATURES_CACHE   = Path("data/processed/movies_features.parquet")
MODELS_DIR       = Path("data/models")
USER_RATINGS_DIR = Path("data/user_ratings")

# Upewnij się, że katalogi istnieją przy imporcie modułu
MODELS_DIR.mkdir(parents=True, exist_ok=True)
USER_RATINGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hiperparametry ────────────────────────────────────────────────────────────

NUM_USERS        = 10   # ilu użytkowników obsługujemy w trybie batch
MIN_RATINGS      = 5    # minimalna liczba ocen do trenowania modelu
LIKE_THRESH      = 3.5  # ocena >= próg → "podoba mi się"
MAX_DEPTH        = 6    # maksymalna głębokość drzewa decyzyjnego
TOP_N            = 10   # liczba rekomendacji zwracanych użytkownikowi
TOP_TAGS         = 50   # ile najpopularniejszych tagów brać jako cechy
N_TAG_COMPONENTS = 20   # wymiary PCA dla wektorów semantycznych tagów
