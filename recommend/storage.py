"""
Warstwa I/O: odczyt i zapis ocen użytkowników oraz modeli na dysk.

Modele zapisywane są w dwóch miejscach:
  - ModelRegistry (pamięć Ray) — szybki odczyt O(1), zerowe I/O
  - data/models/user_*.pkl    — trwałość po restarcie klastra/serwera
"""

from pathlib import Path

import joblib
import pandas as pd
import ray

from .config import MODELS_DIR, USER_RATINGS_DIR
from .registry import get_registry


# ── Ścieżki ───────────────────────────────────────────────────────────────────

def model_path(user_id: int) -> Path:
    return MODELS_DIR / f"user_{user_id}.pkl"


def ratings_path(user_id: int) -> Path:
    return USER_RATINGS_DIR / f"user_{user_id}.csv"


# ── Oceny użytkowników ────────────────────────────────────────────────────────

def load_user_ratings(user_id: int) -> pd.DataFrame:
    p = ratings_path(user_id)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=["userId", "movieId", "rating"])


def save_user_ratings(user_id: int, df: pd.DataFrame) -> None:
    df.to_csv(ratings_path(user_id), index=False)


# ── Modele ────────────────────────────────────────────────────────────────────

def load_model(user_id: int):
    """
    Ładuje model z ModelRegistry (pamięć Ray) — O(1), zero I/O.
    Fallback na dysk jeśli registry nie ma modelu (np. po restarcie klastra).
    """
    try:
        registry = get_registry()
        model = ray.get(registry.load.remote(user_id))
        if model is not None:
            return model
    except Exception:
        pass
    p = model_path(user_id)
    return joblib.load(p) if p.exists() else None


def save_model(user_id: int, model) -> None:
    """Zapisuje model do ModelRegistry (RAM) i na dysk (trwałość po restarcie)."""
    registry = get_registry()
    ray.get(registry.save.remote(user_id, model))
    joblib.dump(model, model_path(user_id))


# ── Zarządzanie użytkownikami ─────────────────────────────────────────────────

def list_managed_users() -> list[int]:
    """Zwraca user_id dla których mamy zapisane pliki ocen."""
    return sorted(
        int(p.stem.split("_")[1])
        for p in USER_RATINGS_DIR.glob("user_*.csv")
    )


def create_user(user_id: int | None = None) -> int:
    """
    Tworzy nowego użytkownika z pustą historią ocen i zwraca jego user_id.
    Generuje ID >= 1000, żeby nie kolidować z ID użytkowników MovieLens (1–610).
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
