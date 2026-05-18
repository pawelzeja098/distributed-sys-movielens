"""
Feature engineering: one-hot encoding gatunków + semantyczne embeddingi tagów.

Pipeline:
  1. Gatunki → MultiLabelBinarizer → kolumny binarne (0/1)
  2. Tagi    → sentence-transformers (all-MiniLM-L6-v2) → wektory 384-dim
              → uśrednianie per film → PCA(20) → kolumny tagvec_0..19
  3. Wynik cache'owany w FEATURES_CACHE — wczytywany przy restarcie (~0.1s vs ~30s)

Enkodowanie tagów działa równolegle: 4 Ray workery, każdy ładuje model ST osobno.
"""

import numpy as np
import pandas as pd
import ray
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

from .config import FEATURES_CACHE, N_TAG_COMPONENTS, TOP_TAGS


@ray.remote
def _encode_tags_batch(tags_chunk: list[str]) -> np.ndarray:
    """
    Enkoduje listę tagów przez sentence-transformers w osobnym Ray workerze.
    Wywoływany równolegle dla N chunków — każdy worker ładuje model niezależnie.
    Model jest cache'owany lokalnie po pierwszym pobraniu (~90 MB).
    """
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("all-MiniLM-L6-v2")
    return st.encode(tags_chunk, batch_size=128, show_progress_bar=False)


def build_movie_features(
    movies_df: pd.DataFrame,
    tags_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Buduje macierz cech filmów:
      - gatunki: one-hot encoded przez MultiLabelBinarizer
      - tagi: semantyczne embeddingi (sentence-transformers) zredukowane PCA

    Wynik cache'owany w FEATURES_CACHE (Parquet). Przy restarcie serwera
    wczytywany z dysku zamiast ponownie enkodować (~30s oszczędności).
    """
    if FEATURES_CACHE.exists() and tags_df is not None:
        print(f"  Cechy filmów z cache ({FEATURES_CACHE}). Pomijam enkodowanie tagów.")
        return pd.read_parquet(FEATURES_CACHE)

    # ── Gatunki: one-hot encoding ──────────────────────────────────────────────
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies_df["genres"].str.split("|"))
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, dtype=np.float32)
    base = pd.concat(
        [movies_df[["movieId", "title"]].reset_index(drop=True), genre_df],
        axis=1,
    )

    # ── Tagi: sentence-transformers + PCA ─────────────────────────────────────
    if tags_df is not None and not tags_df.empty:
        from sentence_transformers import SentenceTransformer  # noqa: F401 – trigger cache check

        tags_df = tags_df.copy()
        tags_df["tag"] = tags_df["tag"].str.lower().str.strip()
        unique_tags = tags_df["tag"].dropna().unique().tolist()

        n_workers = min(4, len(unique_tags))
        chunks = [c.tolist() for c in np.array_split(unique_tags, n_workers)]
        print(f"► Enkodowanie {len(unique_tags)} tagów "
              f"w {n_workers} równoległych Ray workers...")
        futures = [_encode_tags_batch.remote(chunk) for chunk in chunks]
        results = ray.get(futures)
        tag_embeddings = np.vstack(results).astype(np.float32)
        tag_to_vec = dict(zip(unique_tags, tag_embeddings))
        emb_dim = tag_embeddings.shape[1]  # 384

        # Per film: uśrednij wektory jego tagów
        movie_tag_groups = tags_df.groupby("movieId")["tag"].apply(list)
        movie_ids = base["movieId"].tolist()
        tag_matrix = np.zeros((len(movie_ids), emb_dim), dtype=np.float32)
        for i, mid in enumerate(movie_ids):
            if mid in movie_tag_groups.index:
                vecs = [tag_to_vec[t] for t in movie_tag_groups[mid] if t in tag_to_vec]
                if vecs:
                    tag_matrix[i] = np.mean(vecs, axis=0)

        # PCA: redukuj do N_TAG_COMPONENTS wymiarów
        n_comp = min(N_TAG_COMPONENTS, len(movie_ids) - 1, emb_dim)
        pca = PCA(n_components=n_comp, random_state=42)
        tagvec = pca.fit_transform(tag_matrix).astype(np.float32)
        tagvec_df = pd.DataFrame(tagvec, columns=[f"tagvec_{i}" for i in range(n_comp)])
        base = pd.concat([base.reset_index(drop=True), tagvec_df], axis=1)
        print(f"  Tagi → PCA {n_comp} komponentów "
              f"(wyjaśnia {pca.explained_variance_ratio_.sum():.1%} wariancji)")

        FEATURES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        base.to_parquet(FEATURES_CACHE, index=False)
        print(f"  Cache zapisany → {FEATURES_CACHE}")

    return base


def genre_cols(movies_features: pd.DataFrame) -> list[str]:
    """Zwraca listę kolumn cech (wszystkie poza movieId i title)."""
    return [c for c in movies_features.columns if c not in ("movieId", "title")]
