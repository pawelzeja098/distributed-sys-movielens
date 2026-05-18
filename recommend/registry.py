"""
Ray Actor: rozproszony rejestr modeli przechowywany w pamięci klastra Ray.

Wzorzec Actor Model:
  - prywatny stan (_models) niedostępny bezpośrednio z zewnątrz
  - komunikacja wyłącznie przez asynchroniczne wywołania metod
  - dostępny globalnie po nazwie: ray.get_actor("ModelRegistry")
  - lifetime="detached" — przeżywa proces który go utworzył
"""

import ray


@ray.remote
class ModelRegistry:
    """
    Stateful distributed service przechowujący wytrenowane modele w pamięci
    klastra Ray. Dysk (joblib) pełni rolę trwałego storage na wypadek restartu.
    """

    def __init__(self) -> None:
        self._models: dict[int, object] = {}

    def save(self, user_id: int, model) -> None:
        self._models[user_id] = model

    def load(self, user_id: int):
        return self._models.get(user_id)

    def list_users(self) -> list[int]:
        return sorted(self._models.keys())

    def has_model(self, user_id: int) -> bool:
        return user_id in self._models

    def remove(self, user_id: int) -> None:
        self._models.pop(user_id, None)

    def count(self) -> int:
        return len(self._models)


def get_registry() -> ModelRegistry:
    """
    Zwraca handle do named actora ModelRegistry.
    Jeśli nie istnieje — tworzy go (lifetime=detached: przeżywa wywołującego).
    """
    try:
        return ray.get_actor("ModelRegistry")
    except ValueError:
        return ModelRegistry.options(
            name="ModelRegistry",
            lifetime="detached",
        ).remote()
