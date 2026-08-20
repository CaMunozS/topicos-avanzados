"""Punto de entrada reproducible para la tarea."""

from __future__ import annotations

import random

import numpy as np

RANDOM_SEED = 42


def configure_reproducibility(seed: int = RANDOM_SEED) -> None:
    """Configura las semillas usadas por Python y NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    """Ejecuta el flujo principal del proyecto."""
    configure_reproducibility()
    print("Proyecto Tópicos Avanzados inicializado.")


if __name__ == "__main__":
    main()
