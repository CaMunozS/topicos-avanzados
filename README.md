# Tópicos Avanzados

Repositorio base para desarrollar y entregar una tarea académica de tópicos avanzados.

## Objetivo

Documentar de forma reproducible:

1. El problema y sus objetivos.
2. La metodología y los supuestos.
3. Los experimentos realizados.
4. Los resultados, conclusiones y trabajo futuro.

> Reemplaza esta sección con el enunciado y los objetivos específicos de la tarea.

## Estructura

```text
.
├── data/
│   ├── raw/            # Datos originales (no versionados)
│   └── processed/      # Datos procesados (no versionados)
├── notebooks/
│   └── 01_exploracion.ipynb
├── outputs/            # Figuras, métricas y otros resultados
├── src/
│   ├── __init__.py
│   └── main.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Inicio rápido

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

## Reproducibilidad

- Mantén los datos originales en `data/raw/`.
- Guarda transformaciones en `data/processed/`.
- Fija semillas aleatorias en los experimentos.
- Registra métricas, parámetros y conclusiones.
- No subas credenciales, datos sensibles ni archivos pesados.

## Entrega

Antes de entregar, completa el objetivo, documenta la fuente de los datos, ejecuta el notebook de principio a fin y verifica que los resultados puedan reproducirse.
