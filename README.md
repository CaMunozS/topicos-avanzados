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

## Tarea 3 — Segmentación MRI con LadderNet

- [Notebook de solución](notebooks/actividad3_Lopez_Munoz.ipynb).
- [Abrir directamente en Colab](https://colab.research.google.com/github/CaMunozS/topicos-avanzados/blob/main/notebooks/actividad3_Lopez_Munoz.ipynb).
- [Plantilla original](notebooks/Tarea%2003.ipynb).
- [Instrucciones y alcance de validación](docs/tarea3_validacion.md).

La solución conserva las 39 celdas originales en su orden y agrega ocho notas metodológicas.
Implementa P1 con 2 etapas, 15 filtros iniciales y profundidad 4; P2 compara tres candidatos
de 3 etapas. P3 elige por validación y P4 muestra 10 predicciones del modelo seleccionado.

En Colab, seleccionar GPU, ejecutar todas las celdas y autorizar Drive si se desea conservar
los pesos. Los experimentos completos se reutilizan al ejecutar nuevamente con la misma
configuración y partición. Guardar el notebook **con sus salidas** para la entrega académica.
La validación técnica documentada no sustituye el entrenamiento completo ni sus métricas finales.
