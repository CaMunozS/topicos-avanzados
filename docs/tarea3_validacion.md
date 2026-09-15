# Tarea 3 — Guía de revisión y ejecución

## Estado

La solución está implementada en `notebooks/actividad3_Lopez_Munoz.ipynb`.
La plantilla recibida se conserva íntegra en `notebooks/Tarea 03.ipynb`.
Los enunciados, la secuencia y los tipos de las 39 celdas originales se preservan.
Se completan las celdas de solución y se agregan ocho notas metodológicas.

Esta versión requiere ejecutar el experimento completo con GPU para obtener los
resultados académicos. Las pruebas de desarrollo no son resultados finales de P1–P4.
No se publican métricas inventadas ni métricas de la prueba reducida como si fueran
del experimento completo. Tampoco puede asegurarse una nota específica.

## Cobertura del enunciado

| Pregunta | Puntos | Implementación y evidencia al ejecutar |
|---|---:|---|
| P1 | 5 | LadderNet de 2 etapas, 15 filtros iniciales y profundidad 4. Curvas, mejor época, IoU/Dice globales de validación, pesos guardados. |
| P2 | 5 | Tres candidatos de 3 etapas: filtros/profundidad 15/4, 24/4 y 16/5. Selección por IoU de validación, tabla con todos los experimentos y sus costos. |
| P3 | 2 | Tabla comparable, brecha train-val con inferencia determinista, cantidad de parámetros, tiempos, bootstrap pareado por paciente y conclusión calculada con resultados reales. |
| P4 | 3 | Diez ejemplos de test con MRI, máscara real, máscara predicha y contornos superpuestos. Métricas adicionales sobre todo test después de fijar el modelo. |

El archivo adjunto solo contiene este enunciado, que suma 15 puntos. No se recibió una
rúbrica adicional. La cobertura se verifica contra estas cuatro preguntas.

## Datos y evaluación

La descarga oficial de Kaggle se probó en este entorno. El ZIP contiene dos copias
de cada par bajo distintas carpetas. La carga calcula SHA-256 y descarta una copia
únicamente cuando coinciden identificador y contenido de imagen y máscara. Una
colisión con contenido distinto provoca un error para revisión.

Se comprobaron **3.929 pares únicos de 110 pacientes**. Con semilla 84, la partición
por paciente produce:

| Partición | Pacientes | Imágenes |
|---|---:|---:|
| Train | 89 | 3.155 |
| Validación | 10 | 361 |
| Test | 11 | 413 |

Se mantienen las proporciones de partición de la plantilla, aplicándolas a pacientes
para evitar fuga de información entre cortes de una misma persona. Se verifican
pacientes disjuntos, ausencia de imágenes idénticas entre conjuntos, cobertura de
todas las filas, dimensiones, canales, legibilidad y máscaras binarias.

Se mantiene la resolución configurada de 512 × 512. Los datos se leen por lotes y
las máscaras se redimensionan con vecino más cercano. El último lote se incluye
en entrenamiento y evaluación. Solo train recibe aumento geométrico sincronizado.

IoU y Dice de selección corresponden al primer plano binario con umbral fijo 0,5,
acumulando TP, FP y FN de todo el conjunto. Las funciones suaves `smooth=100` de la
plantilla se conservan y se informan con otros nombres. Los promedios por imagen,
los cortes con lesión y los promedios por paciente complementan las métricas globales.

## Arquitectura y referencia docente

La definición LadderNet no venía en la plantilla, y no se encontró el código de
clases entre el material disponible. La solución implementa una adaptación Keras
de la arquitectura de Juntang Zhuang, con conexiones por suma entre etapas y una
misma convolución utilizada dos veces en cada bloque residual. La cabeza binaria
usa sigmoid y la pérdida combina BCE y Dice suave.

| Etapas | Filtros iniciales | Profundidad | Parámetros | Bloques con convolución compartida |
|---:|---:|---:|---:|---:|
| 2 | 15 | 4 | 3.107.506 | 19 |
| 3 | 15 | 4 | 4.662.061 | 29 |
| 3 | 24 | 4 | 11.930.497 | 29 |
| 3 | 16 | 5 | 21.233.921 | 35 |

Se verificó la estructura y el funcionamiento de esta implementación. **Queda
pendiente contrastarla con la definición exacta utilizada por el profesor**,
porque P1 y P2 piden usar el código visto en clase. No se afirma que esa equivalencia
docente esté validada.

Referencias: [artículo](https://arxiv.org/abs/1810.07810),
[implementación del autor](https://github.com/juntang-zhuang/LadderNet/blob/master/src/LadderNetv65.py),
[dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation),
[métricas Keras](https://keras.io/api/metrics/segmentation_metrics/),
[EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping).

## Pruebas técnicas

Entorno de pruebas: Python 3.12.14, TensorFlow CPU 2.18.1, Keras 3.15.1,
NumPy 2.0.2, OpenCV 4.11.0 y pandas 2.2.3. No hubo GPU disponible en este entorno.

El script `tests/validate_tarea3.py` comprueba:

- Formato Jupyter, sintaxis de todas las celdas y preservación de los enunciados.
- Lectura y partición del dataset real completo.
- Casos conocidos de IoU/Dice: coincidencia, disyunción, máscaras vacías y solapamiento parcial.
- Acumulación correcta de métricas entre lotes y reinicio de sus contadores.
- Alineación de la transformación geométrica de MRI y máscara.
- Inclusión del último lote incompleto.
- En las cuatro arquitecturas: conexiones entre etapas, pesos realmente compartidos,
  un paso real de optimización con imágenes a 32 px, pérdidas finitas y reconstrucción
  a partir de los pesos guardados con predicciones equivalentes.
- Formas de entrada/salida a 512 px y cantidad de parámetros.
- Restauración de la política de precisión después de `clear_session` y un paso
  de entrenamiento de precisión mixta en CPU, conservando la salida en float32.

Para repetir estas pruebas con TensorFlow y las dependencias del notebook instaladas:

```bash
python tests/validate_tarea3.py --data-root /ruta/a/MRI_Images --report /tmp/tarea3_validacion.json
```

El informe en JSON se guarda en `docs/tarea3_validacion.json`.
Además, se ejecutaron correctamente las 21 celdas de código de principio a fin en
una copia de prueba con datos MRI reales: 8 imágenes de train, 4 de validación y
10 de test, a 32 px y durante 2 épocas por candidato. Se mantuvieron las cuatro
arquitecturas completas. La prueba recorrió descarga/carga, entrenamiento,
checkpoint, selección, tabla comparativa, respuesta dinámica y las diez predicciones.
Las salidas de ese experimento reducido no se incorporan al notebook de entrega.

## Ejecución completa en Colab Pro

1. [Abrir la solución en Colab](https://colab.research.google.com/github/CaMunozS/topicos-avanzados/blob/main/notebooks/actividad3_Lopez_Munoz.ipynb).
2. Seleccionar un entorno con GPU y ejecutar todas las celdas.
3. Autorizar el montaje de Drive si `SAVE_TO_DRIVE=True` para conservar resultados.
   Si se desactiva, los archivos quedan en la sesión temporal de Colab.
4. Revisar la configuración inicial. El experimento completo usa imágenes de 512 px,
   batch 2, hasta 80 épocas por candidato, paciencia 12 y reducción de tasa con paciencia 4.
   El tiempo depende de la GPU asignada y de la época en que se detenga cada entrenamiento.
5. Revisar curvas, resultados de P1/P2, justificación P3 y las diez imágenes de P4.
6. Guardar el notebook con todas las salidas y el nombre `actividad3_Lopez_Munoz.ipynb`.
   Completar los RUT de los integrantes en la copia privada de Webcursos: la versión
   pública del repositorio contiene solo sus nombres.
   La fecha indicada en el enunciado es 20 de septiembre de 2026 a las 23:59 de Chile.

Los resultados se escriben en `MyDrive/TopicosAvanzados/Tarea03` si se usa Drive.
Cada experimento guarda configuración, arquitectura, historial, mejor checkpoint,
métricas y predicciones. Una huella identifica datos, particiones e hiperparámetros.
Solo se reutilizan automáticamente experimentos **completos** que coinciden con esa
configuración. Si un experimento se interrumpe antes de terminar, vuelve a comenzar
al ejecutar su pregunta; los candidatos ya completos se conservan.

Si se cambia el tamaño de imagen o cualquier hiperparámetro, se deben ejecutar de
nuevo P1 y P2 para mantener una comparación coherente. Ante falta de memoria GPU,
reducir `BATCH_SIZE` a 1 y repetir ambos. La configuración predeterminada con GPU
y precisión mixta aún requiere verificación en la sesión concreta de Colab.
