# Predicción de demanda eléctrica con LSTM

Taller de Inteligencia Artificial: diseño, comparación y despliegue de arquitecturas LSTM que predicen la demanda eléctrica (MW) de la **hora siguiente** a partir de una ventana de *n* horas de información multivariada (demanda, clima, precio y calendario).

## Informe

📄 **[Informe_Taller_LSTM.pdf](Informe_Taller_LSTM.pdf)**: informe completo del taller (portada, metodología, resultados, gráficas, preguntas de análisis y conclusiones). También se puede ver y exportar a PDF desde la app: https://red-neuronal-app.vercel.app/reporte.html

**Autores:** Valery Nickol Rosero Molina y Johan David Delgado · **Docente:** Cristian Ordoñez

## Resultados

Evaluación sobre el conjunto de prueba (13-sep a 31-dic-2026, nunca visto en entrenamiento), con los mismos instantes para todos los modelos:

| Modelo | Ventana | MAE (MW) | RMSE (MW) | MAPE | R² |
|---|---|---|---|---|---|
| **A · LSTM base** (seleccionado) | 12 h | 19.86 | 24.99 | 3.43 % | 0.904 |
| B · LSTM profunda | 12 h | 19.43 | 24.77 | 3.37 % | 0.905 |
| C · CNN-LSTM | 12 h | 19.75 | 25.33 | 3.42 % | 0.901 |
| Persistencia (referencia) | – | 35.50 | 44.31 | 6.07 % | 0.697 |

La tabla completa (3 arquitecturas × ventanas de 12, 24 y 48 h), las gráficas, el análisis de sobreajuste, los experimentos de Dropout, Early Stopping y número de neuronas, y las respuestas a las preguntas de análisis están en el notebook.

## Estructura

```
├── Taller_LSTM.ipynb            # Notebook completo: limpieza, EDA, modelos, métricas, análisis y despliegue
├── dataset_demanda_energia_LSTM_2025_2026.csv
├── src/
│   ├── preprocessing.py         # Limpieza, variables cíclicas, split cronológico, escalado y ventanas
│   ├── modelos.py               # Arquitecturas A (LSTM base), B (LSTM profunda) y C (CNN-LSTM)
│   ├── exportar_modelo.py       # Exporta el modelo a JSON para la app y verifica Keras vs. NumPy
│   ├── exportar_informe.py      # Genera la página de resultados de la app desde el notebook
│   └── generar_reporte.py       # Genera el informe formal (reporte.html) y su PDF
├── reporte/                     # Texto (contenido.md) y plantilla del informe
├── Informe_Taller_LSTM.pdf      # Informe en PDF
├── modelos/                     # Modelos entrenados (.keras) y modelo_produccion.keras
├── artefactos/                  # Historiales, experimentos, dataset limpio y configuración
└── app_vercel/                  # App web desplegable en Vercel
    ├── index.html, app.js, styles.css
    ├── informe.html, informe/   # Página de resultados y análisis (gráficas + preguntas)
    ├── reporte.html             # Informe formal con opción de exportar a PDF
    ├── historico.csv            # Histórico limpio para el modo "desde el histórico"
    ├── api/predict.py           # Función serverless: POST /api/predict
    ├── api/_motor.py            # Validación + LSTM en NumPy
    ├── modelo/                  # Pesos y configuración exportados
    └── servidor_local.py        # Para probar la app localmente
```

## Reproducir el notebook

```bash
pip install -r requirements.txt
jupyter notebook Taller_LSTM.ipynb
```

Por defecto (`REENTRENAR = False`) el notebook carga los modelos ya entrenados de `modelos/`. Con `REENTRENAR = True` vuelve a entrenar todo (unos 15 minutos en CPU).

## App web

**En línea:** https://red-neuronal-app.vercel.app

La app tiene dos páginas:

- **Predicción:** permite ingresar las variables independientes (x) de las últimas 12 horas (desde el histórico, a mano o subiendo un CSV) y ver la predicción de la demanda de la hora siguiente (y).
- **Informe** (`reporte.html`): el informe formal con portada, listo para descargar o imprimir en PDF.
- **Resultados y análisis** (`informe.html`): muestra la limpieza de datos, las arquitecturas, la tabla de resultados, todas las gráficas, el análisis de sobreajuste, los experimentos, las respuestas a las 9 preguntas y las conclusiones. Se genera desde el notebook ejecutado con `python src/exportar_informe.py`.

- **Validación estricta:** si una hora tiene un dato faltante o fuera de rango físico, la app lo señala y no predice. Aplica las mismas reglas de limpieza del entrenamiento y no inventa valores.
- **Inferencia sin TensorFlow:** los pesos de la LSTM se exportan a JSON y el paso hacia adelante se calcula con NumPy, para cumplir el límite de tamaño de las funciones serverless. La diferencia con Keras es menor a 0.01 MW.

### Probar localmente

```bash
pip install numpy
python app_vercel/servidor_local.py
# abrir http://localhost:8000
```

### Desplegar en Vercel

1. En [vercel.com](https://vercel.com), elige **Add New → Project** e importa este repositorio de GitHub.
2. En **Root Directory**, selecciona `app_vercel`.
3. Deja **Framework Preset = Other**, sin comandos de build.
4. Pulsa **Deploy**. Vercel instala `numpy` desde `app_vercel/requirements.txt` y publica la página y la API `/api/predict`.

### API

`POST /api/predict`

```json
{
  "filas": [
    {"timestamp": "2026-11-12 06:00", "demanda_mw": 679.2, "temperatura_c": 9.31, "humedad_pct": 65.37,
     "viento_kmh": 16.56, "radiacion_wm2": 0, "precipitacion_mm": 0.57, "precio_kwh": 26.09, "festivo": 0}
  ]
}
```

La lista `filas` debe tener exactamente 12 horas consecutivas (en el ejemplo solo se muestra una). La respuesta es `{"ok": true, "prediccion_mw": 719.63, "timestamp_objetivo": "2026-11-12 18:00", ...}`. Si hay datos inválidos, la API responde con código 422 y la lista de errores por fila y campo.

`GET /api/predict` devuelve la información y las métricas del modelo.
