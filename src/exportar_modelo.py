"""Exporta el modelo de producción (Keras) a JSON para la app de Vercel y verifica
que la implementación NumPy de la app reproduce exactamente las predicciones de Keras.

Uso (desde la carpeta del taller):  python src/exportar_modelo.py
"""
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import joblib
import keras
import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))
import preprocessing as pp  # noqa: E402

APP = os.path.join(RAIZ, "app_vercel")


def exportar():
    os.makedirs(os.path.join(APP, "modelo"), exist_ok=True)
    cfg = json.load(open(os.path.join(RAIZ, "artefactos", "config_app.json"), encoding="utf-8"))
    modelo = keras.models.load_model(os.path.join(RAIZ, "modelos", "modelo_produccion.keras"))

    capas = [c for c in modelo.layers if c.weights]
    assert [type(c).__name__ for c in capas] == ["LSTM", "Dense"], \
        "La exportación NumPy soporta la arquitectura A (LSTM -> Dense)."
    k, r, b = capas[0].get_weights()
    dk, db = capas[1].get_weights()
    pesos = {"lstm_kernel": k.tolist(), "lstm_recurrent": r.tolist(), "lstm_bias": b.tolist(),
             "dense_kernel": dk.tolist(), "dense_bias": db.tolist()}
    json.dump(pesos, open(os.path.join(APP, "modelo", "pesos.json"), "w"))

    df = pd.read_csv(os.path.join(RAIZ, "artefactos", "dataset_limpio.csv"), parse_dates=["timestamp"])
    i_tr, _ = pp.split_cronologico(len(df))
    cfg_app = {**cfg, "limites": pp.LIMITES, "parametros": int(modelo.count_params()),
               "periodo_entrenamiento": [str(df.timestamp[0]), str(df.timestamp[i_tr - 1])]}
    json.dump(cfg_app, open(os.path.join(APP, "modelo", "config.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # Histórico limpio para el modo "desde el histórico" (inválidos quedan vacíos, no se imputan)
    cols = ["timestamp"] + pp.CONTINUAS + ["festivo", pp.TARGET]
    h = df[cols].copy()
    h["timestamp"] = h.timestamp.dt.strftime("%Y-%m-%d %H:%M")
    h.to_csv(os.path.join(APP, "historico.csv"), index=False, float_format="%.4g")
    return modelo, df


def verificar(modelo, df, n_muestras=300):
    """Compara Keras vs. NumPy (app) sobre ventanas reales de prueba."""
    sys.path.insert(0, os.path.join(APP, "api"))
    import _motor
    _motor._cache.clear()
    datos = joblib.load(os.path.join(RAIZ, "artefactos", "datos_ventanas.joblib"))
    esc = pp.Escalador.desde_dict(datos["escalador"])
    n = json.load(open(os.path.join(APP, "modelo", "config.json"), encoding="utf-8"))["ventana"]
    Xw, _, it = datos["datos"][n]["prueba"]
    sel = np.linspace(0, len(it) - 1, n_muestras).astype(int)
    p_keras = esc.y_inverso(modelo.predict(Xw[sel], verbose=0).ravel())
    p_app = []
    for t in it[sel]:
        filas = df.iloc[t - n + 1: t + 1][["timestamp"] + pp.CONTINUAS + ["festivo"]].copy()
        filas["timestamp"] = filas.timestamp.dt.strftime("%Y-%m-%d %H:%M")
        res = _motor.predecir(filas.to_dict("records"))
        assert res["ok"], res
        p_app.append(res["prediccion_mw"])
    dif = np.abs(np.array(p_app) - p_keras)
    print(f"Verificación Keras vs. NumPy en {len(sel)} ventanas de prueba: "
          f"diferencia máx. {dif.max():.4f} MW, media {dif.mean():.4f} MW")
    assert dif.max() < 0.01, "La implementación NumPy no coincide con Keras"
    return dif


if __name__ == "__main__":
    m, d = exportar()
    verificar(m, d)
    print("Exportación lista en", APP)
