"""Motor de inferencia de la LSTM en NumPy puro (sin TensorFlow).

Reproduce exactamente el preprocesamiento del notebook (src/preprocessing.py):
mismas reglas de validación, mismas variables cíclicas y la normalización
calculada sobre entrenamiento. Los pesos se exportan desde el modelo Keras.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta

import numpy as np

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelo")

CONTINUAS = ["demanda_mw", "temperatura_c", "humedad_pct", "viento_kmh",
             "radiacion_wm2", "precipitacion_mm", "precio_kwh"]
ETIQUETAS = {
    "demanda_mw": "Demanda (MW)", "temperatura_c": "Temperatura (°C)",
    "humedad_pct": "Humedad (%)", "viento_kmh": "Viento (km/h)",
    "radiacion_wm2": "Radiación (W/m²)", "precipitacion_mm": "Precipitación (mm)",
    "precio_kwh": "Precio (kWh)", "festivo": "Festivo", "timestamp": "Fecha y hora",
}

_cache: dict = {}


def cargar():
    """Carga (una sola vez por instancia) configuración y pesos."""
    if not _cache:
        with open(os.path.join(_DIR, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        with open(os.path.join(_DIR, "pesos.json"), encoding="utf-8") as fh:
            w = json.load(fh)
        _cache["cfg"] = cfg
        _cache["W"] = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}
        e = cfg["escalador"]
        _cache["media"] = np.array([e["media"][c] for c in CONTINUAS])
        _cache["std"] = np.array([e["std"][c] for c in CONTINUAS])
    return _cache


# --------------------------------------------------------------------------
# Validación (mismas reglas R5/R6 del notebook: no se corrige ni se imputa)
# --------------------------------------------------------------------------
def _parse_ts(v) -> datetime:
    s = str(v).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"formato de fecha no válido: '{v}' (use AAAA-MM-DD HH:MM)")


def validar(filas: list[dict]) -> tuple[list[dict], list[dict]]:
    """Devuelve (filas_normalizadas_ordenadas, errores). Cada error indica fila y campo."""
    S = cargar()
    n = S["cfg"]["ventana"]
    limites = S["cfg"]["limites"]
    errores: list[dict] = []

    if not isinstance(filas, list) or len(filas) != n:
        return [], [{"fila": None, "campo": None,
                     "mensaje": f"Se requieren exactamente {n} horas consecutivas (llegaron "
                                f"{len(filas) if isinstance(filas, list) else 0})."}]

    norm = []
    for i, f in enumerate(filas):
        r = {}
        try:
            r["timestamp"] = _parse_ts(f.get("timestamp"))
        except Exception as ex:  # noqa: BLE001
            errores.append({"fila": i, "campo": "timestamp", "mensaje": str(ex)})
            r["timestamp"] = None
        for c in CONTINUAS:
            v = f.get(c)
            try:
                x = float(v)
                if math.isnan(x) or math.isinf(x):
                    raise ValueError
            except (TypeError, ValueError):
                errores.append({"fila": i, "campo": c, "mensaje": f"{ETIQUETAS[c]}: valor faltante o no numérico."})
                r[c] = None
                continue
            lo, hi = limites[c]
            if not (lo <= x <= hi):
                errores.append({"fila": i, "campo": c,
                                "mensaje": f"{ETIQUETAS[c]} = {x:g} fuera del rango válido [{lo:g}, {hi:g}]."})
            r[c] = x
        fv = f.get("festivo", 0)
        if fv in (True, False, 0, 1, "0", "1", "true", "false", "True", "False"):
            r["festivo"] = 1 if str(fv).lower() in ("1", "true") else 0
        else:
            errores.append({"fila": i, "campo": "festivo", "mensaje": "Festivo debe ser 0 o 1."})
            r["festivo"] = 0
        r["_orig"] = i
        norm.append(r)

    if any(r["timestamp"] is None for r in norm):
        return norm, errores

    norm.sort(key=lambda r: r["timestamp"])                    # R3: ordenar
    vistos = set()
    for k, r in enumerate(norm):
        if r["timestamp"] in vistos:                           # R2: duplicados
            errores.append({"fila": r["_orig"], "campo": "timestamp", "mensaje": "Hora duplicada."})
        vistos.add(r["timestamp"])
        if k and r["timestamp"] - norm[k - 1]["timestamp"] != timedelta(hours=1):
            errores.append({"fila": r["_orig"], "campo": "timestamp",
                            "mensaje": "Las horas deben ser consecutivas (sin saltos ni minutos)."})
        if r["timestamp"].minute or r["timestamp"].second:
            errores.append({"fila": r["_orig"], "campo": "timestamp", "mensaje": "Use horas en punto (HH:00)."})
    return norm, errores


# --------------------------------------------------------------------------
# Variables de entrada (mismo orden que pp.FEATURES)
# --------------------------------------------------------------------------
def matriz_entrada(norm: list[dict]) -> np.ndarray:
    S = cargar()
    X = []
    for r in norm:
        t = r["timestamp"]
        hora, dia, mes = t.hour, t.weekday(), t.month       # weekday(): lunes=0, igual que pandas
        cont = (np.array([r[c] for c in CONTINUAS]) - S["media"]) / S["std"]
        X.append(np.concatenate([cont, [
            math.sin(2 * math.pi * hora / 24), math.cos(2 * math.pi * hora / 24),
            math.sin(2 * math.pi * dia / 7), math.cos(2 * math.pi * dia / 7),
            math.sin(2 * math.pi * (mes - 1) / 12), math.cos(2 * math.pi * (mes - 1) / 12),
            1.0 if dia >= 5 else 0.0, float(r["festivo"]),
        ]]))
    return np.asarray(X)


# --------------------------------------------------------------------------
# Paso hacia adelante de la LSTM (Keras: compuertas en orden i, f, c, o)
# --------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def lstm_forward(X: np.ndarray) -> float:
    W = cargar()["W"]
    K, R, b = W["lstm_kernel"], W["lstm_recurrent"], W["lstm_bias"]
    u = R.shape[0]
    h = np.zeros(u); c = np.zeros(u)
    for x_t in X:
        z = x_t @ K + h @ R + b
        i, f = _sigmoid(z[:u]), _sigmoid(z[u:2 * u])
        g, o = np.tanh(z[2 * u:3 * u]), _sigmoid(z[3 * u:])
        c = f * c + i * g
        h = o * np.tanh(c)
    return float(h @ W["dense_kernel"][:, 0] + W["dense_bias"][0])


def predecir(filas: list[dict]) -> dict:
    """Valida, construye la ventana y devuelve la predicción en MW para t+1."""
    S = cargar()
    norm, errores = validar(filas)
    if errores:
        return {"ok": False, "errores": errores}
    y_esc = lstm_forward(matriz_entrada(norm))
    e = S["cfg"]["escalador"]
    pred = y_esc * e["y_std"] + e["y_media"]
    t_obj = norm[-1]["timestamp"] + timedelta(hours=1)
    return {"ok": True, "prediccion_mw": round(pred, 2),
            "timestamp_objetivo": t_obj.strftime("%Y-%m-%d %H:%M"),
            "ventana_desde": norm[0]["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "ventana_hasta": norm[-1]["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "modelo": S["cfg"]["arquitectura"]}
