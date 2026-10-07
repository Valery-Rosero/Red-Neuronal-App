"""Preprocesamiento compartido entre el notebook de entrenamiento y la app web.

Reglas de diseño (ver notebook, sección 3):
  * Nunca se imputan ni se inventan valores: lo inválido pasa a NaN y las
    ventanas que lo contengan se descartan.
  * Los parámetros de normalización se calculan SOLO con entrenamiento.
  * Ninguna variable de entrada contiene información posterior a la hora t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
TARGET = "demanda_objetivo"          # demanda de la hora t+1
CONTINUAS = [
    "demanda_mw", "temperatura_c", "humedad_pct", "viento_kmh",
    "radiacion_wm2", "precipitacion_mm", "precio_kwh",
]
CICLICAS = ["hora_sin", "hora_cos", "dia_sin", "dia_cos", "mes_sin", "mes_cos"]
BINARIAS = ["fin_semana", "festivo"]
FEATURES = CONTINUAS + CICLICAS + BINARIAS     # 15 entradas por hora
N_CONT = len(CONTINUAS)

# Límites físicos / operativos (justificación en el notebook).
# Fuera de rango => valor inválido => NaN (no se corrige).
LIMITES = {
    "demanda_mw":       (300.0, 1000.0),   # objetivo (limpio) varía en 314-927
    "temperatura_c":    (-10.0, 45.0),
    "humedad_pct":      (0.0, 100.0),
    "viento_kmh":       (0.0, 100.0),
    "radiacion_wm2":    (0.0, 1200.0),
    "precipitacion_mm": (0.0, 100.0),
    "precio_kwh":       (0.0, 100.0),
}


# --------------------------------------------------------------------------
# Limpieza
# --------------------------------------------------------------------------
def cargar_crudo(ruta: str) -> pd.DataFrame:
    return pd.read_csv(ruta, parse_dates=["timestamp"])


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Devuelve (df_limpio, reporte). El df queda ordenado, sin duplicados y con
    una fila por hora; los valores inválidos quedan como NaN (no se imputan)."""
    rep: dict = {"filas_crudas": len(df)}
    df = df.copy()

    # 1) Texto: periodo_dia con mayúsculas/espacios inconsistentes
    rep["periodo_dia_variantes_antes"] = df["periodo_dia"].nunique()
    df["periodo_dia"] = df["periodo_dia"].str.strip().str.lower()
    rep["periodo_dia_variantes_despues"] = df["periodo_dia"].nunique()

    # 2) Duplicados exactos y desorden temporal
    rep["desordenado"] = not df["timestamp"].is_monotonic_increasing
    rep["duplicados"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    rep["timestamps_repetidos_restantes"] = int(df["timestamp"].duplicated().sum())
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 3) Continuidad horaria
    esperado = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    rep["horas_faltantes"] = int(len(esperado.difference(df["timestamp"])))

    # 4) Anomalías -> NaN (sin corregir)
    rep["nan_originales"] = df[CONTINUAS].isna().sum().to_dict()
    rep["anomalias"] = {}
    for col, (lo, hi) in LIMITES.items():
        malo = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        rep["anomalias"][col] = int(malo.sum())
        df.loc[malo, col] = np.nan
    return df, rep


# --------------------------------------------------------------------------
# Ingeniería de variables
# --------------------------------------------------------------------------
def agregar_ciclicas(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica hora, día de semana y mes como seno/coseno (24, 7 y 12 son
    periodos): evita el salto artificial 23 -> 0."""
    df = df.copy()
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["dia_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["dia_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)
    df["mes_sin"] = np.sin(2 * np.pi * (df["mes"] - 1) / 12)
    df["mes_cos"] = np.cos(2 * np.pi * (df["mes"] - 1) / 12)
    return df


# --------------------------------------------------------------------------
# Split cronológico, normalización y ventanas
# --------------------------------------------------------------------------
def split_cronologico(n: int, p_train: float = 0.70, p_val: float = 0.15):
    """Índices de corte (fin de train, fin de val) sobre una serie ordenada."""
    i_tr = int(n * p_train)
    i_va = int(n * (p_train + p_val))
    return i_tr, i_va


class Escalador:
    """Estandarización z-score ajustada ÚNICAMENTE con filas de entrenamiento
    (ignora NaN). Solo se escalan las variables continuas y el objetivo."""

    def ajustar(self, df_train: pd.DataFrame):
        self.media = df_train[CONTINUAS].mean()
        self.std = df_train[CONTINUAS].std(ddof=0)
        self.y_media = float(df_train[TARGET].mean())
        self.y_std = float(df_train[TARGET].std(ddof=0))
        return self

    def x(self, df: pd.DataFrame) -> np.ndarray:
        out = df[FEATURES].copy()
        out[CONTINUAS] = (out[CONTINUAS] - self.media) / self.std
        return out.to_numpy(dtype="float32")

    def y(self, serie) -> np.ndarray:
        return ((np.asarray(serie, dtype="float64") - self.y_media) / self.y_std).astype("float32")

    def y_inverso(self, y_esc) -> np.ndarray:
        return np.asarray(y_esc, dtype="float64") * self.y_std + self.y_media

    def a_dict(self) -> dict:
        return {"media": self.media.to_dict(), "std": self.std.to_dict(),
                "y_media": self.y_media, "y_std": self.y_std}

    @classmethod
    def desde_dict(cls, d: dict):
        e = cls()
        e.media = pd.Series(d["media"])[CONTINUAS]
        e.std = pd.Series(d["std"])[CONTINUAS]
        e.y_media, e.y_std = d["y_media"], d["y_std"]
        return e


def crear_ventanas(X: np.ndarray, y: np.ndarray, n: int, desde: int, hasta: int):
    """Ventanas de n horas [t-n+1 .. t] -> objetivo en la fila t (demanda t+1).

    Solo se generan para filas objetivo t en [desde, hasta). La entrada puede
    recorrer horas anteriores a `desde` (es pasado, no fuga). Se descartan las
    ventanas con cualquier NaN en la entrada o en el objetivo (no se imputa).
    Devuelve (Xw, yw, idx_t) con idx_t = fila t de cada ventana.
    """
    idx = np.arange(max(desde, n - 1), hasta)
    bloques = np.stack([X[i - n + 1 : i + 1] for i in idx])
    ok = ~np.isnan(bloques).any(axis=(1, 2)) & ~np.isnan(y[idx])
    return bloques[ok], y[idx][ok], idx[ok]
