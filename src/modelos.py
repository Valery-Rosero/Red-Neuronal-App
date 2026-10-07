"""Arquitecturas LSTM del taller (entrada: ventana de n horas x 15 variables)."""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
from keras import layers


def modelo_A(n: int, f: int, unidades: int = 32) -> keras.Model:
    """A. LSTM base: una capa LSTM y una salida lineal. Línea base sencilla."""
    return keras.Sequential([
        layers.Input((n, f)),
        layers.LSTM(unidades),
        layers.Dense(1),
    ], name="A_LSTM_base")


def modelo_B(n: int, f: int, dropout: float = 0.2) -> keras.Model:
    """B. LSTM profunda: dos LSTM apiladas con Dropout y una capa densa."""
    return keras.Sequential([
        layers.Input((n, f)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(32),
        layers.Dropout(dropout),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name="B_LSTM_profunda")


def modelo_C(n: int, f: int, dropout: float = 0.2) -> keras.Model:
    """C. Propuesta del grupo, CNN-LSTM: una Conv1D causal extrae patrones locales
    (p. ej. subidas/bajadas de pocas horas) y la LSTM modela la dependencia larga."""
    return keras.Sequential([
        layers.Input((n, f)),
        layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu"),
        layers.LSTM(48),
        layers.Dropout(dropout),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name="C_CNN_LSTM")


ARQUITECTURAS = {"A": modelo_A, "B": modelo_B, "C": modelo_C}
