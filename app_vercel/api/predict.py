"""Función serverless de Vercel: POST /api/predict  ·  GET /api/predict (info del modelo).

Entrada (JSON):  {"filas": [ {timestamp, demanda_mw, temperatura_c, humedad_pct, viento_kmh,
                              radiacion_wm2, precipitacion_mm, precio_kwh, festivo}, ... ]}
                 con exactamente `ventana` horas consecutivas (12 para el modelo desplegado).
Salida (JSON):   {"ok": true, "prediccion_mw": ..., "timestamp_objetivo": ...}
                 o {"ok": false, "errores": [{fila, campo, mensaje}, ...]} con código 422.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _motor  # noqa: E402


def info_modelo() -> dict:
    cfg = _motor.cargar()["cfg"]
    return {k: cfg[k] for k in ("arquitectura", "ventana", "metricas_prueba", "limites",
                                "periodo_entrenamiento", "parametros")}


class handler(BaseHTTPRequestHandler):
    def _responder(self, codigo: int, cuerpo: dict):
        datos = json.dumps(cuerpo, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def do_OPTIONS(self):
        self._responder(204, {})

    def do_GET(self):
        self._responder(200, info_modelo())

    def do_POST(self):
        try:
            largo = int(self.headers.get("Content-Length", 0))
            if largo > 200_000:
                return self._responder(413, {"ok": False, "errores": [{"fila": None, "campo": None,
                                              "mensaje": "Solicitud demasiado grande."}]})
            cuerpo = json.loads(self.rfile.read(largo) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._responder(400, {"ok": False, "errores": [{"fila": None, "campo": None,
                                          "mensaje": "El cuerpo debe ser JSON válido."}]})
        res = _motor.predecir(cuerpo.get("filas"))
        self._responder(200 if res["ok"] else 422, res)
