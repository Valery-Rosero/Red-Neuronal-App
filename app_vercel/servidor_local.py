"""Servidor local para probar la app antes de desplegar en Vercel.

Sirve los archivos estáticos y enruta /api/predict a la MISMA función que usa Vercel.
Uso:  python servidor_local.py      ->  abrir http://localhost:8000
"""
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DIR, "api"))
import predict  # noqa: E402


class Local(SimpleHTTPRequestHandler):
    _responder = predict.handler._responder

    def _es_api(self):
        return self.path.split("?")[0].rstrip("/") == "/api/predict"

    def do_GET(self):
        return predict.handler.do_GET(self) if self._es_api() else super().do_GET()

    def do_POST(self):
        return predict.handler.do_POST(self) if self._es_api() else self.send_error(404)

    def do_OPTIONS(self):
        return predict.handler.do_OPTIONS(self)


if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 8000))
    print(f"App en http://localhost:{puerto}  (Ctrl+C para detener)")
    ThreadingHTTPServer(("127.0.0.1", puerto), partial(Local, directory=DIR)).serve_forever()
