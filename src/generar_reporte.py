"""Genera el informe formal del taller (app_vercel/reporte.html) y su PDF.

Fuentes: reporte/contenido.md (texto), reporte/plantilla.html (portada y estilos),
el notebook ejecutado (tablas), app_vercel/informe/fig_XX.png (figuras, ver
exportar_informe.py) y artefactos/ (resultados).

Uso (desde la carpeta del taller):  python src/generar_reporte.py
Requiere playwright (python -m playwright install chromium) para la captura y el PDF.
"""
import html
import io
import os
import re
import shutil
import sys
import threading
from functools import partial
from http.server import ThreadingHTTPServer

import joblib
import mistune
import nbformat
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(RAIZ, "app_vercel")
REPO = "https://github.com/Valery-Rosero/Red-Neuronal-App"
URL_APP = "https://red-neuronal-app.vercel.app"
FECHA = "Octubre de 2026"
PDF = "Informe_Taller_LSTM.pdf"

md = mistune.create_markdown(plugins=["table", "strikethrough"], escape=False)


# --------------------------------------------------------------------------
# utilidades
# --------------------------------------------------------------------------
def num(x, d=2):
    """Número con formato es-CO: coma decimal y espacio fino de miles."""
    s = f"{x:,.{d}f}"
    return s.replace(",", " ").replace(".", ",")


def slug(t):
    t = re.sub(r"[^\w\s-]", "", t.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", "-", t).strip("-")


def tabla_html(cab, filas, numericas=(), clases=None, envolver=False):
    th = "".join(f'<th class="{"n" if i in numericas else ""}">{c}</th>' for i, c in enumerate(cab))
    tr = []
    for k, f in enumerate(filas):
        cl = f' class="{clases[k]}"' if clases and clases[k] else ""
        tds = "".join(f'<td class="{"n" if i in numericas else ""}">{v}</td>' for i, v in enumerate(f))
        tr.append(f"<tr{cl}>{tds}</tr>")
    return f'<div class="tabla-wrap"><table class="datos{" envolver" if envolver else ""}"><thead><tr>{th}</tr></thead><tbody>{"".join(tr)}</tbody></table></div>'


def tabla_notebook(nb, clave):
    """DataFrame de la primera salida HTML de la celda cuyo código contiene `clave`."""
    for c in nb.cells:
        if c.cell_type == "code" and clave in c.source:
            for o in c.outputs:
                h = o.get("data", {}).get("text/html")
                if h:
                    return pd.read_html(io.StringIO(h))[0]
    raise KeyError(clave)


# --------------------------------------------------------------------------
# tablas
# --------------------------------------------------------------------------
def construir_tablas(nb):
    T = {}
    res = pd.read_csv(os.path.join(RAIZ, "artefactos", "tabla_resultados.csv"))
    cfg = pd.read_json(os.path.join(RAIZ, "artefactos", "config_app.json"), typ="series")
    sel_arq, sel_n = cfg["arquitectura"], int(cfg["ventana"])
    filas, clases = [], []
    for _, r in res.iterrows():
        ref = r.Modelo.startswith("Persistencia")
        filas.append([r.Modelo + (" ✓" if (r.Modelo == sel_arq and r.Ventana == sel_n) else ""),
                      "–" if ref else f"{int(r.Ventana)} h", num(r.MAE), num(r.MSE, 1), num(r.RMSE),
                      num(r.MAPE) + " %", num(r["R²"], 4), "–" if ref else int(r["Épocas"])])
        clases.append("destacado" if (r.Modelo == sel_arq and r.Ventana == sel_n) else ("referencia" if ref else ""))
    T["resultados"] = tabla_html(["Modelo", "Ventana", "MAE", "MSE", "RMSE", "MAPE", "R²", "Épocas"],
                                 filas, numericas=range(2, 8), clases=clases) + \
        '<p class="tabla-cap" style="margin-top:4px">✓ Modelo seleccionado por validación. MAE, MSE y RMSE en MW (MSE en MW²).</p>'

    ent = joblib.load(os.path.join(RAIZ, "artefactos", "entrenamiento.joblib"))["resultados"]
    nombres = {"A": "A · LSTM base", "B": "B · LSTM profunda", "C": "C · CNN-LSTM"}
    T["entrenamiento"] = tabla_html(
        ["Modelo", "Ventana", "Épocas ejecutadas", "Mejor época", "MSE validación*", "Tiempo (s)"],
        [[nombres[r["arq"]], f'{r["ventana"]} h', r["epocas_ejecutadas"], r["mejor_epoca"],
          num(r["loss_val_mejor"], 4), num(r["segundos"], 0)] for r in ent],
        numericas=range(2, 6)) +         '<p class="tabla-cap" style="margin-top:4px">* MSE sobre datos normalizados en la mejor época. Tiempo en CPU.</p>'

    v = tabla_notebook(nb, "VENTANAS = [12, 24, 48]")
    v.columns = ["Ventana", "Entrenamiento", "Validación", "Prueba"]
    T["ventanas"] = tabla_html(list(v.columns),
                               [[(f"{x} h" if str(x).isdigit() else "Conjunto común (válidas para 48 h)"),
                                 num(a, 0), num(b, 0), num(c, 0)] for x, a, b, c in v.itertuples(index=False)],
                               numericas=(1, 2, 3))

    s = tabla_notebook(nb, "sobreaj = pd.DataFrame(filas)")
    T["sobreajuste"] = tabla_html(["Modelo", "Ventana", "RMSE entren.", "RMSE valid.", "RMSE prueba", "Prueba / entren."],
                                  [[r[0], f"{int(r[1])} h", num(r[2]), num(r[3]), num(r[4]), num(r[5])]
                                   for r in s.itertuples(index=False)], numericas=range(2, 6))

    e = tabla_notebook(nb, "def entrenar_variante")
    e = e.rename(columns={e.columns[0]: "Variante"})
    T["experimentos"] = tabla_html(["Variante", "Parám.", "Épocas", "RMSE entren.", "RMSE prueba", "R² prueba", "Brecha*"],
                                   [[r[0].replace(" (referencia)", " <em>(referencia)</em>"), num(r[1], 0), int(r[2]),
                                     num(r[3]), num(r[4]), num(r[5], 4), num(r[6])] for r in e.itertuples(index=False)],
                                   numericas=range(1, 7), envolver=True) +         '<p class="tabla-cap" style="margin-top:4px">* Brecha = RMSE de prueba − RMSE de entrenamiento (MW).</p>'

    b = tabla_notebook(nb, "from sklearn.linear_model import Ridge")
    b = b.rename(columns={b.columns[0]: "Modelo"})
    T["baselines"] = tabla_html(["Modelo", "MAE", "MSE", "RMSE", "MAPE", "R²"],
                                [[r[0], num(r[1]), num(r[2], 1), num(r[3]), num(r[4]) + " %", num(r[5], 4)]
                                 for r in b.itertuples(index=False)], numericas=range(1, 6),
                                clases=["referencia", "", "destacado"])

    g = tabla_notebook(nb, "it_s1 =")
    g = g.rename(columns={g.columns[0]: "Grupo"})
    T["errores"] = tabla_html(["Hora predicha", "MAE (MW)", "Horas en prueba"],
                              [[r[0].capitalize(), num(r[1]), num(r[2], 0)] for r in g.itertuples(index=False)],
                              numericas=(1, 2))
    return T


# --------------------------------------------------------------------------
# curva decorativa de la portada (demanda real de una semana de prueba)
# --------------------------------------------------------------------------
def curva_portada():
    h = pd.read_csv(os.path.join(APP, "historico.csv"))
    h = h[(h.timestamp >= "2026-11-09 00:00") & (h.timestamp < "2026-11-16 00:00")].demanda_mw.interpolate().to_numpy()
    lo, hi = h.min(), h.max()
    W, H = 600, 90
    pts = [(i * W / (len(h) - 1), H - 6 - (v - lo) / (hi - lo) * (H - 12)) for i, v in enumerate(h)]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = d + f" L{W},{H} L0,{H} Z"
    return (f'<svg class="curva" viewBox="0 0 {W} {H}" preserveAspectRatio="none" aria-hidden="true">'
            f'<defs><linearGradient id="g" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#2a78d6" stop-opacity=".18"/>'
            f'<stop offset="1" stop-color="#2a78d6" stop-opacity="0"/></linearGradient></defs>'
            f'<path d="{area}" fill="url(#g)"/><path d="{d}" fill="none" stroke="#2a78d6" stroke-width="1.6" '
            f'vector-effect="non-scaling-stroke"/></svg>')


# --------------------------------------------------------------------------
# cuerpo
# --------------------------------------------------------------------------
def construir_cuerpo(tablas):
    texto = open(os.path.join(RAIZ, "reporte", "contenido.md"), encoding="utf-8").read()
    texto = texto.replace("{{LINK_REPO}}", f"[{REPO}]({REPO})").replace("{{LINK_APP}}", f"[{URL_APP}]({URL_APP})")
    partes = re.split(r"(?m)^(?=## )", texto)
    secciones, indice = [], []
    n_fig = n_tab = 0

    def reemplazar(m):
        nonlocal n_fig, n_tab
        tipo, ref, cap = m.group(1), m.group(2), m.group(3)
        if tipo in ("FIG", "IMG"):
            n_fig += 1
            src = f"informe/fig_{int(ref):02d}.png" if tipo == "FIG" else f"informe/{ref}"
            return (f'<figure><img src="{src}" alt="{html.escape(cap)}">'
                    f'<figcaption><strong>Figura {n_fig}.</strong> {cap}</figcaption></figure>')
        n_tab += 1
        return f'<p class="tabla-cap"><strong>Tabla {n_tab}.</strong> {cap}</p>{tablas[ref]}'

    for p in partes:
        if not p.strip():
            continue
        primera, _, resto = p.partition("\n")
        nueva = "{.nueva-pagina}" in primera
        titulo = primera.replace("{.nueva-pagina}", "").lstrip("# ").strip()
        sid = slug(titulo)
        indice.append(f'<li><a href="#{sid}">{html.escape(titulo)}</a></li>')
        cuerpo = md(f"## {titulo}\n{resto}")
        cuerpo = re.sub(r"<p>\{\{(FIG|TABLA|IMG):([^|]+)\|(.*?)\}\}</p>", reemplazar, cuerpo, flags=re.S)
        # tablas escritas en el markdown: alinear columnas numéricas no es necesario; solo envolver
        cuerpo = re.sub(r"(<table>.*?</table>)", r'<div class="tabla-wrap">\1</div>', cuerpo, flags=re.S)
        secciones.append(f'<section id="{sid}"{" class=\"nueva-pagina\"" if nueva else ""}>{cuerpo}</section>')
    return "\n".join(secciones), "".join(indice), n_fig, n_tab


# --------------------------------------------------------------------------
# servidor local + captura + PDF
# --------------------------------------------------------------------------
def servidor():
    sys.path.insert(0, APP)
    from servidor_local import Local  # noqa: E402
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(Local, directory=APP))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def capturar_y_pdf(base):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 1000}, device_scale_factor=1.5)
        pg.goto(base + "/")
        pg.wait_for_selector("#resultado:not(.oculto)", timeout=20000)
        pg.wait_for_timeout(800)
        alto = pg.evaluate("document.querySelector('#sec-resultado').getBoundingClientRect().bottom + window.scrollY")
        pg.screenshot(path=os.path.join(APP, "informe", "app_prediccion.png"),
                      clip={"x": 0, "y": 0, "width": 1280, "height": min(alto + 16, 2600)}, full_page=True)

        doc = b.new_page()
        doc.goto(base + "/reporte.html", wait_until="networkidle")
        doc.wait_for_timeout(600)
        pie = ('<div style="width:100%;font-family:Arial,sans-serif;font-size:8px;color:#898781;padding:0 17mm;'
               'display:flex;justify-content:space-between"><span>Diseño de una arquitectura LSTM para predicción de '
               'demanda energética</span><span>Página <span class="pageNumber"></span> de <span class="totalPages">'
               '</span></span></div>')
        doc.pdf(path=os.path.join(APP, PDF), format="A4", print_background=True, prefer_css_page_size=True,
                display_header_footer=True, header_template="<span></span>", footer_template=pie)
        b.close()
    shutil.copy(os.path.join(APP, PDF), os.path.join(RAIZ, PDF))


def main():
    nb = nbformat.read(os.path.join(RAIZ, "Taller_LSTM.ipynb"), as_version=4)
    tablas = construir_tablas(nb)
    cuerpo, indice, n_fig, n_tab = construir_cuerpo(tablas)
    plantilla = open(os.path.join(RAIZ, "reporte", "plantilla.html"), encoding="utf-8").read()
    salida = (plantilla.replace("{{CUERPO}}", cuerpo).replace("{{INDICE}}", indice)
              .replace("{{CURVA}}", curva_portada()).replace("{{REPO}}", REPO)
              .replace("{{APP}}", URL_APP).replace("{{FECHA}}", FECHA))
    with open(os.path.join(APP, "reporte.html"), "w", encoding="utf-8") as fh:
        fh.write(salida)
    print(f"reporte.html: {indice.count('<li>')} secciones, {n_fig} figuras, {n_tab} tablas")

    srv, base = servidor()
    try:
        capturar_y_pdf(base)
    finally:
        srv.shutdown()
    from pypdf import PdfReader
    print(f"{PDF}: {len(PdfReader(os.path.join(APP, PDF)).pages)} páginas")


if __name__ == "__main__":
    main()
