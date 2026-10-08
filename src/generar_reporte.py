"""Genera el informe del taller en normas APA 7.ª ed. (app_vercel/reporte.html) y su PDF.

Fuentes: reporte/contenido.md (texto), reporte/plantilla.html (portada y estilos APA),
el notebook ejecutado (tablas), app_vercel/informe/fig_XX.png (figuras, ver
exportar_informe.py) y artefactos/ (resultados).

Marcadores en contenido.md:
  {{FIG:n|Título|Nota}}        figura n del notebook (app_vercel/informe/fig_0n.png)
  {{IMG:archivo|Título|Nota}}  imagen de app_vercel/informe/
  {{TABLA:clave|Título|Nota}}  tabla generada desde los resultados
  {{TTEXTO:Título|Nota}}       título APA para la tabla markdown que sigue
Tablas y figuras se numeran en orden de aparición; se verifica que coincida con las
citas "Tabla N" / "Figura N" del texto.

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
TITULO = "Diseño de una arquitectura LSTM para la predicción de demanda energética"
FECHA = "7 de octubre de 2026"
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
    t = re.sub(r"<.*?>", "", t)
    t = re.sub(r"[^\w\s-]", "", t.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", "-", t).strip("-")


def tabla_html(cab, filas, numericas=(), envolver=False):
    th = "".join(f'<th class="{"n" if i in numericas else ""}">{c}</th>' for i, c in enumerate(cab))
    tr = "".join("<tr>" + "".join(f'<td class="{"n" if i in numericas else ""}">{v}</td>'
                                  for i, v in enumerate(f)) + "</tr>" for f in filas)
    return (f'<div class="tabla-wrap"><table class="datos{" envolver" if envolver else ""}">'
            f'<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>')


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
# tablas de resultados
# --------------------------------------------------------------------------
def construir_tablas(nb):
    T = {}
    res = pd.read_csv(os.path.join(RAIZ, "artefactos", "tabla_resultados.csv"))
    cfg = pd.read_json(os.path.join(RAIZ, "artefactos", "config_app.json"), typ="series")
    sel_arq, sel_n = cfg["arquitectura"], int(cfg["ventana"])
    filas = []
    for _, r in res.iterrows():
        ref = r.Modelo.startswith("Persistencia")
        nombre = "Persistencia" if ref else r.Modelo.replace(" · ", ". ")
        filas.append([nombre + (" ✓" if (r.Modelo == sel_arq and r.Ventana == sel_n) else ""),
                      "–" if ref else f"{int(r.Ventana)} h", num(r.MAE), num(r.MSE, 1), num(r.RMSE),
                      num(r.MAPE) + " %", num(r["R²"], 4), "–" if ref else int(r["Épocas"])])
    T["resultados"] = tabla_html(["Modelo", "Ventana", "MAE", "MSE", "RMSE", "MAPE", "R²", "Épocas"],
                                 filas, numericas=range(2, 8))

    ent = joblib.load(os.path.join(RAIZ, "artefactos", "entrenamiento.joblib"))["resultados"]
    nombres = {"A": "A. LSTM base", "B": "B. LSTM profunda", "C": "C. CNN-LSTM"}
    T["entrenamiento"] = tabla_html(
        ["Modelo", "Ventana", "Épocas ejecutadas", "Mejor época", "MSE de validación", "Tiempo (s)"],
        [[nombres[r["arq"]], f'{r["ventana"]} h', r["epocas_ejecutadas"], r["mejor_epoca"],
          num(r["loss_val_mejor"], 4), num(r["segundos"], 0)] for r in ent],
        numericas=range(2, 6))

    v = tabla_notebook(nb, "VENTANAS = [12, 24, 48]")
    v.columns = ["Ventana", "Entrenamiento", "Validación", "Prueba"]
    T["ventanas"] = tabla_html(list(v.columns),
                               [[(f"{x} h" if str(x).isdigit() else "Conjunto común"),
                                 num(a, 0), num(b, 0), num(c, 0)] for x, a, b, c in v.itertuples(index=False)],
                               numericas=(1, 2, 3))

    s = tabla_notebook(nb, "sobreaj = pd.DataFrame(filas)")
    T["sobreajuste"] = tabla_html(["Modelo", "Ventana", "Entrenamiento", "Validación", "Prueba", "Prueba / entrenamiento"],
                                  [[r[0].replace(" · ", ". "), f"{int(r[1])} h", num(r[2]), num(r[3]), num(r[4]), num(r[5])]
                                   for r in s.itertuples(index=False)], numericas=range(2, 6))

    e = tabla_notebook(nb, "def entrenar_variante")
    T["experimentos"] = tabla_html(["Variante", "Parámetros", "Épocas", "RMSE entrenamiento", "RMSE prueba", "R² prueba", "Brecha"],
                                   [[r[0].replace(" (referencia)", " (referencia)").replace("0.2", "0,2"),
                                     num(r[1], 0), int(r[2]), num(r[3]), num(r[4]), num(r[5], 4), num(r[6])]
                                    for r in e.itertuples(index=False)], numericas=range(1, 7), envolver=True)

    b = tabla_notebook(nb, "from sklearn.linear_model import Ridge")
    T["baselines"] = tabla_html(["Modelo", "MAE", "MSE", "RMSE", "MAPE", "R²"],
                                [[r[0], num(r[1]), num(r[2], 1), num(r[3]), num(r[4]) + " %", num(r[5], 4)]
                                 for r in b.itertuples(index=False)], numericas=range(1, 6))

    g = tabla_notebook(nb, "it_s1 =")
    T["errores"] = tabla_html(["Hora predicha", "MAE (MW)", "Horas en prueba"],
                              [[r[0].capitalize(), num(r[1]), num(r[2], 0)] for r in g.itertuples(index=False)],
                              numericas=(1, 2))
    return T


# --------------------------------------------------------------------------
# cuerpo APA
# --------------------------------------------------------------------------
PATRON = re.compile(
    r"<p>\{\{(FIG|IMG|TABLA):([^|}]*)\|([^|}]*)(?:\|(.*?))?\}\}</p>"
    r"|<p>\{\{TTEXTO:([^|}]*)(?:\|(.*?))?\}\}</p>\s*(<table>.*?</table>)", re.S)


def construir_cuerpo(tablas):
    texto = open(os.path.join(RAIZ, "reporte", "contenido.md"), encoding="utf-8").read()
    texto = (texto.replace("{{LINK_REPO}}", f"[{REPO}]({REPO})").replace("{{LINK_APP}}", f"[{URL_APP}]({URL_APP})")
             .replace("{{TITULO}}", TITULO))
    n = {"fig": 0, "tab": 0}

    def nota(t):
        return f'<p class="nota"><em>Nota.</em> {t}</p>' if t and t.strip() else ""

    def reemplazar(m):
        if m.group(1) in ("FIG", "IMG"):
            n["fig"] += 1
            ref, tit, no = m.group(2), m.group(3), m.group(4)
            src = f"informe/fig_{int(ref):02d}.png" if m.group(1) == "FIG" else f"informe/{ref}"
            return (f'<figure><p class="t-num">Figura {n["fig"]}</p><p class="t-tit">{tit}</p>'
                    f'<img src="{src}" alt="{html.escape(tit)}">{nota(no)}</figure>')
        n["tab"] += 1
        if m.group(1) == "TABLA":
            tit, no, cuerpo = m.group(3), m.group(4), tablas[m.group(2)]
        else:
            tit, no, cuerpo = m.group(5), m.group(6), f'<div class="tabla-wrap">{m.group(7)}</div>'
        return (f'<div class="tabla-apa"><p class="t-num">Tabla {n["tab"]}</p><p class="t-tit">{tit}</p>'
                f'{cuerpo}{nota(no)}</div>')

    paginas = {"resumen": [], "cuerpo": [], "referencias": []}
    indice = []
    for parte in re.split(r"(?m)^(?=## )", texto):
        if not parte.strip():
            continue
        primera, _, resto = parte.partition("\n")
        clase = (re.search(r"\{\.(\w+)\}", primera) or [None, None])[1]
        titulo = re.sub(r"\{\.\w+\}", "", primera).lstrip("# ").strip()
        h = md(f"## {titulo}\n{resto}")
        h = PATRON.sub(reemplazar, h)
        # ids para el índice en pantalla
        def con_id(mm):
            nivel, cont = mm.group(1), mm.group(2)
            sid = slug(cont)
            indice.append(f'<a class="{"n2" if nivel == "3" else ""}" href="#{sid}">{re.sub("<.*?>", "", cont)}</a>')
            return f'<h{nivel} id="{sid}">{cont}</h{nivel}>'
        h = re.sub(r"<h([23])>(.*?)</h\1>", con_id, h)
        if clase == "resumen":
            h = h.replace("<p><em>Palabras clave:</em>", '<p class="palabras"><em>Palabras clave:</em>')
        destino = clase if clase in ("resumen", "referencias") else "cuerpo"
        paginas[destino].append(f'<section class="{clase or ""}">{h}</section>')

    cuerpo = "".join(f'<div class="pagina {k}">{"".join(v)}</div>' for k, v in paginas.items() if v)
    return cuerpo, "".join(indice), n["fig"], n["tab"], texto


def verificar_citas(texto, n_fig, n_tab):
    """Las citas 'Tabla N'/'Figura N' del texto deben existir y seguir el orden de numeración."""
    tab = sorted({int(x) for x in re.findall(r"Tabla (\d+)", texto)})
    fig = sorted({int(x) for x in re.findall(r"Figuras? (\d+)", texto)} |
                 {int(x) for x in re.findall(r"Figuras \d+ y (\d+)", texto)})
    assert tab == list(range(1, n_tab + 1)), f"Tablas citadas {tab} vs {n_tab} tablas"
    assert fig == list(range(1, n_fig + 1)), f"Figuras citadas {fig} vs {n_fig} figuras"
    # cada tabla/figura debe citarse antes (o justo antes) de aparecer
    orden_marc = [m.group(0) for m in re.finditer(r"\{\{(TTEXTO|TABLA):", texto)]
    for k in range(1, n_tab + 1):
        pos_cita = min(m.start() for m in re.finditer(rf"Tabla {k}(?!\d)", texto))
        pos_marc = [m.start() for m in re.finditer(r"\{\{(TTEXTO|TABLA):", texto)][k - 1]
        assert pos_cita < pos_marc, f"Tabla {k} se cita después de aparecer"
    for k in range(1, n_fig + 1):
        pos_cita = min(m.start() for m in re.finditer(rf"Figuras? {k}(?!\d)|Figuras \d+ y {k}(?!\d)", texto))
        pos_marc = [m.start() for m in re.finditer(r"\{\{(FIG|IMG):", texto)][k - 1]
        assert pos_cita < pos_marc, f"Figura {k} se cita después de aparecer"
    return len(orden_marc)


# --------------------------------------------------------------------------
# servidor local + captura + PDF
# --------------------------------------------------------------------------
def servidor():
    sys.path.insert(0, APP)
    from servidor_local import Local  # noqa: E402

    class Silencioso(Local):
        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(Silencioso, directory=APP))
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
        encabezado = ('<div style="width:100%;font-family:\'Times New Roman\',Tinos,serif;font-size:12pt;'
                      'text-align:right;padding:0 1in 0 0;margin-top:.45in;color:#000">'
                      '<span class="pageNumber"></span></div>')
        doc.pdf(path=os.path.join(APP, PDF), format="Letter", print_background=True, prefer_css_page_size=True,
                display_header_footer=True, header_template=encabezado, footer_template="<span></span>")
        b.close()
    shutil.copy(os.path.join(APP, PDF), os.path.join(RAIZ, PDF))


def main():
    nb = nbformat.read(os.path.join(RAIZ, "Taller_LSTM.ipynb"), as_version=4)
    tablas = construir_tablas(nb)
    cuerpo, indice, n_fig, n_tab, texto = construir_cuerpo(tablas)
    verificar_citas(texto, n_fig, n_tab)
    plantilla = open(os.path.join(RAIZ, "reporte", "plantilla.html"), encoding="utf-8").read()
    salida = (plantilla.replace("{{CUERPO}}", cuerpo).replace("{{INDICE}}", indice)
              .replace("{{TITULO}}", TITULO).replace("{{REPO}}", REPO)
              .replace("{{APP}}", URL_APP).replace("{{FECHA}}", FECHA))
    assert "{{" not in salida, re.findall(r"\{\{.{0,40}", salida)[:3]
    with open(os.path.join(APP, "reporte.html"), "w", encoding="utf-8") as fh:
        fh.write(salida)
    print(f"reporte.html: {n_fig} figuras y {n_tab} tablas, numeración verificada contra las citas del texto")

    srv, base = servidor()
    try:
        capturar_y_pdf(base)
    finally:
        srv.shutdown()
    from pypdf import PdfReader
    print(f"{PDF}: {len(PdfReader(os.path.join(APP, PDF)).pages)} páginas (carta, APA)")


if __name__ == "__main__":
    main()
