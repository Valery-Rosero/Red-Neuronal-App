"""Genera la página "Resultados y análisis" de la app (app_vercel/informe.html) a partir
del notebook ejecutado: textos, tablas y gráficas salen tal cual del notebook.

Uso (desde la carpeta del taller):  python src/exportar_informe.py
"""
import base64
import html
import os
import re

import mistune
import nbformat

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(RAIZ, "app_vercel")
FIG_DIR = os.path.join(APP, "informe")

# Celdas de código cuyo texto impreso aporta al informe (se identifican por su contenido)
TEXTO_UTIL = ["auditoria = pd.DataFrame", "Verificación de que la regla R6", "df = df.reset_index",
              "Comprobaciones anti-fuga", "rmse_val = {}", "err = yr - p", "it_s1 =", "import exportar_modelo",
              "comp = pd.DataFrame", "from sklearn.metrics import"]
# Celdas cuyas tablas HTML no se muestran (vista previa cruda y resumen de Keras)
SIN_TABLA = ["crudo = pp.cargar_crudo", "import os, time, joblib"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")

md = mistune.create_markdown(plugins=["table", "strikethrough"], escape=False)


def slug(t):
    t = re.sub(r"[^\w\s-]", "", t.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", "-", t).strip("-")


def limpiar_texto(t):
    lineas = [l for l in ANSI.sub("", t).splitlines()
              if l.strip() and "WARNING" not in l and "oneDNN" not in l and not l.startswith("I0000")]
    return "\n".join(lineas)


def construir():
    nb = nbformat.read(os.path.join(RAIZ, "Taller_LSTM.ipynb"), as_version=4)
    os.makedirs(FIG_DIR, exist_ok=True)
    for f in os.listdir(FIG_DIR):
        os.remove(os.path.join(FIG_DIR, f))

    cuerpo, toc, n_fig = [], [], 0
    titulo_actual = ""
    for c in nb.cells[1:]:
        if c.cell_type == "markdown":
            src = c.source
            for linea in src.splitlines():
                m = re.match(r"^(##|###) (.+)$", linea)
                if m:
                    titulo_actual = m.group(2).strip()
                    if m.group(1) == "##":
                        toc.append((slug(titulo_actual), titulo_actual))
            h = md(src)
            # ids en los h2/h3 para el índice
            h = re.sub(r"<h([23])>(.*?)</h\1>",
                       lambda mm: f'<h{mm.group(1)} id="{slug(re.sub("<.*?>", "", mm.group(2)))}">{mm.group(2)}</h{mm.group(1)}>', h)
            cuerpo.append(f'<div class="md">{h}</div>')
            continue

        src = c.source
        texto = any(k in src for k in TEXTO_UTIL)
        tabla_ok = not any(k in src for k in SIN_TABLA)
        for o in c.get("outputs", []):
            d = o.get("data", {})
            if "image/png" in d:
                n_fig += 1
                nombre = f"fig_{n_fig:02d}.png"
                with open(os.path.join(FIG_DIR, nombre), "wb") as fh:
                    fh.write(base64.b64decode(d["image/png"]))
                primera = src.strip().splitlines()[0]
                alt = primera.lstrip("# ").strip() if primera.startswith("#") else titulo_actual
                cuerpo.append(f'<figure class="lamina"><img src="informe/{nombre}" loading="lazy" '
                              f'alt="{html.escape(alt)}"><figcaption>Figura {n_fig}. {html.escape(alt)}</figcaption></figure>')
            elif "text/html" in d and tabla_ok:
                cuerpo.append(f'<div class="lamina tabla-informe">{d["text/html"]}</div>')
            elif texto and o.get("output_type") == "stream":
                t = limpiar_texto(o.get("text", ""))
                if t:
                    cuerpo.append(f'<pre class="salida">{html.escape(t)}</pre>')
            elif texto and "text/plain" in d and "text/html" not in d and "image/png" not in d:
                t = limpiar_texto(d["text/plain"])
                if t and not t.startswith("<"):
                    cuerpo.append(f'<pre class="salida">{html.escape(t)}</pre>')

    toc_html = "".join(f'<a href="#{s}">{html.escape(re.sub(r"^\d+\. ", "", t))}</a>' for s, t in toc)
    plantilla = open(os.path.join(APP, "informe_plantilla.html"), encoding="utf-8").read()
    salida = plantilla.replace("{{TOC}}", toc_html).replace("{{CUERPO}}", "\n".join(cuerpo))
    with open(os.path.join(APP, "informe.html"), "w", encoding="utf-8") as fh:
        fh.write(salida)
    print(f"informe.html generado: {len(toc)} secciones, {n_fig} figuras.")


if __name__ == "__main__":
    construir()
