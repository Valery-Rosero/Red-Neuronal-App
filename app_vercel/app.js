"use strict";
/* Frontend de la app de predicción. La inferencia ocurre en /api/predict (función Python en Vercel). */

const API = "/api/predict";
const CAMPOS = ["demanda_mw", "temperatura_c", "humedad_pct", "viento_kmh",
                "radiacion_wm2", "precipitacion_mm", "precio_kwh"];
const DIAS = ["dom", "lun", "mar", "mié", "jue", "vie", "sáb"];
const INICIO_PRUEBA = "2026-09-13 12:00";
const POR_DEFECTO = "2026-11-12 17:00";

const estado = {
  ventana: 12, modo: "historico", filas: [], original: [], historico: null,
  realObjetivo: null, info: null, grafica: null,
};
const $ = (s) => document.querySelector(s);

/* ---------- fechas (en UTC para evitar desfases de zona horaria) ---------- */
const aMs = (s) => { const [d, h] = s.replace("T", " ").split(" "); const [Y, M, D] = d.split("-").map(Number);
  const [hh, mm] = (h || "00:00").split(":").map(Number); return Date.UTC(Y, M - 1, D, hh, mm || 0); };
const pad = (n) => String(n).padStart(2, "0");
const deMs = (ms) => { const t = new Date(ms);
  return `${t.getUTCFullYear()}-${pad(t.getUTCMonth() + 1)}-${pad(t.getUTCDate())} ${pad(t.getUTCHours())}:00`; };
const HORA = 3600e3;
const diaDe = (s) => DIAS[new Date(aMs(s)).getUTCDay()];
const finDeSemana = (s) => [0, 6].includes(new Date(aMs(s)).getUTCDay());
const tInput = () => $("#inp-t").value ? deMs(aMs($("#inp-t").value)) : null;
const fijarTInput = (s) => { $("#inp-t").value = s.replace(" ", "T"); };
const fmt = (v, d = 1) => Number(v).toLocaleString("es-CO", { minimumFractionDigits: d, maximumFractionDigits: d });

/* ---------- datos ---------- */
async function cargarHistorico() {
  if (estado.historico) return estado.historico;
  const txt = await (await fetch("historico.csv")).text();
  const [cab, ...lineas] = txt.trim().split(/\r?\n/);
  const cols = cab.split(",");
  const mapa = new Map();
  for (const l of lineas) {
    const v = l.split(","); const o = {};
    cols.forEach((c, i) => { o[c] = v[i]; });
    mapa.set(o.timestamp, o);
  }
  const claves = [...mapa.keys()];
  $("#inp-t").min = claves[estado.ventana - 1].replace(" ", "T");
  $("#inp-t").max = claves[claves.length - 2].replace(" ", "T");
  estado.historico = mapa;
  return mapa;
}

function ventanaHistorica(t) {
  const H = estado.historico, filas = [];
  for (let k = estado.ventana - 1; k >= 0; k--) {
    const ts = deMs(aMs(t) - k * HORA), o = H.get(ts);
    if (!o) return null;
    const f = { timestamp: ts, festivo: Number(o.festivo) || 0 };
    CAMPOS.forEach((c) => { f[c] = o[c] === "" || o[c] === undefined ? "" : o[c]; });
    filas.push(f);
  }
  const o = H.get(t);
  estado.realObjetivo = o && o.demanda_objetivo !== "" ? Number(o.demanda_objetivo) : null;
  return filas;
}

function ventanaVacia(t) {
  const filas = [];
  for (let k = estado.ventana - 1; k >= 0; k--) {
    const f = { timestamp: deMs(aMs(t) - k * HORA), festivo: 0 };
    CAMPOS.forEach((c) => { f[c] = ""; });
    filas.push(f);
  }
  return filas;
}

/* ---------- tabla ---------- */
function pintarTabla() {
  const tb = $("#tabla tbody");
  tb.innerHTML = "";
  estado.filas.forEach((f, i) => {
    const tr = document.createElement("tr");
    if (i === estado.filas.length - 1) tr.className = "ultima";
    tr.innerHTML = `<td class="fecha">${f.timestamp}${i === estado.filas.length - 1 ? " <small>(t)</small>" : ""}</td>
      <td class="dia">${diaDe(f.timestamp)}${finDeSemana(f.timestamp) ? " · fds" : ""}</td>` +
      CAMPOS.map((c) => `<td data-campo="${c}" class="${f[c] === "" ? "vacio-dato" : ""}">
        <input type="text" inputmode="decimal" data-i="${i}" data-c="${c}" value="${f[c]}"
          placeholder="falta" aria-label="${c} ${f.timestamp}"></td>`).join("") +
      `<td data-campo="festivo"><input type="checkbox" data-i="${i}" data-c="festivo" ${f.festivo ? "checked" : ""}
          aria-label="festivo ${f.timestamp}"></td>`;
    tb.appendChild(tr);
  });
}

$("#tabla").addEventListener("input", (e) => {
  const el = e.target, i = Number(el.dataset.i), c = el.dataset.c;
  if (c === undefined) return;
  estado.filas[i][c] = c === "festivo" ? (el.checked ? 1 : 0) : el.value.trim().replace(",", ".");
  const td = el.closest("td"); td.classList.remove("invalido");
  td.classList.toggle("vacio-dato", c !== "festivo" && el.value.trim() === "");
});

/* ---------- errores ---------- */
function mostrarErrores(errores) {
  document.querySelectorAll("td.invalido").forEach((td) => td.classList.remove("invalido"));
  const caja = $("#errores");
  if (!errores || !errores.length) { caja.classList.add("oculto"); return; }
  const filas = $("#tabla tbody").rows;
  errores.forEach((er) => {
    if (er.fila === null || er.fila === undefined || !filas[er.fila]) return;
    const td = filas[er.fila].querySelector(`td[data-campo="${er.campo}"]`);
    if (td) td.classList.add("invalido");
  });
  const items = errores.slice(0, 8).map((er) =>
    `<li>${er.fila !== null && er.fila !== undefined && estado.filas[er.fila] ? `<b>${estado.filas[er.fila].timestamp}</b> · ` : ""}${er.mensaje}</li>`).join("");
  caja.innerHTML = `<strong>No se puede predecir sin inventar valores.</strong> Corrige o reemplaza estos datos:
    <ul>${items}${errores.length > 8 ? `<li>… y ${errores.length - 8} más.</li>` : ""}</ul>`;
  caja.classList.remove("oculto");
}

/* ---------- predicción ---------- */
async function predecir(desplazar = true) {
  const btn = $("#btn-predecir");
  btn.disabled = true; btn.textContent = "Calculando…";
  try {
    const filas = estado.filas.map((f) => {
      const o = { timestamp: f.timestamp, festivo: f.festivo };
      CAMPOS.forEach((c) => { o[c] = f[c] === "" ? null : Number(f[c]); });
      return o;
    });
    const r = await fetch(API, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filas }) });
    const res = await r.json();
    if (!res.ok) { mostrarErrores(res.errores); return; }
    mostrarErrores([]);
    mostrarResultado(res);
    if (desplazar) $("#sec-resultado").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    mostrarErrores([{ fila: null, mensaje: `No se pudo contactar la API (${err.message}).` }]);
  } finally {
    btn.disabled = false; btn.textContent = "Predecir demanda de la hora siguiente";
  }
}

function sinCambios() {
  return JSON.stringify(estado.filas) === JSON.stringify(estado.original);
}

function mostrarResultado(res) {
  $("#resultado-vacio").classList.add("oculto");
  $("#resultado").classList.remove("oculto");
  const pred = res.prediccion_mw;
  $("#r-hora").textContent = res.timestamp_objetivo;
  $("#r-pred").textContent = fmt(pred);
  $("#r-rmse").textContent = estado.info ? fmt(estado.info.metricas_prueba.RMSE, 0) : "25";

  const real = estado.modo === "historico" ? estado.realObjetivo : null;
  const kReal = $("#kpi-real");
  if (real !== null) {
    kReal.classList.remove("oculto");
    $("#r-real").textContent = fmt(real);
    const e = pred - real;
    $("#r-error").textContent = `Error: ${e >= 0 ? "+" : ""}${fmt(e)} MW (${fmt(Math.abs(e) / real * 100)} %)` +
      (sinCambios() ? "" : " · entradas editadas");
  } else kReal.classList.add("oculto");

  const actual = Number(estado.filas[estado.filas.length - 1].demanda_mw);
  const d = pred - actual;
  $("#r-delta").textContent = `${d >= 0 ? "+" : ""}${fmt(d)}`;
  $("#r-delta-pct").textContent = `${d >= 0 ? "Sube" : "Baja"} ${fmt(Math.abs(d) / actual * 100)} % respecto a ${estado.filas.at(-1).timestamp.slice(11)}`;
  graficar(pred, real, res.timestamp_objetivo);
}

function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

function graficar(pred, real, tObj) {
  const etiquetas = [...estado.filas.map((f) => f.timestamp.slice(5)), tObj.slice(5) + " (t+1)"];
  const hist = [...estado.filas.map((f) => (f.demanda_mw === "" ? null : Number(f.demanda_mw))), null];
  const n = estado.filas.length;
  const lineaPred = Array(n + 1).fill(null); lineaPred[n - 1] = hist[n - 1]; lineaPred[n] = pred;
  const puntoReal = Array(n + 1).fill(null); if (real !== null) puntoReal[n] = real;
  const rmse = estado.info ? estado.info.metricas_prueba.RMSE : 25;
  const sup = Array(n + 1).fill(null), inf = Array(n + 1).fill(null);
  sup[n] = pred + rmse; inf[n] = pred - rmse;

  const ds = [
    { label: "Demanda observada (entrada)", data: hist, borderColor: css("--tinta"), backgroundColor: css("--tinta"),
      borderWidth: 2, pointRadius: 3, tension: .25 },
    { label: "Predicción", data: lineaPred, borderColor: css("--acento"), backgroundColor: css("--acento"),
      borderWidth: 2, borderDash: [6, 4], pointRadius: (c) => (c.dataIndex === n ? 7 : 0), pointStyle: "circle" },
    { label: "± RMSE", data: sup, borderColor: "transparent", backgroundColor: css("--acento"),
      pointStyle: "line", pointRadius: (c) => (c.dataIndex === n ? 10 : 0), pointBorderColor: css("--acento"),
      pointBorderWidth: 2, showLine: false },
    { label: "_inf", data: inf, borderColor: "transparent", pointStyle: "line",
      pointRadius: (c) => (c.dataIndex === n ? 10 : 0), pointBorderColor: css("--acento"), pointBorderWidth: 2, showLine: false },
  ];
  if (real !== null) ds.push({ label: "Demanda real (t+1)", data: puntoReal, borderColor: css("--naranja"),
    backgroundColor: css("--naranja"), pointRadius: 7, pointStyle: "rectRot", showLine: false });

  if (estado.grafica) estado.grafica.destroy();
  estado.grafica = new Chart($("#grafica"), {
    type: "line", data: { labels: etiquetas, datasets: ds },
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: {
        y: { title: { display: true, text: "MW", color: css("--tinta-3") }, grid: { color: css("--borde") },
             ticks: { color: css("--tinta-3") } },
        x: { grid: { display: false }, ticks: { color: css("--tinta-3"), maxRotation: 50, autoSkip: true } },
      },
      plugins: {
        legend: { labels: { color: css("--tinta-2"), usePointStyle: true, filter: (it) => !it.text.startsWith("_") } },
        tooltip: { filter: (it) => it.raw !== null && !it.dataset.label.startsWith("_") && !(it.datasetIndex === 1 && it.dataIndex < n),
          callbacks: { label: (c) => ` ${c.dataset.label}: ${fmt(c.raw)} MW` } },
      },
    },
  });

  const filasT = estado.filas.map((f) => `<tr><td>${f.timestamp}</td><td>${f.demanda_mw === "" ? "—" : fmt(f.demanda_mw)}</td><td>entrada</td></tr>`).join("");
  $("#tabla-grafica").innerHTML = `<thead><tr><th>Hora</th><th>Demanda (MW)</th><th>Tipo</th></tr></thead><tbody>${filasT}
    <tr><td>${tObj}</td><td><b>${fmt(pred)}</b></td><td>predicción</td></tr>
    ${real !== null ? `<tr><td>${tObj}</td><td>${fmt(real)}</td><td>real</td></tr>` : ""}</tbody>`;
}

/* ---------- acciones ---------- */
async function cargarDesdeHistorico(t, auto = false) {
  await cargarHistorico();
  const filas = ventanaHistorica(t);
  if (!filas) { mostrarErrores([{ fila: null, mensaje: `No hay ${estado.ventana} horas de histórico antes de ${t}.` }]); return; }
  fijarTInput(t);
  estado.filas = filas; estado.original = JSON.parse(JSON.stringify(filas));
  pintarTabla(); mostrarErrores([]);
  if (auto) predecir(auto === "desplazar");
}

function ventanaValida(t) {
  const f = ventanaHistorica(t);
  return f && estado.realObjetivo !== null && f.every((r) => CAMPOS.every((c) => r[c] !== ""));
}

async function aleatorio() {
  await cargarHistorico();
  const claves = [...estado.historico.keys()].filter((k) => k >= INICIO_PRUEBA);
  for (let intento = 0; intento < 200; intento++) {
    const t = claves[Math.floor(Math.random() * (claves.length - 1))];
    if (ventanaValida(t)) return cargarDesdeHistorico(t, "desplazar");
  }
}

function cambiarModo(modo) {
  estado.modo = modo;
  document.querySelectorAll(".tab").forEach((b) => {
    const on = b.dataset.modo === modo; b.classList.toggle("activo", on); b.setAttribute("aria-selected", on);
  });
  document.querySelectorAll("[data-solo]").forEach((el) => el.classList.toggle("oculto", el.dataset.solo !== modo));
  $("#ayuda-modo").textContent = modo === "historico"
    ? "Elige un instante del histórico (2025-2026). Puedes editar cualquier valor de la tabla antes de predecir."
    : "Escribe los valores de cada hora, o sube un CSV con columnas: timestamp, demanda_mw, temperatura_c, humedad_pct, viento_kmh, radiacion_wm2, precipitacion_mm, precio_kwh, festivo. Las horas deben ser consecutivas.";
}

function descargarCSV() {
  const cab = ["timestamp", ...CAMPOS, "festivo"];
  const txt = [cab.join(","), ...estado.filas.map((f) => cab.map((c) => f[c]).join(","))].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([txt], { type: "text/csv" }));
  a.download = `ventana_${estado.filas.at(-1).timestamp.replace(/[ :]/g, "-")}.csv`; a.click();
  URL.revokeObjectURL(a.href);
}

function subirCSV(archivo) {
  const lector = new FileReader();
  lector.onload = () => {
    const [cab, ...lineas] = lector.result.trim().split(/\r?\n/);
    const cols = cab.split(/[;,]/).map((c) => c.trim().toLowerCase());
    const faltan = ["timestamp", ...CAMPOS].filter((c) => !cols.includes(c));
    if (faltan.length) { mostrarErrores([{ fila: null, mensaje: `Faltan columnas en el CSV: ${faltan.join(", ")}.` }]); return; }
    const filas = lineas.filter((l) => l.trim()).map((l) => {
      const v = l.split(/[;,]/); const o = {};
      cols.forEach((c, i) => { o[c] = (v[i] ?? "").trim(); });
      const f = { timestamp: deMs(aMs(o.timestamp)), festivo: ["1", "true"].includes((o.festivo || "0").toLowerCase()) ? 1 : 0 };
      CAMPOS.forEach((c) => { f[c] = o[c] ?? ""; });
      return f;
    }).sort((a, b) => aMs(a.timestamp) - aMs(b.timestamp)).slice(-estado.ventana);
    if (filas.length !== estado.ventana) {
      mostrarErrores([{ fila: null, mensaje: `El CSV debe tener al menos ${estado.ventana} filas (tiene ${filas.length}).` }]); return;
    }
    estado.filas = filas; estado.original = JSON.parse(JSON.stringify(filas));
    fijarTInput(filas.at(-1).timestamp); pintarTabla(); mostrarErrores([]);
  };
  lector.readAsText(archivo);
}

function aplicarEscenario() {
  const dt = Number($("#rng-dt").value), esc = Number($("#rng-rad").value) / 100;
  estado.filas = JSON.parse(JSON.stringify(estado.original)).map((f, i) => {
    const g = { ...f, festivo: estado.filas[i].festivo };
    if (g.temperatura_c !== "") g.temperatura_c = (Number(g.temperatura_c) + dt).toFixed(2);
    if (g.radiacion_wm2 !== "") g.radiacion_wm2 = (Number(g.radiacion_wm2) * esc).toFixed(2);
    return g;
  });
  pintarTabla();
}

/* ---------- inicio ---------- */
async function iniciar() {
  try {
    const info = await (await fetch(API)).json();
    estado.info = info; estado.ventana = info.ventana;
    const m = info.metricas_prueba;
    $("#txt-ventana").textContent = info.ventana;
    $("#m-arq").textContent = `${info.arquitectura} (${info.parametros.toLocaleString("es-CO")} parámetros, ventana de ${info.ventana} h)`;
    $("#m-periodo").textContent = `${info.periodo_entrenamiento[0].slice(0, 10)} a ${info.periodo_entrenamiento[1].slice(0, 10)}`;
    $("#chips-modelo").innerHTML = [
      `Modelo <b>${info.arquitectura}</b>`, `Ventana <b>${info.ventana} h</b>`,
      `MAE <b>${fmt(m.MAE)} MW</b>`, `RMSE <b>${fmt(m.RMSE)} MW</b>`,
      `MAPE <b>${fmt(m.MAPE, 2)} %</b>`, `R² <b>${fmt(m["R²"], 3)}</b>`,
    ].map((t) => `<span class="chip">${t}</span>`).join("") + `<span class="chip">métricas en prueba (sept-dic 2026)</span>`;
  } catch {
    $("#chips-modelo").innerHTML = `<span class="chip">No se pudo leer /api/predict</span>`;
  }
  await cargarHistorico();
  cargarDesdeHistorico(POR_DEFECTO, true);
}

document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => cambiarModo(b.dataset.modo)));
$("#btn-cargar").addEventListener("click", () => tInput() && cargarDesdeHistorico(tInput()));
$("#btn-aleatorio").addEventListener("click", aleatorio);
$("#btn-plantilla").addEventListener("click", () => {
  const t = tInput() || POR_DEFECTO;
  estado.filas = ventanaVacia(t); estado.original = JSON.parse(JSON.stringify(estado.filas));
  pintarTabla(); mostrarErrores([]);
});
$("#inp-t").addEventListener("change", () => {
  if (estado.modo === "historico") cargarDesdeHistorico(tInput());
  else { const t = tInput(); estado.filas = estado.filas.map((f, i) => ({ ...f, timestamp: deMs(aMs(t) - (estado.ventana - 1 - i) * HORA) })); pintarTabla(); }
});
$("#inp-csv").addEventListener("change", (e) => { if (e.target.files[0]) subirCSV(e.target.files[0]); e.target.value = ""; });
$("#btn-descargar").addEventListener("click", descargarCSV);
$("#rng-dt").addEventListener("input", (e) => { $("#out-dt").textContent = (e.target.value > 0 ? "+" : "") + e.target.value; });
$("#rng-rad").addEventListener("input", (e) => { $("#out-rad").textContent = e.target.value; });
$("#btn-aplicar").addEventListener("click", aplicarEscenario);
$("#btn-festivo").addEventListener("click", () => { estado.filas.forEach((f) => { f.festivo = 1; }); pintarTabla(); });
$("#btn-restaurar").addEventListener("click", () => {
  estado.filas = JSON.parse(JSON.stringify(estado.original));
  $("#rng-dt").value = 0; $("#out-dt").textContent = "0"; $("#rng-rad").value = 100; $("#out-rad").textContent = "100";
  pintarTabla(); mostrarErrores([]);
});
$("#btn-predecir").addEventListener("click", () => predecir(true));
iniciar();
