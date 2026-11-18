// Vista de métricas: evaluación offline (reports/evaluation_report.json) y
// estado del servicio en vivo. Las gráficas usan Chart.js.

const COLORES = ["#f2b134", "#5aa9e6", "#4cc38a", "#ef5b6b"];
const METRICAS = { "precision@10": "Precisión", "recall@10": "Exhaustividad (recall)", "ndcg@10": "NDCG" };
const SEGMENTOS = { low: "Poco historial", medium: "Historial medio", high: "Mucho historial" };
const graficas = {};

function dibujar(id, tipo, etiquetas, series) {
  const datos = { labels: etiquetas, datasets: series };
  if (graficas[id]) {
    graficas[id].data = datos;
    graficas[id].update("none");
    return;
  }
  graficas[id] = new Chart(document.getElementById(id), {
    type: tipo,
    data: datos,
    options: {
      responsive: true,
      aspectRatio: id === "g-actividad" ? 4 : 1.6,
      plugins: { legend: { labels: { color: "#e9eaee" } } },
      scales: {
        x: { ticks: { color: "#9aa1ad" }, grid: { color: "#2e333d" } },
        y: { beginAtZero: true, ticks: { color: "#9aa1ad", precision: 2 }, grid: { color: "#2e333d" } },
      },
    },
  });
}

function escapar(texto) {
  const div = document.createElement("div");
  div.textContent = texto ?? "";
  return div.innerHTML;
}

function pintarPorSexo(porSexo) {
  const columna = (sexo, titulo) => {
    const items = porSexo[sexo] || [];
    const lista = items.length
      ? `<ol>${items.map((i) => `<li>${escapar(i.titulo)}<span>${i.puntos} pts</span></li>`).join("")}</ol>`
      : '<p class="nota">Aún sin actividad.</p>';
    return `<div><h3>${titulo}</h3>${lista}</div>`;
  };
  document.getElementById("por-sexo").innerHTML = columna("mujer", "Mujeres") + columna("hombre", "Hombres");
}

export async function mostrarMetricas() {
  const nota = document.getElementById("metricas-nota");
  const respuesta = await fetch("/metricas");
  const { evaluacion, servicio, arranque_en_frio: arranque, por_sexo: porSexo } = await respuesta.json();
  pintarPorSexo(porSexo);

  const recientes = servicio.interacciones_por_minuto.reduce((a, b) => a + b, 0);
  document.getElementById("kpi-estrategia").textContent = servicio.estrategia_activa.toUpperCase();
  document.getElementById("kpi-latencia").textContent =
    servicio.latencia_p95_ms === null ? "Aún sin peticiones" : `${servicio.latencia_p95_ms} ms`;
  document.getElementById("kpi-usuarios").textContent = servicio.usuarios_activos;
  document.getElementById("kpi-interacciones").textContent = recientes;

  if (typeof Chart === "undefined") {
    nota.textContent = "No se pudo cargar Chart.js (se necesita internet para las gráficas).";
    return;
  }

  if (evaluacion) {
    const nombres = evaluacion.estrategias.map((e) => e.nombre);
    dibujar("g-calidad", "bar", nombres, Object.entries(METRICAS).map(([clave, etiqueta], i) => ({
      label: etiqueta,
      data: evaluacion.estrategias.map((e) => e.metricas[clave]),
      backgroundColor: COLORES[i],
    })));
    dibujar("g-segmentos", "bar", Object.values(SEGMENTOS), evaluacion.estrategias.map((e, i) => ({
      label: e.nombre,
      data: Object.keys(SEGMENTOS).map((s) => e.por_segmento[s] ?? 0),
      backgroundColor: COLORES[i % COLORES.length],
    })));
    nota.textContent = `Evaluación offline con ${evaluacion.usuarios_evaluados} usuarios de MovieLens, ranking de ${evaluacion.k} películas.`;
  }

  if (arranque) {
    dibujar("g-arranque", "bar", arranque.estrategias.map((e) => e.nombre), [
      { label: "Precisión@10", data: arranque.estrategias.map((e) => e["precision@10"]), backgroundColor: COLORES[0] },
      { label: "Aciertos (hit@10)", data: arranque.estrategias.map((e) => e["hit@10"]), backgroundColor: COLORES[1] },
    ]);
    document.getElementById("arranque-nota").textContent =
      `Simulación con ${arranque.usuarios} usuarios de MovieLens que se registran: eligen 3 géneros y marcan, ` +
      `de ${arranque.peliculas_mostradas_en_registro} películas, las que les gustaban (promedio ${arranque.promedio_elegidas}).`;
  }

  const minutos = servicio.interacciones_por_minuto.length;
  dibujar(
    "g-actividad",
    "line",
    servicio.interacciones_por_minuto.map((_, i) => (i === minutos - 1 ? "ahora" : `-${minutos - 1 - i} min`)),
    [{
      label: "Interacciones",
      data: servicio.interacciones_por_minuto,
      borderColor: COLORES[0],
      backgroundColor: "rgba(242, 177, 52, .15)",
      fill: true,
      tension: 0.3,
    }],
  );
}
