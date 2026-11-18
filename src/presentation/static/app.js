// Cliente web de la Cartelera Inteligente. No tiene lógica de recomendación:
// solo llama a la API pública y muestra lo que responde.
import { mostrarMetricas } from "./metricas.js";

const SEGUNDOS_TENDENCIAS = 10;

const POR_QUE = {
  popularity: "Es de lo más visto en <b>{genero}</b>, el género que elegiste. Todavía no tenemos tu historial, así que empezamos por la popularidad.",
  content: "Se parece a las películas que marcaste con <b>Me gusta</b> (géneros y etiquetas similares). Recomendación basada en contenido.",
  als: "A usuarios con gustos parecidos a los tuyos les encantó. Filtrado colaborativo (ALS) con las películas que te gustaron.",
  ncf: "La eligió una red neuronal entrenada con el historial de todos los usuarios.",
  parecidas: "A quienes les gustó una película que marcaste también les gustó esta (filtrado colaborativo, ALS).",
  genero: "Es de las más vistas de <b>{genero}</b> en MovieLens.",
  tendencia: "Está entre las más populares de los últimos minutos entre <b>todos los usuarios conectados</b>.",
  busqueda: "La encontraste con el buscador. Si la marcas con <b>Me gusta</b>, la tomaremos en cuenta para tus recomendaciones.",
};

const $ = (selector) => document.querySelector(selector);
let sesion = JSON.parse(sessionStorage.getItem("sesion") || "null");
const gustadas = new Set(); // ids con "me gusta"; el servidor los manda en /auth/yo
let generoPrincipal = "tu género favorito";
// Géneros del usuario como [clave, nombre en español], para las filas por género.
let misGeneros = [];
// Sin géneros elegidos se muestran estos, para que el inicio no quede vacío.
const GENEROS_DE_RESPALDO = [["Comedy", "Comedia"], ["Action", "Acción"]];
let ultimaRespuesta = null;
let temporizador = null;

function escapar(texto) {
  const div = document.createElement("div");
  div.textContent = texto ?? "";
  return div.innerHTML;
}

async function api(ruta, { metodo = "GET", cuerpo } = {}) {
  const encabezados = { "Content-Type": "application/json" };
  if (sesion) encabezados.Authorization = `Bearer ${sesion.token}`;
  const respuesta = await fetch(ruta, {
    method: metodo,
    headers: encabezados,
    body: cuerpo ? JSON.stringify(cuerpo) : undefined,
  });
  const datos = await respuesta.json().catch(() => ({}));
  // Un 401 por contraseña o código equivocado no cierra la sesión; solo uno
  // por token vencido o inválido.
  if (respuesta.status === 401 && sesion && datos.code === "sesion_invalida") {
    cerrarSesion("Tu sesión expiró. Vuelve a iniciar sesión.");
  }
  if (!respuesta.ok) {
    const fallo = new Error(datos.detail || datos.message || "No se pudo completar la solicitud.");
    fallo.code = datos.code;
    throw fallo;
  }
  return datos;
}

/* ---------------- Acceso ---------------- */

function mostrarPestana(crear) {
  $("#pestana-entrar").setAttribute("aria-selected", String(!crear));
  $("#pestana-crear").setAttribute("aria-selected", String(crear));
  $("#form-entrar").hidden = crear;
  $("#form-crear").hidden = !crear;
  $("#form-correo").hidden = true;
}

// Correo de la cuenta que espera el código de confirmación.
let correoPorConfirmar = null;

function mostrarConfirmacionCorreo(correo, aviso = "") {
  correoPorConfirmar = correo;
  $("#correo-destino").textContent = correo;
  $("#aviso-correo").textContent = aviso;
  $("#form-entrar").hidden = true;
  $("#form-crear").hidden = true;
  $("#form-correo").hidden = false;
  $("#vista-acceso").classList.remove("amplio");
  $("#form-correo [name=codigo]").focus();
}

async function cargarGeneros() {
  const contenedor = $("#generos");
  const generos = await api("/generos");
  contenedor.innerHTML = generos
    .map((g) => `<button type="button" aria-pressed="false" data-clave="${g.clave}">${escapar(g.nombre)}</button>`)
    .join("");
  contenedor.addEventListener("click", (evento) => {
    const chip = evento.target.closest("button");
    if (!chip) return;
    const elegidos = contenedor.querySelectorAll('[aria-pressed="true"]').length;
    const activo = chip.getAttribute("aria-pressed") === "true";
    if (!activo && elegidos >= 3) return;
    chip.setAttribute("aria-pressed", String(!activo));
  });
}

// Misma política que el servidor (presentation/schemas.py): 8+ caracteres,
// una mayúscula, un número y un símbolo, sin espacios ni guiones.
const REGLAS_CONTRASENA = {
  largo: { cumple: (v) => v.length >= 8, falta: "al menos 8 caracteres" },
  mayuscula: { cumple: (v) => /\p{Lu}/u.test(v), falta: "una mayúscula" },
  numero: { cumple: (v) => /\p{Nd}/u.test(v), falta: "un número" },
  simbolo: { cumple: (v) => /[^\p{L}\p{N}\s-]/u.test(v), falta: "un símbolo" },
  prohibidos: { cumple: (v) => !/[\s-]/.test(v), falta: null }, // tiene su propio mensaje
};

function faltasContrasena(valor) {
  return Object.values(REGLAS_CONTRASENA)
    .filter((r) => r.falta && !r.cumple(valor))
    .map((r) => r.falta);
}

function evaluarContrasena(valor) {
  document.querySelectorAll("#reglas-contrasena li").forEach((li) => {
    li.dataset.estado = valor ? String(REGLAS_CONTRASENA[li.dataset.regla].cumple(valor)) : "";
  });
  evaluarConfirmacion();
}

function evaluarConfirmacion() {
  const contrasena = $("#form-crear [name=contrasena]").value;
  const confirmar = $("#form-crear [name=confirmar]").value;
  const aviso = $("#coincide");
  aviso.classList.remove("aviso-mal", "aviso-bien");
  if (!confirmar) {
    aviso.textContent = "Escríbela otra vez.";
  } else if (confirmar === contrasena) {
    aviso.textContent = "Las contraseñas coinciden.";
    aviso.classList.add("aviso-bien");
  } else {
    aviso.textContent = "Las contraseñas no coinciden.";
    aviso.classList.add("aviso-mal");
  }
}

async function enviarFormulario(evento, accion) {
  evento.preventDefault();
  const formulario = evento.target;
  const boton = formulario.querySelector("button[type=submit]");
  const error = formulario.querySelector(".paso:not([hidden]) .error") || formulario.querySelector(":scope > .error");
  error.textContent = "";
  boton.disabled = true;
  try {
    await accion(Object.fromEntries(new FormData(formulario)));
  } catch (fallo) {
    error.textContent = fallo.message;
  } finally {
    boton.disabled = false;
  }
}

async function guardarSesionYEntrar(nueva) {
  sesion = nueva;
  sessionStorage.setItem("sesion", JSON.stringify(sesion));
  $("#form-entrar").reset();
  $("#form-crear").reset();
  $("#form-correo").reset();
  $("#form-correo").hidden = true;
  mostrarPasoPeliculas(false);
  mostrarPasoCodigo(false);
  await entrar();
}

// Token del primer paso cuando la cuenta tiene verificación en dos pasos.
let tokenDobleFactor = null;

function mostrarPasoCodigo(visible) {
  if (!visible) tokenDobleFactor = null;
  $("#entrar-datos").hidden = visible;
  $("#entrar-codigo").hidden = !visible;
  $("#volver-entrar").hidden = !visible;
  if (visible) $("#form-entrar [name=codigo]").focus();
}

async function iniciarSesion(datos) {
  if (tokenDobleFactor) {
    const cuerpo = { token_doble_factor: tokenDobleFactor, codigo: datos.codigo.trim() };
    await guardarSesionYEntrar(await api("/auth/sesion/codigo", { metodo: "POST", cuerpo }));
    return;
  }
  let respuesta;
  try {
    respuesta = await api("/auth/sesion", {
      metodo: "POST",
      cuerpo: { correo: datos.correo, contrasena: datos.contrasena },
    });
  } catch (fallo) {
    if (fallo.code !== "correo_sin_verificar") throw fallo;
    mostrarConfirmacionCorreo(datos.correo.trim().toLowerCase(), "Tu correo aún no está confirmado.");
    return;
  }
  if (respuesta.doble_factor_requerido) {
    tokenDobleFactor = respuesta.token_doble_factor;
    mostrarPasoCodigo(true);
  } else {
    await guardarSesionYEntrar(respuesta);
  }
}

function cerrarSesion(mensaje = "") {
  sesion = null;
  gustadas.clear();
  sessionStorage.clear();
  clearInterval(temporizador);
  $("#vista-app").hidden = true;
  $("#vista-acceso").hidden = false;
  mostrarPestana(false);
  mostrarPasoCodigo(false);
  $("#form-entrar .error").textContent = mensaje;
}

function mostrarPasoPeliculas(visible) {
  $("#paso-datos").hidden = visible;
  $("#paso-peliculas").hidden = !visible;
  $("#vista-acceso").classList.toggle("amplio", visible);
}

function generosElegidos() {
  return [...document.querySelectorAll('#generos [aria-pressed="true"]')].map((b) => b.dataset.clave);
}

async function pasoPeliculas() {
  const error = $("#paso-datos .error");
  const datos = new FormData($("#form-crear"));
  const faltas = faltasContrasena(datos.get("contrasena"));
  if (!/^[A-Za-z0-9_]{3,32}$/.test(datos.get("usuario"))) {
    error.textContent = "El usuario debe tener de 3 a 32 letras, números o guion bajo.";
  } else if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(datos.get("correo").trim())) {
    error.textContent = "Escribe un correo válido, como nombre@dominio.com.";
  } else if (/[\s-]/.test(datos.get("contrasena"))) {
    error.textContent = "La contraseña no puede llevar espacios ni guiones (-).";
  } else if (faltas.length) {
    const lista = faltas.length > 1 ? `${faltas.slice(0, -1).join(", ")} y ${faltas.at(-1)}` : faltas[0];
    error.textContent = `A tu contraseña le falta ${lista}.`;
  } else if (datos.get("confirmar") !== datos.get("contrasena")) {
    error.textContent = "Las contraseñas no coinciden.";
  } else if (!generosElegidos().length) {
    error.textContent = "Elige al menos un género.";
  } else {
    error.textContent = "";
    const consulta = generosElegidos().map((g) => `generos=${encodeURIComponent(g)}`).join("&");
    const peliculas = await api(`/peliculas/para-registro?${consulta}`);
    $("#peliculas-registro").replaceChildren(...peliculas.map(opcionDePelicula));
    $("#contador-elegidas").textContent = "0 elegidas";
    mostrarPasoPeliculas(true);
  }
}

function opcionDePelicula(item) {
  const boton = document.createElement("button");
  boton.type = "button";
  boton.dataset.item = item.item_id;
  boton.setAttribute("aria-pressed", "false");
  boton.innerHTML = `<div class="poster"></div><p>${escapar(item.titulo)}</p>`;
  poster(boton.querySelector(".poster"), item);
  boton.addEventListener("click", () => {
    boton.setAttribute("aria-pressed", String(boton.getAttribute("aria-pressed") !== "true"));
    const total = document.querySelectorAll('#peliculas-registro [aria-pressed="true"]').length;
    $("#contador-elegidas").textContent = `${total} elegida${total === 1 ? "" : "s"}`;
  });
  return boton;
}

/* ---------------- Verificación en dos pasos ---------------- */

function mostrarSeccionSeguridad(id) {
  ["df-inactivo", "df-alta", "df-respaldo", "df-activo"].forEach((s) => { $(`#${s}`).hidden = s !== id; });
  $("#seguridad .error").textContent = "";
}

async function abrirSeguridad() {
  // Nada de lo escrito antes (contraseña, códigos) debe quedar para la siguiente vez.
  ["#df-codigo-alta", "#df-codigo-baja", "#baja-contrasena", "#baja-codigo"].forEach((s) => { $(s).value = ""; });
  $("#seguridad .zona-peligro").open = false;
  const yo = await api("/auth/yo");
  mostrarSeccionSeguridad(yo.doble_factor_activo ? "df-activo" : "df-inactivo");
  $("#seguridad").showModal();
}

async function accionSeguridad(accion) {
  try {
    await accion();
  } catch (fallo) {
    $("#seguridad .error").textContent = fallo.message;
  }
}

/* ---------------- Inicio ---------------- */

async function entrar() {
  $("#vista-acceso").hidden = true;
  $("#vista-app").hidden = false;
  $("#nombre-usuario").textContent = sesion.usuario;
  const yo = await api("/auth/yo");
  if (yo.generos.length) generoPrincipal = yo.generos[0];
  misGeneros = yo.generos_clave.map((clave, i) => [clave, yo.generos[i]]);
  yo.me_gusta.forEach((id) => gustadas.add(id));
  mostrarVista("inicio");
  temporizador = setInterval(() => {
    if (!$("#inicio").hidden) cargarTendencias();
    else mostrarMetricas();
  }, SEGUNDOS_TENDENCIAS * 1000);
}

function mostrarVista(nombre) {
  document.querySelectorAll(".barra nav button").forEach((boton) => {
    if (boton.dataset.vista === nombre) boton.setAttribute("aria-current", "page");
    else boton.removeAttribute("aria-current");
  });
  $("#inicio").hidden = nombre !== "inicio";
  $("#metricas").hidden = nombre !== "metricas";
  if (nombre === "inicio") {
    cargarRecomendaciones();
    cargarTendencias();
    cargarPorqueTeGusto();
    cargarFilasGenero();
  } else {
    mostrarMetricas();
  }
}

function esqueletos(contenedor) {
  contenedor.innerHTML = '<div class="esqueleto"></div>'.repeat(7);
}

async function cargarRecomendaciones() {
  const contenedor = $("#para-ti");
  if (!ultimaRespuesta) esqueletos(contenedor);
  try {
    ultimaRespuesta = await api(`/recommendations/${sesion.user_id}?k=20`);
    pintarFase(ultimaRespuesta);
    pintarFila(contenedor, ultimaRespuesta.items, ultimaRespuesta.strategy_id);
    pintarDestacada(ultimaRespuesta.items[0], ultimaRespuesta.strategy_id);
    $("#estrategia").textContent = `${ultimaRespuesta.items.length} películas · ${ultimaRespuesta.latency_ms} ms`;
  } catch (fallo) {
    contenedor.innerHTML = `<p class="error">${escapar(fallo.message)}</p>`;
  }
}

// Portada: la primera recomendación, en grande y con su póster de fondo.
function pintarDestacada(item, origen) {
  const seccion = $("#destacada");
  seccion.hidden = !item;
  if (!item) return;
  const sinopsis = item.sinopsis || "Sin sinopsis disponible.";
  seccion.innerHTML = `
    ${item.poster ? `<img class="fondo" src="${escapar(item.poster)}" alt="">` : ""}
    <div class="poster"></div>
    <div class="destacada-texto">
      <span class="etiqueta">Destacada para ti</span>
      <h2 id="destacada-titulo">${escapar(item.titulo)}</h2>
      <p class="nota">${[item.anio, item.generos.join(", ")].filter(Boolean).join(" · ")}</p>
      <p class="sinopsis">${escapar(sinopsis.length > 280 ? `${sinopsis.slice(0, 277)}…` : sinopsis)}</p>
      <div class="botones">
        <button type="button" class="primario ver-ficha">Ver ficha</button>
        <button type="button" class="secundario corazon-texto" data-item="${item.item_id}"></button>
      </div>
    </div>`;
  poster(seccion.querySelector(".poster"), item);
  seccion.querySelector(".ver-ficha").addEventListener("click", () => abrirFicha(item, origen));
  seccion.querySelector(".corazon-texto").addEventListener("click", () => alternarMeGusta(item));
  actualizarCorazones(item.item_id);
}

// "Porque te gustó X": parecidas a la última película que marcó.
async function cargarPorqueTeGusto() {
  const seccion = $("#fila-porque");
  const ultima = [...gustadas].at(-1);
  if (ultima === undefined) {
    seccion.hidden = true;
    return;
  }
  try {
    const respuesta = await api(`/peliculas/${ultima}/parecidas?k=12`);
    seccion.hidden = !respuesta.items.length;
    $("#porque-titulo").textContent = `Porque te gustó ${respuesta.base?.titulo ?? "una película"}`;
    pintarFila($("#porque"), respuesta.items, "parecidas");
  } catch {
    seccion.hidden = true;
  }
}

// Una fila con lo más visto de cada género del usuario.
async function cargarFilasGenero() {
  const contenedor = $("#filas-genero");
  if (contenedor.childElementCount) return; // no cambian durante la sesión
  const generos = misGeneros.length ? misGeneros : GENEROS_DE_RESPALDO;
  for (const [clave, nombre] of generos) {
    const seccion = document.createElement("section");
    seccion.className = "fila";
    seccion.innerHTML = `
      <div class="fila-titulo"><h2>Lo más visto en ${escapar(nombre)}</h2></div>
      <div class="carrusel"></div>`;
    contenedor.append(seccion);
    try {
      const items = await api(`/peliculas/populares?genero=${encodeURIComponent(clave)}&k=16`);
      pintarFila(seccion.querySelector(".carrusel"), items, "genero");
    } catch {
      seccion.remove();
    }
  }
}

async function cargarTendencias() {
  const contenedor = $("#tendencias");
  try {
    const respuesta = await api("/tendencias?k=10");
    pintarFila(contenedor, respuesta.items, "tendencia", true);
    const nota = $("#tendencias-nota");
    nota.classList.toggle("en-vivo", respuesta.en_vivo);
    nota.textContent = respuesta.en_vivo
      ? `En vivo · actividad de los últimos ${respuesta.ventana_minutos} minutos`
      : "Sin actividad reciente: se muestra la popularidad histórica de MovieLens";
  } catch (fallo) {
    contenedor.innerHTML = `<p class="error">${escapar(fallo.message)}</p>`;
  }
}

function pintarFase(respuesta) {
  const elegidas = gustadas.size === 1
    ? "la película que te gustó"
    : `las ${gustadas.size} películas que te gustaron`;
  const textos = {
    arranque_en_frio: [
      "Arranque en frío",
      "Todavía no te conocemos",
      `Te mostramos lo más popular de ${escapar(generoPrincipal)}. Marca con ♥ cualquier película y tus recomendaciones se personalizan al instante.`,
    ],
    calentamiento: [
      "Calentamiento",
      "Ya sabemos algo de ti",
      `Calculamos tu perfil con ${elegidas}, con el mismo modelo colaborativo (ALS) que usan los demás usuarios, sin esperar a reentrenarlo.`,
    ],
    personalizado: [
      "Personalizado",
      "Recomendaciones hechas para ti",
      "Filtrado colaborativo: comparamos tu historial con el de usuarios de gustos parecidos.",
    ],
  };
  const [etiqueta, titulo, detalle] = textos[respuesta.fase];
  $("#fase").innerHTML = `<span class="etiqueta">${etiqueta}</span><div><strong>${titulo}</strong><p class="nota">${detalle}</p></div>`;
}

function pintarFila(contenedor, items, origen, conPuesto = false) {
  if (!items.length) {
    contenedor.innerHTML = '<p class="nota">No hay películas para mostrar todavía.</p>';
    return;
  }
  contenedor.replaceChildren(...items.map((item, i) => tarjeta(item, origen, conPuesto ? i + 1 : null)));
  if (items.some((item) => item.poster)) $("#aviso-tmdb").hidden = false;
}

function poster(elemento, item) {
  const sinImagen = `<span>${escapar(item.titulo)}</span>`;
  elemento.style.setProperty("--tono", (item.item_id * 47) % 360);
  elemento.innerHTML = item.poster ? `<img src="${escapar(item.poster)}" alt="" loading="lazy">` : sinImagen;
  // Sin internet el póster de TMDB no carga: se vuelve al cartel con el título.
  elemento.querySelector("img")?.addEventListener("error", () => { elemento.innerHTML = sinImagen; });
}

function tarjeta(item, origen, puesto) {
  const elemento = document.createElement("article");
  elemento.className = "tarjeta";
  elemento.innerHTML = `
    <button type="button" aria-label="Ver ficha de ${escapar(item.titulo)}">
      <div class="poster"></div>
      <h3>${escapar(item.titulo)}</h3>
      <p class="nota">${[item.anio, item.generos.slice(0, 2).join(", ")].filter(Boolean).join(" · ")}</p>
    </button>
    ${puesto ? `<span class="puesto">#${puesto}</span>` : ""}
    <button type="button" class="corazon" data-item="${item.item_id}" aria-label="Me gusta: ${escapar(item.titulo)}" aria-pressed="${gustadas.has(item.item_id)}">♥</button>`;
  poster(elemento.querySelector(".poster"), item);
  elemento.querySelector("button").addEventListener("click", () => abrirFicha(item, origen));
  elemento.querySelector(".corazon").addEventListener("click", () => alternarMeGusta(item));
  return elemento;
}

/* ---------------- Buscador ---------------- */

async function buscar(texto) {
  const buscando = texto.length >= 2;
  $("#filas").hidden = buscando;
  $("#resultados").hidden = !buscando;
  if (!buscando) return;
  if ($("#inicio").hidden) mostrarVista("inicio");
  const items = await api(`/peliculas/buscar?q=${encodeURIComponent(texto)}&k=24`);
  if ($("#buscar").value.trim() !== texto) return; // ya escribió otra cosa
  $("#resultados-titulo").textContent = items.length
    ? `Resultados para “${texto}”`
    : `No encontramos películas con “${texto}”`;
  pintarFila($("#resultados-lista"), items, "busqueda");
}

/* ---------------- Ficha e interacciones ---------------- */

function registrar(tipo, item) {
  return api("/interactions", {
    metodo: "POST",
    cuerpo: { user_id: sesion.user_id, event_type: tipo, item_id: item.item_id },
  });
}

function abrirFicha(item, origen) {
  poster($("#ficha-poster"), item);
  $("#ficha-titulo").textContent = item.titulo;
  $("#ficha-datos").textContent = [item.anio, item.generos.join(", ")].filter(Boolean).join(" · ");
  $("#ficha-sinopsis").textContent = item.sinopsis || "Sin sinopsis disponible.";
  $("#ficha-porque").innerHTML = `<b>¿Por qué te la recomendamos?</b> ${
    (POR_QUE[origen] || POR_QUE.popularity).replace("{genero}", escapar(generoPrincipal))}`;
  const boton = $("#ficha-me-gusta");
  boton.dataset.item = item.item_id;
  boton.onclick = () => alternarMeGusta(item);
  actualizarCorazones(item.item_id);
  $("#ficha").showModal();
  registrar("clic", item).catch(() => {});
}

// Marca o quita el "me gusta". Quitarlo también es un evento del historial, así
// que las recomendaciones dejan de tomar en cuenta esa película.
async function alternarMeGusta(item) {
  const yaLeGusta = gustadas.has(item.item_id);
  if (yaLeGusta) gustadas.delete(item.item_id);
  else gustadas.add(item.item_id);
  actualizarCorazones(item.item_id);
  await registrar(yaLeGusta ? "ya_no_me_gusta" : "me_gusta", item);
  // El historial se guarda de forma asíncrona (cola + worker): se espera un
  // momento antes de pedir de nuevo las recomendaciones.
  setTimeout(() => { cargarRecomendaciones(); cargarTendencias(); cargarPorqueTeGusto(); }, 800);
}

function actualizarCorazones(itemId) {
  const leGusta = gustadas.has(itemId);
  document.querySelectorAll(`.corazon[data-item="${itemId}"]`).forEach((c) => {
    c.setAttribute("aria-pressed", String(leGusta));
  });
  document.querySelectorAll(`.corazon-texto[data-item="${itemId}"]`).forEach((b) => {
    b.textContent = leGusta ? "♥ Te gusta · Quitar" : "♥ Me gusta";
    b.setAttribute("aria-pressed", String(leGusta));
  });
}

/* ---------------- Arranque ---------------- */

$("#pestana-entrar").addEventListener("click", () => mostrarPestana(false));
$("#pestana-crear").addEventListener("click", () => mostrarPestana(true));
$("#form-entrar").addEventListener("submit", (e) => enviarFormulario(e, iniciarSesion));
$("#volver-entrar").addEventListener("click", () => mostrarPasoCodigo(false));
$("#form-crear").addEventListener("submit", (e) => enviarFormulario(e, async (datos) => {
  const peliculas = [...document.querySelectorAll('#peliculas-registro [aria-pressed="true"]')]
    .map((b) => Number(b.dataset.item));
  delete datos.confirmar; // solo se revisa en la página; al servidor va una vez
  const cuerpo = { ...datos, generos: generosElegidos(), sexo: datos.sexo || null, peliculas };
  const pendiente = await api("/auth/registro", { metodo: "POST", cuerpo });
  mostrarPasoPeliculas(false);
  $("#form-crear").reset();
  mostrarConfirmacionCorreo(pendiente.correo);
}));
$("#form-correo").addEventListener("submit", (e) => enviarFormulario(e, async (datos) => {
  const cuerpo = { correo: correoPorConfirmar, codigo: datos.codigo.trim() };
  await guardarSesionYEntrar(await api("/auth/correo/confirmar", { metodo: "POST", cuerpo }));
}));
$("#reenviar-codigo").addEventListener("click", async () => {
  try {
    await api("/auth/correo/reenviar", { metodo: "POST", cuerpo: { correo: correoPorConfirmar } });
    $("#aviso-correo").textContent = "Listo. Si no llega en un minuto, revisa spam o vuelve a pedirlo.";
  } catch (fallo) {
    $("#form-correo .error").textContent = fallo.message;
  }
});
$("#siguiente").addEventListener("click", () => pasoPeliculas().catch((f) => { $("#paso-datos .error").textContent = f.message; }));
$("#atras").addEventListener("click", () => mostrarPasoPeliculas(false));
$("#form-crear [name=contrasena]").addEventListener("input", (e) => evaluarContrasena(e.target.value));
$("#form-crear [name=confirmar]").addEventListener("input", evaluarConfirmacion);
// reset() no dispara "input": se limpia la lista de requisitos al terminar.
$("#form-crear").addEventListener("reset", () => setTimeout(() => evaluarContrasena(""), 0));
$("#salir").addEventListener("click", () => cerrarSesion());
let esperaBusqueda;
$("#buscar").addEventListener("input", (e) => {
  // Espera a que deje de teclear 300 ms para no consultar con cada letra.
  clearTimeout(esperaBusqueda);
  esperaBusqueda = setTimeout(() => buscar(e.target.value.trim()), 300);
});
document.querySelectorAll(".barra nav button").forEach((boton) => {
  boton.addEventListener("click", () => mostrarVista(boton.dataset.vista));
});
$("#ficha .cerrar").addEventListener("click", () => $("#ficha").close());
$("#seguridad .cerrar").addEventListener("click", () => $("#seguridad").close());
$("#abrir-seguridad").addEventListener("click", abrirSeguridad);
$("#df-activar").addEventListener("click", () => accionSeguridad(async () => {
  const alta = await api("/auth/doble-factor", { metodo: "POST" });
  $("#df-qr").innerHTML = alta.qr_svg; // SVG generado por el servidor (segno)
  $("#df-secreto").textContent = alta.secreto;
  $("#df-codigo-alta").value = "";
  mostrarSeccionSeguridad("df-alta");
}));
$("#df-confirmar").addEventListener("click", () => accionSeguridad(async () => {
  const cuerpo = { codigo: $("#df-codigo-alta").value.trim() };
  const { codigos } = await api("/auth/doble-factor/confirmar", { metodo: "POST", cuerpo });
  $("#df-codigos").innerHTML = codigos.map((c) => `<li>${escapar(c)}</li>`).join("");
  mostrarSeccionSeguridad("df-respaldo");
}));
$("#df-listo").addEventListener("click", () => mostrarSeccionSeguridad("df-activo"));
$("#eliminar-cuenta").addEventListener("click", () => accionSeguridad(async () => {
  if (!confirm("¿Seguro que quieres eliminar tu cuenta? No se puede deshacer.")) return;
  const cuerpo = { contrasena: $("#baja-contrasena").value, codigo: $("#baja-codigo").value.trim() };
  await api("/auth/cuenta/eliminar", { metodo: "POST", cuerpo });
  $("#seguridad").close();
  cerrarSesion("Tu cuenta se eliminó.");
}));
$("#df-desactivar").addEventListener("click", () => accionSeguridad(async () => {
  const cuerpo = { codigo: $("#df-codigo-baja").value.trim() };
  await api("/auth/doble-factor/desactivar", { metodo: "POST", cuerpo });
  $("#df-codigo-baja").value = "";
  mostrarSeccionSeguridad("df-inactivo");
}));

cargarGeneros();
if (sesion) entrar().catch(() => { if (sesion) cerrarSesion(); });
else $("#vista-acceso").hidden = false;
