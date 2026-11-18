# Entrega final: aplicación web

La entrega final construye la aplicación que usa una persona sobre la arquitectura en capas del Milestone 3. No se reescribió el núcleo: las estrategias, el Feature Store, la cola y los repositorios son los mismos. Lo nuevo se conecta por las mismas interfaces.

## Qué se agregó

| Requisito del profesor | Cómo se resolvió | Archivos |
|---|---|---|
| Presentación para autenticar | Inicio de sesión solo con correo y registro con correo único; sesión con JWT; verificación en dos pasos opcional con TOTP | `application/autenticacion.py`, `application/doble_factor.py`, `persistence/sql_repositorio_cuentas.py`, `static/` |
| Medidas de seguridad | Ver la sección de seguridad | `application/autenticacion.py`, `application/api.py` |
| Usuario de arranque en frío | Registro con películas iniciales y fold-in de ALS; fases arranque en frío → calentamiento → personalizado | `application/recommendation_controller.py`, `ai/strategies/collaborative_filtering_strategy.py` |
| Registro más completo y resultados iniciales más precisos | Registro en dos pasos (datos, sexo opcional y 1 a 3 géneros; luego marcar películas). Precisión inicial 5.3 veces mayor, medida offline | `static/app.js`, `benchmarks/evaluar_arranque_en_frio.py` |
| Popularidad entre hombres y mujeres | Top por sexo con las interacciones de los usuarios que lo indicaron | `application/tendencias.py`, `static/metricas.js` |
| Top del momento por popularidad | Ventana deslizante de 30 minutos con las interacciones de todos | `application/tendencias.py` |
| Gráficas de las métricas | Vista de métricas: evaluación offline y estado en vivo | `application/metricas.py`, `static/metricas.js` |
| Interfaz bonita, en español | Carteles, ficha de película, títulos y géneros en español | `application/catalogo_es.py`, `data/pipeline/enriquecer_tmdb.py`, `static/` |
| Buscador de películas | Búsqueda por título original o en español, sin importar acentos, ordenada por popularidad | `application/buscador.py`, `static/app.js` |
| 2 o 3 máquinas a la vez | Un servidor; las demás máquinas solo abren el navegador | `application/server.py` |

## Cómo se comporta un usuario nuevo

El problema del arranque en frío: ALS necesita el historial de un usuario para calcularle su vector latente, y un usuario recién registrado no tiene ninguno.

**Registro en dos pasos.** Primero usuario, contraseña, sexo (opcional) y de 1 a 3 géneros. Después se muestran las 16 películas *más vistas* de cada género y el usuario marca las que vio y le gustaron. Se muestran las más vistas, y no las mejor calificadas, porque la gente reconoce más títulos: en la simulación marcan 9.8 en promedio contra 5.8. Las elegidas se guardan en el perfil (SQL), así que están disponibles desde la primera petición sin esperar a la cola de eventos.

**Fases** (campo `fase` de la respuesta):

1. **Arranque en frío.** No marcó ninguna película: lo más popular de su primer género.
2. **Calentamiento.** Con las películas del registro y sus "me gusta" se calcula su vector latente con **fold-in de ALS**: el mismo paso de mínimos cuadrados que ALS hace para cada usuario al entrenar, con los factores de las películas fijos:

   `u = (YᵀY + Yₛᵀ(c − 1)Yₛ + λI)⁻¹ · Yₛᵀc`

   Yₛ son los factores de las películas que le gustaron, c = 4.5 es la confianza de un "me gusta" (al entrenar la confianza es la calificación) y λ es la regularización del modelo. Es una sola resolución de un sistema de 64×64; no hay que reentrenar.
3. **Personalizado.** Cuando el pipeline offline se vuelve a correr, el usuario ya tiene su vector entrenado como cualquier usuario de MovieLens.

Las películas marcadas se excluyen de las siguientes recomendaciones.

## Evaluación del arranque en frío

`python benchmarks/evaluar_arranque_en_frio.py` usa la misma división temporal de la entrega 3 (`artifacts/evaluation`). Cada uno de los 592 usuarios evaluables de MovieLens se trata como si se registrara: elige sus 3 géneros más frecuentes y marca, de las 48 películas que le mostraría el registro, las que calificó bien **antes** del corte. Con solo eso se le recomiendan 10 y se comparan con lo que calificó bien **después** del corte. No hay fuga de datos.

| Estrategia para un usuario nuevo | precision@10 | recall@10 | NDCG@10 | hit@10 |
|---|---|---|---|---|
| Popularidad global (como en la entrega 3) | 0.0100 | 0.0071 | 0.0194 | 8.5 % |
| Popularidad del género elegido | 0.0159 | 0.0120 | 0.0193 | 12.7 % |
| Contenido con las películas elegidas | 0.0078 | 0.0101 | 0.0111 | 7.6 % |
| **ALS fold-in con las películas elegidas** | **0.0529** | **0.0514** | **0.0666** | **34.8 %** |

Para comparar: ALS con todo el historial (usuarios de MovieLens) obtiene 0.0865 de precision@10 en `evaluation_report.json`. Con unas 10 películas elegidas, un usuario nuevo ya alcanza el 61 % de esa precisión.

Una primera versión del calentamiento usaba contenido (promedio de embeddings). La evaluación mostró que era peor que la simple popularidad del género, y por eso se cambió a fold-in de ALS.

## Popularidad por sexo

MovieLens `ml-latest-small` no trae datos demográficos (su README: *"No demographic information is included"*), así que no se puede calcular lo más popular entre hombres y mujeres con el histórico. La plataforma lo obtiene de sus propios usuarios: el registro pide el sexo como dato **opcional**, explicando para qué se usa, y la vista de métricas muestra el top de los últimos minutos para mujeres y para hombres. Solo se guarda en el perfil y solo se muestra agregado. Para tener el dato histórico habría que cambiar a MovieLens 1M (sí trae sexo y edad) y reentrenar todo; además ese dataset no trae el id de TMDB, así que se perderían los pósters.

## Seguridad

- Contraseñas guardadas como **hash scrypt con sal aleatoria** (parámetros recomendados por OWASP). Nunca en claro.
- **Política de contraseñas** (pedida por el equipo en la revisión): al menos 8 caracteres, una mayúscula, un número y un símbolo, sin espacios ni guiones. La valida el servidor (`faltas_de_contrasena` en `presentation/schemas.py`, error 400 en español que dice qué falta) y la página la marca mientras se escribe. En el registro la contraseña se escribe dos veces y no avanza si no coinciden.
- Comparación de hashes en **tiempo constante** (`hmac.compare_digest`). Si el usuario no existe se verifica contra un hash de relleno, para que el tiempo de respuesta no revele qué usuarios existen.
- **Bloqueo temporal** tras 5 intentos fallidos (60 s).
- Sesión con **JWT firmado (HS256) que expira** en 60 minutos. El secreto se lee de `RECO_JWT_SECRETO`; si no existe, se genera uno al arrancar.
- **Autorización:** un usuario solo puede pedir sus propias recomendaciones y registrar eventos a su nombre (401 sin token, 403 con el token de otro).
- **Encabezados de seguridad:** `Content-Security-Policy` (solo scripts propios y Chart.js), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy`.
- Validación de entradas con Pydantic y errores en español sin detalles internos (ya venía del Milestone 3).
- **Correo único:** el registro pide correo, se guarda en minúsculas y la base de datos impide repetirlo (índice único). Se inicia sesión **solo con el correo**: el nombre de usuario aparece en pantalla y el correo no, así que para intentar entrar a una cuenta ajena primero hay que conocer su correo. 
- **Correo confirmado con código:** al registrarse, la cuenta queda sin activar y se envía un código de 6 dígitos al correo (`application/correo.py`). Hasta escribirlo no se puede entrar (403 `correo_sin_verificar`; al intentarlo se manda otro código). El código se guarda como hash, sirve 10 minutos, los equivocados cuentan para el bloqueo y se reenvía máximo uno por minuto. Una cuenta sin confirmar no le quita el correo a su dueño: si alguien se registra con él, la cuenta sin confirmar se descarta.
  - **Envío real:** por SMTP con TLS. Con Gmail: activar la verificación en dos pasos de la cuenta, crear una *contraseña de aplicación* y definir `RECO_SMTP_USUARIO` (el correo) y `RECO_SMTP_CONTRASENA` (esa contraseña) antes de arrancar. Opcionales: `RECO_SMTP_HOST` (por defecto `smtp.gmail.com`), `RECO_SMTP_PUERTO` (587), `RECO_SMTP_REMITENTE`.
  - **Sin SMTP** (desarrollo y pruebas): no se envía nada; el mensaje se anota en `artifacts/logs/correos_sin_enviar.log` y se muestra en la consola del servidor. `iniciar_demo.ps1` avisa si falta la configuración.
- **Verificación en dos pasos (TOTP, RFC 6238), opcional:** desde el botón *Seguridad* se escanea un QR con Google Authenticator, Microsoft Authenticator o similar, y se confirma con el primer código. Desde entonces, la contraseña sola no basta: el primer paso devuelve un token que solo sirve para el segundo paso y dura 5 minutos (el campo `typ` del JWT impide usarlo como sesión).
  - Se aceptan el código actual, el anterior y el siguiente (±30 s) por diferencias de reloj, pero **un código no se acepta dos veces**.
  - **8 códigos de respaldo** de un solo uso, guardados como hash SHA-256, por si se pierde el celular.
  - Los códigos equivocados cuentan para el mismo bloqueo por intentos.
  - Desactivarla también pide un código.
  - *Limitación:* el secreto TOTP se guarda en la base sin cifrar; en producción se cifraría con una llave fuera de la base.
- **Aviso de privacidad** (`static/privacidad.html`), enlazado desde el registro y el pie de página: qué datos se piden, para qué, servicios externos, derechos ARCO y la Ley Federal de Protección de Datos Personales en Posesión de los Particulares (DOF 20-03-2025).
- **Eliminar mi cuenta** (derecho de cancelación): en *Seguridad*, con la contraseña y, si aplica, el código. Se borran la cuenta y los datos del perfil (sexo, géneros, películas elegidas); las interacciones quedan solo con el número interno. Ruta `POST /auth/cuenta/eliminar`.
- **Imagen Docker sin datos personales:** el `Dockerfile` copia el catálogo en español y los reportes que usa la vista de métricas, y al construir borra la tabla `cuentas` y los perfiles de quienes se registraron en la demo (id mayor a 610).
- **Varios procesos o réplicas:** con `WORKERS > 1`, `application/server.py` genera un solo secreto de sesión y los procesos lo heredan; antes cada proceso tenía el suyo y el token fallaba en los demás (se detectó al probar el perfil completo de Docker Compose: 4 de 20 peticiones pasaban). En Kubernetes el secreto viene del Secret `recommendation-jwt` (ver `k8s/kustomization.yaml`).
- **Migración automática:** al arrancar, si la base de datos tiene la tabla `cuentas` de antes, se le agregan las columnas nuevas (`persistence/__init__.py`), sin perder las cuentas existentes.

## Cómo correrlo

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python -m application.server
```

Abre `http://127.0.0.1:8000/ui/`.

### Demo en internet desde la laptop (túnel de Cloudflare)

Para la presentación se publica la app con un túnel gratuito de Cloudflare: la laptop sigue siendo el servidor (misma base de datos, misma velocidad, no se duerme) y cualquier dispositivo con internet la abre por HTTPS, sin estar en la misma red.

1. Una sola vez: `winget install Cloudflare.cloudflared` y abrir una terminal nueva.
2. Desde la carpeta del proyecto: `powershell -ExecutionPolicy Bypass -File .\iniciar_demo.ps1`.
3. El script arranca el servidor, espera a que responda y abre el túnel. Entre los mensajes aparece la dirección `https://<palabras>.trycloudflare.com`: esa se comparte (por WhatsApp o un QR).
4. `Ctrl+C` cierra el túnel y el servidor.

La dirección cambia cada vez que se ejecuta, así que se arranca justo antes de presentar. La laptop debe estar conectada a la corriente y sin suspensión. Se descartó un hosting gratuito (Render) porque su plan sin costo apaga la app tras 15 minutos sin visitas y borra la base de datos al reiniciar.

### Demo en la misma red local (sin internet)

1. Todas las máquinas en la misma red. El WiFi de la universidad suele aislar a los dispositivos entre sí; lo más seguro es el **hotspot de un celular**.
2. En la máquina servidor, arranca con `python -m application.server`. Al iniciar imprime la dirección que deben abrir las demás, por ejemplo `http://192.168.1.20:8000/ui/`.
3. La primera vez, Windows pregunta si permite a Python aceptar conexiones: elige **redes privadas**. Si no pregunta, abre el puerto 8000 en el firewall.
4. Cada máquina crea su propia cuenta. Lo que una marca con "me gusta" aparece en **Lo más popular ahora** de las demás (se actualiza cada 10 s), y la vista de métricas muestra los usuarios activos y las interacciones por minuto de todas.

Las demás máquinas no instalan nada. Se deja un solo proceso (`WORKERS=1`): las tendencias y los intentos de inicio de sesión viven en memoria del servidor.

### Pósters y títulos en español (TMDB)

`links.csv` de MovieLens trae el id de TMDB de cada película. Para descargar títulos en español, sinopsis y pósters:

1. Crea una cuenta gratuita en themoviedb.org y copia el **token de acceso de lectura** (Ajustes → API).
2. Corre una sola vez (tarda unos minutos):

```powershell
$env:TMDB_TOKEN = "tu_token"
.\.venv\Scripts\python.exe -m data.pipeline.enriquecer_tmdb
```

Genera `datasets/peliculas_es.csv`. La sinopsis se toma en español de México, si no existe en español de España y, como último recurso, en inglés. Volver a correrlo solo consulta las películas que faltan o que se quedaron sin sinopsis. Sin ese archivo la aplicación funciona igual: usa el título de MovieLens, traduce los géneros y dibuja un cartel de color en lugar del póster. Los pósters se cargan desde `image.tmdb.org`, así que para verlos la demo necesita internet. Los términos de TMDB piden el aviso de atribución que aparece en el pie de página.

### Cuenta para mostrar la fase personalizada

Los usuarios de MovieLens no tienen contraseña ni correo. Para entrar como uno de ellos (con ese correo) y comparar con un usuario nuevo:

```bash
python -m application.cuenta_demo --usuario demo --correo demo@correo.com --id 1
```

## Despliegues probados (16 de noviembre de 2026)

- **Imagen Docker:** construida y probada (salud, búsqueda en español, registro, inicio de sesión con correo, aviso de privacidad, sin cuentas dentro).
- **Docker Compose, perfil completo** (`docker compose --profile full up -d api-full worker`): API con 2 procesos, Redis, RabbitMQ, MongoDB y el worker. Un "me gusta" pasó por RabbitMQ al worker, se guardó en MongoDB y cambió las recomendaciones.
- **Kubernetes** (Docker Desktop, v1.36): 2 réplicas de la API y Redis en marcha; dentro del clúster respondió el Service, se registró un usuario, recibió recomendaciones y un token de una réplica sirvió en la otra. Al borrar un pod, Kubernetes creó otro. *No probado:* el autoescalado, porque el clúster local no tiene metrics-server. Las cuentas (SQLite) y las tendencias son por pod; el Service usa afinidad por cliente.

## Pruebas

`python -m pytest`: 365 pruebas (292 del Milestone 3 más 73 nuevas). Pasan 364 y se omite 1, la del modelo neuronal, si `torch` no está instalado. Las nuevas están en `tests/application/test_autenticacion.py`, `test_doble_factor.py`, `test_verificacion_correo.py`, `test_tendencias.py`, `test_catalogo_es.py`, `test_buscador.py`, `test_fases_usuario_nuevo.py`, y en `tests/test_system_e2e.py` (servidor real: sin token 401, token ajeno 403, usuario nuevo en arranque en frío que mueve las tendencias, registro con películas que entra directo a calentamiento).

Las pruebas del Milestone 3 omiten el token (fixture `_sin_autenticacion` en `conftest.py`) porque prueban la recomendación, no el inicio de sesión. Las que llevan la marca `autenticacion` sí lo exigen.

## Patrones y decisiones nuevas

- **Observer:** `InteractionIngestionService` avisa a sus observadores de cada interacción; el contador de tendencias se suscribe sin que la ingesta sepa qué hace con ella.
- **Segregación de interfaces:** `IEstrategiaConPerfil` extiende `IRecommendationStrategy` solo para las estrategias que saben armar un perfil en vivo (ALS con fold-in y contenido); las demás no cargan con ese método.
- **Repositorio:** `IRepositorioCuentas` con su adaptador SQL, igual que los repositorios del Milestone 3. La regla de importación entre capas sigue verificada por `test_architecture_imports.py`.
- **Caché:** la llave ahora incluye cuántas películas vio el usuario y su género preferido. Sin eso, un usuario nuevo seguiría viendo la película que acaba de marcar, y las consultas de popularidad por género del registro devolvían siempre la misma lista.
- **Interfaz sin caché vieja:** los archivos de `/ui` se sirven con `Cache-Control: no-cache`, así el navegador siempre revisa si cambiaron (si no, recibe un 304).

## Inicio y "me gusta"

- **Portada destacada:** la primera recomendación en grande, con su póster borroso de fondo, sinopsis y botones *Ver ficha* y *Me gusta*.
- **Filas:** *Para ti*, *Porque te gustó …* (parecidas a la última película marcada: `GET /peliculas/{id}/parecidas`, fold-in de ALS con una sola película), *Lo más popular ahora* y *Lo más visto en …* por cada género del usuario (`GET /peliculas/populares`).
- **Quitar "me gusta":** el corazón y el botón de la ficha alternan. Quitarlo registra el evento `ya_no_me_gusta`; `peliculas_gustadas` recorre el historial en orden, así que la película deja de contar para las recomendaciones (también las elegidas en el registro). La llave de caché lleva una huella de las películas vistas, porque quitar una y marcar otra deja la misma cantidad.

