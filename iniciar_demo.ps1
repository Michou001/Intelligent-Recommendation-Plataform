# Arranca la plataforma en esta laptop y la publica en internet con un túnel
# gratuito de Cloudflare. Uso, desde la carpeta del proyecto:
#
#     powershell -ExecutionPolicy Bypass -File .\iniciar_demo.ps1
#
# El túnel imprime una dirección https://<algo>.trycloudflare.com: esa es la que
# se abre en las demás máquinas o celulares. Ctrl+C cierra el túnel y el servidor.
# La dirección cambia cada vez que se ejecuta.

$ErrorActionPreference = "Stop"
$puerto = 8000
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "Falta cloudflared. Instálalo con: winget install Cloudflare.cloudflared" -ForegroundColor Red
    exit 1
}

if (-not $env:RECO_SMTP_USUARIO) {
    Write-Host "Aviso: falta RECO_SMTP_USUARIO. Los códigos de verificación NO se enviarán por correo;" -ForegroundColor Yellow
    Write-Host "       se mostrarán en esta consola (solo sirve para pruebas)." -ForegroundColor Yellow
}

$env:PORT = "$puerto"
$servidor = Start-Process -FilePath $python -ArgumentList "-m", "application.server" `
    -WorkingDirectory $PSScriptRoot -PassThru -NoNewWindow

try {
    # Esperar a que el servidor responda antes de abrir el túnel.
    $listo = $false
    for ($i = 0; $i -lt 60 -and -not $listo; $i++) {
        Start-Sleep -Seconds 1
        try {
            $listo = (Invoke-WebRequest "http://127.0.0.1:$puerto/health" -UseBasicParsing).StatusCode -eq 200
        } catch { }
    }
    if (-not $listo) { throw "El servidor no respondió en 60 s (¿el puerto $puerto está ocupado?)." }

    Write-Host "`nServidor listo. Abriendo el túnel; busca abajo la dirección https://...trycloudflare.com`n" -ForegroundColor Green
    cloudflared tunnel --url "http://127.0.0.1:$puerto"
}
finally {
    Stop-Process -Id $servidor.Id -ErrorAction SilentlyContinue
}
