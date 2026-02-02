#!/usr/bin/env bash
set -e  # si algo falla, el script se para

# ===== CONFIGURACIÓN =====
# Carpeta del proyecto -> la carpeta donde ESTÁS ahora (gesture-recognizer)
PROJECT_DIR="$(pwd)"

# Versión de mediapipe que quieres
MP_VERSION="0.10.18"

# ===== COMPROBAR python3.10 =====
if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
else
    echo "[ERROR] No se ha encontrado python3.10 en el sistema."
    echo "Instálalo, por ejemplo (si tu Ubuntu lo trae en los repos):"
    echo "  sudo apt update"
    echo "  sudo apt install python3.10 python3.10-venv python3.10-dev"
    echo "y vuelve a lanzar este script."
    exit 1
fi

# ===== PAQUETES DEL SISTEMA =====
echo "[*] Actualizando paquetes..."
sudo apt update
sudo apt upgrade -y

# ===== PROYECTO Y VENV =====
echo "[*] Usando carpeta de proyecto: $PROJECT_DIR"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    echo "[*] No se ha encontrado .venv, creando entorno virtual con $PYTHON_BIN (sin pip)..."
    # Tu python3.10 tiene roto ensurepip, así que creamos el venv sin pip
    "$PYTHON_BIN" -m venv --without-pip .venv
else
    echo "[*] Encontrado entorno virtual existente en $PROJECT_DIR/.venv"
fi

echo "[*] Activando entorno virtual..."
# shellcheck disable=SC1091
source .venv/bin/activate

# ===== INSTALAR PIP EN EL VENV (si no existe) =====
if ! python -m pip --version >/dev/null 2>&1; then
    echo "[*] pip no está en el venv, instalándolo..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python /tmp/get-pip.py
fi

echo "[*] Actualizando pip dentro del venv..."
python -m pip install --upgrade pip

echo "[*] Instalando Mediapipe y OpenCV dentro del venv..."
python -m pip install "mediapipe==${MP_VERSION}" opencv-python

echo
echo "[OK] Todo listo."
echo "Para usar tu proyecto luego:"
echo "  cd \"$PROJECT_DIR\""
echo "  source .venv/bin/activate"
echo "  python main.py   # o el script que quieras"
