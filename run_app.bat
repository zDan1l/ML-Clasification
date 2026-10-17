@echo off
REM =============================================================
REM  Launcher Streamlit - Diabetes Classifier (Kelompok 2)
REM  Klik dua kali file ini, atau jalankan:  run_app.bat
REM  Selalu memakai Streamlit dari .venv (bukan Python global).
REM =============================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\streamlit.exe" (
    echo.
    echo [ERROR] Virtual environment belum siap.
    echo Jalankan dulu:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo  Menjalankan Streamlit... browser akan terbuka otomatis.
echo  Jika tidak, buka manual:  http://localhost:8501
echo  Hentikan server dengan Ctrl+C di jendela ini.
echo.

".venv\Scripts\streamlit.exe" run app.py --server.port 8501
pause
