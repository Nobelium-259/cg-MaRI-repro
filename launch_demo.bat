@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
 echo Run setup_windows.bat first.
 pause
 exit /b 1
)
.venv\Scripts\python.exe -m streamlit run source\course_demo\comparison_app.py --server.address 127.0.0.1
pause
