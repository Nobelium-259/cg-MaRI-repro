@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto check_torch
py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>&1
if errorlevel 1 goto use_python
py -3.11 -m venv .venv
if errorlevel 1 goto failed
goto check_torch
:use_python
python -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>&1
if errorlevel 1 goto no_python
python -m venv .venv
if errorlevel 1 goto failed
:check_torch
.venv\Scripts\python.exe -c "import torch, torchvision; print('Existing torch:', torch.__version__, 'torchvision:', torchvision.__version__, 'CUDA:', torch.cuda.is_available())"
if not errorlevel 1 goto install_interface
.venv\Scripts\python.exe -m pip install --cache-dir .cache\pip -r source\course_demo\requirements_torch.txt --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto failed
:install_interface
.venv\Scripts\python.exe -m pip install --cache-dir .cache\pip -r source\course_demo\requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe -c "import torch, torchvision, streamlit; print('torch:',torch.__version__,'torchvision:',torchvision.__version__,'streamlit:',streamlit.__version__,'CUDA:',torch.cuda.is_available())"
if errorlevel 1 goto failed
echo Setup finished. Existing environments were preserved. Download MaRI weights before live MaRI inference.
pause
exit /b 0
:no_python
echo Python 3.11 was not found. Install it or configure an existing Python 3.11 project environment.
:failed
echo Setup failed. Read the usage guide; do not delete an existing environment blindly.
pause
exit /b 1
