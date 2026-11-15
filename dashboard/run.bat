@echo off
setlocal
rem ===================================================================
rem  Hillstrom Campaign Studio - bam dup file nay de chay ung dung.
rem  Lan dau: tu cai uv (neu chua co), tai Python 3.12, tao .venv va
rem  cai dung phien ban thu vien da khoa trong uv.lock (can Internet).
rem  Cac lan sau: mo ung dung ngay.
rem ===================================================================
cd /d "%~dp0"

where uv >nul 2>nul
if not errorlevel 1 goto run

if exist "%USERPROFILE%\.local\bin\uv.exe" goto addpath

echo [Hillstrom] Chua co uv tren may - dang cai dat, chi can lam 1 lan...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if not exist "%USERPROFILE%\.local\bin\uv.exe" goto fail_uv

:addpath
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

:run
echo [Hillstrom] Dang kiem tra moi truong - lan dau co the mat vai phut...
uv run --locked main.py
if errorlevel 1 goto fail_app
goto end

:fail_uv
echo.
echo [Hillstrom] Khong cai duoc uv. Kiem tra ket noi Internet roi chay lai,
echo             hoac cai thu cong theo huong dan: https://docs.astral.sh/uv/
pause
goto end

:fail_app
echo.
echo [Hillstrom] Ung dung dung lai do loi o tren. Gui noi dung cua so nay cho nhom ho tro.
pause

:end
endlocal
