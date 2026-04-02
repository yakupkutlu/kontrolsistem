@echo off
echo ================================================
echo   Kontrol Sistemi - Kurulum
echo ================================================
echo.

cd /d "%~dp0"

echo [1/4] Eski sanal ortam temizleniyor...
if exist venv (
    rmdir /s /q venv
    echo Eski venv silindi.
)

echo [2/4] Sanal ortam olusturuluyor...
python -m venv venv
if errorlevel 1 (
    echo HATA: Python bulunamadi. Python 3.11 kurun ve PATH secenegini isaretleyin.
    pause
    exit /b 1
)

echo [3/4] Paketler yukleniyor...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo HATA: Paket yuklenemedi. Internet baglantisini kontrol edin.
    pause
    exit /b 1
)

echo [4/4] Kurulum tamamlandi!
echo.
echo Programi baslatmak icin proje.bat dosyasini kullanin.
echo.
pause