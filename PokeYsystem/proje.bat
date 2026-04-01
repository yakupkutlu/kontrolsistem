@echo off
:: 1. Proje klasörüne git
cd /d "C:\Users\Acer\Desktop\projeler\ototrim\Kontrol_sistem"

:: 2. Sanal ortamı (myenv) aktif et
:: Not: myenv klasörünün Kontrol_sistem içinde olduğunu varsayıyorum
call myenv\Scripts\activate

:: 3. Programı çalıştır (Artık cv2'yi görecektir)
python monitoring.py

:: 4. Hata olursa pencere kapanmasın
pause