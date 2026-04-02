@echo off
cd /d %~dp0
call venv\Scripts\activate
python monitoring.py
pause
