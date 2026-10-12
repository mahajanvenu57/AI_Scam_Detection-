@echo off
title SCAMSHIELD AI - Cybersecurity Web Portal
echo ===================================================
echo Starting SCAMSHIELD AI (MCARP31)...
echo ===================================================
cd /d "%~dp0"
echo Opening web portal in browser...
start "" http://127.0.0.1:5000
python flask_app.py
pause
