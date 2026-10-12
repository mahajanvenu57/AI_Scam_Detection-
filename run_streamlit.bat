@echo off
title SCAMSHIELD AI - Streamlit Dashboard
echo ===================================================
echo Starting Streamlit Presentation Dashboard...
echo ===================================================
cd /d "%~dp0"
streamlit run app.py
pause
