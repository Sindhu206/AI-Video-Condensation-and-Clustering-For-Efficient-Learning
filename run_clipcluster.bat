@echo off
title ClipCluster - AI Video Summarization and Scene Clustering
echo ======================================================================
echo           ClipCluster - AI Video Summarization & Clustering
echo ======================================================================
echo.
echo Starting backend server on http://127.0.0.1:5000 ...
echo Press Ctrl+C in this window to stop the server.
echo.

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py app.py
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        python app.py
    ) else (
        if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" (
            "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" app.py
        ) else (
            "%LOCALAPPDATA%\Python\bin\python.exe" app.py
        )
    )
)

pause
