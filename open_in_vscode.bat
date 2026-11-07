@echo off
title Open ClipCluster in VS Code
echo Opening ClipCluster project in VS Code...
code "%~dp0"
if %ERRORLEVEL% NEQ 0 (
    echo VS Code 'code' command not in PATH.
    echo Please open VS Code, go to File ^> Open Folder... and select:
    echo %~dp0
    pause
)
