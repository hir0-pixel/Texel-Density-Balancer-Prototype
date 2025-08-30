@echo off
:: Batch file to open CMD in C:\dev\VramGovernor as admin

:: Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process cmd -ArgumentList '/k cd /d C:\dev\VramGovernor' -Verb RunAs"
    exit /b
)

:: If already admin, just open CMD in target folder
cd /d C:\dev\VramGovernor
cmd
