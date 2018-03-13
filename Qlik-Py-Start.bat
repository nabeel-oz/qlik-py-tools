@echo off
cd "%~dp0\qlik-py-env\Scripts"
call activate
cd ..\core
python __main__.py
pause