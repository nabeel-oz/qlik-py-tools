@echo off
echo Setting up the Python virtual environment... & echo.
python -m venv "%~dp0\qlik-py-env"
echo.
echo Moving project files to the new directory... & echo.
move generated "%~dp0\qlik-py-env"
move core "%~dp0\qlik-py-env"
echo.
echo Activating the virtual environment... & echo.
cd /d "%~dp0\qlik-py-env\Scripts"
call activate
cd ..
echo.
echo Installing required packages... & echo.
python -m pip install --upgrade pip
pip install grpcio
pip install numpy
pip install scipy
pip install pandas
pip install fbprophet
pip install -U scikit-learn
pip install hdbscan
echo.
echo All done. Run Qlik-Py-Start.bat to start the SSE Extension Service. & echo.
pause
