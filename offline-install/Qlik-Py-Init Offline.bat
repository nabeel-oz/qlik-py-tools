@echo off
echo Setting up the Python virtual environment... & echo.
python -m venv "%~dp0\qlik-py-env"
echo.
echo Copying project files to the new directory... & echo.
xcopy /E /I "%~dp0\generated" "%~dp0\qlik-py-env\generated"
xcopy /E /I "%~dp0\core" "%~dp0\qlik-py-env\core"
xcopy /E /I "%~dp0\offline" "%~dp0\qlik-py-env\offline"
echo.
echo Activating the virtual environment... & echo.
cd /d "%~dp0\qlik-py-env\Scripts"
call activate
cd ..
echo.
echo Installing required packages... & echo.
pip install --no-index --find-links=offline grpcio grpcio-tools numpy scipy pandas cython
pip install --no-index --find-links=offline pystan==2.17
pip install --no-index --find-links=offline fbprophet
pip install --no-index --find-links=offline scikit-learn
pip install --no-index --find-links=offline setuptools wheel hdbscan
pip install --no-index --find-links=offline skater
echo.
echo Creating a new firewall rule for TCP port 50055... & echo.
netsh advfirewall firewall add rule name="Qlik PyTools" dir=in action=allow protocol=TCP localport=50055
echo.
echo All done. Run Qlik-Py-Start.bat to start the SSE Extension Service. & echo.
pause
